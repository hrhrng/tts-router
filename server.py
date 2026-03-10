"""
Qwen3-TTS API Server
====================
DashScope-compatible + OpenAI-compatible TTS API, powered by MLX on Apple Silicon.

Supported endpoints:
  - POST /api/v1/services/aigc/multimodal-generation/generation  (DashScope 兼容)
  - POST /v1/audio/speech                                        (OpenAI 兼容)
  - GET  /v1/audio/voices                                        (音色列表)
  - GET  /health                                                 (健康检查)
  - POST /v1/audio/references/upload                             (上传参考音频)
  - POST /v1/audio/references/from-url                           (从 URL 下载音频作为参考)
  - GET  /v1/audio/references                                    (列出参考音频)
  - DELETE /v1/audio/references/{ref_id}                         (删除参考音频)
  - POST /v1/audio/references/{ref_id}/trim                       (裁剪参考音频)
  - POST /v1/audio/clone                                         (声音复刻生成)

Usage:
    uv run tts-router serve [--port 8091] [--host 0.0.0.0]
    python server.py  # fallback: uses default model from HF cache

Example (DashScope style):
    curl -X POST http://localhost:8091/api/v1/services/aigc/multimodal-generation/generation \\
      -H "Content-Type: application/json" \\
      -d '{
        "model": "qwen3-tts-instruct-flash",
        "input": {
          "text": "你好，世界",
          "voice": "Vivian",
          "language_type": "Chinese",
          "instructions": "用开心的语气"
        }
      }' --output output.wav

Example (OpenAI style):
    curl -X POST http://localhost:8091/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"input": "Hello world", "voice": "Vivian"}' --output output.wav
"""

import os
import io
import gc
import json
import time
import uuid
import shutil
import base64
import asyncio
import tempfile
import warnings
import subprocess

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import uvicorn
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import Response, JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field
from typing import Optional

from backends import get_backend

from extractors import create_extractor_chain, get_extractor_names

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "api")
REFERENCES_DIR = os.path.join(BASE_DIR, "outputs", "references")
REFERENCES_INDEX = os.path.join(REFERENCES_DIR, "index.json")

# Default HuggingFace repo IDs — used when running `python server.py` directly.
# When launched via `tts-router serve`, these are overridden by configure().
DEFAULT_MODEL_REPO = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
DEFAULT_CLONE_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(REFERENCES_DIR, exist_ok=True)

# ── Speaker / Language mapping ─────────────────────────────────────────────────
SPEAKER_MAP = {
    "English":  ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
    "Chinese":  ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
    "Japanese": ["Ono_Anna"],
    "Korean":   ["Sohee"],
}
ALL_VOICES = sorted({v for voices in SPEAKER_MAP.values() for v in voices})

# DashScope language_type → lang_code mapping
LANG_TYPE_MAP = {
    "auto": None, "chinese": "zh", "english": "en",
    "japanese": "ja", "korean": "ko", "german": "de",
    "french": "fr", "russian": "ru", "portuguese": "pt",
    "spanish": "es", "italian": "it",
}

# Supported audio output formats
FORMAT_MIME = {
    "wav": "audio/wav", "mp3": "audio/mpeg",
    "flac": "audio/flac", "aac": "audio/aac", "opus": "audio/opus",
}

# DashScope model name aliases → local model (all route to the same local model)
DASHSCOPE_MODEL_ALIASES = {
    "qwen3-tts-flash", "qwen3-tts-instruct-flash",
    "qwen3-tts-instruct-flash-2026-01-26",
    "qwen3-tts-flash-2025-11-27",
    "qwen3-tts",  # generic name
}


# ── App & global model holder ──────────────────────────────────────────────────
app = FastAPI(title="tts-router API Server", version="1.0.0")
_backend = get_backend()            # TTS backend (auto-detected)
_model = None           # Primary TTS model, loaded at startup
_model_repo = None      # HF repo ID for the primary model — set by configure() or defaults
_clone_model = None                  # Base model for voice cloning, lazy-loaded on first use
_clone_model_id = DEFAULT_CLONE_MODEL  # HF repo ID or local path, overridable via configure()
_clone_disabled = False              # set via configure() or --no-clone flag
_gen_lock = asyncio.Lock()  # serialize generation (MLX model is not thread-safe)
_extractor_chain = create_extractor_chain()  # available URL audio extractors


def configure(model_repo: str, clone_repo: str = None, no_clone: bool = False):
    """
    Inject model configuration from CLI before the server starts.
    Called by cli.py's `serve` command to decouple server from hardcoded paths.
    """
    global _model_repo, _clone_model_id, _clone_disabled
    _model_repo = model_repo
    if clone_repo:
        _clone_model_id = clone_repo
    _clone_disabled = no_clone


def _detect_language(text: str) -> str:
    """Simple heuristic: if >30% CJK chars → zh, else en."""
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    return "zh" if cjk / max(len(text), 1) > 0.3 else "en"


def _resolve_lang_code(language_type: Optional[str], text: str) -> str:
    """Convert DashScope language_type to 2-letter lang code, with auto-detect fallback."""
    if not language_type or language_type.lower() == "auto":
        return _detect_language(text)
    return LANG_TYPE_MAP.get(language_type.lower(), language_type[:2].lower())


def _convert_wav(src_path: str, fmt: str) -> bytes:
    """Convert WAV to the requested format via ffmpeg, return raw bytes."""
    if fmt == "wav":
        with open(src_path, "rb") as f:
            return f.read()
    codec_map = {"mp3": "libmp3lame", "flac": "flac", "aac": "aac", "opus": "libopus"}
    codec = codec_map.get(fmt, fmt)
    proc = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", src_path, "-c:a", codec, "-f", fmt, "pipe:1"],
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode()}")
    return proc.stdout


def _generate_to_wav(text: str, voice: str, lang_code: str,
                     instructions: str = "", speed: float = 1.0,
                     temperature: float | None = None,
                     exaggeration: float | None = None) -> tuple[str, float]:
    """
    Core generation: produce a WAV file and return (wav_path, elapsed_seconds).
    The caller is responsible for cleanup.
    """
    tmp_dir = tempfile.mkdtemp(prefix="tts_")
    t0 = time.time()
    wav_path = _backend.generate(
        _model,
        text,
        voice=voice,
        instruct=instructions,
        speed=speed,
        lang_code=lang_code,
        output_dir=tmp_dir,
        temperature=temperature,
        exaggeration=exaggeration,
    )
    elapsed = time.time() - t0
    return wav_path, elapsed


# ── Reference audio index helpers ─────────────────────────────────────────────

def _load_ref_index() -> list[dict]:
    """Load the reference audio index from disk, returning a list of entries."""
    if not os.path.exists(REFERENCES_INDEX):
        return []
    with open(REFERENCES_INDEX, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_ref_index(entries: list[dict]):
    """Persist the reference audio index to disk."""
    with open(REFERENCES_INDEX, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


# ── Clone model lazy-loading ─────────────────────────────────────────────────

def _ensure_clone_model():
    """
    Lazy-load the Base model for voice cloning.
    Uses HuggingFace auto-download via mlx_audio's load_model().
    """
    global _clone_model
    if _clone_model is not None:
        return _clone_model

    print(f"[Voice Clone] Loading Base model: {_clone_model_id} ...")
    print("[Voice Clone] First load will download ~2GB from HuggingFace. Please wait.")
    _clone_model = _backend.load_model(_clone_model_id)
    print("[Voice Clone] Base model loaded and ready.")
    return _clone_model


def _generate_clone_wav(text: str, ref_audio_path: str, ref_text: str,
                        lang_code: str, speed: float = 1.0) -> tuple[str, float]:
    """
    Generate audio using voice cloning (ICL mode with the Base model).
    Returns (wav_path, elapsed_seconds).
    """
    model = _ensure_clone_model()
    tmp_dir = tempfile.mkdtemp(prefix="tts_clone_")
    t0 = time.time()
    wav_path = _backend.generate(
        model,
        text,
        ref_audio=ref_audio_path,
        ref_text=ref_text or "",
        speed=speed,
        lang_code=lang_code,
        output_dir=tmp_dir,
    )
    elapsed = time.time() - t0
    return wav_path, elapsed


# ═══════════════════════════════════════════════════════════════════════════════
#  DashScope-compatible endpoint
#  POST /api/v1/services/aigc/multimodal-generation/generation
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/v1/services/aigc/multimodal-generation/generation")
async def dashscope_generation(request: Request):
    """
    DashScope-compatible TTS endpoint.

    Supports two response modes:
      - Non-streaming (default): returns JSON with audio URL
      - Streaming (X-DashScope-SSE: enable): returns SSE with base64 audio chunks
    """
    if _model is None:
        raise HTTPException(503, detail="Model not loaded yet")

    body = await request.json()
    request_id = str(uuid.uuid4())

    # ── Parse request body (DashScope format) ──────────────────────────────
    model_name = body.get("model", "qwen3-tts-flash")
    input_data = body.get("input", {})

    text = input_data.get("text", "")
    if not text:
        return JSONResponse(status_code=400, content={
            "status_code": 400, "request_id": request_id,
            "code": "InvalidParameter", "message": "'input.text' is required",
        })

    voice = input_data.get("voice", "Vivian")
    language_type = input_data.get("language_type", "Auto")
    instructions = input_data.get("instructions", "")
    lang_code = _resolve_lang_code(language_type, text)

    # ── Check if streaming is requested via SSE header ─────────────────────
    is_sse = request.headers.get("X-DashScope-SSE", "").lower() == "enable"

    # ── Generate audio (serialized — MLX model is not thread-safe) ────────
    try:
        async with _gen_lock:
            wav_path, elapsed = await asyncio.to_thread(
                _generate_to_wav, text, voice, lang_code, instructions,
            )
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status_code": 500, "request_id": request_id,
            "code": "InternalError", "message": str(e),
        })

    audio_id = f"audio_{request_id}"
    info = sf.info(wav_path)

    if is_sse:
        # ── Streaming response: SSE with base64-encoded PCM chunks ─────
        def _sse_generator():
            """Yield SSE events with base64-encoded audio data in chunks."""
            chunk_size = 24000 * 2  # ~1 second of 24kHz 16-bit mono PCM
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
            # Skip WAV header (44 bytes), send raw PCM in chunks
            pcm_data = audio_bytes[44:]
            offset = 0
            while offset < len(pcm_data):
                chunk = pcm_data[offset:offset + chunk_size]
                b64_chunk = base64.b64encode(chunk).decode("ascii")
                event_data = {
                    "status_code": 200,
                    "request_id": request_id,
                    "output": {
                        "finish_reason": "null" if offset + chunk_size < len(pcm_data) else "stop",
                        "audio": {"data": b64_chunk, "id": audio_id},
                    },
                    "usage": {"characters": len(text)},
                }
                yield f"data: {__import__('json').dumps(event_data, ensure_ascii=False)}\n\n"
                offset += chunk_size
            # Cleanup temp file
            shutil.rmtree(os.path.dirname(wav_path), ignore_errors=True)

        gc.collect()
        return StreamingResponse(_sse_generator(), media_type="text/event-stream")

    else:
        # ── Non-streaming response: save WAV and return URL ────────────
        # Save audio to outputs/api/ with a stable filename for URL serving
        filename = f"{audio_id}.wav"
        saved_path = os.path.join(OUTPUTS_DIR, filename)
        shutil.move(wav_path, saved_path)
        shutil.rmtree(os.path.dirname(wav_path) if os.path.dirname(wav_path) != saved_path else "", ignore_errors=True)

        # Build a local URL for the audio file
        host = request.headers.get("host", "localhost:8091")
        scheme = "https" if request.url.scheme == "https" else "http"
        audio_url = f"{scheme}://{host}/outputs/{filename}"
        expires_at = int(time.time()) + 86400  # 24h expiry (same as DashScope)

        gc.collect()

        # Return DashScope-format JSON response
        return JSONResponse(content={
            "status_code": 200,
            "request_id": request_id,
            "code": "",
            "message": "",
            "output": {
                "text": None,
                "finish_reason": "stop",
                "choices": None,
                "audio": {
                    "data": "",
                    "url": audio_url,
                    "id": audio_id,
                    "expires_at": expires_at,
                },
            },
            "usage": {
                "input_tokens": 0,
                "output_tokens": 0,
                "characters": len(text),
            },
        })


# ── Static file serving for generated audio (non-streaming mode) ──────────────
from fastapi.staticfiles import StaticFiles
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")


# ═══════════════════════════════════════════════════════════════════════════════
#  OpenAI-compatible endpoint (kept for broad compatibility)
#  POST /v1/audio/speech
# ═══════════════════════════════════════════════════════════════════════════════

class OpenAISpeechRequest(BaseModel):
    """OpenAI /v1/audio/speech request body, with TTS-specific extras."""
    input: str = Field(..., description="Text to synthesise")
    voice: str = Field("", description="Speaker name (empty = model default)")
    model: str = Field("qwen3-tts", description="Model identifier (ignored)")
    response_format: str = Field("wav", description="Output format: wav, mp3, flac, aac, opus")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed multiplier")
    language: Optional[str] = Field(None, description="Language hint: Chinese, English, Japanese, …")
    instructions: Optional[str] = Field(None, description="Emotion / style instruction")
    temperature: Optional[float] = Field(None, description="Sampling temperature (model-specific default)")
    exaggeration: Optional[float] = Field(None, ge=0.0, le=1.0, description="Emotion exaggeration 0.0-1.0 (Chatterbox)")


@app.post("/v1/audio/speech")
async def openai_create_speech(req: OpenAISpeechRequest):
    """OpenAI-compatible speech generation endpoint. Returns raw audio bytes."""
    if _model is None:
        raise HTTPException(503, "Model not loaded yet")
    if req.response_format not in FORMAT_MIME:
        raise HTTPException(400, f"Unsupported format '{req.response_format}', use: {list(FORMAT_MIME)}")

    lang_code = _resolve_lang_code(req.language, req.input)

    try:
        async with _gen_lock:
            wav_path, elapsed = await asyncio.to_thread(
                _generate_to_wav, req.input, req.voice, lang_code,
                req.instructions or "", req.speed, req.temperature, req.exaggeration,
            )
    except Exception as e:
        raise HTTPException(500, str(e))

    info = sf.info(wav_path)
    audio_bytes = _convert_wav(wav_path, req.response_format)

    # Cleanup temp directory
    shutil.rmtree(os.path.dirname(wav_path), ignore_errors=True)
    gc.collect()

    return Response(
        content=audio_bytes,
        media_type=FORMAT_MIME[req.response_format],
        headers={
            "X-Audio-Duration": f"{info.duration:.3f}",
            "X-Processing-Time": f"{elapsed:.3f}",
            "X-Realtime-Factor": f"{info.duration / elapsed:.2f}" if elapsed > 0 else "inf",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Utility endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/v1/audio/voices")
async def list_voices():
    """List all available preset voices, grouped by language."""
    return JSONResponse({
        "voices": ALL_VOICES,
        "by_language": {lang: voices for lang, voices in SPEAKER_MAP.items()},
    })


@app.get("/v1/models")
async def list_models():
    """List all available models with their metadata (features, voices, parameters)."""
    registry = _backend.available_models()
    # Find the short name of the currently loaded model
    current_name = None
    for name, info in registry.items():
        if info["repo"] == _model_repo:
            current_name = name
            break
    return JSONResponse({
        "current_model": current_name,
        "current_model_repo": _model_repo,
        "models": {
            name: {
                "repo": info["repo"],
                "description": info.get("description", ""),
                "features": info.get("features", []),
                "voices": info.get("voices", {}),
                "parameters": info.get("parameters", {}),
                "default": info.get("default", False),
            }
            for name, info in registry.items()
        },
    })


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "model_id": _model_repo,
        "clone_enabled": not _clone_disabled,
        "clone_model_loaded": _clone_model is not None,
        "clone_model_id": _clone_model_id if not _clone_disabled else None,
        "url_extractors": get_extractor_names(_extractor_chain),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Voice Clone — Reference Audio Management
#  Manage uploaded reference audio files for voice cloning.
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/v1/audio/references/upload")
async def upload_reference(file: UploadFile = File(...)):
    """
    Upload a reference audio file for voice cloning.
    Accepts common audio formats (wav, mp3, flac, m4a, ogg, webm).
    Converts to WAV internally for consistent processing.
    Returns the ref_id for subsequent clone requests.
    """
    if _clone_disabled:
        raise HTTPException(403, "Voice cloning is disabled on this server (--no-clone)")

    # Validate file extension
    allowed_exts = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".webm", ".aac"}
    orig_name = file.filename or "unknown.wav"
    ext = os.path.splitext(orig_name)[1].lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported audio format '{ext}'. Allowed: {sorted(allowed_exts)}")

    ref_id = str(uuid.uuid4())[:8]
    raw_path = os.path.join(REFERENCES_DIR, f"{ref_id}_raw{ext}")
    wav_path = os.path.join(REFERENCES_DIR, f"{ref_id}.wav")

    # Save uploaded file
    content = await file.read()
    if len(content) > 50 * 1024 * 1024:  # 50MB limit
        raise HTTPException(400, "File too large (max 50MB)")
    with open(raw_path, "wb") as f:
        f.write(content)

    # Convert to WAV (16kHz mono) via ffmpeg for consistent model input
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", raw_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav_path],
            capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode())
    except Exception as e:
        # Cleanup on failure
        for p in (raw_path, wav_path):
            if os.path.exists(p):
                os.remove(p)
        raise HTTPException(500, f"Failed to process audio: {e}")
    finally:
        # Remove raw upload, keep only the WAV
        if os.path.exists(raw_path) and os.path.exists(wav_path):
            os.remove(raw_path)

    # Get audio duration for display
    try:
        info = sf.info(wav_path)
        duration = round(info.duration, 2)
    except Exception:
        duration = 0

    # Update index
    entries = _load_ref_index()
    entries.append({
        "ref_id": ref_id,
        "original_name": orig_name,
        "duration": duration,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_ref_index(entries)

    return JSONResponse({
        "ref_id": ref_id,
        "original_name": orig_name,
        "duration": duration,
    })


class FromUrlRequest(BaseModel):
    """Request body for downloading reference audio from a URL."""
    url: str = Field(..., description="URL to download audio from (direct link, YouTube, Bilibili, etc.)")


@app.post("/v1/audio/references/from-url")
async def reference_from_url(req: FromUrlRequest):
    """
    Download audio from a URL and save it as a reference for voice cloning.
    Supports direct audio links (.mp3/.wav/etc.) and, when yt-dlp is installed,
    YouTube/Bilibili/1000+ video sites.
    Returns the same format as the upload endpoint: { ref_id, original_name, duration }.
    """
    if _clone_disabled:
        raise HTTPException(403, "Voice cloning is disabled on this server (--no-clone)")

    url = req.url.strip()
    if not url:
        raise HTTPException(400, "URL is required")

    # Find the first extractor that can handle this URL
    extractor = None
    for ext in _extractor_chain:
        if ext.can_handle(url):
            extractor = ext
            break

    if extractor is None:
        # Build a helpful error message based on available extractors
        names = get_extractor_names(_extractor_chain)
        if "yt-dlp" not in names:
            raise HTTPException(
                400,
                "This URL is not a direct audio link. "
                "Install yt-dlp (`pip install yt-dlp`) to support YouTube/Bilibili and 1000+ sites."
            )
        raise HTTPException(400, "No extractor can handle this URL.")

    # Run extraction in a temp directory (can be slow — run in thread)
    tmp_dir = tempfile.mkdtemp(prefix="tts_extract_")
    try:
        result = await asyncio.to_thread(extractor.extract, url, tmp_dir)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"Audio extraction failed: {e}")

    # Move the extracted WAV into the references directory with a new ref_id
    ref_id = str(uuid.uuid4())[:8]
    wav_path = os.path.join(REFERENCES_DIR, f"{ref_id}.wav")

    try:
        shutil.move(result.path, wav_path)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(500, f"Failed to save reference: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Get accurate duration from the final WAV
    try:
        info = sf.info(wav_path)
        duration = round(info.duration, 2)
    except Exception:
        duration = result.duration

    # Update the reference index
    entries = _load_ref_index()
    entries.append({
        "ref_id": ref_id,
        "original_name": result.title,
        "duration": duration,
        "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    _save_ref_index(entries)

    return JSONResponse({
        "ref_id": ref_id,
        "original_name": result.title,
        "duration": duration,
    })


@app.get("/v1/audio/references")
async def list_references():
    """List all uploaded reference audio files."""
    if _clone_disabled:
        raise HTTPException(403, "Voice cloning is disabled on this server (--no-clone)")
    entries = _load_ref_index()
    return JSONResponse({"references": entries})


@app.delete("/v1/audio/references/{ref_id}")
async def delete_reference(ref_id: str):
    """Delete a previously uploaded reference audio file."""
    if _clone_disabled:
        raise HTTPException(403, "Voice cloning is disabled on this server (--no-clone)")

    entries = _load_ref_index()
    found = None
    for entry in entries:
        if entry["ref_id"] == ref_id:
            found = entry
            break
    if not found:
        raise HTTPException(404, f"Reference '{ref_id}' not found")

    # Remove WAV file from disk
    wav_path = os.path.join(REFERENCES_DIR, f"{ref_id}.wav")
    if os.path.exists(wav_path):
        os.remove(wav_path)

    # Update index
    entries = [e for e in entries if e["ref_id"] != ref_id]
    _save_ref_index(entries)

    return JSONResponse({"deleted": ref_id})


# ── Serve reference audio files for playback in the UI ────────────────────────
@app.get("/v1/audio/references/{ref_id}/play")
async def play_reference(ref_id: str):
    """Stream a reference audio file for playback in the browser."""
    wav_path = os.path.join(REFERENCES_DIR, f"{ref_id}.wav")
    if not os.path.exists(wav_path):
        raise HTTPException(404, f"Reference '{ref_id}' not found")
    return FileResponse(wav_path, media_type="audio/wav")


# ── Trim reference audio ──────────────────────────────────────────────────────

class TrimRequest(BaseModel):
    """Request body for trimming a reference audio file."""
    start: float = Field(..., ge=0, description="Trim start position in seconds")
    end: float = Field(..., gt=0, description="Trim end position in seconds")


@app.post("/v1/audio/references/{ref_id}/trim")
async def trim_reference(ref_id: str, req: TrimRequest):
    """
    Trim a reference audio file to the specified [start, end] interval.
    Uses ffmpeg to cut the WAV in-place (atomic replace via temp file).
    Updates the duration in index.json accordingly.
    """
    if _clone_disabled:
        raise HTTPException(403, "Voice cloning is disabled on this server (--no-clone)")

    wav_path = os.path.join(REFERENCES_DIR, f"{ref_id}.wav")
    if not os.path.exists(wav_path):
        raise HTTPException(404, f"Reference '{ref_id}' not found")

    # Validate against actual file duration
    try:
        info = sf.info(wav_path)
        actual_duration = info.duration
    except Exception as e:
        raise HTTPException(500, f"Cannot read audio info: {e}")

    if req.start >= actual_duration:
        raise HTTPException(400, f"start ({req.start}s) must be less than audio duration ({actual_duration:.2f}s)")
    if req.start >= req.end:
        raise HTTPException(400, f"start ({req.start}s) must be less than end ({req.end}s)")

    # Clamp end to actual duration (user may overshoot slightly)
    trimmed_end = min(req.end, actual_duration)

    # ffmpeg trim to a temp file, then atomic-replace the original
    tmp_path = wav_path + ".trim.tmp.wav"
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error",
             "-i", wav_path,
             "-ss", str(req.start), "-to", str(trimmed_end),
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
             tmp_path],
            capture_output=True, timeout=30,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode())
    except subprocess.TimeoutExpired:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(500, "ffmpeg timed out during trim")
    except Exception as e:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise HTTPException(500, f"Trim failed: {e}")

    # Atomic replace: move trimmed output over the original
    os.replace(tmp_path, wav_path)

    # Read new duration from the trimmed file
    try:
        new_info = sf.info(wav_path)
        new_duration = round(new_info.duration, 2)
    except Exception:
        new_duration = round(trimmed_end - req.start, 2)

    # Update index.json with the new duration
    entries = _load_ref_index()
    for entry in entries:
        if entry["ref_id"] == ref_id:
            entry["duration"] = new_duration
            break
    _save_ref_index(entries)

    return JSONResponse({
        "ref_id": ref_id,
        "duration": new_duration,
        "trimmed_from": round(req.start, 3),
        "trimmed_to": round(trimmed_end, 3),
    })


# ═══════════════════════════════════════════════════════════════════════════════
#  Voice Clone — Generation
#  Uses the Base model (ICL mode) with a reference audio to clone voice timbre.
# ═══════════════════════════════════════════════════════════════════════════════

class CloneRequest(BaseModel):
    """Request body for voice cloning generation."""
    text: str = Field(..., description="Text to synthesise with cloned voice")
    ref_id: str = Field(..., description="Reference audio ID from upload")
    ref_text: Optional[str] = Field(None, description="Transcript of the reference audio (optional, auto-detected if omitted)")
    language: Optional[str] = Field(None, description="Language hint: Chinese, English, Japanese, …")
    speed: float = Field(1.0, ge=0.5, le=2.0, description="Playback speed multiplier")
    response_format: str = Field("wav", description="Output format: wav, mp3, flac, aac, opus")


@app.post("/v1/audio/clone")
async def clone_speech(req: CloneRequest):
    """
    Generate speech using voice cloning.
    Requires a previously uploaded reference audio (ref_id).
    On first call, the Base model (~2GB) will be downloaded from HuggingFace.
    """
    if _clone_disabled:
        raise HTTPException(403, "Voice cloning is disabled on this server (--no-clone)")

    if not req.text.strip():
        raise HTTPException(400, "Text is required")
    if req.response_format not in FORMAT_MIME:
        raise HTTPException(400, f"Unsupported format '{req.response_format}', use: {list(FORMAT_MIME)}")

    # Verify reference audio exists
    ref_wav = os.path.join(REFERENCES_DIR, f"{req.ref_id}.wav")
    if not os.path.exists(ref_wav):
        raise HTTPException(404, f"Reference audio '{req.ref_id}' not found. Upload one first.")

    lang_code = _resolve_lang_code(req.language, req.text)

    try:
        async with _gen_lock:
            wav_path, elapsed = await asyncio.to_thread(
                _generate_clone_wav,
                req.text, ref_wav, req.ref_text or "", lang_code, req.speed,
            )
    except Exception as e:
        raise HTTPException(500, f"Voice clone generation failed: {e}")

    info = sf.info(wav_path)
    audio_bytes = _convert_wav(wav_path, req.response_format)

    # Cleanup temp directory
    shutil.rmtree(os.path.dirname(wav_path), ignore_errors=True)
    gc.collect()

    return Response(
        content=audio_bytes,
        media_type=FORMAT_MIME[req.response_format],
        headers={
            "X-Audio-Duration": f"{info.duration:.3f}",
            "X-Processing-Time": f"{elapsed:.3f}",
            "X-Realtime-Factor": f"{info.duration / elapsed:.2f}" if elapsed > 0 else "inf",
            "X-Clone-Model": _clone_model_id or "",
        },
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Playground UI
#  GET / and GET /playground serve the single-file playground HTML
# ═══════════════════════════════════════════════════════════════════════════════

PLAYGROUND_HTML = os.path.join(BASE_DIR, "playground.html")


@app.get("/")
async def playground_root():
    """Serve the Playground UI at the root path."""
    return FileResponse(PLAYGROUND_HTML, media_type="text/html")


@app.get("/playground")
async def playground_alias():
    """Alias route for the Playground UI."""
    return FileResponse(PLAYGROUND_HTML, media_type="text/html")


# ── Startup ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def load_model_on_startup():
    global _model, _model_repo
    # Fall back to default repo if configure() was never called (direct `python server.py`)
    if _model_repo is None:
        _model_repo = DEFAULT_MODEL_REPO
    print(f"Loading model: {_model_repo} ...")
    _model = _backend.load_model(_model_repo)
    print("Model loaded and ready to serve.")

    if not _clone_disabled:
        print(f"[Voice Clone] Enabled. Base model: {_clone_model_id}")
        print("[Voice Clone] Model will be downloaded on first clone request (lazy-load).")
    else:
        print("[Voice Clone] Disabled via --no-clone flag.")


# ── Direct execution fallback ──────────────────────────────────────────────────
# `python server.py` still works — uses default model repos from HF cache.
# For the full CLI experience, use: uv run tts-router serve

if __name__ == "__main__":
    configure(
        model_repo=DEFAULT_MODEL_REPO,
        clone_repo=DEFAULT_CLONE_MODEL,
    )
    uvicorn.run(app, host="0.0.0.0", port=8091)
