"""
tts-router CLI
==============
Productized CLI for managing and serving local TTS models on Apple Silicon.

Commands:
    tts-router pull <model>     Download model weights from HuggingFace Hub
    tts-router list             List available models and download status
    tts-router serve            Start the TTS API server
    tts-router say <text>       Synthesize speech directly from the command line

Usage:
    uv run tts-router list
    uv run tts-router pull qwen3-tts
    uv run tts-router serve --model qwen3-tts --port 8091
    uv run tts-router say "你好世界" -o hello.wav
"""

import sys

import typer

# ── Backend abstraction ───────────────────────────────────────────────────────
from backends import get_backend

_backend = get_backend()
_model_registry = _backend.available_models()


def _resolve_repo(model: str) -> str:
    """
    Resolve a model name to a HuggingFace repo ID.
    Accepts both short names (e.g. "qwen3-tts") and raw repo IDs (e.g. "mlx-community/...").
    """
    if model in _model_registry:
        return _model_registry[model]["repo"]
    # Looks like a raw HF repo ID (contains a slash)
    if "/" in model:
        return model
    # Unknown short name — show available options
    available = ", ".join(_model_registry.keys())
    print(f"[error] Unknown model '{model}'. Available: {available}")
    print(f"[hint]  You can also pass a full HuggingFace repo ID (e.g. mlx-community/...)")
    raise typer.Exit(1)


def _is_model_cached(repo_id: str) -> bool:
    """Check if a HuggingFace model is already downloaded to local cache."""
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo_info in cache_info.repos:
            if repo_info.repo_id == repo_id:
                return True
    except Exception:
        pass
    return False


# ── Typer app ─────────────────────────────────────────────────────────────────

app = typer.Typer(
    name="tts-router",
    help="Local TTS router for Apple Silicon (MLX). Pull models, then serve.",
    add_completion=False,
    no_args_is_help=True,
)


@app.command()
def pull(
    model: str = typer.Argument(
        help="Model short name (e.g. 'qwen3-tts') or HuggingFace repo ID",
    ),
):
    """Download model weights from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    repo_id = _resolve_repo(model)
    display_name = model if model in _model_registry else repo_id

    print(f"Pulling '{display_name}' ({repo_id}) ...")
    try:
        local_path = snapshot_download(repo_id=repo_id)
    except Exception as e:
        print(f"[error] Download failed: {e}")
        raise typer.Exit(1)

    print(f"Cached at: {local_path}")

    # Pre-load the model to fetch any additional dependencies (codec, tokenizer, etc.)
    print(f"Verifying model loads correctly (fetching dependencies) ...")
    try:
        _backend.load_model(repo_id)
        print(f"Done. '{display_name}' is ready to use.")
    except Exception as e:
        print(f"[warn] Model downloaded but load_model() failed: {e}")
        print(f"[hint] The model weights are cached, but inference may need extra deps.")


@app.command("list")
def list_models():
    """List available models and their download status."""
    # Header
    print(f"\n{'Name':<22} {'Status':<10} {'Description'}")
    print(f"{'─' * 22} {'─' * 10} {'─' * 50}")

    for name, info in _model_registry.items():
        repo_id = info["repo"]
        cached = _is_model_cached(repo_id)
        status = "cached" if cached else "not pulled"
        marker = "  *" if info.get("default") else "   "
        default_tag = " (default)" if info.get("default") else ""
        print(f"{marker}{name:<19} {status:<10} {info['description']}{default_tag}")

    print(f"\n  * = default model")
    print(f"  Pull a model:  tts-router pull <name>")
    print()


@app.command()
def serve(
    model: str = typer.Option(
        "qwen3-tts",
        help="Primary TTS model (short name or HF repo ID)",
    ),
    host: str = typer.Option("0.0.0.0", help="Bind address"),
    port: int = typer.Option(8091, help="Port number"),
    no_clone: bool = typer.Option(False, "--no-clone", help="Disable voice cloning"),
    clone_model: str = typer.Option(
        "qwen3-tts-clone",
        help="Override the voice cloning model (short name or HF repo ID)",
        rich_help_panel="Advanced Options",
    ),
):
    """Start the TTS API server."""
    import uvicorn

    model_repo = _resolve_repo(model)
    clone_repo = _resolve_repo(clone_model) if not no_clone else None

    # ── Verify models are downloaded before launching ─────────────────────
    if not _is_model_cached(model_repo):
        print(f"[error] Model '{model}' ({model_repo}) is not downloaded.")
        print(f"[hint]  Run: tts-router pull {model}")
        raise typer.Exit(1)

    if clone_repo and not _is_model_cached(clone_repo):
        print(f"[warn]  Clone model '{clone_model}' ({clone_repo}) is not downloaded.")
        print(f"[hint]  Voice cloning will fail until you run: tts-router pull {clone_model}")
        print(f"[hint]  Or start with --no-clone to disable voice cloning.\n")

    # ── Inject config into server module and launch ───────────────────────
    from server import configure
    configure(model_repo=model_repo, clone_repo=clone_repo, no_clone=no_clone)

    print(f"Starting tts-router on {host}:{port}")
    print(f"  Model:       {model} ({model_repo})")
    if not no_clone:
        print(f"  Clone model: {clone_model} ({clone_repo})")
    else:
        print(f"  Clone model: disabled")
    print(f"  Playground:  http://localhost:{port}/\n")

    uvicorn.run("server:app", host=host, port=port)


@app.command()
def say(
    text: str = typer.Argument(help="Text to synthesize"),
    output: str = typer.Option(None, "--output", "-o", help="Output file path (default: stdout wav)"),
    voice: str = typer.Option("", help="Voice/speaker name (model-specific, e.g. Vivian, af_heart)"),
    model: str = typer.Option("qwen3-tts", help="Model short name or HF repo ID"),
    format: str = typer.Option(None, help="Output format: wav, mp3, flac, aac, opus"),
    language: str = typer.Option(None, help="Language hint (default: auto-detect)"),
    speed: float = typer.Option(1.0, help="Speed multiplier 0.5–2.0"),
    instruct: str = typer.Option("", help="Emotion/style instruction, e.g. 'speak sadly'", rich_help_panel="Voice Cloning & Advanced"),
    ref_audio: str = typer.Option(None, "--ref-audio", help="Reference audio file for voice cloning", rich_help_panel="Voice Cloning & Advanced"),
    ref_text: str = typer.Option(None, "--ref-text", help="Transcript of ref audio (ICL mode). Omit for x-vector mode (no ASR needed)", rich_help_panel="Voice Cloning & Advanced"),
    temperature: float = typer.Option(None, "--temperature", "-t", help="Sampling temperature (default: model-specific)", rich_help_panel="Voice Cloning & Advanced"),
    exaggeration: float = typer.Option(None, "--exaggeration", help="Emotion exaggeration 0.0–1.0 (Chatterbox only)", rich_help_panel="Voice Cloning & Advanced"),
):
    """Synthesize speech directly from the command line."""
    import os
    import shutil
    import subprocess
    import tempfile
    import time

    # ── Infer format from output filename if not explicitly set ────────────
    if format is None:
        if output:
            ext = os.path.splitext(output)[1].lstrip(".").lower()
            format = ext if ext in ("wav", "mp3", "flac", "aac", "opus") else "wav"
        else:
            format = "wav"

    # ── Resolve and verify model ──────────────────────────────────────────
    # Auto-switch to clone model when --ref-audio is used with a model that
    # doesn't support voice cloning (e.g. qwen3-tts → qwen3-tts-clone)
    if ref_audio and model in _model_registry:
        features = _model_registry[model].get("features", [])
        if "voice_clone" not in features:
            # Find a clone-capable model in the same family
            clone_model = None
            for name, info in _model_registry.items():
                if "voice_clone" in info.get("features", []):
                    clone_model = name
                    break
            if clone_model:
                print(f"[say] '{model}' doesn't support voice cloning, switching to '{clone_model}'", file=sys.stderr)
                model = clone_model

    repo_id = _resolve_repo(model)
    if not _is_model_cached(repo_id):
        print(f"[error] Model '{model}' ({repo_id}) is not downloaded.", file=sys.stderr)
        print(f"[hint]  Run: tts-router pull {model}", file=sys.stderr)
        raise typer.Exit(1)

    # ── Load model ────────────────────────────────────────────────────────
    print(f"[say] Loading model: {repo_id} ...", file=sys.stderr)
    tts_model = _backend.load_model(repo_id)
    print(f"[say] Model loaded.", file=sys.stderr)

    # ── Generate WAV ──────────────────────────────────────────────────────
    tmp_dir = tempfile.mkdtemp(prefix="tts_say_")
    try:
        print(f"[say] Generating audio ...", file=sys.stderr)
        t0 = time.time()
        wav_path = _backend.generate(
            tts_model,
            text,
            voice=voice,
            instruct=instruct,
            speed=speed,
            lang_code=language or "",
            output_dir=tmp_dir,
            ref_audio=ref_audio,
            ref_text=ref_text,
            temperature=temperature,
            exaggeration=exaggeration,
        )
        elapsed = time.time() - t0
        print(f"[say] Generated in {elapsed:.1f}s", file=sys.stderr)

        # ── Convert format if needed ──────────────────────────────────────
        if format == "wav":
            audio_bytes = open(wav_path, "rb").read()
        else:
            codec_map = {"mp3": "libmp3lame", "flac": "flac", "aac": "aac", "opus": "libopus"}
            codec = codec_map.get(format, format)
            proc = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-i", wav_path,
                 "-c:a", codec, "-f", format, "pipe:1"],
                capture_output=True,
            )
            if proc.returncode != 0:
                print(f"[error] ffmpeg conversion failed: {proc.stderr.decode()}", file=sys.stderr)
                raise typer.Exit(1)
            audio_bytes = proc.stdout

        # ── Write output ──────────────────────────────────────────────────
        if output:
            with open(output, "wb") as f:
                f.write(audio_bytes)
            print(f"[say] Saved to {output} ({len(audio_bytes)} bytes, {format})", file=sys.stderr)
        else:
            sys.stdout.buffer.write(audio_bytes)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    app()
