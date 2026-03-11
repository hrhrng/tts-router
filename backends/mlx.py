"""
MLXBackend — TTS backend powered by mlx-audio on Apple Silicon.

All mlx_audio imports are contained here so that the rest of the codebase
never touches the library directly.  If mlx-audio is not installed the
import error is caught and surfaced as a clear message.
"""

import os
from typing import Any

from backends.base import TTSBackend

# ── Model registry (migrated from cli.py) ────────────────────────────────────

_MODEL_REGISTRY: dict[str, dict] = {
    "qwen3-tts": {
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit",
        "description": "Qwen3-TTS 1.7B CustomVoice — multi-speaker, emotion control",
        "features": ["custom_voice", "multi_speaker", "instruct"],
        "default": True,
        "voices": {
            "English": ["Ryan", "Aiden", "Ethan", "Chelsie", "Serena", "Vivian"],
            "Chinese": ["Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric"],
            "Japanese": ["Ono_Anna"],
            "Korean": ["Sohee"],
        },
    },
    "qwen3-tts-design": {
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit",
        "description": "Qwen3-TTS 1.7B VoiceDesign — free-form voice description",
        "features": ["voice_design", "instruct"],
        "voices": {},
    },
    "qwen3-tts-clone": {
        "repo": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "description": "Qwen3-TTS 1.7B Base — voice cloning with ref audio",
        "features": ["voice_clone"],
        "voices": {},
    },
    "kokoro": {
        "repo": "prince-canuma/Kokoro-82M",
        "description": "Kokoro 82M — fast, lightweight, multi-language",
        "features": ["multi_speaker"],
        "voices": {
            "_all": ["af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
                     "am_adam", "am_michael",
                     "bf_emma", "bf_isabella",
                     "bm_george", "bm_lewis"],
        },
    },
    "dia": {
        "repo": "mlx-community/Dia-1.6B-4bit",
        "description": "Dia 1.6B — multi-speaker dialogue, laughter/emotion sounds",
        "features": ["multi_speaker", "dialogue"],
        "voices": {},
    },
    "chatterbox": {
        "repo": "mlx-community/chatterbox-fp16",
        "description": "Chatterbox — 23 languages, emotion control, voice cloning",
        "features": ["multi_speaker", "voice_clone", "instruct"],
        "voices": {},
        "parameters": {"exaggeration": {"default": 0.1, "min": 0.0, "max": 1.0}},
    },
    "orpheus": {
        "repo": "mlx-community/orpheus-3b-0.1-ft-6bit",
        "description": "Orpheus 3B — emotive TTS with emotion tags (<laugh>, <sigh>)",
        "features": ["instruct"],
        "voices": {
            "_all": ["tara", "zoe", "leah", "jess", "leo", "dan", "mia", "zac"],
        },
    },
}

# Map repo ID fragments to model family for parameter routing
_MODEL_FAMILIES = {
    "Qwen3-TTS": "qwen3",
    "Kokoro": "kokoro",
    "Dia-": "dia",
    "chatterbox": "chatterbox",
    "orpheus": "orpheus",
}

# Default voices per model family (when user doesn't specify)
_DEFAULT_VOICES = {
    "qwen3": "Vivian",
    "kokoro": "af_heart",
    "orpheus": "tara",
}


def _detect_family(repo_id: str) -> str:
    """Detect model family from repo ID for parameter routing."""
    repo_lower = repo_id.lower()
    for pattern, family in _MODEL_FAMILIES.items():
        if pattern.lower() in repo_lower:
            return family
    return "unknown"


def _import_mlx_audio():
    """Lazily import mlx_audio, raising a friendly error on failure."""
    try:
        from mlx_audio.tts.utils import load_model
        from mlx_audio.tts.generate import generate_audio
        return load_model, generate_audio
    except ImportError as exc:
        raise ImportError(
            "mlx-audio is not installed. "
            "On Apple Silicon, install with: uv pip install -e '.[mlx]'\n"
            "On other platforms, mlx-audio is not supported yet."
        ) from exc


class MLXBackend(TTSBackend):
    """Apple-Silicon TTS backend using mlx-audio."""

    def load_model(self, repo_id: str) -> Any:
        load_model_fn, _ = _import_mlx_audio()
        return load_model_fn(repo_id)

    def generate(
        self,
        model: Any,
        text: str,
        *,
        voice: str = "",
        instruct: str = "",
        speed: float = 1.0,
        lang_code: str = "",
        output_dir: str,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        temperature: float | None = None,
        exaggeration: float | None = None,
        verbose: bool = False,
    ) -> str:
        _, generate_audio_fn = _import_mlx_audio()

        # Detect model family for parameter routing
        repo_id = getattr(model, "model_path", "") or ""
        if not repo_id:
            # Try to get from config
            repo_id = str(getattr(model, "_model_path", ""))
        family = _detect_family(repo_id)

        # Build kwargs with model-aware parameter mapping
        kwargs: dict[str, Any] = {
            "model": model,
            "text": text,
            "output_path": output_dir,
            "verbose": verbose,
        }

        # ── voice ─────────────────────────────────────────────────────────
        # Use model-appropriate default if not specified
        if voice:
            kwargs["voice"] = voice
        elif family in _DEFAULT_VOICES:
            kwargs["voice"] = _DEFAULT_VOICES[family]

        # ── instruct (Qwen3 only) ────────────────────────────────────────
        if instruct:
            kwargs["instruct"] = instruct

        # ── speed (Kokoro mainly) ─────────────────────────────────────────
        if speed != 1.0:
            kwargs["speed"] = speed

        # ── lang_code ─────────────────────────────────────────────────────
        if lang_code:
            kwargs["lang_code"] = lang_code

        # ── temperature ───────────────────────────────────────────────────
        if temperature is not None:
            kwargs["temperature"] = temperature

        # ── exaggeration (Chatterbox only) ────────────────────────────────
        if exaggeration is not None and family == "chatterbox":
            kwargs["exaggeration"] = exaggeration

        # ── ref_audio / ref_text (voice cloning) ──────────────────────────
        # Call model.generate() directly to support both:
        #  - ICL mode (ref_audio + ref_text): best quality clone
        #  - X-vector only mode (ref_audio only): speaker embedding, no ASR needed
        # The generate_audio() wrapper requires ref_text or a Whisper STT model,
        # so we bypass it for voice cloning.
        if ref_audio:
            return self._generate_clone(
                model, text, ref_audio=ref_audio, ref_text=ref_text,
                voice=voice if voice else (_DEFAULT_VOICES.get(family) or ""),
                speed=speed, lang_code=lang_code, temperature=temperature,
                output_dir=output_dir, verbose=verbose,
            )

        generate_audio_fn(**kwargs)

        wav_path = os.path.join(output_dir, "audio_000.wav")
        if not os.path.exists(wav_path):
            raise RuntimeError("Audio generation produced no output")
        return wav_path

    @staticmethod
    def _generate_clone(
        model: Any, text: str, *, ref_audio: str, ref_text: str | None,
        voice: str, speed: float, lang_code: str,
        temperature: float | None, output_dir: str, verbose: bool,
    ) -> str:
        """Generate audio with voice cloning via model.generate() directly.

        Two modes depending on whether ref_text is provided:
        - ICL mode (ref_text given): full in-context learning, best quality
        - X-vector mode (no ref_text): speaker embedding only, no Whisper needed
        """
        import numpy as np
        import mlx.core as mx
        from mlx_audio.audio_io import write as audio_write

        gen_kwargs: dict[str, Any] = {
            "text": text,
            "ref_audio": ref_audio,
            "lang_code": lang_code or "auto",
            "verbose": verbose,
        }
        if ref_text:
            gen_kwargs["ref_text"] = ref_text
        if voice:
            gen_kwargs["voice"] = voice
        if speed != 1.0:
            gen_kwargs["speed"] = speed
        if temperature is not None:
            gen_kwargs["temperature"] = temperature

        audio_chunks = []
        for result in model.generate(**gen_kwargs):
            audio_chunks.append(result.audio)

        if not audio_chunks:
            raise RuntimeError("Voice clone generation produced no output")

        audio = (
            mx.concatenate(audio_chunks, axis=0)
            if len(audio_chunks) > 1
            else audio_chunks[0]
        )

        os.makedirs(output_dir, exist_ok=True)
        wav_path = os.path.join(output_dir, "audio_000.wav")
        audio_write(wav_path, np.array(audio), model.sample_rate)
        return wav_path

    def available_models(self) -> dict[str, dict]:
        return dict(_MODEL_REGISTRY)
