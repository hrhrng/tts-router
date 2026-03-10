"""
TTSBackend — abstract base class for all TTS inference backends.

Every backend (MLX, transformers, vLLM, …) implements this interface so that
cli.py and server.py never import a specific framework directly.
"""

from abc import ABC, abstractmethod
from typing import Any


class TTSBackend(ABC):
    """Unified TTS inference interface."""

    @abstractmethod
    def load_model(self, repo_id: str) -> Any:
        """Load a model by HuggingFace repo ID (or local path). Returns an opaque model object."""

    @abstractmethod
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
        """
        Generate audio from text using *model*.

        Parameters
        ----------
        voice : str
            Speaker/voice identifier.  Meaning varies by model:
            - Qwen3-TTS CustomVoice: speaker name (Vivian, Chelsie, Ethan, …)
            - Kokoro: voice file name (af_heart, bf_emma, …)
            - Orpheus: prompt prefix (zoe, tara, leah, …)
            - Dia, Chatterbox: ignored (use ref_audio for cloning)
        instruct : str
            Emotion/style instruction (Qwen3 CustomVoice) or voice description
            (Qwen3 VoiceDesign).  Ignored by other models.
        speed : float
            Speech speed multiplier.  Supported by Kokoro; accepted but
            not yet implemented by Qwen3; ignored by others.
        lang_code : str
            Language hint. Qwen3: "auto"/"chinese"/"english"/…;
            Kokoro: "a"/"b"/"e"/"f"/"j"/"z"/…;
            Chatterbox: "en"/ISO codes; others: ignored.
        ref_audio : str or None
            Path to reference WAV for voice cloning.  Used by Qwen3 Base,
            Chatterbox, Dia (audio continuation), Orpheus (unreliable).
        ref_text : str or None
            Transcript of ref_audio.  Auto-transcribed via Whisper if omitted.
        temperature : float or None
            Sampling temperature.  None = model default
            (Qwen3: 0.9, Dia: 1.3, Chatterbox: 0.8, Orpheus: 0.6).
        exaggeration : float or None
            Emotion exaggeration (Chatterbox only, 0.0–1.0, default 0.1).
            Ignored by other models.

        Returns the path to the generated WAV file.
        The caller owns *output_dir* and is responsible for cleanup.
        """

    @abstractmethod
    def available_models(self) -> dict[str, dict]:
        """
        Return the model registry for this backend.

        Keys are short names (e.g. "qwen3-tts"), values are dicts with at
        least ``repo`` and ``description``.
        """
