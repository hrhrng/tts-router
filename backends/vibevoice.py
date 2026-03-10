"""
VibeVoiceBackend — TTS backend powered by Microsoft VibeVoice.

Multi-speaker, long-form TTS with voice cloning, built on PyTorch + Qwen2.5.
Uses the community fork: https://github.com/vibevoice-community/VibeVoice
"""

import os
import platform
from typing import Any

from backends.base import TTSBackend

_MODEL_REGISTRY: dict[str, dict] = {
    "vibevoice-1.5b": {
        "repo": "microsoft/VibeVoice-1.5B",
        "description": "VibeVoice 1.5B — multi-speaker, long-form TTS (up to 90min)",
        "features": ["multi_speaker", "voice_clone", "long_form"],
        "default": True,
        "voices": {},
    },
    "vibevoice-realtime": {
        "repo": "microsoft/VibeVoice-Realtime-0.5B",
        "description": "VibeVoice Realtime 0.5B — streaming, low-latency TTS",
        "features": ["streaming", "realtime"],
        "voices": {},
    },
}

_REALTIME_REPOS = {"microsoft/VibeVoice-Realtime-0.5B"}


def _is_realtime(repo_id: str) -> bool:
    return repo_id in _REALTIME_REPOS


def _get_device_config():
    """Pick torch dtype, device, and attention implementation for the current platform."""
    import torch

    if torch.cuda.is_available():
        dtype = torch.bfloat16
        device_map = "cuda"
        # Try flash_attention_2, fall back to sdpa
        try:
            import flash_attn  # noqa: F401
            attn = "flash_attention_2"
        except ImportError:
            attn = "sdpa"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dtype = torch.float32
        device_map = None  # load to CPU first, then .to("mps")
        attn = "sdpa"
    else:
        dtype = torch.float32
        device_map = "cpu"
        attn = "sdpa"

    return dtype, device_map, attn


def _import_vibevoice_batch():
    """Import classes for the 1.5B / 7B batch models."""
    try:
        from vibevoice.modular.modeling_vibevoice_inference import (
            VibeVoiceForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
        return VibeVoiceProcessor, VibeVoiceForConditionalGenerationInference
    except ImportError as exc:
        raise ImportError(
            "vibevoice is not installed. "
            "Install with: uv pip install -e '.[vibevoice]'\n"
            "See https://github.com/vibevoice-community/VibeVoice for details."
        ) from exc


def _import_vibevoice_streaming():
    """Import classes for the Realtime 0.5B streaming model."""
    try:
        from vibevoice.modular.modeling_vibevoice_streaming_inference import (
            VibeVoiceStreamingForConditionalGenerationInference,
        )
        from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
        return VibeVoiceStreamingProcessor, VibeVoiceStreamingForConditionalGenerationInference
    except ImportError as exc:
        raise ImportError(
            "vibevoice is not installed. "
            "Install with: uv pip install -e '.[vibevoice]'\n"
            "See https://github.com/vibevoice-community/VibeVoice for details."
        ) from exc


class _VibeVoiceModel:
    """Wrapper holding processor + model + metadata for a loaded VibeVoice variant."""

    def __init__(self, processor, model, repo_id: str, device: str):
        self.processor = processor
        self.model = model
        self.repo_id = repo_id
        self.device = device
        self.is_realtime = _is_realtime(repo_id)


class VibeVoiceBackend(TTSBackend):
    """PyTorch TTS backend using Microsoft VibeVoice."""

    def load_model(self, repo_id: str) -> Any:
        import torch

        dtype, device_map, attn = _get_device_config()
        is_rt = _is_realtime(repo_id)

        if is_rt:
            ProcessorCls, ModelCls = _import_vibevoice_streaming()
        else:
            ProcessorCls, ModelCls = _import_vibevoice_batch()

        processor = ProcessorCls.from_pretrained(repo_id)
        model = ModelCls.from_pretrained(
            repo_id,
            torch_dtype=dtype,
            device_map=device_map,
            attn_implementation=attn,
        )
        model.eval()

        # Determine actual device
        if device_map is None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            model = model.to("mps")
            device = "mps"
        elif device_map == "cpu":
            device = "cpu"
        else:
            device = "cuda"

        if is_rt:
            model.set_ddpm_inference_steps(num_steps=5)

        return _VibeVoiceModel(processor, model, repo_id, device)

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
        import torch

        vm: _VibeVoiceModel = model
        wav_path = os.path.join(output_dir, "audio_000.wav")

        # Format text as single-speaker script
        script = f"Speaker 1: {text}"

        if vm.is_realtime:
            return self._generate_realtime(vm, script, ref_audio, wav_path)
        else:
            return self._generate_batch(vm, script, ref_audio, wav_path)

    def _generate_batch(
        self, vm: _VibeVoiceModel, script: str, ref_audio: str | None, wav_path: str
    ) -> str:
        import torch

        voice_samples = [ref_audio] if ref_audio else []
        is_prefill = bool(ref_audio)

        inputs = vm.processor(
            text=[script],
            voice_samples=[voice_samples] if voice_samples else None,
            padding=True,
            return_tensors="pt",
        )
        # Move tensors to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(vm.device)

        with torch.no_grad():
            outputs = vm.model.generate(
                **inputs,
                cfg_scale=1.3,
                tokenizer=vm.processor.tokenizer,
                is_prefill=is_prefill,
            )

        vm.processor.save_audio(outputs.speech_outputs[0], output_path=wav_path)

        if not os.path.exists(wav_path):
            raise RuntimeError("VibeVoice generation produced no output")
        return wav_path

    def _generate_realtime(
        self, vm: _VibeVoiceModel, script: str, ref_audio: str | None, wav_path: str
    ) -> str:
        import copy
        import torch

        if ref_audio and ref_audio.endswith(".pt"):
            # Pre-computed voice embedding
            cached_prompt = torch.load(ref_audio, map_location=vm.device, weights_only=False)
        elif ref_audio:
            raise ValueError(
                "VibeVoice Realtime model requires pre-computed voice embeddings (.pt files), "
                "not raw audio. Use the 1.5B model for WAV-based voice cloning."
            )
        else:
            # Download default voice embedding if none provided
            cached_prompt = self._get_default_streaming_voice(vm.device)

        inputs = vm.processor.process_input_with_cached_prompt(
            text=script,
            cached_prompt=cached_prompt,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(vm.device)

        with torch.no_grad():
            outputs = vm.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.5,
                tokenizer=vm.processor.tokenizer,
                generation_config={"do_sample": False},
                all_prefilled_outputs=copy.deepcopy(cached_prompt) if cached_prompt else None,
            )

        vm.processor.save_audio(outputs.speech_outputs[0], output_path=wav_path)

        if not os.path.exists(wav_path):
            raise RuntimeError("VibeVoice Realtime generation produced no output")
        return wav_path

    @staticmethod
    def _get_default_streaming_voice(device: str):
        """Download and cache the default voice embedding for the Realtime model."""
        import torch

        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "vibevoice", "voices")
        os.makedirs(cache_dir, exist_ok=True)
        local_path = os.path.join(cache_dir, "en-Emma_woman.pt")

        if not os.path.exists(local_path):
            import urllib.request
            url = "https://raw.githubusercontent.com/vibevoice-community/VibeVoice/main/demo/voices/streaming_model/en-Emma_woman.pt"
            print(f"[VibeVoice] Downloading default voice embedding...")
            urllib.request.urlretrieve(url, local_path)

        return torch.load(local_path, map_location=device, weights_only=False)

    def available_models(self) -> dict[str, dict]:
        return dict(_MODEL_REGISTRY)
