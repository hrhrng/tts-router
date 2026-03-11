"""
Tests for backends/mlx.py — model registry, family detection, voice cloning modes.

All mlx-audio / mlx imports are mocked so tests run without GPU / model weights.
"""

import os
import pytest
from unittest.mock import patch, MagicMock, call

from backends.mlx import (
    _MODEL_REGISTRY,
    _detect_family,
    _DEFAULT_VOICES,
    MLXBackend,
)


# ═════════════════════════════════════════════════════════════════════════════
#  Model registry sanity checks
# ═════════════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    def test_clone_model_has_voice_clone_feature(self):
        assert "voice_clone" in _MODEL_REGISTRY["qwen3-tts-clone"]["features"]

    def test_all_entries_have_repo(self):
        for name, info in _MODEL_REGISTRY.items():
            assert "/" in info["repo"], f"{name} missing valid repo"


class TestDetectFamily:
    def test_qwen3(self):
        assert _detect_family("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit") == "qwen3"

    def test_kokoro(self):
        assert _detect_family("prince-canuma/Kokoro-82M") == "kokoro"

    def test_chatterbox(self):
        assert _detect_family("mlx-community/chatterbox-fp16") == "chatterbox"

    def test_unknown(self):
        assert _detect_family("someone/random-model") == "unknown"


# ═════════════════════════════════════════════════════════════════════════════
#  _generate_clone — x-vector vs ICL mode
# ═════════════════════════════════════════════════════════════════════════════

def _make_fake_model(repo_id="mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"):
    """Create a mock model that yields one GenerationResult."""
    fake_audio = MagicMock(name="audio_array")
    fake_result = MagicMock()
    fake_result.audio = fake_audio

    model = MagicMock()
    model.model_path = repo_id
    model.sample_rate = 24000
    model.generate.return_value = iter([fake_result])
    return model, fake_audio


class TestGenerateCloneXVector:
    """Voice cloning without ref_text → x-vector only mode (no ASR)."""

    def test_xvector_mode_does_not_pass_ref_text(self):
        model, _ = _make_fake_model()

        import tempfile
        tmp = tempfile.mkdtemp()
        with (
            patch("numpy.array", return_value="np_audio"),
            patch("mlx.core.concatenate"),
            patch("mlx_audio.audio_io.write"),
        ):
            MLXBackend._generate_clone(
                model, "hello",
                ref_audio="/tmp/ref.wav", ref_text=None,
                voice="Vivian", speed=1.0, lang_code="",
                temperature=None, output_dir=tmp, verbose=False,
            )

        # model.generate() should NOT have ref_text in kwargs
        gen_call_kwargs = model.generate.call_args
        assert "ref_text" not in gen_call_kwargs.kwargs
        assert gen_call_kwargs.kwargs["ref_audio"] == "/tmp/ref.wav"

    def test_icl_mode_passes_ref_text(self):
        model, fake_audio = _make_fake_model()

        import tempfile
        tmp = tempfile.mkdtemp()
        with (
            patch("numpy.array", return_value="np_audio"),
            patch("mlx.core.concatenate"),
            patch("mlx_audio.audio_io.write"),
        ):
            MLXBackend._generate_clone(
                model, "hello",
                ref_audio="/tmp/ref.wav", ref_text="transcript of ref",
                voice="Vivian", speed=1.0, lang_code="",
                temperature=None, output_dir=tmp, verbose=False,
            )

        gen_call_kwargs = model.generate.call_args
        assert gen_call_kwargs.kwargs["ref_text"] == "transcript of ref"


class TestGenerateCloneParams:
    """Verify optional params are forwarded correctly."""

    def _run_clone(self, **overrides):
        model, _ = _make_fake_model()
        defaults = dict(
            ref_audio="/tmp/ref.wav", ref_text=None,
            voice="", speed=1.0, lang_code="",
            temperature=None, output_dir="/tmp/out", verbose=False,
        )
        defaults.update(overrides)

        import tempfile
        tmp = overrides.get("output_dir", tempfile.mkdtemp())
        defaults["output_dir"] = tmp

        with (
            patch("numpy.array", return_value="np_audio"),
            patch("mlx.core.concatenate"),
            patch("mlx_audio.audio_io.write"),
        ):
            MLXBackend._generate_clone(model, "hello", **defaults)
        return model

    def test_speed_forwarded(self):
        model = self._run_clone(speed=1.5)
        assert model.generate.call_args.kwargs["speed"] == 1.5

    def test_speed_1_not_forwarded(self):
        model = self._run_clone(speed=1.0)
        assert "speed" not in model.generate.call_args.kwargs

    def test_temperature_forwarded(self):
        model = self._run_clone(temperature=0.7)
        assert model.generate.call_args.kwargs["temperature"] == 0.7

    def test_temperature_none_not_forwarded(self):
        model = self._run_clone(temperature=None)
        assert "temperature" not in model.generate.call_args.kwargs

    def test_lang_code_defaults_to_auto(self):
        model = self._run_clone(lang_code="")
        assert model.generate.call_args.kwargs["lang_code"] == "auto"

    def test_lang_code_forwarded(self):
        model = self._run_clone(lang_code="chinese")
        assert model.generate.call_args.kwargs["lang_code"] == "chinese"

    def test_voice_forwarded(self):
        model = self._run_clone(voice="Ryan")
        assert model.generate.call_args.kwargs["voice"] == "Ryan"

    def test_empty_voice_not_forwarded(self):
        model = self._run_clone(voice="")
        assert "voice" not in model.generate.call_args.kwargs


class TestGenerateCloneOutput:
    """Verify audio output handling."""

    def test_single_chunk_not_concatenated(self):
        model, fake_audio = _make_fake_model()
        import tempfile
        tmp = tempfile.mkdtemp()

        with (
            patch("numpy.array", return_value="np_audio") as mock_np,
            patch("mlx.core.concatenate") as mock_concat,
            patch("mlx_audio.audio_io.write") as mock_write,
        ):
            MLXBackend._generate_clone(
                model, "hello",
                ref_audio="/tmp/ref.wav", ref_text=None,
                voice="", speed=1.0, lang_code="",
                temperature=None, output_dir=tmp, verbose=False,
            )
        # Single chunk → no concatenation needed
        mock_concat.assert_not_called()
        # audio_write called with the single chunk
        mock_write.assert_called_once()
        assert mock_write.call_args[0][0] == os.path.join(tmp, "audio_000.wav")

    def test_multiple_chunks_concatenated(self):
        chunk1 = MagicMock(name="chunk1")
        chunk2 = MagicMock(name="chunk2")
        result1 = MagicMock(); result1.audio = chunk1
        result2 = MagicMock(); result2.audio = chunk2

        model = MagicMock()
        model.model_path = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
        model.sample_rate = 24000
        model.generate.return_value = iter([result1, result2])

        import tempfile
        tmp = tempfile.mkdtemp()

        with (
            patch("numpy.array", return_value="np_audio"),
            patch("mlx.core.concatenate", return_value="joined") as mock_concat,
            patch("mlx_audio.audio_io.write"),
        ):
            MLXBackend._generate_clone(
                model, "hello",
                ref_audio="/tmp/ref.wav", ref_text=None,
                voice="", speed=1.0, lang_code="",
                temperature=None, output_dir=tmp, verbose=False,
            )
        mock_concat.assert_called_once_with([chunk1, chunk2], axis=0)

    def test_empty_output_raises(self):
        model = MagicMock()
        model.model_path = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
        model.generate.return_value = iter([])  # no results

        import tempfile
        tmp = tempfile.mkdtemp()

        with (
            patch("numpy.array"),
            patch("mlx.core.concatenate"),
            patch("mlx_audio.audio_io.write"),
        ):
            with pytest.raises(RuntimeError, match="no output"):
                MLXBackend._generate_clone(
                    model, "hello",
                    ref_audio="/tmp/ref.wav", ref_text=None,
                    voice="", speed=1.0, lang_code="",
                    temperature=None, output_dir=tmp, verbose=False,
                )


# ═════════════════════════════════════════════════════════════════════════════
#  generate() routing — ref_audio triggers _generate_clone, not generate_audio
# ═════════════════════════════════════════════════════════════════════════════

class TestGenerateRouting:
    """Verify that generate() routes to _generate_clone when ref_audio is set."""

    @patch("backends.mlx._import_mlx_audio")
    @patch.object(MLXBackend, "_generate_clone", return_value="/tmp/out/audio_000.wav")
    def test_ref_audio_routes_to_clone(self, mock_clone, mock_import):
        mock_import.return_value = (MagicMock(), MagicMock())
        model = MagicMock()
        model.model_path = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

        backend = MLXBackend()
        result = backend.generate(
            model, "hello", output_dir="/tmp/out",
            ref_audio="/tmp/ref.wav",
        )

        mock_clone.assert_called_once()
        assert result == "/tmp/out/audio_000.wav"

    @patch("backends.mlx._import_mlx_audio")
    @patch.object(MLXBackend, "_generate_clone")
    def test_no_ref_audio_uses_generate_audio(self, mock_clone, mock_import):
        """Without ref_audio, should use generate_audio (not _generate_clone)."""
        fake_generate_audio = MagicMock()
        mock_import.return_value = (MagicMock(), fake_generate_audio)
        model = MagicMock()
        model.model_path = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

        backend = MLXBackend()
        with patch("os.path.exists", return_value=True):
            backend.generate(model, "hello", output_dir="/tmp/out")

        mock_clone.assert_not_called()
        fake_generate_audio.assert_called_once()

    @patch("backends.mlx._import_mlx_audio")
    @patch.object(MLXBackend, "_generate_clone", return_value="/tmp/out/audio_000.wav")
    def test_clone_receives_default_voice_for_qwen3(self, mock_clone, mock_import):
        """When voice is empty and family is qwen3, default voice should be passed."""
        mock_import.return_value = (MagicMock(), MagicMock())
        model = MagicMock()
        model.model_path = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

        backend = MLXBackend()
        backend.generate(
            model, "hello", output_dir="/tmp/out",
            ref_audio="/tmp/ref.wav",
        )

        _, kwargs = mock_clone.call_args
        assert kwargs["voice"] == "Vivian"  # qwen3 default

    @patch("backends.mlx._import_mlx_audio")
    @patch.object(MLXBackend, "_generate_clone", return_value="/tmp/out/audio_000.wav")
    def test_clone_passes_explicit_voice(self, mock_clone, mock_import):
        mock_import.return_value = (MagicMock(), MagicMock())
        model = MagicMock()
        model.model_path = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

        backend = MLXBackend()
        backend.generate(
            model, "hello", output_dir="/tmp/out",
            ref_audio="/tmp/ref.wav", voice="Ryan",
        )

        _, kwargs = mock_clone.call_args
        assert kwargs["voice"] == "Ryan"
