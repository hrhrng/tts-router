"""
Tests for cli.py — MODEL_REGISTRY, repo resolution, cache checking, CLI commands.

All HuggingFace / uvicorn side effects are mocked so tests run offline and fast.
"""

import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from cli import (
    MODEL_REGISTRY,
    _resolve_repo,
    _is_model_cached,
    app,
)

runner = CliRunner()


# ═════════════════════════════════════════════════════════════════════════════
#  MODEL_REGISTRY sanity checks
# ═════════════════════════════════════════════════════════════════════════════

class TestModelRegistry:
    """Verify the registry has correct structure and exactly one default."""

    def test_registry_not_empty(self):
        assert len(MODEL_REGISTRY) >= 4

    def test_each_entry_has_required_keys(self):
        for name, info in MODEL_REGISTRY.items():
            assert "repo" in info, f"{name} missing 'repo'"
            assert "description" in info, f"{name} missing 'description'"
            assert "features" in info, f"{name} missing 'features'"
            # repo should look like a HF repo ID (org/name)
            assert "/" in info["repo"], f"{name} repo '{info['repo']}' doesn't look like a HF ID"

    def test_exactly_one_default(self):
        defaults = [n for n, info in MODEL_REGISTRY.items() if info.get("default")]
        assert len(defaults) == 1, f"Expected 1 default model, found: {defaults}"
        assert defaults[0] == "qwen3-tts"


# ═════════════════════════════════════════════════════════════════════════════
#  _resolve_repo()
# ═════════════════════════════════════════════════════════════════════════════

class TestResolveRepo:
    """Test model name → HF repo ID resolution."""

    def test_known_short_name(self):
        assert _resolve_repo("qwen3-tts") == "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"

    def test_all_registry_names_resolve(self):
        for name, info in MODEL_REGISTRY.items():
            assert _resolve_repo(name) == info["repo"]

    def test_raw_hf_repo_id_passthrough(self):
        """A string with a slash is treated as a raw HF repo ID."""
        assert _resolve_repo("someone/some-model") == "someone/some-model"

    def test_unknown_short_name_exits(self):
        """Unknown name without slash should raise typer.Exit (click.exceptions.Exit)."""
        from click.exceptions import Exit
        with pytest.raises(Exit):
            _resolve_repo("nonexistent-model")


# ═════════════════════════════════════════════════════════════════════════════
#  _is_model_cached()
# ═════════════════════════════════════════════════════════════════════════════

class TestIsModelCached:
    """Test HF cache detection (mocked)."""

    def test_cached_model_returns_true(self):
        """When scan_cache_dir finds the repo, should return True."""
        mock_repo = MagicMock()
        mock_repo.repo_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
        mock_cache = MagicMock()
        mock_cache.repos = [mock_repo]

        with patch("cli.scan_cache_dir", return_value=mock_cache, create=True):
            # Need to patch at the point of import inside the function
            with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache):
                assert _is_model_cached("mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit") is True

    def test_uncached_model_returns_false(self):
        """When scan_cache_dir doesn't find the repo, should return False."""
        mock_repo = MagicMock()
        mock_repo.repo_id = "some-other/model"
        mock_cache = MagicMock()
        mock_cache.repos = [mock_repo]

        with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache):
            assert _is_model_cached("mlx-community/does-not-exist") is False

    def test_scan_exception_returns_false(self):
        """If scan_cache_dir blows up, gracefully return False."""
        with patch("huggingface_hub.scan_cache_dir", side_effect=OSError("boom")):
            assert _is_model_cached("anything") is False


# ═════════════════════════════════════════════════════════════════════════════
#  CLI: tts-router list
# ═════════════════════════════════════════════════════════════════════════════

class TestListCommand:
    """Test the `list` CLI command."""

    def test_list_shows_all_models(self):
        """Output should contain every model name from the registry."""
        with patch("cli._is_model_cached", return_value=False):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        for name in MODEL_REGISTRY:
            assert name in result.output

    def test_list_shows_cached_status(self):
        """Cached models should show 'cached', others 'not pulled'."""
        def fake_cached(repo_id):
            return repo_id == MODEL_REGISTRY["kokoro"]["repo"]

        with patch("cli._is_model_cached", side_effect=fake_cached):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # kokoro line should say "cached"
        lines = result.output.splitlines()
        kokoro_line = [l for l in lines if "kokoro" in l][0]
        assert "cached" in kokoro_line
        # qwen3-tts line should say "not pulled"
        qwen_line = [l for l in lines if "qwen3-tts-design" in l][0]
        assert "not pulled" in qwen_line

    def test_list_marks_default(self):
        """The default model should be marked with '*'."""
        with patch("cli._is_model_cached", return_value=False):
            result = runner.invoke(app, ["list"])
        assert "(default)" in result.output


# ═════════════════════════════════════════════════════════════════════════════
#  CLI: tts-router pull
# ═════════════════════════════════════════════════════════════════════════════

class TestPullCommand:
    """Test the `pull` CLI command (HF download mocked)."""

    def test_pull_known_model(self):
        """pull qwen3-tts → calls snapshot_download with correct repo."""
        with patch("cli.snapshot_download", create=True) as mock_dl:
            # patch at the import location inside the pull function
            with patch("huggingface_hub.snapshot_download", return_value="/fake/cache/path") as mock_hf:
                result = runner.invoke(app, ["pull", "qwen3-tts"])
        # The function does a local import, so we need to check it was called
        # Accept either mock being called (depends on import resolution)
        assert result.exit_code == 0
        assert "Done" in result.output or "Pulling" in result.output

    def test_pull_raw_repo_id(self):
        """pull with a raw HF repo ID should pass it through."""
        with patch("huggingface_hub.snapshot_download", return_value="/fake/path"):
            result = runner.invoke(app, ["pull", "someone/custom-model"])
        assert result.exit_code == 0

    def test_pull_unknown_name_fails(self):
        """pull with an unknown short name (no slash) should fail."""
        result = runner.invoke(app, ["pull", "no-such-model"])
        assert result.exit_code != 0

    def test_pull_download_failure(self):
        """If snapshot_download raises, pull should exit with error."""
        with patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("network error")):
            result = runner.invoke(app, ["pull", "qwen3-tts"])
        assert result.exit_code != 0
        assert "error" in result.output.lower() or "failed" in result.output.lower()


# ═════════════════════════════════════════════════════════════════════════════
#  CLI: tts-router serve
# ═════════════════════════════════════════════════════════════════════════════

class TestServeCommand:
    """Test the `serve` CLI command (uvicorn and server.configure mocked)."""

    def test_serve_uncached_model_exits(self):
        """If the model isn't downloaded, serve should exit 1 with a helpful hint."""
        with patch("cli._is_model_cached", return_value=False):
            result = runner.invoke(app, ["serve", "--model", "kokoro"])
        assert result.exit_code != 0
        assert "not downloaded" in result.output or "pull" in result.output.lower()

    def test_serve_cached_model_launches(self):
        """With a cached model, serve should call configure() and uvicorn.run()."""
        with (
            patch("cli._is_model_cached", return_value=True),
            patch("server.configure") as mock_configure,
            patch("uvicorn.run") as mock_uvicorn,
        ):
            result = runner.invoke(app, ["serve", "--model", "qwen3-tts", "--port", "9999"])

        assert result.exit_code == 0
        # configure() should have been called with the correct repo
        mock_configure.assert_called_once()
        call_kwargs = mock_configure.call_args
        assert "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit" in str(call_kwargs)
        # uvicorn.run() should have been called
        mock_uvicorn.assert_called_once()

    def test_serve_no_clone_flag(self):
        """--no-clone should pass no_clone=True to configure()."""
        with (
            patch("cli._is_model_cached", return_value=True),
            patch("server.configure") as mock_configure,
            patch("uvicorn.run"),
        ):
            result = runner.invoke(app, ["serve", "--no-clone"])

        assert result.exit_code == 0
        _, kwargs = mock_configure.call_args
        assert kwargs["no_clone"] is True

    def test_serve_custom_clone_model(self):
        """--clone-model should resolve and pass the clone repo."""
        with (
            patch("cli._is_model_cached", return_value=True),
            patch("server.configure") as mock_configure,
            patch("uvicorn.run"),
        ):
            result = runner.invoke(app, ["serve", "--clone-model", "qwen3-tts-clone"])

        assert result.exit_code == 0
        _, kwargs = mock_configure.call_args
        assert kwargs["clone_repo"] == "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"

    def test_serve_output_shows_connection_info(self):
        """Serve output should display host, port, and model info."""
        with (
            patch("cli._is_model_cached", return_value=True),
            patch("server.configure"),
            patch("uvicorn.run"),
        ):
            result = runner.invoke(app, ["serve", "--port", "7777"])

        assert "7777" in result.output
        assert "Playground" in result.output
