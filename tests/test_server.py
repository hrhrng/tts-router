"""
Tests for server.py — configure(), startup behavior, API endpoints.

The MLX model loading is mocked so tests run without GPU / model weights.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport

import server


# ═════════════════════════════════════════════════════════════════════════════
#  configure()
# ═════════════════════════════════════════════════════════════════════════════

class TestConfigure:
    """Test that configure() properly injects model settings into server globals."""

    def setup_method(self):
        """Reset server globals before each test."""
        server._model_repo = None
        server._clone_model_id = server.DEFAULT_CLONE_MODEL
        server._clone_disabled = False

    def test_configure_sets_model_repo(self):
        server.configure(model_repo="org/my-model")
        assert server._model_repo == "org/my-model"

    def test_configure_sets_clone_repo(self):
        server.configure(model_repo="org/main", clone_repo="org/clone")
        assert server._clone_model_id == "org/clone"

    def test_configure_no_clone(self):
        server.configure(model_repo="org/main", no_clone=True)
        assert server._clone_disabled is True

    def test_configure_without_clone_repo_keeps_default(self):
        """If clone_repo is None, the default clone model ID should be preserved."""
        original = server._clone_model_id
        server.configure(model_repo="org/main")
        assert server._clone_model_id == original

    def test_configure_overrides_previous(self):
        """Calling configure() twice should overwrite the first call."""
        server.configure(model_repo="org/first")
        server.configure(model_repo="org/second")
        assert server._model_repo == "org/second"


# ═════════════════════════════════════════════════════════════════════════════
#  FastAPI endpoints (model loading mocked)
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_model():
    """Provide a fake model object and patch load_model to return it."""
    fake_model = MagicMock(name="FakeTTSModel")
    with patch("server.load_model", return_value=fake_model):
        yield fake_model


@pytest.fixture
def configured_app(mock_model):
    """Configure server with a test repo and return the FastAPI app."""
    server.configure(model_repo="test-org/test-model")
    # Pre-set the model so startup doesn't actually load
    server._model = mock_model
    return server.app


@pytest.mark.anyio
async def test_health_endpoint(configured_app):
    """GET /health should return status ok and model info."""
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert "clone_enabled" in data


@pytest.mark.anyio
async def test_voices_endpoint(configured_app):
    """GET /v1/audio/voices should return voice list."""
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/audio/voices")
    assert resp.status_code == 200
    data = resp.json()
    assert "voices" in data
    assert "Vivian" in data["voices"]
    assert "by_language" in data


@pytest.mark.anyio
async def test_speech_returns_503_when_model_not_loaded():
    """POST /v1/audio/speech should return 503 if model is None."""
    server._model = None  # force unloaded state
    transport = ASGITransport(app=server.app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/audio/speech", json={
            "input": "hello",
            "voice": "Vivian",
        })
    assert resp.status_code == 503


@pytest.mark.anyio
async def test_speech_bad_format(configured_app):
    """POST /v1/audio/speech with unsupported format should 400."""
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/audio/speech", json={
            "input": "hello",
            "voice": "Vivian",
            "response_format": "xyz_invalid",
        })
    assert resp.status_code == 400


@pytest.mark.anyio
async def test_dashscope_missing_text(configured_app):
    """DashScope endpoint should reject requests without text (JSON body, status_code 400)."""
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/v1/services/aigc/multimodal-generation/generation",
            json={"model": "qwen3-tts", "input": {}},
        )
    # The endpoint returns HTTP 400 with a JSON body containing the error detail
    assert resp.status_code == 400
    data = resp.json()
    assert data["status_code"] == 400
    assert "text" in data["message"].lower()


@pytest.mark.anyio
async def test_references_list_empty(configured_app):
    """GET /v1/audio/references should return empty list initially."""
    server._clone_disabled = False
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/audio/references")
    assert resp.status_code == 200
    data = resp.json()
    assert "references" in data


@pytest.mark.anyio
async def test_references_blocked_when_clone_disabled(configured_app):
    """Reference endpoints should 403 when clone is disabled."""
    server._clone_disabled = True
    transport = ASGITransport(app=configured_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/audio/references")
    assert resp.status_code == 403


# ═════════════════════════════════════════════════════════════════════════════
#  Startup event
# ═════════════════════════════════════════════════════════════════════════════

class TestStartup:
    """Test the startup event handler."""

    def setup_method(self):
        server._model = None
        server._model_repo = None

    @pytest.mark.anyio
    async def test_startup_uses_default_when_not_configured(self):
        """If configure() was never called, startup should fall back to DEFAULT_MODEL_REPO."""
        fake_model = MagicMock()
        with patch("server.load_model", return_value=fake_model) as mock_load:
            await server.load_model_on_startup()
        mock_load.assert_called_once_with(server.DEFAULT_MODEL_REPO)
        assert server._model is fake_model

    @pytest.mark.anyio
    async def test_startup_uses_configured_repo(self):
        """If configure() was called, startup should use that repo."""
        server.configure(model_repo="custom-org/custom-model")
        fake_model = MagicMock()
        with patch("server.load_model", return_value=fake_model) as mock_load:
            await server.load_model_on_startup()
        mock_load.assert_called_once_with("custom-org/custom-model")
