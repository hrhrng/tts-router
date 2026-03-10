# Skill: tts-router — Local TTS Router for Apple Silicon

## What is tts-router?

A productized CLI that manages and serves multiple TTS models locally on Apple Silicon (MLX).
Models are downloaded from HuggingFace Hub and served via DashScope + OpenAI compatible APIs.

## Prerequisites

- macOS with Apple Silicon (M1/M2/M3/M4)
- `uv` installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- ffmpeg installed (`brew install ffmpeg`)

## Commands

### `tts-router list` — Show available models

```bash
uv run tts-router list
```

Output shows all models from `MODEL_REGISTRY` in `cli.py`, with download status (cached / not pulled).

### `tts-router pull <model>` — Download model weights

```bash
# By short name (from MODEL_REGISTRY)
uv run tts-router pull qwen3-tts
uv run tts-router pull kokoro

# By raw HuggingFace repo ID
uv run tts-router pull mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit
```

Models are cached in `~/.cache/huggingface/hub/`. No need to re-download.

### `tts-router serve` — Start the TTS API server

```bash
# Default: qwen3-tts on port 8091
uv run tts-router serve

# Custom model and port
uv run tts-router serve --model kokoro --port 9000

# Disable voice cloning
uv run tts-router serve --no-clone

# Custom clone model
uv run tts-router serve --clone-model qwen3-tts-clone
```

The server **requires models to be pulled first**. If not pulled, it exits with a hint.

## Available Models

| Short Name         | HF Repo                                                   | Features                        |
| ------------------ | --------------------------------------------------------- | ------------------------------- |
| `qwen3-tts`        | `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit`     | multi-speaker, emotion, instruct (default) |
| `qwen3-tts-design` | `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit`     | free-form voice description     |
| `qwen3-tts-clone`  | `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`            | voice cloning with ref audio    |
| `kokoro`           | `prince-canuma/Kokoro-82M`                                | fast, lightweight, multi-lang   |

## API Endpoints (when serving)

| Endpoint                                                  | Method | Description                    |
| --------------------------------------------------------- | ------ | ------------------------------ |
| `GET /`                                                   | GET    | Playground UI                  |
| `POST /v1/audio/speech`                                   | POST   | OpenAI-compatible TTS          |
| `POST /api/v1/services/aigc/multimodal-generation/generation` | POST | DashScope-compatible TTS       |
| `GET /v1/audio/voices`                                    | GET    | List available voices          |
| `GET /health`                                             | GET    | Health check                   |
| `POST /v1/audio/references/upload`                        | POST   | Upload reference audio (clone) |
| `POST /v1/audio/references/from-url`                      | POST   | Download ref audio from URL    |
| `POST /v1/audio/clone`                                    | POST   | Voice clone generation         |

## Quick Start for Agent

```bash
# 1. Pull the default model
uv run tts-router pull qwen3-tts

# 2. (Optional) Pull clone model for voice cloning
uv run tts-router pull qwen3-tts-clone

# 3. Start the server
uv run tts-router serve

# 4. Generate speech (OpenAI format)
curl -X POST http://localhost:8091/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "Vivian"}' \
  --output output.wav

# 5. Generate speech with emotion (DashScope format)
curl -X POST http://localhost:8091/api/v1/services/aigc/multimodal-generation/generation \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": {
      "text": "这是一段开心的语音",
      "voice": "Vivian",
      "instructions": "用开心的语气"
    }
  }' --output happy.wav
```

## Project Architecture

```
cli.py          — Typer CLI entry point (MODEL_REGISTRY + pull/list/serve)
server.py       — FastAPI server (DashScope + OpenAI endpoints, configure())
extractors.py   — URL audio extraction (HTTP direct + yt-dlp)
playground.html — Browser-based TTS playground UI
pyproject.toml  — Dependencies and project metadata
tests/          — pytest test suite
```

## Key Design Decisions

- **HuggingFace Hub cache**: Models live in `~/.cache/huggingface/`, not in the project tree.
  `tts-router pull` calls `snapshot_download()`, `tts-router serve` passes the repo ID
  directly to `mlx_audio.tts.utils.load_model()`.
- **configure() injection**: `cli.py` calls `server.configure()` before launching uvicorn
  to decouple server from hardcoded paths. `python server.py` still works as a fallback.
- **uv-first workflow**: `pyproject.toml` + `uv.lock` for reproducible deps.
  First `uv run` auto-creates `.venv` and installs everything.

## Running Tests

```bash
uv run --group dev pytest tests/ -v
```
