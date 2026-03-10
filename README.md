# tts-router

Local TTS router for Apple Silicon — multi-backend, OpenAI-compatible API.

Pull models, serve an API, or synthesize speech directly from the command line.

## Install

```bash
pip install tts-router
# or
uv pip install tts-router
```

Requires Python 3.11+ and macOS with Apple Silicon (M1/M2/M3/M4).

## Quick Start

```bash
# List available models
tts-router list

# Download a model
tts-router pull qwen3-tts

# Synthesize speech
tts-router say "Hello world" -o hello.wav

# Start the API server (OpenAI-compatible)
tts-router serve
```

## Commands

| Command | Description |
|---------|-------------|
| `tts-router list` | List available models and download status |
| `tts-router pull <model>` | Download model weights from HuggingFace |
| `tts-router say <text>` | Synthesize speech from the command line |
| `tts-router serve` | Start the TTS API server with web playground |

## Models

Supports multiple TTS models through a unified interface:

- **Qwen3-TTS** — Multi-speaker with emotion control and voice cloning (MLX)
- **Kokoro** — Fast, lightweight TTS with many voices (MLX)
- **Dia** — Dialogue generation with speaker tags (MLX)
- **Orpheus** — Expressive speech with emotion tags (MLX)
- **Chatterbox** — Voice cloning with emotion exaggeration (MLX)
- **VibeVoice** — Microsoft's multi-speaker long-form TTS (PyTorch, optional)

Run `tts-router list` to see all available models.

## Voice Cloning

```bash
# Clone a voice from a reference audio clip
tts-router say "Text to speak" --ref-audio reference.wav -o output.wav
```

The model automatically switches to a clone-capable variant when `--ref-audio` is provided.

## API Server

```bash
tts-router serve --port 8091
```

Starts an OpenAI-compatible `/v1/audio/speech` endpoint with a built-in web playground at `http://localhost:8091/`.

## Options

```
tts-router say --help
```

Key options: `--voice`, `--model`, `--format` (wav/mp3/flac/aac/opus), `--speed`, `--instruct`, `--ref-audio`, `--ref-text`, `--temperature`, `--exaggeration`.

## License

MIT
