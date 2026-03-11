# Agent Development Guide — tts-router

## Project Structure

```
cli.py              Typer CLI entry point (MODEL_REGISTRY + pull/list/serve commands)
server.py           FastAPI server (DashScope + OpenAI TTS endpoints, configure())
extractors.py       URL audio extraction (HTTP direct + yt-dlp)
playground.html     Single-file Web UI (HTML + CSS + JS, ~2600 lines)
pyproject.toml      Project metadata, dependencies, CLI entry point
tests/              pytest test suite (test_cli.py, test_server.py)
.claude/skills/     Agent skill documentation
```

## Single-File UI Convention — `playground.html`

The playground is intentionally kept as a **single HTML file** (no build step, no bundler).
This makes it trivial to serve (`FileResponse`) and deploy, but requires discipline when editing.

### Module Index

The file starts with a **MODULE INDEX** comment block that maps logical sections to approximate
line numbers. This index exists so agents (and humans) can navigate the 2600+ line file quickly.

**MANDATORY RULE**: After any edit to `playground.html`, you MUST update the MODULE INDEX
line numbers at the top of the file to reflect the current state. Stale line numbers are
worse than no index at all.

How to update:
1. Finish your edits
2. Search for the key section headers (they use `═══` or `──` comment delimiters)
3. Update the `~Lnnn` numbers in the MODULE INDEX to match actual line numbers

### Section Conventions

- **CSS**: All styles live inside `<style>` in `<head>`. Organized by component with
  `/* ── Section Name ───── */` headers.
- **HTML**: Semantic structure with `id` attributes for JS binding. Tab panels use
  `class="tab-panel"` with `active` toggle.
- **JavaScript**: All logic inside `<script>` at the bottom. Organized with
  `/* ═══ Section Name ═══ */` headers. Uses a centralized `state` object and
  a `els` DOM cache populated on `DOMContentLoaded`.

### Adding New Features

1. Add CSS in the appropriate section (or create a new section with a header comment)
2. Add HTML in the correct tab panel or results section
3. Add JS handler, cache new DOM refs in the `els` object
4. Wire up event listeners in the `DOMContentLoaded` init block
5. **Update the MODULE INDEX**

## Backend Architecture

### `cli.py` — CLI Layer

- `MODEL_REGISTRY` dict maps short names to HF repo IDs + metadata
- `_resolve_repo(model)` — resolves short name or passthrough for raw HF IDs
- `_is_model_cached(repo_id)` — checks HuggingFace local cache via `scan_cache_dir()`
- `serve` command calls `server.configure()` before `uvicorn.run()`

### `server.py` — API Layer

- `configure(model_repo, clone_repo, no_clone)` — injected by CLI before startup
- `load_model_on_startup()` — FastAPI startup event, loads from `_model_repo`
- `_gen_lock` — async lock serializing all generation (MLX is not thread-safe)
- `_ensure_clone_model()` — lazy-loads clone model on first clone request

Key endpoints:
- `POST /v1/audio/speech` — OpenAI-compatible
- `POST /api/v1/services/aigc/multimodal-generation/generation` — DashScope-compatible
- `POST /v1/audio/clone` — voice cloning
- `GET /health` — returns model status, model_id, clone status

### `extractors.py` — Audio Extraction

Dependency-inverted chain of responsibility pattern:
- `BaseExtractor` ABC with `can_handle()` + `extract()`
- `HttpExtractor` — direct audio file URLs (zero deps)
- `YtDlpExtractor` — YouTube/Bilibili/1000+ sites (optional yt-dlp)
- `create_extractor_chain()` — factory based on available deps

## Testing

```bash
uv run --group dev pytest tests/ -v
```

All HuggingFace / uvicorn side effects are mocked. Tests run offline and fast (~0.6s).

## Workflow

```bash
uv run tts-router pull qwen3-tts       # download model
uv run tts-router serve                 # start server
uv run tts-router list                  # check model status
uv run --group dev pytest tests/ -v     # run tests
```

## Release SOP

Follow these steps in order when cutting a new release:

1. **Bump version** in `pyproject.toml`:
   ```
   version = "x.y.z"
   ```

2. **Commit and push** the version bump to `main`.

3. **Delete the old tag if it exists** (e.g. if you tagged before bumping pyproject.toml):
   ```bash
   git tag -d vx.y.z
   git push origin :refs/tags/vx.y.z
   ```

4. **Re-tag at the latest commit**:
   ```bash
   git tag vx.y.z
   git push origin vx.y.z
   ```

5. **Create a GitHub Release** from the tag — this triggers `publish.yml` which:
   - Stamps `cli_version` in `skills/tts-router/SKILL.md`
   - Builds and publishes to PyPI via trusted publishing
   ```bash
   gh release create vx.y.z --title "vx.y.z" --generate-notes
   ```

6. **Update the global skill** after release:
   ```bash
   npx skills install ./skills/tts-router --yes --global
   ```

> **Why this order matters**: `pyproject.toml` must be bumped before tagging, because
> the CI reads the version from `pyproject.toml` to stamp the skill. Tagging before
> bumping results in the skill and PyPI package having mismatched versions.
