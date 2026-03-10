"""
Audio Extractors — Dependency-Inverted URL Audio Extraction
============================================================
Defines an abstract interface + concrete implementations for downloading
audio from URLs and converting to WAV format for voice cloning references.

Architecture:
  BaseExtractor (ABC)         — abstract interface: can_handle(url), extract(url)
  HttpExtractor               — direct HTTP download for .mp3/.wav/.flac etc. (zero extra deps)
  YtDlpExtractor              — YouTube/Bilibili/1000+ sites via yt-dlp (optional dependency)
  create_extractor_chain()    — factory: returns list of available extractors based on installed deps
"""

import os
import re
import subprocess
import tempfile
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlparse, unquote

# ── Optional yt-dlp import ────────────────────────────────────────────────────
# yt-dlp is an optional dependency. If not installed, YtDlpExtractor won't be
# registered in the extractor chain — the system gracefully degrades.
_YT_DLP_AVAILABLE = False
try:
    import yt_dlp  # type: ignore
    _YT_DLP_AVAILABLE = True
except ImportError:
    pass


# ── Result dataclass ──────────────────────────────────────────────────────────

@dataclass
class AudioExtractResult:
    """Holds the result of a successful audio extraction."""
    path: str              # absolute path to the converted WAV file
    title: str             # human-readable name (filename or video title)
    duration: float = 0.0  # duration in seconds (populated after ffmpeg conversion)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseExtractor(ABC):
    """
    Abstract base class for URL audio extractors.
    Subclasses must implement `name`, `can_handle()`, and `extract()`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this extractor (shown in /health)."""
        ...

    @abstractmethod
    def can_handle(self, url: str) -> bool:
        """Return True if this extractor knows how to process the given URL."""
        ...

    @abstractmethod
    def extract(self, url: str, output_dir: str) -> AudioExtractResult:
        """
        Download and convert the audio at `url` into a 16kHz mono WAV file.
        The WAV file should be placed under `output_dir`.
        Raises RuntimeError on failure.
        """
        ...

    def _convert_to_wav(self, src_path: str, dst_path: str) -> float:
        """
        Convert any audio file to 16kHz mono WAV via ffmpeg.
        Returns the duration in seconds.
        """
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src_path,
             "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", dst_path],
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg conversion failed: {proc.stderr.decode()}")

        # Read duration from the converted file via ffprobe
        try:
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", dst_path],
                capture_output=True, timeout=10,
            )
            return float(probe.stdout.decode().strip())
        except Exception:
            return 0.0


# ── HttpExtractor — direct audio link download (zero extra deps) ──────────────

# Audio extensions recognized for direct download
_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".webm", ".opus"}

class HttpExtractor(BaseExtractor):
    """
    Downloads direct audio file links (URLs ending in .mp3/.wav/.flac/etc.
    or serving audio/* Content-Type). Uses only stdlib urllib + ffmpeg.
    """

    @property
    def name(self) -> str:
        return "http"

    def can_handle(self, url: str) -> bool:
        """
        Match if the URL path ends with a known audio extension,
        or if a HEAD request returns an audio/* Content-Type.
        """
        parsed = urlparse(url)
        # Must be http or https
        if parsed.scheme not in ("http", "https"):
            return False

        # Check extension in URL path (ignoring query params)
        path_lower = parsed.path.lower()
        for ext in _AUDIO_EXTENSIONS:
            if path_lower.endswith(ext):
                return True

        # Fallback: probe Content-Type with a HEAD request
        try:
            req = urllib.request.Request(url, method="HEAD")
            req.add_header("User-Agent", "Mozilla/5.0 (Qwen-TTS/1.0)")
            with urllib.request.urlopen(req, timeout=10) as resp:
                ct = resp.headers.get("Content-Type", "")
                if ct.startswith("audio/"):
                    return True
        except Exception:
            pass

        return False

    def extract(self, url: str, output_dir: str) -> AudioExtractResult:
        """Download the audio file and convert to 16kHz mono WAV."""
        parsed = urlparse(url)
        # Derive a human-readable filename from the URL
        filename = unquote(os.path.basename(parsed.path)) or "download"

        # Determine extension for the temp download
        ext = os.path.splitext(filename)[1].lower()
        if ext not in _AUDIO_EXTENSIONS:
            ext = ".tmp"

        raw_path = os.path.join(output_dir, f"raw_download{ext}")
        wav_path = os.path.join(output_dir, "extracted.wav")

        # Download the file
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "Mozilla/5.0 (Qwen-TTS/1.0)")
            with urllib.request.urlopen(req, timeout=120) as resp:
                # Enforce 50MB limit
                content_len = resp.headers.get("Content-Length")
                if content_len and int(content_len) > 50 * 1024 * 1024:
                    raise RuntimeError("File too large (max 50MB)")
                with open(raw_path, "wb") as f:
                    total = 0
                    while True:
                        chunk = resp.read(8192)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > 50 * 1024 * 1024:
                            raise RuntimeError("File too large (max 50MB)")
                        f.write(chunk)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Download failed: {e}")

        # Convert to WAV
        duration = self._convert_to_wav(raw_path, wav_path)

        # Clean up raw download
        if os.path.exists(raw_path):
            os.remove(raw_path)

        return AudioExtractResult(path=wav_path, title=filename, duration=round(duration, 2))


# ── YtDlpExtractor — YouTube/Bilibili/etc via yt-dlp (optional dep) ──────────

class YtDlpExtractor(BaseExtractor):
    """
    Extracts audio from YouTube, Bilibili, and 1000+ sites using yt-dlp.
    This extractor is only available when yt-dlp is installed.
    """

    @property
    def name(self) -> str:
        return "yt-dlp"

    def can_handle(self, url: str) -> bool:
        """
        Match any http/https URL that looks like a video/streaming site.
        Specifically targets YouTube and Bilibili, but also handles any URL
        that yt-dlp might support (we let yt-dlp try and fail gracefully).
        """
        if not _YT_DLP_AVAILABLE:
            return False

        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return False

        host = parsed.netloc.lower()
        # Known video platforms — match generously
        known_hosts = [
            "youtube.com", "youtu.be", "www.youtube.com", "m.youtube.com",
            "bilibili.com", "www.bilibili.com", "b23.tv",
            "vimeo.com", "dailymotion.com", "soundcloud.com",
            "twitter.com", "x.com", "tiktok.com",
        ]
        for h in known_hosts:
            if host == h or host.endswith("." + h):
                return True

        # For unknown hosts, return True as a fallback —
        # yt-dlp supports 1000+ sites, let it try
        # But only if HttpExtractor didn't already claim it (audio file extension)
        path_lower = parsed.path.lower()
        for ext in _AUDIO_EXTENSIONS:
            if path_lower.endswith(ext):
                return False  # let HttpExtractor handle direct audio links
        return True

    def extract(self, url: str, output_dir: str) -> AudioExtractResult:
        """Download best audio via yt-dlp, then convert to 16kHz mono WAV."""
        if not _YT_DLP_AVAILABLE:
            raise RuntimeError("yt-dlp is not installed. Run: pip install yt-dlp")

        raw_path = os.path.join(output_dir, "yt_audio.%(ext)s")
        wav_path = os.path.join(output_dir, "extracted.wav")

        # Configure yt-dlp to extract best audio
        ydl_opts = {
            "format": "bestaudio/best",
            "outtmpl": raw_path,
            "max_filesize": 50 * 1024 * 1024,  # 50MB limit
            "noplaylist": True,                 # single video only
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
        }

        title = url  # fallback title

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get("title", url)
                # Find the actual downloaded file (extension may vary)
                downloaded = ydl.prepare_filename(info)
                if not os.path.exists(downloaded):
                    # yt-dlp sometimes post-processes and changes extension
                    # Search the output dir for the downloaded file
                    for f in os.listdir(output_dir):
                        if f.startswith("yt_audio") and not f.endswith(".wav"):
                            downloaded = os.path.join(output_dir, f)
                            break
                if not os.path.exists(downloaded):
                    raise RuntimeError("yt-dlp downloaded no file")

        except Exception as e:
            error_msg = str(e)
            if "is not a valid URL" in error_msg or "Unsupported URL" in error_msg:
                raise RuntimeError(f"URL not supported by yt-dlp: {url}")
            raise RuntimeError(f"yt-dlp extraction failed: {error_msg}")

        # Convert to WAV
        duration = self._convert_to_wav(downloaded, wav_path)

        # Clean up raw download
        for f in os.listdir(output_dir):
            fpath = os.path.join(output_dir, f)
            if fpath != wav_path and os.path.isfile(fpath):
                os.remove(fpath)

        return AudioExtractResult(path=wav_path, title=title, duration=round(duration, 2))


# ── Factory function ──────────────────────────────────────────────────────────

def create_extractor_chain() -> list[BaseExtractor]:
    """
    Build the ordered list of available extractors.
    Order matters: more specific extractors go first.

    - YtDlpExtractor is only included when yt-dlp is importable
    - HttpExtractor is always available (zero extra deps)
    """
    chain: list[BaseExtractor] = []

    # YtDlpExtractor — optional, higher priority for video URLs
    if _YT_DLP_AVAILABLE:
        chain.append(YtDlpExtractor())

    # HttpExtractor — always available as fallback for direct audio links
    chain.append(HttpExtractor())

    return chain


def get_extractor_names(chain: list[BaseExtractor]) -> list[str]:
    """Return the names of all extractors in the chain (for /health endpoint)."""
    return [e.name for e in chain]
