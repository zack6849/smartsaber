"""YouTube audio fetching via yt-dlp."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from urllib.parse import urlparse, parse_qs

import yt_dlp

from smartsaber.models import Track
from smartsaber.utils import normalize_string

if TYPE_CHECKING:
    from smartsaber.yt_cache import YTCache

logger = logging.getLogger(__name__)

_DURATION_TOLERANCE_S = 30


@dataclass
class YouTubeResolution:
    """Result of resolving a YouTube URL for one track (before download)."""
    track: Track
    url: Optional[str]
    duration_diff_s: float   # seconds off from the Spotify duration (0 if N/A)
    source: str              # "search" | "cache_file" | "cache_url" | "override" | "not_found"


def _build_ydl_opts(out_path: Path, prefer_ogg: bool = True) -> dict:
    postprocessors = []
    if prefer_ogg:
        postprocessors.append(
            {"key": "FFmpegExtractAudio", "preferredcodec": "vorbis", "preferredquality": "0"}
        )
    else:
        postprocessors.append(
            {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
        )
    return {
        "format": "bestaudio/best",
        "outtmpl": str(out_path / "%(title)s.%(ext)s"),
        "postprocessors": postprocessors,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
    }


def _build_query(track: Track) -> str:
    return f"{normalize_string(track.artist)} {normalize_string(track.title)} audio"


def _search_best_url(track: Track) -> tuple[Optional[str], float]:
    """
    Search YouTube (top 3 results) and return (best_url, duration_diff_seconds).
    Returns (None, 0.0) on failure.
    """
    query = f"ytsearch3:{_build_query(track)}"
    info_opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "noplaylist": True}

    target_s = track.duration_s
    best_url: Optional[str] = None
    best_diff = float("inf")

    with yt_dlp.YoutubeDL(info_opts) as ydl:
        try:
            result = ydl.extract_info(query, download=False)
        except Exception as exc:
            logger.warning("yt-dlp search failed for '%s': %s", track.title, exc)
            return None, 0.0

        for entry in result.get("entries") or []:
            dur = entry.get("duration") or 0
            diff = abs(dur - target_s)
            if diff < best_diff:
                best_diff = diff
                best_url = entry.get("url") or entry.get("webpage_url")

    if not best_url:
        logger.warning("No YouTube results for '%s'", track.title)
        return None, 0.0

    if best_diff > _DURATION_TOLERANCE_S:
        logger.warning(
            "Duration mismatch for '%s': best match differs by %.0fs",
            track.title, best_diff,
        )

    return best_url, best_diff


def resolve_url(
    track: Track,
    cache: Optional["YTCache"] = None,
    override_url: Optional[str] = None,
) -> YouTubeResolution:
    """
    Resolve a YouTube URL for a track WITHOUT downloading.
    Used for the interactive review step before the generation phase begins.
    """
    # Override takes absolute priority
    if override_url:
        return YouTubeResolution(track=track, url=override_url, duration_diff_s=0.0, source="override")

    if cache:
        # Best case: file still on disk
        cached_path = cache.get_audio_path(track)
        if cached_path:
            return YouTubeResolution(
                track=track, url=cache.get_url(track),
                duration_diff_s=0.0, source="cache_file",
            )
        # Second best: URL is known, just needs re-download
        cached_url = cache.get_url(track)
        if cached_url:
            return YouTubeResolution(track=track, url=cached_url, duration_diff_s=0.0, source="cache_url")

    # Fall through to live search
    url, diff = _search_best_url(track)
    if url:
        return YouTubeResolution(track=track, url=url, duration_diff_s=diff, source="search")
    return YouTubeResolution(track=track, url=None, duration_diff_s=0.0, source="not_found")


def _yt_video_id(url: str) -> Optional[str]:
    """Extract the YouTube video ID from any common URL format."""
    parsed = urlparse(url)
    if parsed.hostname in ("youtu.be",):
        return parsed.path.lstrip("/").split("/")[0] or None
    if parsed.hostname in ("www.youtube.com", "youtube.com", "m.youtube.com"):
        return parse_qs(parsed.query).get("v", [None])[0]
    return None


def _download_url(url: str, track: Track, tmp_dir: Path, prefer_ogg: bool) -> Optional[Path]:
    """Download a specific YouTube URL. Returns the local file path or None."""
    video_id = _yt_video_id(url)
    safe_name = f"yt_{video_id}" if video_id else f"audio_{track.source_id}"
    dl_opts = _build_ydl_opts(tmp_dir, prefer_ogg=prefer_ogg)
    dl_opts["outtmpl"] = str(tmp_dir / f"{safe_name}.%(ext)s")

    with yt_dlp.YoutubeDL(dl_opts) as ydl:
        try:
            ydl.download([url])
        except Exception as exc:
            logger.warning("yt-dlp download failed for '%s': %s", track.title, exc)
            return None

    for candidate in tmp_dir.glob(f"{safe_name}.*"):
        return candidate

    logger.warning("Downloaded file not found for '%s'", track.title)
    return None


def fetch_audio(
    track: Track,
    output_dir: Path,
    prefer_ogg: bool = True,
    cache: Optional["YTCache"] = None,
    resolution: Optional[YouTubeResolution] = None,
) -> Optional[Path]:
    """
    Fetch audio for a track.

    If a pre-resolved YouTubeResolution is supplied (from the interactive review
    step), it is used directly — skipping the search entirely.  Otherwise the
    full search → cache → download flow runs.
    """
    tmp_dir = output_dir / "_audio_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # --- Path 1: pre-resolved result from the interactive review step ---
    if resolution is not None:
        if resolution.source == "cache_file" and cache:
            cached = cache.get_audio_path(track)
            if cached:
                logger.info("Using cached file for '%s'", track.title)
                return cached
        # File was deleted or not a file-cache hit — download the resolved URL
        if not resolution.url:
            return None
        audio_path = _download_url(resolution.url, track, tmp_dir, prefer_ogg)
        if audio_path and cache:
            cache.put(track, resolution.url, audio_path)
        return audio_path

    # --- Path 2: no pre-resolved result — full search + cache flow ---
    if cache:
        cached = cache.get_audio_path(track)
        if cached:
            return cached
        cached_url = cache.get_url(track)
        if cached_url:
            audio_path = _download_url(cached_url, track, tmp_dir, prefer_ogg)
            if audio_path:
                cache.put(track, cached_url, audio_path)
            return audio_path

    url, _ = _search_best_url(track)
    if not url:
        return None

    audio_path = _download_url(url, track, tmp_dir, prefer_ogg)
    if audio_path and cache:
        cache.put(track, url, audio_path)
    return audio_path
