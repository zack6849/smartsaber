"""Persistent cache for YouTube audio: avoids re-searching and re-downloading.

Cache file: ~/.smartsaber/yt_cache.json

Two levels:
- URL cache  : stores the resolved YouTube URL so re-runs skip the ytsearch step.
- File cache : if the downloaded audio file still exists on disk, return it directly
               (useful when keep_audio=True, or within the same run).

Cache key uses light_norm(title)::light_norm(artist) from utils.py so that
CSV imports (which get unstable file_0, file_1 IDs) stay cache-hit across re-runs
of the same playlist.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from smartsaber.models import Track
from smartsaber.utils import cache_key as _cache_key_raw

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path.home() / ".smartsaber" / "yt_cache.json"


def _cache_key(track: Track) -> str:
    """Stable cache key regardless of source (Spotify ID or CSV file_N)."""
    return _cache_key_raw(track.title, track.artist)


class YTCache:
    """Maps track → {url, audio_path} to speed up repeated imports."""

    def __init__(self, path: Path = _DEFAULT_CACHE_PATH) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load YT cache from %s: %s", self._path, exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save YT cache: %s", exc)

    def get_audio_path(self, track: Track) -> Optional[Path]:
        """Return the cached local audio file if it still exists on disk."""
        entry = self._data.get(_cache_key(track))
        if not entry:
            return None
        path_str = entry.get("audio_path")
        if not path_str:
            return None
        p = Path(path_str)
        return p if p.exists() else None

    def get_url(self, track: Track) -> Optional[str]:
        """Return the cached YouTube URL (even if the local file is gone)."""
        entry = self._data.get(_cache_key(track))
        return entry.get("url") if entry else None

    def put_url(self, track: Track, url: str) -> None:
        """Persist a resolved URL immediately, before the audio is downloaded.
        Preserves any existing audio_path so a later put() doesn't lose it."""
        with self._lock:
            key = _cache_key(track)
            entry = self._data.get(key, {})
            entry.update({"title": track.title, "artist": track.artist, "url": url})
            self._data[key] = entry
            self._save()

    def put(self, track: Track, url: str, audio_path: Path) -> None:
        """Store a resolved URL and downloaded file path."""
        with self._lock:
            self._data[_cache_key(track)] = {
                "title": track.title,
                "artist": track.artist,
                "url": url,
                "audio_path": str(audio_path),
            }
            self._save()
