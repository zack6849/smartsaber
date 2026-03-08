"""Persistent cache for BeatSaver search results.

Cache file: ~/.smartsaber/bs_cache.json

Stores both hits (map found) and misses (no map found) to avoid
redundant API calls on repeated imports of the same playlist.

Cache key: normalize_string(title)::normalize_string(artist)
(stable across CSV re-imports with unstable source_ids)

To force a fresh search for all tracks, delete the cache file.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from smartsaber.models import BeatSaverMap, BeatSaverMatch, Track
from smartsaber.utils import normalize_string

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path.home() / ".smartsaber" / "bs_cache.json"


def _cache_key(track: Track) -> str:
    return f"{normalize_string(track.title)}::{normalize_string(track.artist)}"


def _map_to_dict(m: BeatSaverMap) -> dict:
    return {
        "id": m.id,
        "name": m.name,
        "artist": m.artist,
        "bpm": m.bpm,
        "duration_s": m.duration_s,
        "upvotes": m.upvotes,
        "downvotes": m.downvotes,
        "download_url": m.download_url,
        "difficulties": m.difficulties,
        "hash": m.hash,
    }


def _map_from_dict(d: dict) -> BeatSaverMap:
    return BeatSaverMap(
        id=d["id"],
        name=d["name"],
        artist=d["artist"],
        bpm=d["bpm"],
        duration_s=d["duration_s"],
        upvotes=d["upvotes"],
        downvotes=d["downvotes"],
        download_url=d["download_url"],
        difficulties=d["difficulties"],
        hash=d["hash"],
    )


class BSCache:
    """Maps track → BeatSaverMatch (or confirmed miss) to skip repeat searches."""

    def __init__(self, path: Path = _DEFAULT_CACHE_PATH) -> None:
        self._path = path
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Could not load BS cache from %s: %s", self._path, exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save BS cache: %s", exc)

    def get(self, track: Track) -> tuple[bool, Optional[BeatSaverMatch]]:
        """
        Returns (cached, match).
        - cached=False: not in cache — caller should search.
        - cached=True, match=None: previously searched, nothing found.
        - cached=True, match=BeatSaverMatch: cached hit.
        """
        key = _cache_key(track)
        if key not in self._data:
            return False, None
        entry = self._data[key]
        if entry.get("miss"):
            return True, None
        try:
            match = BeatSaverMatch(
                track=track,
                map=_map_from_dict(entry["map"]),
                title_score=entry["title_score"],
                artist_score=entry["artist_score"],
            )
            return True, match
        except Exception as exc:
            logger.warning("Corrupted BS cache entry for '%s': %s", track.title, exc)
            return False, None

    def put_match(self, track: Track, match: BeatSaverMatch) -> None:
        self._data[_cache_key(track)] = {
            "title": track.title,
            "artist": track.artist,
            "map": _map_to_dict(match.map),
            "title_score": match.title_score,
            "artist_score": match.artist_score,
        }
        self._save()

    def put_miss(self, track: Track) -> None:
        self._data[_cache_key(track)] = {
            "title": track.title,
            "artist": track.artist,
            "miss": True,
        }
        self._save()
