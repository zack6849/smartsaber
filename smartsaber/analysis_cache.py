"""Persistent cache for AudioAnalysis results — avoids re-running librosa.

Cache file: ~/.smartsaber/analysis_cache.json

Key: audio filename stem (e.g. "yt_dQw4w9WgXcQ") — stable for a given
YouTube video, so re-runs with keep_audio=True skip the expensive librosa
analysis entirely even when --force is used to regenerate notes.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Optional

from smartsaber.models import AudioAnalysis

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_PATH = Path.home() / ".smartsaber" / "analysis_cache.json"


class AnalysisCache:
    """Maps audio file stem → AudioAnalysis to skip librosa on re-runs."""

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
                logger.warning("Could not load analysis cache from %s: %s", self._path, exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._path.write_text(
                json.dumps(self._data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Could not save analysis cache: %s", exc)

    def get(self, key: str) -> Optional[AudioAnalysis]:
        """Return cached AudioAnalysis or None if not cached."""
        entry = self._data.get(key)
        if not entry:
            return None
        try:
            return AudioAnalysis(
                tempo=entry["tempo"],
                beat_times=entry["beat_times"],
                onset_times=entry["onset_times"],
                rms_curve=entry["rms_curve"],
                rms_times=entry["rms_times"],
                segment_times=entry["segment_times"],
                duration_s=entry["duration_s"],
            )
        except Exception as exc:
            logger.warning("Corrupt analysis cache entry for '%s': %s", key, exc)
            return None

    def put(self, key: str, analysis: AudioAnalysis) -> None:
        """Persist an AudioAnalysis result."""
        with self._lock:
            self._data[key] = {
                "tempo": analysis.tempo,
                "beat_times": analysis.beat_times,
                "onset_times": analysis.onset_times,
                "rms_curve": analysis.rms_curve,
                "rms_times": analysis.rms_times,
                "segment_times": analysis.segment_times,
                "duration_s": analysis.duration_s,
            }
            self._save()
