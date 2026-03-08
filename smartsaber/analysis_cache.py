"""Persistent cache for AudioAnalysis results — avoids re-running librosa.

Cache directory: ~/.smartsaber/analysis/
Each entry is stored as a separate JSON file named by its cache key,
so lookups and writes are O(1) instead of loading a giant monolithic file.

Key: audio filename stem (e.g. "yt_dQw4w9WgXcQ") — stable for a given
YouTube video, so re-runs with keep_audio=True skip the expensive librosa
analysis entirely even when --force is used to regenerate notes.

Each entry stores a "_version" field.  When the analyser changes (new features,
tuned parameters) we bump _CACHE_VERSION and stale entries are treated as
cache-misses so they get re-analysed automatically.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path
from typing import Optional

from smartsaber.models import AudioAnalysis

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path.home() / ".smartsaber" / "analysis"

# Legacy monolithic cache file — migrated on first use then deleted.
_LEGACY_CACHE_PATH = Path.home() / ".smartsaber" / "analysis_cache.json"

# Bump this whenever the analyser output format changes (new fields, tuned
# delta thresholds, etc.) so that old cached entries are automatically
# re-analysed instead of silently returning stale data.
_CACHE_VERSION = 5

# Allowed characters in cache filenames (safe for all OSes)
_SAFE_RE = re.compile(r'[^a-zA-Z0-9_\-]')


def _safe_key(key: str) -> str:
    """Convert a cache key to a safe filename component."""
    return _SAFE_RE.sub('_', key)[:200]


class AnalysisCache:
    """Maps audio file stem → AudioAnalysis to skip librosa on re-runs.

    Each entry is stored as a separate small JSON file under the cache dir.
    This avoids loading/writing a single 100MB+ file on every operation.
    """

    def __init__(self, cache_dir: Path = _DEFAULT_CACHE_DIR) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # In-memory LRU so we don't re-read the same file within one process
        self._mem: dict[str, AudioAnalysis] = {}
        # Legacy monolithic file lives alongside the cache dir
        self._legacy_path = self._dir.parent / "analysis_cache.json"
        self._migrate_legacy()

    def _migrate_legacy(self) -> None:
        """One-time migration from the old monolithic analysis_cache.json."""
        if not self._legacy_path.exists():
            return
        try:
            data = json.loads(self._legacy_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            migrated = 0
            for key, entry in data.items():
                if not isinstance(entry, dict):
                    continue
                dest = self._dir / f"{_safe_key(key)}.json"
                if not dest.exists():
                    dest.write_text(
                        json.dumps(entry, separators=(",", ":")),
                        encoding="utf-8",
                    )
                    migrated += 1
            logger.info("Migrated %d entries from legacy analysis_cache.json", migrated)
            # Remove old file so we don't migrate again
            self._legacy_path.unlink(missing_ok=True)
        except Exception as exc:
            logger.warning("Legacy cache migration failed: %s", exc)

    def _path_for(self, key: str) -> Path:
        return self._dir / f"{_safe_key(key)}.json"

    def get(self, key: str) -> Optional[AudioAnalysis]:
        """Return cached AudioAnalysis or None if not cached / stale."""
        # Check in-memory first
        if key in self._mem:
            return self._mem[key]

        path = self._path_for(key)
        if not path.exists():
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Could not read analysis cache for '%s': %s", key, exc)
            return None

        # Version check — stale entries are treated as cache-misses
        if entry.get("_version", 0) < _CACHE_VERSION:
            logger.debug("Stale analysis cache entry for '%s' (v%s < v%s), will re-analyse",
                         key, entry.get("_version", 0), _CACHE_VERSION)
            return None
        try:
            analysis = AudioAnalysis(
                tempo=entry["tempo"],
                beat_times=entry["beat_times"],
                onset_times=entry["onset_times"],
                rms_curve=entry["rms_curve"],
                rms_times=entry["rms_times"],
                segment_times=entry["segment_times"],
                duration_s=entry["duration_s"],
                perc_onset_times=entry.get("perc_onset_times", []),
                harm_onset_times=entry.get("harm_onset_times", []),
                spectral_centroid_curve=entry.get("spectral_centroid_curve", []),
                bass_energy_curve=entry.get("bass_energy_curve", []),
                mid_energy_curve=entry.get("mid_energy_curve", []),
                treble_energy_curve=entry.get("treble_energy_curve", []),
                onset_strengths=entry.get("onset_strengths", []),
                onset_metrical_weights=entry.get("onset_metrical_weights", []),
                segment_energies=entry.get("segment_energies", []),
                spectral_novelty_curve=entry.get("spectral_novelty_curve", []),
            )
            self._mem[key] = analysis
            return analysis
        except Exception as exc:
            logger.warning("Corrupt analysis cache entry for '%s': %s", key, exc)
            return None

    def put(self, key: str, analysis: AudioAnalysis) -> None:
        """Persist an AudioAnalysis result."""
        entry = {
            "_version": _CACHE_VERSION,
            "tempo": analysis.tempo,
            "beat_times": analysis.beat_times,
            "onset_times": analysis.onset_times,
            "perc_onset_times": analysis.perc_onset_times,
            "harm_onset_times": analysis.harm_onset_times,
            "rms_curve": analysis.rms_curve,
            "rms_times": analysis.rms_times,
            "spectral_centroid_curve": analysis.spectral_centroid_curve,
            "bass_energy_curve": analysis.bass_energy_curve,
            "mid_energy_curve": analysis.mid_energy_curve,
            "treble_energy_curve": analysis.treble_energy_curve,
            "segment_times": analysis.segment_times,
            "duration_s": analysis.duration_s,
            "onset_strengths": analysis.onset_strengths,
            "onset_metrical_weights": analysis.onset_metrical_weights,
            "segment_energies": analysis.segment_energies,
            "spectral_novelty_curve": analysis.spectral_novelty_curve,
        }
        with self._lock:
            self._mem[key] = analysis
            path = self._path_for(key)
            try:
                path.write_text(
                    json.dumps(entry, separators=(",", ":")),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("Could not save analysis cache for '%s': %s", key, exc)
