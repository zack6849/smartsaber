"""Audio analysis using librosa."""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np

from smartsaber.models import AudioAnalysis

logger = logging.getLogger(__name__)

_SR = 22050         # sample rate
_HOP = 512          # hop length for most analyses
_BEAT_SUBDIVISION_THRESHOLDS = [1 / 4, 1 / 8, 1 / 16]  # note grid units


def analyze(audio_path: Path) -> AudioAnalysis:
    """
    Load an audio file and extract all features needed for map generation.
    Returns an AudioAnalysis dataclass.
    """
    y, sr = librosa.load(str(audio_path), sr=_SR, mono=True)
    duration_s = librosa.get_duration(y=y, sr=sr)

    # --- Tempo + beats ---
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=_HOP)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=_HOP).tolist()
    # librosa 0.11 + NumPy 2.x may return tempo as a non-scalar ndarray
    tempo = float(np.asarray(tempo).flat[0])

    # --- Onsets ---
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=_HOP, backtrack=True
    )
    onset_times_raw = librosa.frames_to_time(onset_frames, sr=sr, hop_length=_HOP)

    # Quantize onsets to nearest beat subdivision (1/4, 1/8, 1/16 beat)
    beat_duration = 60.0 / tempo  # seconds per beat
    onset_times = _quantize_onsets(onset_times_raw.tolist(), beat_times, beat_duration)

    # --- RMS energy ---
    rms = librosa.feature.rms(y=y, hop_length=_HOP)[0]
    rms_max = rms.max()
    if rms_max > 0:
        rms_norm = (rms / rms_max).tolist()
    else:
        rms_norm = rms.tolist()
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=_HOP
    ).tolist()

    # --- Structural segmentation ---
    segment_times = _segment(y, sr, duration_s)

    return AudioAnalysis(
        tempo=tempo,
        beat_times=beat_times,
        onset_times=onset_times,
        rms_curve=rms_norm,
        rms_times=rms_times,
        segment_times=segment_times,
        duration_s=duration_s,
    )


def _quantize_onsets(
    onset_times: list[float],
    beat_times: list[float],
    beat_duration: float,
    tolerance_s: float = 0.05,
) -> list[float]:
    """Snap onset times to the nearest beat subdivision within tolerance."""
    if not beat_times:
        return onset_times

    quantized = []
    seen: set[float] = set()

    for onset in onset_times:
        # Find nearest beat
        nearest_beat = min(beat_times, key=lambda b: abs(b - onset))
        offset = onset - nearest_beat

        # Try each subdivision
        snapped = onset
        for sub in _BEAT_SUBDIVISION_THRESHOLDS:
            grid = beat_duration * sub
            if grid == 0:
                continue
            remainder = offset % grid
            candidate_offset = offset - remainder
            candidate = nearest_beat + candidate_offset
            if abs(candidate - onset) <= tolerance_s:
                snapped = candidate
                break

        # Round to 4 decimal places to avoid floating-point duplicates
        snapped = round(snapped, 4)
        if snapped not in seen:
            seen.add(snapped)
            quantized.append(snapped)

    quantized.sort()
    return quantized


def _segment(y: np.ndarray, sr: int, duration_s: float) -> list[float]:
    """
    Use librosa's agglomerative segmentation to find structural boundaries.
    Clamps k between 4 and 12, scaled by duration.
    """
    try:
        # Rough heuristic: ~1 segment per 30s, min 4, max 12
        k = int(max(4, min(12, duration_s / 30)))
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=_HOP)
        bound_frames = librosa.segment.agglomerative(chroma, k)
        bound_times = librosa.frames_to_time(bound_frames, sr=sr, hop_length=_HOP)
        # Always include start and end
        times = [0.0] + bound_times.tolist() + [duration_s]
        return sorted(set(round(t, 3) for t in times))
    except Exception as exc:
        logger.warning("Segmentation failed: %s", exc)
        return [0.0, duration_s]


def rms_at(analysis: AudioAnalysis, time_s: float) -> float:
    """Interpolate RMS energy at a given time (0-1)."""
    times = analysis.rms_times
    curve = analysis.rms_curve
    if not times:
        return 0.5
    if time_s <= times[0]:
        return curve[0]
    if time_s >= times[-1]:
        return curve[-1]
    # Linear search (fast enough for typical curve sizes)
    for i in range(len(times) - 1):
        if times[i] <= time_s <= times[i + 1]:
            t0, t1 = times[i], times[i + 1]
            r0, r1 = curve[i], curve[i + 1]
            alpha = (time_s - t0) / (t1 - t0) if t1 != t0 else 0.0
            return r0 + alpha * (r1 - r0)
    return curve[-1]


def time_to_beat(time_s: float, tempo: float, offset_s: float = 0.0) -> float:
    """Convert a time in seconds to a beat number."""
    return (time_s - offset_s) * tempo / 60.0
