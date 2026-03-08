"""Audio analysis using librosa."""

from __future__ import annotations

import bisect
import logging
from pathlib import Path

import librosa
import numpy as np

from smartsaber.models import AudioAnalysis

logger = logging.getLogger(__name__)

_SR = 22050         # sample rate
_HOP = 512          # hop length for most analyses
_N_FFT = 2048       # STFT window size
_BEAT_SUBDIVISION_THRESHOLDS = [1 / 4, 1 / 8, 1 / 16]  # note grid units


def analyze(audio_path: Path) -> AudioAnalysis:
    """
    Load an audio file and extract all features needed for map generation.

    Uses HPSS (Harmonic-Percussive Source Separation) to detect drum hits and
    melodic onsets independently, giving better coverage of the full mix.
    Spectral centroid is stored normalised per-song so the generator can map
    frequency content to row (bass → bottom, treble/vocals → top).
    """
    y, sr = librosa.load(str(audio_path), sr=_SR, mono=True)
    duration_s = librosa.get_duration(y=y, sr=sr)

    # Guard: if the audio is empty or shorter than one FFT window the STFT
    # will produce zero frames and everything downstream (tempo, centroid, …)
    # will crash with division-by-zero or empty-array errors.
    if len(y) < _N_FFT:
        logger.warning(
            "Audio too short (%d samples, need >= %d). "
            "Returning empty analysis for %s",
            len(y), _N_FFT, audio_path,
        )
        return AudioAnalysis(
            tempo=120.0,
            beat_times=[],
            onset_times=[],
            perc_onset_times=[],
            harm_onset_times=[],
            rms_curve=[],
            rms_times=[],
            spectral_centroid_curve=[],
            bass_energy_curve=[],
            mid_energy_curve=[],
            treble_energy_curve=[],
            segment_times=[0.0, duration_s],
            duration_s=duration_s,
        )

    # --- STFT + HPSS ---
    # margin=2 is a soft Wiener mask — leaves ambiguous content (e.g. plucked
    # bass guitar, chord stabs) shared between H and P rather than forcing it
    # into one component.  Works well for electronic music.
    D = librosa.stft(y, n_fft=_N_FFT, hop_length=_HOP)
    S = np.abs(D)
    H, P = librosa.decompose.hpss(S, margin=2.0)

    # --- Tempo + beats ---
    # Beat-track on the combined onset envelope for best rhythmic accuracy.
    onset_env_perc = librosa.onset.onset_strength(S=P, sr=sr, hop_length=_HOP)
    onset_env_harm = librosa.onset.onset_strength(S=H, sr=sr, hop_length=_HOP)
    onset_env_combined = onset_env_perc + onset_env_harm

    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env_combined, sr=sr, hop_length=_HOP
    )
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=_HOP).tolist()
    # librosa 0.11 + NumPy 2.x may return tempo as a non-scalar ndarray
    tempo = float(np.asarray(tempo).flat[0])
    # Guard: beat_track can return 0 BPM on very quiet / degenerate signals
    if tempo <= 0:
        tempo = 120.0
    beat_duration = 60.0 / tempo

    # --- Percussive onsets (kick, snare, hi-hat, transients) ---
    # delta=0.08: drums should be clear spikes — slightly higher threshold
    # reduces ghost onsets from bleed between H/P components.
    perc_onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env_perc, sr=sr, hop_length=_HOP, backtrack=True,
        delta=0.08,
    )
    perc_onset_times_raw = librosa.frames_to_time(perc_onset_frames, sr=sr, hop_length=_HOP)
    perc_onset_times = _quantize_onsets(perc_onset_times_raw.tolist(), beat_times, beat_duration)

    # --- Harmonic onsets (melody, chords, vocals, bass notes) ---
    # delta=0.05: melodic onsets are softer — lower threshold catches chord
    # changes and vocal entries that would be missed at the default 0.07.
    harm_onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env_harm, sr=sr, hop_length=_HOP, backtrack=True,
        delta=0.05,
    )
    harm_onset_times_raw = librosa.frames_to_time(harm_onset_frames, sr=sr, hop_length=_HOP)
    harm_onset_times = _quantize_onsets(harm_onset_times_raw.tolist(), beat_times, beat_duration)

    # --- Merged onset set (deduped, for backward compatibility) ---
    onset_times = sorted({round(t, 4) for t in perc_onset_times + harm_onset_times})

    # --- RMS energy (full mix) ---
    rms = librosa.feature.rms(y=y, hop_length=_HOP)[0]
    rms_max = rms.max()
    if rms_max > 0:
        rms_norm = (rms / rms_max).tolist()
    else:
        rms_norm = rms.tolist()
    rms_times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=_HOP
    ).tolist()

    # --- Spectral centroid, normalised per-song to [0, 1] ---
    # 0 = bass-heavy moment, 1 = treble/vocals-heavy moment.
    # Computed from the full STFT magnitude so it's aligned with the HPSS output.
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    # Normalise using 5th/95th percentiles to be robust to extreme outliers
    c5, c95 = float(np.percentile(centroid, 5)), float(np.percentile(centroid, 95))
    if c95 > c5:
        centroid_norm = np.clip((centroid - c5) / (c95 - c5), 0.0, 1.0)
    else:
        centroid_norm = np.full_like(centroid, 0.5)
    # Trim or pad to match rms_times length (STFT vs RMS frame counts can differ by 1)
    n_rms = len(rms_norm)
    centroid_list = centroid_norm.tolist()
    if len(centroid_list) > n_rms:
        centroid_list = centroid_list[:n_rms]
    elif len(centroid_list) < n_rms:
        centroid_list = centroid_list + [centroid_list[-1]] * (n_rms - len(centroid_list))

    # --- Per-band energy ratios (bass / mid / treble) ---
    # More robust than scalar centroid for row mapping — centroid can be pulled
    # by high-energy outlier bins, whereas band ratios give a clean dominant-band
    # signal per frame.  Frequency splits: bass < 250 Hz, mid 250-2000 Hz, treble 2000+ Hz.
    freqs = librosa.fft_frequencies(sr=sr, n_fft=_N_FFT)
    bass_mask = freqs < 250
    mid_mask = (freqs >= 250) & (freqs < 2000)
    treble_mask = freqs >= 2000

    bass_energy = S[bass_mask, :].sum(axis=0)
    mid_energy = S[mid_mask, :].sum(axis=0)
    treble_energy = S[treble_mask, :].sum(axis=0)
    total_energy = bass_energy + mid_energy + treble_energy
    # Avoid division by zero in silent frames
    total_energy = np.maximum(total_energy, 1e-10)

    bass_ratio = (bass_energy / total_energy).tolist()
    mid_ratio = (mid_energy / total_energy).tolist()
    treble_ratio = (treble_energy / total_energy).tolist()

    # Align band energy lists to rms_times length (same frame-count mismatch as centroid)
    bass_ratio = _align_to_length(bass_ratio, n_rms)
    mid_ratio = _align_to_length(mid_ratio, n_rms)
    treble_ratio = _align_to_length(treble_ratio, n_rms)

    # --- Structural segmentation ---
    segment_times = _segment(y, sr, duration_s)

    return AudioAnalysis(
        tempo=tempo,
        beat_times=beat_times,
        onset_times=onset_times,
        perc_onset_times=perc_onset_times,
        harm_onset_times=harm_onset_times,
        rms_curve=rms_norm,
        rms_times=rms_times,
        spectral_centroid_curve=centroid_list,
        bass_energy_curve=bass_ratio,
        mid_energy_curve=mid_ratio,
        treble_energy_curve=treble_ratio,
        segment_times=segment_times,
        duration_s=duration_s,
    )


def _align_to_length(lst: list[float], target: int) -> list[float]:
    """Trim or pad *lst* to exactly *target* elements (repeat last value)."""
    if not lst or target <= 0:
        return lst[:target] if lst else []
    if len(lst) > target:
        return lst[:target]
    elif len(lst) < target:
        return lst + [lst[-1]] * (target - len(lst))
    return lst


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
    return _interpolate_curve(analysis.rms_times, analysis.rms_curve, time_s, default=0.5)


def centroid_at(analysis: AudioAnalysis, time_s: float) -> float:
    """
    Interpolate the normalised spectral centroid (0-1) at a given time.

    0 = bass-heavy (kick, bass guitar) → bottom row
    1 = treble/vocal-heavy (hi-hats, lead vocals) → top row

    Returns 0.5 (midrange) if spectral_centroid_curve is not populated
    (e.g. old cached entries or test fixtures).
    """
    if not analysis.spectral_centroid_curve:
        return 0.5
    return _interpolate_curve(analysis.rms_times, analysis.spectral_centroid_curve, time_s, default=0.5)


def band_row_at(analysis: AudioAnalysis, time_s: float, energy: float = 0.5) -> int:
    """
    Return the preferred Beat Saber row (0-2) for a note at *time_s*.

    In well-designed Beat Saber maps, note rows follow an ergonomic
    distribution — most notes sit at waist (0) and chest (1) height where
    arms naturally rest.  Overhead notes (row 2) are used sparingly for
    emphasis during high-energy moments.

    Mapping strategy:
      • Row 0 (bottom / waist) — default home position, used most often.
        Bass-dominant frames get a strong nudge here.
      • Row 1 (middle / chest) — second most common.  Mid-range dominant
        frames prefer this row.
      • Row 2 (top / overhead)  — reserved for HIGH energy + treble-dominant
        moments only (big drops, cymbal crashes, vocal peaks).

    The raw frequency band is used as a *bias*, not a hard mapping.
    Energy gates row 2 so it only appears during intense passages.
    """
    if analysis.bass_energy_curve and analysis.mid_energy_curve and analysis.treble_energy_curve:
        bass = _interpolate_curve(analysis.rms_times, analysis.bass_energy_curve, time_s, default=0.33)
        mid = _interpolate_curve(analysis.rms_times, analysis.mid_energy_curve, time_s, default=0.34)
        treble = _interpolate_curve(analysis.rms_times, analysis.treble_energy_curve, time_s, default=0.33)

        # Bass-dominant → always row 0 (kicks, sub-bass feel natural low)
        if bass >= mid and bass >= treble:
            return 0

        # Treble-dominant AND high energy → row 2 (overhead for emphasis)
        if treble >= mid and treble >= bass and energy >= 0.7:
            return 2

        # Treble-dominant but not enough energy → row 1 (don't force overhead)
        if treble >= mid and treble >= bass:
            return 1

        # Mid-dominant — split between row 0 and row 1 based on energy.
        # Lower energy mid → row 0 (arms down), higher energy mid → row 1.
        if energy >= 0.5:
            return 1
        return 0

    # Fallback: centroid thresholds (legacy / empty band curves)
    c = centroid_at(analysis, time_s)
    if c < 0.4:
        return 0
    if c >= 0.7 and energy >= 0.7:
        return 2
    if energy >= 0.5:
        return 1
    return 0


def _interpolate_curve(times: list[float], curve: list[float], time_s: float, default: float) -> float:
    """Linear interpolation of a sampled curve at an arbitrary time.

    Uses bisect for O(log n) lookup instead of O(n) linear scan.
    With ~10K frames per song and thousands of lookups per difficulty,
    this is the difference between seconds and minutes of generation time.
    """
    if not times or not curve:
        return default
    if time_s <= times[0]:
        return curve[0]
    if time_s >= times[-1]:
        return curve[-1]
    # bisect_right returns the index where time_s would be inserted to keep
    # the list sorted, i.e. times[i-1] <= time_s < times[i].
    i = bisect.bisect_right(times, time_s)
    if i <= 0:
        return curve[0]
    if i >= len(times):
        return curve[-1]
    t0, t1 = times[i - 1], times[i]
    v0, v1 = curve[i - 1], curve[i]
    alpha = (time_s - t0) / (t1 - t0) if t1 != t0 else 0.0
    return v0 + alpha * (v1 - v0)


def time_to_beat(time_s: float, tempo: float, offset_s: float = 0.0) -> float:
    """Convert a time in seconds to a beat number."""
    return (time_s - offset_s) * tempo / 60.0
