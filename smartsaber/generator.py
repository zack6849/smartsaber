"""Note placement engine — converts AudioAnalysis into MapDifficulty objects."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from smartsaber.analyzer import AudioAnalysis, centroid_at, band_row_at, novelty_at, segment_energy_at, rms_at, bass_energy_at
from smartsaber.models import (
    CutDirection,
    Difficulty,
    Event,
    MapDifficulty,
    Note,
    NoteType,
)
from smartsaber.patterns import (
    Pattern,
    PatternNote,
    FLOW_MAP,
    is_good_flow,
    patterns_for,
    reachable_positions,
)

# Seconds of silence before any notes — gives player time to orient
_INTRO_QUIET_S = 3.0

# Minimum normalised RMS energy to place any note — scales with difficulty.
# Quiet passages (piano intros, vocal-only sections, fadeouts) shouldn't
# have notes because there's nothing audible to "hit".  Higher difficulties
# map more musical detail so their floor is lower.
_MIN_NOTE_ENERGY: dict[Difficulty, float] = {
    Difficulty.EASY: 0.02,       # Trust onset detector — only filter true silence/noise.
    Difficulty.NORMAL: 0.02,     # Note spam is already prevented by min_gap_beats.
    Difficulty.HARD: 0.02,
    Difficulty.EXPERT: 0.02,
    Difficulty.EXPERT_PLUS: 0.02,
}

# ---------------------------------------------------------------------------
# Difficulty parameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DiffParams:
    """Controls how dense and complex the note chart is for one difficulty."""

    # Minimum gap between consecutive same-hand notes, expressed in BEATS.
    # Scaled by beat duration at runtime so it stays proportional to song tempo.
    # A hard floor of _MIN_GAP_FLOOR_S is always applied regardless of BPM.
    min_gap_beats: float
    # Fraction of available onsets to use (0-1)
    onset_density: float
    # Allow doubles (both hands at the same time)
    allow_doubles: bool
    # Minimum RMS energy to place a double.
    # Corpus analysis (500 maps): doubles appear at RMS ~0.64-0.67, singles at
    # ~0.57-0.63.  Threshold sits just above the singles mean so doubles only
    # fire during genuinely elevated passages.
    double_energy_threshold: float
    # Probability that a crossover is attempted when energy is sufficient.
    # Corpus: crossovers are 5% of Easy notes → 24% of ExpertPlus notes.
    crossover_chance: float


# ---------------------------------------------------------------------------
# Timing conversion utilities (reusable for calibration tools)  
# ---------------------------------------------------------------------------

def time_to_beat(time_seconds: float, analysis) -> float:
    """Convert a time in seconds to a Beat Saber beat number.

    Beat Saber's beat grid starts at beat 0 = second 0 of the audio file
    (when _songTimeOffset is 0, which we always set).  So the conversion
    is simply: beat = time * BPM / 60.

    Previous implementations subtracted ``first_onset`` and added a fixed
    correction factor, which caused every song to be shifted by a different
    amount (equal to its first_onset).  That was the root cause of the
    timing offset the user reported.
    """
    return time_seconds * analysis.tempo / 60.0


def beat_to_time(beat_number: float, analysis) -> float:
    """Convert a Beat Saber beat number back to seconds."""
    return beat_number * 60.0 / analysis.tempo


# ---------------------------------------------------------------------------
# Difficulty parameters
# ---------------------------------------------------------------------------

# Absolute minimum gap in seconds — no note can follow another faster than this
# regardless of BPM, so patterns are always physically hittable.
_MIN_GAP_FLOOR_S = 0.2


_DIFF_PARAMS: dict[Difficulty, DiffParams] = {
    Difficulty.EASY: DiffParams(
        min_gap_beats=2.0,        # corpus median 3.0 — wide spacing for readability
        onset_density=0.32,       # corpus ~268 notes/map
        allow_doubles=True,       # corpus: 30 doubles/map even at Easy
        double_energy_threshold=0.50,
        crossover_chance=0.0,
    ),
    Difficulty.NORMAL: DiffParams(
        min_gap_beats=1.25,       # corpus median 2.0 beats
        onset_density=0.40,       # corpus ~392 notes/map
        allow_doubles=True,       # corpus: 50 doubles/map
        double_energy_threshold=0.45,
        crossover_chance=0.0,
    ),
    Difficulty.HARD: DiffParams(
        min_gap_beats=0.85,       # corpus median 1.0 beat
        onset_density=0.55,       # corpus ~555 notes/map
        allow_doubles=True,
        double_energy_threshold=0.40,
        crossover_chance=0.04,
    ),
    Difficulty.EXPERT: DiffParams(
        min_gap_beats=0.65,       # corpus median 1.0 beat
        onset_density=0.70,       # corpus ~742 notes/map
        allow_doubles=True,
        double_energy_threshold=0.35,
        crossover_chance=0.12,
    ),
    Difficulty.EXPERT_PLUS: DiffParams(
        min_gap_beats=0.50,       # corpus median 0.93 beats
        onset_density=0.85,       # corpus ~940 notes/map
        allow_doubles=True,
        double_energy_threshold=0.30,
        crossover_chance=0.20,
    ),
}

# ---------------------------------------------------------------------------
# Difficulty-dependent direction and row scaling
# ---------------------------------------------------------------------------
# Corpus: diags 7% (Easy) → 34% (E+), L/R 8% → 2%, DOT 7% → 4%.
_DIAG_DIRS = frozenset({CutDirection.UP_LEFT, CutDirection.UP_RIGHT,
                         CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT})
_HORIZ_DIRS = frozenset({CutDirection.LEFT, CutDirection.RIGHT})

# (diag_scale, dot_scale, horiz_scale) per difficulty
_DIFF_DIR_SCALE: dict[Difficulty, tuple[float, float, float]] = {
    Difficulty.EASY:        (0.15, 1.5, 1.5),   # corpus: 7% diag
    Difficulty.NORMAL:      (0.5,  0.8, 0.8),   # corpus: 11% diag
    Difficulty.HARD:        (0.9,  0.4, 0.4),   # corpus: 20% diag
    Difficulty.EXPERT:      (1.3,  0.3, 0.2),   # corpus: 27% diag
    Difficulty.EXPERT_PLUS: (1.6,  0.5, 0.15),  # corpus: 34% diag
}

# Row 2 usage scales with difficulty (corpus: 7% Easy → 19% ExpertPlus)
_DIFF_ROW2_SCALE: dict[Difficulty, float] = {
    Difficulty.EASY: 0.5,
    Difficulty.NORMAL: 0.7,
    Difficulty.HARD: 1.0,
    Difficulty.EXPERT: 1.4,
    Difficulty.EXPERT_PLUS: 2.0,
}

# ---------------------------------------------------------------------------
# Hand state
# ---------------------------------------------------------------------------

@dataclass
class HandState:
    last_col: int = 1
    last_row: int = 1
    last_direction: CutDirection = CutDirection.DOWN
    last_time_s: float = -999.0
    # Parity: tracks whether the arm is in "up" or "down" position after the
    # last swing.  A DOWN swing leaves parity="down"; next swing should be UP.
    # Violating parity causes awkward arm tangling — the #1 reason auto-maps
    # feel bad.  Starts as "down" (arms at rest = down position), so the first
    # note will naturally be an upswing.
    parity: Optional[str] = "down"


# Parity produced by each cut direction (where the arm ends up after the swing)
_PARITY_AFTER: dict[CutDirection, Optional[str]] = {
    CutDirection.UP: "up",
    CutDirection.UP_LEFT: "up",
    CutDirection.UP_RIGHT: "up",
    CutDirection.DOWN: "down",
    CutDirection.DOWN_LEFT: "down",
    CutDirection.DOWN_RIGHT: "down",
    CutDirection.LEFT: None,       # horizontal — no strong parity
    CutDirection.RIGHT: None,
    CutDirection.DOT: None,        # dot resets parity (any-direction)
}

# Which parity the arm must be in to execute a given cut direction
_PARITY_REQUIRED: dict[CutDirection, Optional[str]] = {
    CutDirection.UP: "down",       # need arm down to swing up
    CutDirection.UP_LEFT: "down",
    CutDirection.UP_RIGHT: "down",
    CutDirection.DOWN: "up",       # need arm up to swing down
    CutDirection.DOWN_LEFT: "up",
    CutDirection.DOWN_RIGHT: "up",
    CutDirection.LEFT: None,
    CutDirection.RIGHT: None,
    CutDirection.DOT: None,
}


# ---------------------------------------------------------------------------
# Core placement
# ---------------------------------------------------------------------------

def generate_difficulty(
    analysis: AudioAnalysis,
    difficulty: Difficulty,
    rng: Optional[random.Random] = None,
    shared_events: Optional[list[Event]] = None,
) -> MapDifficulty:
    """
    Generate a single difficulty's note chart from an AudioAnalysis.
    Returns a MapDifficulty with notes sorted by beat time.
    """
    if rng is None:
        rng = random.Random(42)

    params = _DIFF_PARAMS[difficulty]
    md = MapDifficulty(difficulty=difficulty)

    if not analysis.beat_times:
        return md

    # Compute the actual minimum gap in seconds, scaled to this song's tempo.
    # Hard floor ensures no pattern is physically impossible regardless of BPM.
    beat_s = 60.0 / analysis.tempo
    min_gap_s = max(params.min_gap_beats * beat_s, _MIN_GAP_FLOOR_S)

    # Select onsets using sliding window approach for local representation.
    # Instead of global ranking, this ensures each section gets proportional
    # notes based on local intensity, preventing loud sections from dominating.
    onsets = _select_onsets_sliding_window(
        analysis, params.onset_density, rng,
    )

    # State per hand — hands start at center columns, waist height (row 0)
    # which is the natural resting position with arms at sides.
    hands: dict[NoteType, HandState] = {
        NoteType.LEFT: HandState(last_col=1, last_row=0),
        NoteType.RIGHT: HandState(last_col=2, last_row=0),
    }

    pending_beat_times: set[float] = set()

    def _beat(t: float) -> float:
        # Use the shared timing conversion utility
        return time_to_beat(t, analysis)

    current_hand = NoteType.LEFT  # alternating hand tracker

    min_energy = _MIN_NOTE_ENERGY.get(difficulty, 0.10)

    for onset_t in onsets:
        if onset_t < _INTRO_QUIET_S:
            continue  # give player time to orient

        energy = rms_at(analysis, onset_t)

        # Skip onsets in very quiet passages — no audible event to punctuate
        if energy < min_energy:
            continue
        beat_t = _beat(onset_t)
        beat_t = round(beat_t, 4)

        if beat_t in pending_beat_times:
            continue  # vision rule: no stacked notes on same beat

        # --- Double placement with energy spike detection ---
        # Enhanced logic to detect "big hits" - dramatic energy spikes that
        # deserve emphasis with both hands hitting simultaneously.

        # Calculate energy context for spike detection
        energy_spike_factor = 0.0
        if len(analysis.rms_curve) > 1 and len(analysis.rms_times) > 1:
            # Look at energy trend over the past 2-5 seconds
            lookback_time = 3.0  # seconds
            past_energies = []

            for i, rms_time in enumerate(analysis.rms_times):
                if onset_t - lookback_time <= rms_time <= onset_t:
                    past_energies.append(analysis.rms_curve[i])

            if len(past_energies) >= 3:
                # Calculate energy spike: current vs recent average
                recent_avg = sum(past_energies[:-1]) / max(len(past_energies) - 1, 1)
                current_energy = past_energies[-1] if past_energies else energy

                if recent_avg > 0:
                    energy_spike_factor = max(0, (current_energy - recent_avg) / recent_avg)
                    # Cap the spike factor to reasonable range
                    energy_spike_factor = min(energy_spike_factor, 3.0)

        # Base double probability from existing logic
        nov = novelty_at(analysis, onset_t)
        base_double_prob = 0.0

        if params.allow_doubles and energy >= params.double_energy_threshold:
            energy_excess = min(
                (energy - params.double_energy_threshold)
                / max(1.0 - params.double_energy_threshold, 0.01),
                1.0,
            )
            base_double_prob = 0.25 + 0.35 * energy_excess
            # Novelty boost: high spectral change → strongly prefer doubles
            base_double_prob = min(1.0, base_double_prob + nov * 0.25)

        # Energy spike boost - dramatic increases get major double preference
        spike_boost = 0.0
        if energy_spike_factor > 0.3:  # Lower threshold for spike detection
            # For high-energy tracks, be more aggressive with doubles on spikes
            if energy > 0.5:  # High energy context
                if energy_spike_factor > 1.0:
                    spike_boost = 0.7  # Almost guarantee double on extreme spikes
                else:
                    spike_boost = 0.5 * energy_spike_factor  # Strong boost
            else:  # Lower energy context
                if energy_spike_factor > 1.0:
                    spike_boost = 0.6  # Strong boost even in quiet sections
                else:
                    spike_boost = 0.3 * energy_spike_factor  # Moderate boost

        # Combine all factors for final double probability
        double_prob = min(1.0, base_double_prob + spike_boost)

        # Force doubles on extreme spikes in high-energy contexts
        force_double_on_spike = (
            ((energy_spike_factor > 1.5 and energy >= 0.3) or  # Extreme spike + reasonable energy
             (energy_spike_factor > 1.0 and energy >= 0.6)) and  # Strong spike + high energy
            params.allow_doubles
        )

        place_double = force_double_on_spike or (double_prob > 0 and rng.random() < double_prob)

        # --- Crossover flag ---
        # Passed into _place_note so it can allow the hand to cross over the
        # other hand's column.  Only triggered at higher difficulties and only
        # on single notes (doubles in crossover positions are disorienting).
        allow_crossover = (
            not place_double
            and params.crossover_chance > 0
            and energy >= 0.55  # corpus: crossovers appear at moderate+ energy
            and rng.random() < params.crossover_chance
        )

        centroid = centroid_at(analysis, onset_t)
        # Use per-band energy ratios for row placement.  Energy gates
        # row 2 (overhead) so it only appears during intense passages.
        preferred_row = band_row_at(analysis, onset_t, energy=energy)

        if place_double:
            left_note = _place_note(NoteType.LEFT, beat_t, onset_t, energy, centroid, preferred_row, hands, min_gap_s, rng, difficulty, allow_crossover=False)
            right_note = _place_note(NoteType.RIGHT, beat_t, onset_t, energy, centroid, preferred_row, hands, min_gap_s, rng, difficulty, allow_crossover=False)
            # For doubles, force same direction (parallel swing) — much more
            # comfortable than random independent directions.
            if left_note and right_note:
                right_note = Note(
                    time=right_note.time,
                    line_index=right_note.line_index,
                    line_layer=right_note.line_layer,
                    type=right_note.type,
                    cut_direction=left_note.cut_direction,
                )
                # Update right hand state to match
                hands[NoteType.RIGHT].last_direction = left_note.cut_direction
                new_p = _PARITY_AFTER.get(left_note.cut_direction)
                if new_p is not None:
                    hands[NoteType.RIGHT].parity = new_p
            if (left_note and right_note
                    and not _vision_conflict(left_note, right_note, md.notes)
                    and not _opposing_directions(left_note, right_note)):
                md.notes.append(left_note)
                md.notes.append(right_note)
                pending_beat_times.add(beat_t)
            elif left_note:
                md.notes.append(left_note)
                pending_beat_times.add(beat_t)
        else:
            note = _place_note(current_hand, beat_t, onset_t, energy, centroid, preferred_row, hands, min_gap_s, rng, difficulty, allow_crossover=allow_crossover)
            if note:
                md.notes.append(note)
                pending_beat_times.add(beat_t)
            
            # Hand alternation with slight bias toward right hand (humans: 56.5% right vs 43.5% left)
            # Use 65% chance to switch hands (vs 100% strict alternation) and 70% chance to pick right when switching
            if rng.random() < 0.65:  # 65% chance to alternate
                current_hand = _other_hand(current_hand)
            else:  # 35% chance to stay on same hand
                # When not alternating, bias toward right hand
                if rng.random() < 0.70:
                    current_hand = NoteType.RIGHT
                else:
                    current_hand = NoteType.LEFT

    md.notes.sort(key=lambda n: n.time)

    # Post-processing: fix near-simultaneous opposing directions.
    # When two notes from different hands are within ~0.25 beats, opposing
    # inward cuts (L→RIGHT + R→LEFT, etc.) would cause the player to slam
    # their wrists together.  Fix by changing the later note to match the
    # earlier note's direction.
    _fix_near_simultaneous_opposing(md.notes, threshold_beats=0.25)

    md.events = shared_events if shared_events is not None else _generate_lighting(analysis, rng)
    return md


def generate_all_difficulties(
    analysis: AudioAnalysis,
    difficulties: Optional[list[Difficulty]] = None,
    rng: Optional[random.Random] = None,
) -> list[MapDifficulty]:
    if difficulties is None:
        difficulties = list(Difficulty)
    if rng is None:
        rng = random.Random(42)
    # Lighting events are difficulty-independent — compute once and reuse.
    shared_events = _generate_lighting(analysis, rng)
    return [generate_difficulty(analysis, d, rng, shared_events=shared_events) for d in difficulties]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _select_onsets(
    onset_times: list[float],
    density: float,
    rng: random.Random,
) -> list[float]:
    """Return a subset of onset_times based on density (0-1)."""
    if density >= 1.0:
        return list(onset_times)
    n = max(1, int(len(onset_times) * density))
    # Always keep the strongest onsets — since librosa sorts by time,
    # use evenly spaced indices to preserve rhythmic feel.
    step = max(1, len(onset_times) / n)
    indices = sorted(set(int(i * step) for i in range(n)))
    return [onset_times[i] for i in indices if i < len(onset_times)]


def _select_onsets_sliding_window(
    analysis: AudioAnalysis,
    density: float,
    rng: random.Random,
    window_size_s: float = 10.0,
    window_step_s: float = 2.0,
) -> list[float]:
    """Select onsets using sliding window to ensure local representation.

    Instead of ranking onsets globally (which favors loud sections), this uses
    a sliding window approach where we select the best onsets within each local
    time window. This ensures quiet sections get representation based on their
    local peaks, preventing loud sections from dominating the entire map.

    Args:
        window_size_s: Size of each sliding window in seconds
        window_step_s: How far to advance the window each step (smaller = more overlap)
    """
    onset_times = analysis.onset_times
    if not onset_times or density <= 0:
        return []

    # Fallback: no salience data available (old cache, test fixtures)
    if not analysis.onset_strengths or len(analysis.onset_strengths) != len(onset_times):
        return _select_onsets_ramped(onset_times, density, analysis.duration_s, rng)

    total_target = max(1, int(len(onset_times) * density))
    if total_target >= len(onset_times):
        return list(onset_times)

    # Pre-compute salience scores for all onsets
    strengths = analysis.onset_strengths
    metrical = analysis.onset_metrical_weights if len(analysis.onset_metrical_weights) == len(onset_times) else [0.5] * len(onset_times)
    seg_mults = _segment_density_mults(analysis)

    salience: list[float] = []
    for i, t in enumerate(onset_times):
        energy = rms_at(analysis, t)
        audio_importance = strengths[i] * 0.5 + energy * 0.5
        audio_importance = max(audio_importance, strengths[i] * 0.8)

        s = audio_importance * metrical[i]

        # Bass energy boost — bass lines (walking bass, kick drums) produce
        # low onset_strength because their energy is concentrated below 250Hz
        # and gets drowned out by broadband vocal/string content.  Boost
        # salience when bass is the dominant frequency band so these onsets
        # aren't dropped by the sliding window.
        bass_ratio = bass_energy_at(analysis, t)
        if bass_ratio > 0.35:
            s *= 1.0 + (bass_ratio - 0.35) * 3.0  # up to ~2x boost at bass_ratio=0.7

        # Segment energy scaling
        seg_mult = _seg_mult_at(t, analysis.segment_times, seg_mults)
        s *= seg_mult

        # Position ramp
        s *= _position_mult(t, analysis.duration_s)

        salience.append(s)

    # Sliding window selection
    selected_indices = set()

    # Calculate how many windows we'll process
    num_windows = int((analysis.duration_s - window_size_s) / window_step_s) + 1
    if num_windows <= 0:
        num_windows = 1
        window_step_s = analysis.duration_s

    for window_idx in range(num_windows):
        window_start = window_idx * window_step_s
        window_end = min(window_start + window_size_s, analysis.duration_s)

        # Find onsets in this window
        window_onset_indices = []
        for i, t in enumerate(onset_times):
            if window_start <= t < window_end:
                window_onset_indices.append(i)

        if not window_onset_indices:
            continue

        # Calculate target notes for this window based on duration proportion
        window_duration = window_end - window_start
        window_proportion = window_duration / analysis.duration_s
        window_target = max(1, int(total_target * window_proportion))

        # Don't exceed available onsets in window
        window_target = min(window_target, len(window_onset_indices))

        # Rank onsets in this window by salience
        window_salience_pairs = [(salience[i], i) for i in window_onset_indices]

        # Add small jitter to prevent deterministic selection
        jittered_pairs = [(sal * (0.95 + rng.random() * 0.10), idx) for sal, idx in window_salience_pairs]
        jittered_pairs.sort(reverse=True)

        # Select top N from this window
        for _, idx in jittered_pairs[:window_target]:
            selected_indices.add(idx)

    # Handle early onset minimum (for quiet intros like Bohemian Rhapsody)
    early_cutoff = 60.0
    min_early_notes = 2 if analysis.duration_s > 180 else 1
    
    early_indices = [i for i, t in enumerate(onset_times)
                    if t <= early_cutoff and rms_at(analysis, t) >= 0.08]

    if early_indices:
        early_with_salience = [(salience[i], i) for i in early_indices]
        early_with_salience.sort(reverse=True)
        
        early_count = sum(1 for idx in selected_indices if onset_times[idx] <= early_cutoff)
        if early_count < min_early_notes:
            needed = min_early_notes - early_count
            for sal, idx in early_with_salience[:needed]:
                selected_indices.add(idx)

    # Trim overage BEFORE gap-filling so that gap-filled notes are never
    # removed (they have low salience and would be first to go).
    if len(selected_indices) > total_target * 1.2:
        all_with_salience = [(salience[i], i) for i in selected_indices]
        all_with_salience.sort(reverse=True)
        selected_indices = set(idx for _, idx in all_with_salience[:int(total_target * 1.1)])

    # Gap-filling pass — scan for gaps larger than max_gap_s and force-select
    # the best unselected onset in each gap.  Prevents silent holes where
    # audible musical events (bass lines, piano notes) were detected but
    # ranked too low by the salience formula.
    max_gap_s = 1.5
    for _gap_pass in range(5):  # enough passes to subdivide long gaps
        sorted_sel = sorted(selected_indices)
        filled_any = False
        prev_t = 0.0
        for sel_idx in list(sorted_sel) + [len(onset_times)]:
            cur_t = onset_times[sel_idx] if sel_idx < len(onset_times) else analysis.duration_s
            gap = cur_t - prev_t
            if gap > max_gap_s:
                # Find best unselected onset in this gap
                best_sal, best_idx = -1.0, -1
                for i, t in enumerate(onset_times):
                    if t <= prev_t or t >= cur_t:
                        continue
                    if i in selected_indices:
                        continue
                    if salience[i] > best_sal:
                        best_sal = salience[i]
                        best_idx = i
                if best_idx >= 0:
                    selected_indices.add(best_idx)
                    filled_any = True
            prev_t = cur_t
        if not filled_any:
            break

    final_indices = sorted(selected_indices)
    return [onset_times[i] for i in final_indices]


def _position_mult(t: float, duration_s: float) -> float:
    """Song-position density multiplier (intro/outro taper)."""
    if duration_s <= 0:
        return 1.0
    pos = t / duration_s
    if pos < 0.125:
        return 0.70
    if pos < 0.25:
        return 0.90
    if pos < 0.75:
        return 1.00
    if pos < 0.875:
        return 0.90
    return 0.80


def _segment_density_mults(analysis: AudioAnalysis) -> list[float]:
    """Compute per-segment density multipliers from segment energies.

    Louder segments get more notes (mult > 1), quieter get fewer (mult < 1).
    Mean segment energy maps to mult=1.0.  Clamped to [0.5, 1.3] to avoid
    extreme sparseness or overmapping.
    """
    if not analysis.segment_energies:
        return []
    mean_e = sum(analysis.segment_energies) / len(analysis.segment_energies) if analysis.segment_energies else 0.5
    if mean_e <= 0:
        return [1.0] * len(analysis.segment_energies)
    return [
        max(0.5, min(1.3, e / mean_e))
        for e in analysis.segment_energies
    ]


def _seg_mult_at(
    time_s: float,
    segment_times: list[float],
    seg_mults: list[float],
) -> float:
    """Look up the segment density multiplier for a given time."""
    if not seg_mults or len(segment_times) < 2:
        return 1.0
    import bisect
    idx = bisect.bisect_right(segment_times, time_s) - 1
    idx = max(0, min(idx, len(seg_mults) - 1))
    return seg_mults[idx]


def _select_onsets_ramped(
    onset_times: list[float],
    density: float,
    duration_s: float,
    rng: random.Random,
) -> list[float]:
    """Fallback onset selection with a song-position density ramp.

    Used when salience data is not available (old cache, test fixtures).

    Position multipliers (derived from corpus density profiles):
      intro  (0-12.5%):  0.70× base density
      early  (12.5-25%): 0.90×
      mid    (25-75%):   1.00× (peak)
      late   (75-87.5%): 0.90×
      outro  (87.5-100%): 0.80×
    """
    if not onset_times or duration_s <= 0:
        return _select_onsets(onset_times, density, rng)

    result = []
    for t in onset_times:
        effective_density = density * _position_mult(t, duration_s)
        if effective_density >= 1.0 or rng.random() < effective_density:
            result.append(t)
    return result


def _other_hand(h: NoteType) -> NoteType:
    return NoteType.RIGHT if h == NoteType.LEFT else NoteType.LEFT


def _direction_position_ok(direction: CutDirection, col: int, row: int) -> bool:
    """
    Return True if this cut direction makes physical sense at this grid position.

    In Beat Saber, UP and DOWN strokes work at every row — the saber simply
    swings through the note regardless of position.  What DOESN'T work is
    diagonal strokes that would send the saber off the playable grid:
      • DOWN_LEFT / DOWN_RIGHT at the top row look odd but are still hittable
      • UP_LEFT / UP_RIGHT at the bottom row are similarly fine

    We only block truly awkward combinations where the approach angle has
    no physical space — this is very rare in practice so we allow everything
    and let the flow/parity system handle ergonomics instead.
    """
    # All directions are physically valid at all positions in Beat Saber.
    # The flow map and parity system handle ergonomic correctness.
    return True


def _place_note(
    hand: NoteType,
    beat_t: float,
    onset_t: float,
    energy: float,
    centroid: float,
    preferred_row: int,
    hands: dict[NoteType, HandState],
    min_gap_s: float,
    rng: random.Random,
    difficulty: Difficulty,
    allow_crossover: bool = False,
) -> Optional[Note]:
    """Place a single note for the given hand, respecting flow, parity, and reachability.

    Parity tracking ensures consecutive same-hand swings alternate between
    up and down arm positions, preventing awkward arm tangling.
    """
    state = hands[hand]
    other_hand_type = NoteType.RIGHT if hand == NoteType.LEFT else NoteType.LEFT
    other_state = hands[other_hand_type]

    # Enforce minimum time gap between same-hand notes
    if onset_t - state.last_time_s < min_gap_s:
        return None

    # 1. Pick position — reachable from last note, filtered by hand zone
    positions = reachable_positions(state.last_col, state.last_row)

    # At Hard+, allow direct row 0↔2 jumps for more vertical variety.
    # Without this, row 2 is underrepresented because _ROW_REACH only
    # allows adjacent rows (0→1→2 takes two notes to traverse).
    if difficulty.rank >= 5:  # Hard+
        cols_available = set(c for c, _ in positions)
        positions = [(c, r) for c in cols_available for r in range(3)]

    # --- Hand zones + crossover control ---
    if hand == NoteType.LEFT:
        if allow_crossover:
            cross_col = other_state.last_col + 1
            strict = [(c, r) for c, r in positions if c >= cross_col and c <= 3]
            positions = strict or [(c, r) for c, r in positions if c >= 2] or positions
        else:
            max_col = min(2, other_state.last_col)
            strict = [(c, r) for c, r in positions if c <= 1 and c <= max_col]
            positions = strict or [(c, r) for c, r in positions if c <= max_col] or positions
    else:
        if allow_crossover:
            cross_col = other_state.last_col - 1
            strict = [(c, r) for c, r in positions if c <= cross_col and c >= 0]
            positions = strict or [(c, r) for c, r in positions if c <= 1] or positions
        else:
            min_col = max(1, other_state.last_col)
            strict = [(c, r) for c, r in positions if c >= 2 and c >= min_col]
            positions = strict or [(c, r) for c, r in positions if c >= min_col] or positions

    # --- Row weighting (difficulty-dependent) ---
    # Corpus row distribution shifts: Easy 70/22/7 → ExpertPlus 56/25/19.
    # Row 2 (overhead) scales up with difficulty via _DIFF_ROW2_SCALE.
    _ROW_WEIGHTS = {
        0: {0: 10, 1: 3, 2: 0},   # bass → heavily bottom
        1: {0: 6, 1: 3, 2: 1},    # mid → mostly bottom, some chest
        2: {0: 2, 1: 3, 2: 5},    # treble+energy → prefer overhead
    }
    row_weights = _ROW_WEIGHTS.get(preferred_row, {0: 7, 1: 2, 2: 1})
    row2_scale = _DIFF_ROW2_SCALE.get(difficulty, 1.0)

    _COL_WEIGHT = {0: 2, 1: 3, 2: 3, 3: 2}

    weighted_positions: list[tuple[int, int]] = []
    position_weights: list[float] = []
    for c, r in positions:
        row_w = row_weights.get(r, 1)
        if r == 2:
            row_w *= row2_scale
        col_w = _COL_WEIGHT.get(c, 1)
        w = row_w * col_w
        if w > 0:
            weighted_positions.append((c, r))
            position_weights.append(w)

    if not weighted_positions:
        weighted_positions = list(positions)
        position_weights = [1.0] * len(weighted_positions)

    col, row = rng.choices(weighted_positions, weights=position_weights, k=1)[0]

    # 2. Pick direction — parity-driven with difficulty-dependent weighting.
    flow_candidates = FLOW_MAP.get(state.last_direction, list(CutDirection))
    flow_candidates = [d for d in flow_candidates if d != state.last_direction] or flow_candidates
    candidates = list(flow_candidates)

    # --- Parity enforcement (soft) ---
    # At row 0, suppress parity enforcement to avoid forcing upstrokes at
    # waist level — reaching below the play field is ergonomically terrible.
    parity_chance = 0.10 if row == 0 else 0.50
    if state.parity is not None and rng.random() < parity_chance:
        parity_ok = [d for d in candidates if _PARITY_REQUIRED.get(d) == state.parity
                     or _PARITY_REQUIRED.get(d) is None]
        if parity_ok:
            candidates = parity_ok

    # --- Position-aware direction weighting ---
    # Diagonals now have weight at center columns (were 0 before).
    # DOT weight reduced from 3-4 to 1 (was massively overrepresented
    # because it passes parity in both states).
    _DIR_WEIGHT_BY_COL = {
        0: {
            CutDirection.LEFT: 6, CutDirection.DOWN: 6, CutDirection.UP: 5,
            CutDirection.DOT: 1, CutDirection.DOWN_LEFT: 4, CutDirection.UP_LEFT: 3,
            CutDirection.DOWN_RIGHT: 2, CutDirection.UP_RIGHT: 1, CutDirection.RIGHT: 1,
        },
        1: {
            CutDirection.DOWN: 14, CutDirection.UP: 8, CutDirection.DOT: 1,
            CutDirection.LEFT: 1, CutDirection.RIGHT: 0,
            CutDirection.DOWN_LEFT: 3, CutDirection.DOWN_RIGHT: 2,
            CutDirection.UP_LEFT: 2, CutDirection.UP_RIGHT: 1,
        },
        2: {
            CutDirection.DOWN: 14, CutDirection.UP: 8, CutDirection.DOT: 1,
            CutDirection.RIGHT: 1, CutDirection.LEFT: 0,
            CutDirection.DOWN_RIGHT: 3, CutDirection.DOWN_LEFT: 2,
            CutDirection.UP_RIGHT: 2, CutDirection.UP_LEFT: 1,
        },
        3: {
            CutDirection.RIGHT: 6, CutDirection.DOWN: 6, CutDirection.UP: 5,
            CutDirection.DOT: 1, CutDirection.DOWN_RIGHT: 4, CutDirection.UP_RIGHT: 3,
            CutDirection.DOWN_LEFT: 2, CutDirection.UP_LEFT: 1, CutDirection.LEFT: 1,
        },
    }

    # Row multipliers — UP at row 0 strongly suppressed to prevent
    # uncomfortable upstrokes at waist level (user complaint).
    _ROW_MULTIPLIER = {
        0: {  # waist — DOWN dominant, UP virtually eliminated
            CutDirection.DOWN: 4.0, CutDirection.DOWN_LEFT: 2.5, CutDirection.DOWN_RIGHT: 2.5,
            CutDirection.UP: 0.05, CutDirection.UP_LEFT: 0.05, CutDirection.UP_RIGHT: 0.05,
            CutDirection.DOT: 0.4, CutDirection.LEFT: 0.5, CutDirection.RIGHT: 0.5,
        },
        1: {  # chest — balanced, diags welcome
            CutDirection.DOWN: 1.0, CutDirection.UP: 1.0,
            CutDirection.LEFT: 2.0, CutDirection.RIGHT: 2.0,
            CutDirection.DOT: 0.5,
            CutDirection.DOWN_LEFT: 1.5, CutDirection.DOWN_RIGHT: 1.5,
            CutDirection.UP_LEFT: 1.5, CutDirection.UP_RIGHT: 1.5,
        },
        2: {  # overhead — UP dominant
            CutDirection.UP: 5.0, CutDirection.UP_LEFT: 3.0, CutDirection.UP_RIGHT: 3.0,
            CutDirection.DOT: 0.3,
            CutDirection.DOWN: 0.0, CutDirection.DOWN_LEFT: 0.0, CutDirection.DOWN_RIGHT: 0.0,
            CutDirection.LEFT: 0.2, CutDirection.RIGHT: 0.2,
        },
    }

    col_weights = _DIR_WEIGHT_BY_COL.get(col, _DIR_WEIGHT_BY_COL[1])
    row_mults = _ROW_MULTIPLIER.get(row, _ROW_MULTIPLIER[0])

    # Difficulty-dependent scaling: diags increase, DOT/horizontals decrease
    diag_s, dot_s, horiz_s = _DIFF_DIR_SCALE.get(difficulty, (1.0, 1.0, 1.0))

    # Cross-body penalty: left hand cutting right (or vice versa) feels
    # unnatural, especially at center columns where the saber must cross
    # the player's torso.  Suppress cross-body diagonals and horizontals.
    _CROSS_BODY_LEFT = {  # penalise for left hand
        CutDirection.DOWN_RIGHT: 0.15, CutDirection.UP_RIGHT: 0.15, CutDirection.RIGHT: 0.15,
    }
    _CROSS_BODY_RIGHT = {  # penalise for right hand
        CutDirection.DOWN_LEFT: 0.15, CutDirection.UP_LEFT: 0.15, CutDirection.LEFT: 0.15,
    }
    cross_body = _CROSS_BODY_LEFT if hand == NoteType.LEFT else _CROSS_BODY_RIGHT

    dir_options: list[CutDirection] = []
    dir_weights: list[float] = []
    for d in candidates:
        base_w = col_weights.get(d, 1)
        mult = row_mults.get(d, 1.0)
        w = base_w * mult
        # Apply difficulty scaling
        if d in _DIAG_DIRS:
            w *= diag_s
        elif d in _HORIZ_DIRS:
            w *= horiz_s
        elif d == CutDirection.DOT:
            w *= dot_s
        # Apply cross-body penalty
        if d in cross_body:
            w *= cross_body[d]
        if w > 0:
            dir_options.append(d)
            dir_weights.append(w)

    if dir_options:
        direction = rng.choices(dir_options, weights=dir_weights, k=1)[0]
    else:
        direction = rng.choice(list(candidates))

    note = Note(
        time=beat_t,
        line_index=col,
        line_layer=row,
        type=hand,
        cut_direction=direction,
    )

    # Update state (including parity)
    state.last_col = col
    state.last_row = row
    state.last_direction = direction
    state.last_time_s = onset_t
    new_parity = _PARITY_AFTER.get(direction)
    if new_parity is not None:
        state.parity = new_parity

    return note


def _fix_near_simultaneous_opposing(notes: list[Note], threshold_beats: float = 0.25) -> None:
    """Fix opposing directions on near-simultaneous notes from different hands.

    When left and right hand notes are within threshold_beats of each other and
    swing in opposing directions (LEFT vs RIGHT, UP vs DOWN, etc.), the player
    would need to cross or collide their arms.  Fix by changing the later note's
    direction to match the earlier note.
    """
    for i in range(len(notes)):
        for j in range(i + 1, len(notes)):
            if notes[j].time - notes[i].time > threshold_beats:
                break
            # Only check different-hand pairs
            if notes[i].type == notes[j].type:
                continue
            if _OPPOSING_DIRS.get(notes[i].cut_direction) == notes[j].cut_direction:
                # Change later note to match earlier note's direction
                notes[j] = Note(
                    time=notes[j].time,
                    line_index=notes[j].line_index,
                    line_layer=notes[j].line_layer,
                    type=notes[j].type,
                    cut_direction=notes[i].cut_direction,
                )


def _vision_conflict(
    new_left: Note,
    new_right: Note,
    existing: list[Note],
) -> bool:
    """True if both notes share a column (vision block)."""
    return new_left.line_index == new_right.line_index


# Pairs of cut directions that are physically impossible to perform simultaneously.
_OPPOSING_DIRS: dict[CutDirection, CutDirection] = {
    CutDirection.UP: CutDirection.DOWN,
    CutDirection.DOWN: CutDirection.UP,
    CutDirection.LEFT: CutDirection.RIGHT,
    CutDirection.RIGHT: CutDirection.LEFT,
    CutDirection.UP_LEFT: CutDirection.DOWN_RIGHT,
    CutDirection.DOWN_RIGHT: CutDirection.UP_LEFT,
    CutDirection.UP_RIGHT: CutDirection.DOWN_LEFT,
    CutDirection.DOWN_LEFT: CutDirection.UP_RIGHT,
}


def _opposing_directions(left: Note, right: Note) -> bool:
    """True if both hands are swinging in directly opposing directions simultaneously."""
    return _OPPOSING_DIRS.get(left.cut_direction) == right.cut_direction


def _generate_lighting(analysis: AudioAnalysis, rng: random.Random) -> list[Event]:
    """Generate rich beat-synced lighting events.

    Human-made maps have dense lighting — every beat has multiple event types
    firing.  The old implementation was too sparse (many events only every
    4-16 beats) with energy thresholds so high that quiet sections got almost
    no lighting at all.

    Beat Saber lighting event types:
      0 = back lasers          4 = center/road lights
      1 = ring lights          8 = ring rotation
      2 = left rotating laser  9 = ring zoom
      3 = right rotating laser 12 = left laser speed
                               13 = right laser speed

    Values: 0=off, 1=blue on, 2=blue flash, 3=blue fade,
            5=red on, 6=red flash, 7=red fade
    """
    events: list[Event] = []
    if not analysis.beat_times:
        return events

    def _bt(t: float) -> float:
        return round(time_to_beat(t, analysis), 4)

    # Detect segment boundaries for big lighting changes
    segment_set = set()
    for st in analysis.segment_times:
        # Find the nearest beat to each segment boundary using bisect
        if analysis.beat_times:
            import bisect
            idx = bisect.bisect_left(analysis.beat_times, st)
            # Check idx and idx-1 to find which beat is truly nearest
            best = idx
            if idx > 0 and (idx >= len(analysis.beat_times) or
                            abs(analysis.beat_times[idx - 1] - st) <= abs(analysis.beat_times[idx] - st)):
                best = idx - 1
            if best < len(analysis.beat_times):
                segment_set.add(best)

    for i, beat_t in enumerate(analysis.beat_times):
        bt = _bt(beat_t)
        energy = rms_at(analysis, beat_t)
        blue_beat = (i % 2 == 0)
        is_segment = i in segment_set

        # --- Every beat: ring lights + center lights (the core atmosphere) ---
        if energy > 0.4:
            ring_val = 2 if blue_beat else 6    # flash
            center_val = 6 if blue_beat else 2
        elif energy > 0.15:
            ring_val = 1 if blue_beat else 5    # steady on
            center_val = 5 if blue_beat else 1
        else:
            ring_val = 3 if blue_beat else 7    # fade (gentle for quiet parts)
            center_val = 7 if blue_beat else 3

        events.append(Event(time=bt, type=1, value=ring_val))
        events.append(Event(time=bt, type=4, value=center_val))

        # --- Every beat: back lasers ---
        if energy > 0.35:
            events.append(Event(time=bt, type=0, value=2 if blue_beat else 6))
        else:
            events.append(Event(time=bt, type=0, value=1 if blue_beat else 5))

        # --- Every 2 beats: rotating lasers (gives movement) ---
        if i % 2 == 0:
            events.append(Event(time=bt, type=2, value=6 if energy > 0.3 else (1 if blue_beat else 5)))
            events.append(Event(time=bt, type=3, value=2 if energy > 0.3 else (5 if blue_beat else 1)))

        # --- Every 4 beats: ring rotation (keeps the ring spinning) ---
        if i % 4 == 0:
            events.append(Event(time=bt, type=8, value=1))

        # --- Every 8 beats: ring zoom ---
        if i % 8 == 0:
            events.append(Event(time=bt, type=9, value=1))

        # --- Every 4 beats: laser speed (scaled to energy) ---
        if i % 4 == 0:
            speed = max(1, min(10, int(energy * 10) + 2))
            events.append(Event(time=bt, type=12, value=speed))
            events.append(Event(time=bt, type=13, value=speed))

        # --- Segment boundaries: full lighting burst ---
        if is_segment:
            events.append(Event(time=bt, type=0, value=6))   # back laser red flash
            events.append(Event(time=bt, type=1, value=2))   # ring blue flash
            events.append(Event(time=bt, type=4, value=6))   # center red flash
            events.append(Event(time=bt, type=8, value=1))   # ring rotation
            events.append(Event(time=bt, type=9, value=1))   # ring zoom

    # Extra flashes on strong onsets (lowered threshold from 0.85 → 0.6)
    for onset_t in analysis.onset_times:
        bt = _bt(onset_t)
        energy = rms_at(analysis, onset_t)
        if energy > 0.6:
            events.append(Event(time=bt, type=0, value=6))
            events.append(Event(time=bt, type=4, value=2))
        elif energy > 0.4:
            events.append(Event(time=bt, type=0, value=2))

    events.sort(key=lambda e: (e.time, e.type))
    return events
