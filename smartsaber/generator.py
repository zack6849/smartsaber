"""Note placement engine — converts AudioAnalysis into MapDifficulty objects."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from smartsaber.analyzer import AudioAnalysis, centroid_at, band_row_at, rms_at, time_to_beat
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


# Absolute minimum gap in seconds — no note can follow another faster than this
# regardless of BPM, so patterns are always physically hittable.
_MIN_GAP_FLOOR_S = 0.2


_DIFF_PARAMS: dict[Difficulty, DiffParams] = {
    Difficulty.EASY: DiffParams(
        min_gap_beats=1.5,        # ~1.5 quarter notes — slow, very readable
        onset_density=0.35,
        allow_doubles=False,
        double_energy_threshold=1.0,
        crossover_chance=0.0,     # no crossovers at beginner level
    ),
    Difficulty.NORMAL: DiffParams(
        min_gap_beats=1.0,        # quarter note minimum
        onset_density=0.5,
        allow_doubles=False,
        double_energy_threshold=1.0,
        crossover_chance=0.0,
    ),
    Difficulty.HARD: DiffParams(
        min_gap_beats=0.75,       # corpus: Hard p25=1.0 beat — 0.75 gives some headroom
        onset_density=0.7,
        allow_doubles=True,
        # Corpus: Hard doubles rms=0.654, singles rms=0.603 — threshold midpoint
        double_energy_threshold=0.63,
        crossover_chance=0.04,    # rare crossovers start appearing at Hard
    ),
    Difficulty.EXPERT: DiffParams(
        min_gap_beats=0.5,        # corpus: Expert p25=0.75 — 0.5 allows occasional tight pairs
        onset_density=0.85,
        allow_doubles=True,
        # Corpus: Expert doubles rms=0.648, singles rms=0.585
        double_energy_threshold=0.60,
        crossover_chance=0.12,    # ~12% — corpus shows ~20% of Expert notes are crossovers
    ),
    Difficulty.EXPERT_PLUS: DiffParams(
        min_gap_beats=0.375,      # corpus: ExpertPlus p25=0.5 — allows fast streams at peaks
        onset_density=1.0,
        allow_doubles=True,
        # Corpus: ExpertPlus doubles rms=0.641, singles rms=0.571
        double_energy_threshold=0.58,
        crossover_chance=0.20,    # corpus shows ~24% of ExpertPlus notes are crossovers
    ),
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

    # First strong onset defines the offset for beat conversion
    first_onset = analysis.onset_times[0] if analysis.onset_times else 0.0

    # Select onsets based on difficulty density, with a song-position ramp.
    # Corpus: human maps use ~70% of peak density in the intro, ramp up through
    # the song, and taper ~20% in the outro.  This is consistent across all five
    # difficulties (density profile index 0 ≈ 75% of index 2, index 7 ≈ 85%).
    onsets = _select_onsets_ramped(
        analysis.onset_times, params.onset_density,
        analysis.duration_s, rng,
    )

    # State per hand — hands start at center columns, waist height (row 0)
    # which is the natural resting position with arms at sides.
    hands: dict[NoteType, HandState] = {
        NoteType.LEFT: HandState(last_col=1, last_row=0),
        NoteType.RIGHT: HandState(last_col=2, last_row=0),
    }

    pending_beat_times: set[float] = set()

    def _beat(t: float) -> float:
        return time_to_beat(t, analysis.tempo, first_onset)

    current_hand = NoteType.LEFT  # alternating hand tracker

    for onset_t in onsets:
        if onset_t < _INTRO_QUIET_S:
            continue  # give player time to orient

        energy = rms_at(analysis, onset_t)
        beat_t = _beat(onset_t)
        beat_t = round(beat_t, 4)

        if beat_t in pending_beat_times:
            continue  # vision rule: no stacked notes on same beat

        # --- Double placement ---
        # Probability scales with how far energy exceeds the threshold.
        # Corpus: doubles occur at +0.04 to +0.07 RMS above singles, and the
        # gap grows with difficulty — so higher-difficulty doubles feel more
        # "earned" by the music.  At threshold energy: 15% chance.  At max: 40%.
        if params.allow_doubles and energy >= params.double_energy_threshold:
            energy_excess = min(
                (energy - params.double_energy_threshold)
                / max(1.0 - params.double_energy_threshold, 0.01),
                1.0,
            )
            double_prob = 0.15 + 0.25 * energy_excess
        else:
            double_prob = 0.0
        place_double = double_prob > 0 and rng.random() < double_prob

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
            left_note = _place_note(NoteType.LEFT, beat_t, onset_t, energy, centroid, preferred_row, hands, min_gap_s, rng, allow_crossover=False)
            right_note = _place_note(NoteType.RIGHT, beat_t, onset_t, energy, centroid, preferred_row, hands, min_gap_s, rng, allow_crossover=False)
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
            note = _place_note(current_hand, beat_t, onset_t, energy, centroid, preferred_row, hands, min_gap_s, rng, allow_crossover=allow_crossover)
            if note:
                md.notes.append(note)
                pending_beat_times.add(beat_t)
            current_hand = _other_hand(current_hand)

    md.notes.sort(key=lambda n: n.time)
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


def _select_onsets_ramped(
    onset_times: list[float],
    density: float,
    duration_s: float,
    rng: random.Random,
) -> list[float]:
    """
    Select onsets with a song-position density ramp.

    Corpus analysis shows human maps use ~70-75% of peak density in the intro
    and ~80-85% in the outro, ramping up through the first quarter and tapering
    in the last.  This prevents the map from feeling equally busy from start to
    finish — a common problem with naive auto-generators.

    Position multipliers (derived from corpus density profiles):
      intro  (0-12.5%):  0.70× base density
      early  (12.5-25%): 0.90×
      mid    (25-75%):   1.00× (peak)
      late   (75-87.5%): 0.90×
      outro  (87.5-100%): 0.80×
    """
    if not onset_times or duration_s <= 0:
        return _select_onsets(onset_times, density, rng)

    def _position_mult(t: float) -> float:
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

    result = []
    for t in onset_times:
        effective_density = density * _position_mult(t)
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
    allow_crossover: bool = False,
) -> Optional[Note]:
    """Place a single note for the given hand, respecting flow, parity, and reachability.

    Row selection is driven by per-band energy ratios (preferred_row):
      0 → row 0 (bottom) — bass-dominant (kick, sub-bass, bass guitar)
      1 → row 1 (middle) — mid-range dominant (snare, chords, rhythm)
      2 → row 2 (top)    — treble-dominant (hi-hats, lead melody, vocals)

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

    # --- Hand zones + crossover control ---
    # Normal: left stays left of right hand, right stays right of left hand.
    # Crossover: the crossing hand is pushed *past* the other hand's column —
    # left goes to col 2-3, right goes to col 0-1.  This matches the ~24% of
    # ExpertPlus notes that cross the center line in well-made human maps.
    if hand == NoteType.LEFT:
        if allow_crossover:
            # Cross over: left hand goes right of the other hand
            cross_col = other_state.last_col + 1
            strict = [(c, r) for c, r in positions if c >= cross_col and c <= 3]
            positions = strict or [(c, r) for c, r in positions if c >= 2] or positions
        else:
            max_col = min(2, other_state.last_col)  # can't go right of right hand
            strict = [(c, r) for c, r in positions if c <= 1 and c <= max_col]
            positions = strict or [(c, r) for c, r in positions if c <= max_col] or positions
    else:
        if allow_crossover:
            # Cross over: right hand goes left of the other hand
            cross_col = other_state.last_col - 1
            strict = [(c, r) for c, r in positions if c <= cross_col and c >= 0]
            positions = strict or [(c, r) for c, r in positions if c <= 1] or positions
        else:
            min_col = max(1, other_state.last_col)  # can't go left of left hand
            strict = [(c, r) for c, r in positions if c >= 2 and c >= min_col]
            positions = strict or [(c, r) for c, r in positions if c >= min_col] or positions

    # Row preference: human-made maps place ~75% of notes at waist (row 0),
    # ~20% at chest (row 1), and only ~5% overhead (row 2).  We use the
    # audio band analysis as a mild nudge but keep the overall distribution
    # heavily biased toward the bottom — arms naturally rest at the sides.
    #
    # Weight table (preferred_row → weight per row):
    #   preferred 0: row0=8, row1=2, row2=0  (bass → heavily low)
    #   preferred 1: row0=6, row1=3, row2=1  (mid  → mostly low, some chest + rare overhead)
    #   preferred 2: row0=3, row1=3, row2=3  (treble+energy → allow overhead)
    _ROW_WEIGHTS = {
        0: {0: 8, 1: 2, 2: 0},
        1: {0: 6, 1: 3, 2: 1},
        2: {0: 3, 1: 3, 2: 3},
    }
    row_weights = _ROW_WEIGHTS.get(preferred_row, {0: 6, 1: 3, 2: 1})

    # Build weighted position list — combines row AND column preference.
    # Center columns (1, 2) are the natural resting zone for arms at sides.
    # Outer columns (0, 3) require reaching but human maps still use them ~20%
    # each, so weight 2 (vs 3 for center) gives a natural ~60/40 center/outer split.
    _COL_WEIGHT = {0: 2, 1: 3, 2: 3, 3: 2}

    weighted_positions = []
    for c, r in positions:
        row_w = row_weights.get(r, 1)
        col_w = _COL_WEIGHT.get(c, 1)
        w = row_w * col_w
        if w > 0:
            weighted_positions.extend([(c, r)] * w)

    # Fallback: if weighting eliminated everything, use unweighted positions
    if not weighted_positions:
        weighted_positions = list(positions)

    col, row = rng.choice(weighted_positions)

    # 2. Pick direction — parity-driven, with a strong DOWN bias.
    #
    # In well-made Beat Saber maps, DOWN strokes are the bread-and-butter
    # (~50-60% of notes).  UP strokes are less common (~25-30%), with
    # diagonals filling the rest.  This matches natural arm mechanics —
    # the downward swing is the strongest, most comfortable motion.
    #
    # The old code used FLOW_MAP ordering + [:4] slice which accidentally
    # created ~90% upstrokes.  Now we:
    #   1. Get parity-correct candidates
    #   2. Weight DOWN/DOWN_diag heavily over UP/UP_diag
    #   3. Pick from the weighted list

    flow_candidates = FLOW_MAP.get(state.last_direction, list(CutDirection))
    # Avoid repeating the same direction (no wrist resets)
    flow_candidates = [d for d in flow_candidates if d != state.last_direction] or flow_candidates

    candidates = list(flow_candidates)

    # --- Parity enforcement (soft) ---
    # Human mappers follow parity ~50% of the time and use "resets" the other
    # half (wrist flicks, DOT notes, horizontal swings that don't set parity).
    # Strict parity forces UP after every DOWN, inflating UP to ~37%.
    # At 50% enforcement, the position-aware weights can naturally favor DOWN
    # since it has 4x row multiplier at waist level.
    if state.parity is not None and rng.random() < 0.50:
        parity_ok = [d for d in candidates if _PARITY_REQUIRED.get(d) == state.parity
                     or _PARITY_REQUIRED.get(d) is None]
        if parity_ok:
            candidates = parity_ok

    # --- Position-aware direction weighting ---
    #
    # Analysis of popular human-made BeatSaver maps reveals very different
    # direction usage depending on column and row position:
    #
    # COLUMNS:
    #   Col 0 (far left):  LEFT 30%, UP 19%, DOWN 18%, DOT 11%, diags 12%
    #   Col 1 (center-L):  DOWN 48%, UP 30%, DOT 12%, diags ~6%
    #   Col 2 (center-R):  DOWN 49%, UP 29%, DOT 12%, diags ~6%
    #   Col 3 (far right): RIGHT 30%, UP 20%, DOWN 17%, DOT 10%, diags 12%
    #
    # ROWS:
    #   Row 0 (waist):     DOWN 48%, UP 26%, DOT 12%, L/R 7%
    #   Row 1 (chest):     LEFT 24%, RIGHT 24%, UP 18%, DOT 11%, diags 16%
    #   Row 2 (overhead):  UP 47%, UP_LEFT 19%, UP_RIGHT 21%, DOT 7%
    #
    # KEY RULES:
    #   1. Diagonals belong at outer columns (0, 3) and extreme rows — NOT at
    #      center columns (1, 2) at waist level.  Human mappers put only ~2%
    #      diagonals in center columns vs 4% at outer columns.
    #   2. DOT notes are used ~11% of the time (we had 0%).
    #   3. Horizontal (LEFT/RIGHT) sweeps belong at outer columns and row 1.
    #   4. Row 0 is overwhelmingly DOWN strokes.
    #   5. Row 2 is overwhelmingly UP/UP_diagonal strokes.

    # Base weights by column position
    _DIR_WEIGHT_BY_COL = {
        # Col 0 (far left): LEFT-heavy + verticals + diags allowed
        0: {
            CutDirection.LEFT: 8, CutDirection.DOWN: 5, CutDirection.UP: 5,
            CutDirection.DOT: 3, CutDirection.DOWN_LEFT: 3, CutDirection.UP_LEFT: 2,
            CutDirection.DOWN_RIGHT: 1, CutDirection.UP_RIGHT: 1, CutDirection.RIGHT: 1,
        },
        # Col 1 (center-left): DOWN/UP dominant, small DOT presence
        1: {
            CutDirection.DOWN: 14, CutDirection.UP: 6, CutDirection.DOT: 4,
            CutDirection.LEFT: 2, CutDirection.RIGHT: 1,
            CutDirection.DOWN_LEFT: 1, CutDirection.DOWN_RIGHT: 1,
            CutDirection.UP_LEFT: 0, CutDirection.UP_RIGHT: 0,
        },
        # Col 2 (center-right): DOWN/UP dominant, small DOT presence
        2: {
            CutDirection.DOWN: 14, CutDirection.UP: 6, CutDirection.DOT: 4,
            CutDirection.RIGHT: 2, CutDirection.LEFT: 1,
            CutDirection.DOWN_RIGHT: 1, CutDirection.DOWN_LEFT: 1,
            CutDirection.UP_RIGHT: 0, CutDirection.UP_LEFT: 0,
        },
        # Col 3 (far right): RIGHT-heavy + verticals + diags allowed
        3: {
            CutDirection.RIGHT: 8, CutDirection.DOWN: 5, CutDirection.UP: 5,
            CutDirection.DOT: 3, CutDirection.DOWN_RIGHT: 3, CutDirection.UP_RIGHT: 2,
            CutDirection.DOWN_LEFT: 1, CutDirection.UP_LEFT: 1, CutDirection.LEFT: 1,
        },
    }

    # Row multipliers — amplify/suppress directions based on vertical position
    # Row 0: heavily favor DOWN.  Row 2: heavily favor UP/UP-diags.
    #
    # NOTE: DOT and horizontal directions pass parity in BOTH states (up and
    # down), so they effectively get picked twice as often as directional notes
    # that only pass one parity state.  Their row multiplier must be kept modest
    # to compensate — otherwise DOT dominates the "wrong parity" state.
    _ROW_MULTIPLIER = {
        0: {  # waist — DOWN is king
            CutDirection.DOWN: 4.0, CutDirection.DOWN_LEFT: 2.0, CutDirection.DOWN_RIGHT: 2.0,
            CutDirection.UP: 1.0, CutDirection.UP_LEFT: 0.5, CutDirection.UP_RIGHT: 0.5,
            CutDirection.DOT: 0.8, CutDirection.LEFT: 0.7, CutDirection.RIGHT: 0.7,
        },
        1: {  # chest — horizontals and verticals mix, diags allowed
            CutDirection.DOWN: 1.0, CutDirection.UP: 1.0,
            CutDirection.LEFT: 2.5, CutDirection.RIGHT: 2.5,
            CutDirection.DOT: 1.0,
            CutDirection.DOWN_LEFT: 1.5, CutDirection.DOWN_RIGHT: 1.5,
            CutDirection.UP_LEFT: 1.5, CutDirection.UP_RIGHT: 1.5,
        },
        2: {  # overhead — UP is king, diags only upward
            CutDirection.UP: 5.0, CutDirection.UP_LEFT: 3.0, CutDirection.UP_RIGHT: 3.0,
            CutDirection.DOT: 0.5,
            CutDirection.DOWN: 0.0, CutDirection.DOWN_LEFT: 0.0, CutDirection.DOWN_RIGHT: 0.0,
            CutDirection.LEFT: 0.2, CutDirection.RIGHT: 0.2,
        },
    }

    col_weights = _DIR_WEIGHT_BY_COL.get(col, _DIR_WEIGHT_BY_COL[1])
    row_mults = _ROW_MULTIPLIER.get(row, _ROW_MULTIPLIER[0])

    weighted_dirs = []
    for d in candidates:
        base_w = col_weights.get(d, 1)
        mult = row_mults.get(d, 1.0)
        w = int(base_w * mult)
        if w > 0:
            weighted_dirs.extend([d] * w)

    # Fallback: if position filtering eliminated everything, use parity candidates
    if not weighted_dirs:
        weighted_dirs = list(candidates)

    direction = rng.choice(weighted_dirs)

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
    # else: horizontal/dot — parity unchanged (acts as a reset)

    return note


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

    first_onset = analysis.onset_times[0] if analysis.onset_times else 0.0

    def _bt(t: float) -> float:
        return round(time_to_beat(t, analysis.tempo, first_onset), 4)

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
