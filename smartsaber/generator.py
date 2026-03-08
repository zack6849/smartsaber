"""Note placement engine — converts AudioAnalysis into MapDifficulty objects."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from smartsaber.analyzer import AudioAnalysis, rms_at, time_to_beat
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
    # Minimum RMS energy to place a double
    double_energy_threshold: float


# Absolute minimum gap in seconds — no note can follow another faster than this
# regardless of BPM, so patterns are always physically hittable.
_MIN_GAP_FLOOR_S = 0.2


_DIFF_PARAMS: dict[Difficulty, DiffParams] = {
    Difficulty.EASY: DiffParams(
        min_gap_beats=1.5,        # ~1.5 quarter notes — slow, very readable
        onset_density=0.35,
        allow_doubles=False,
        double_energy_threshold=1.0,
    ),
    Difficulty.NORMAL: DiffParams(
        min_gap_beats=1.0,        # quarter note minimum
        onset_density=0.5,
        allow_doubles=False,
        double_energy_threshold=1.0,
    ),
    Difficulty.HARD: DiffParams(
        min_gap_beats=0.5,        # 8th note minimum
        onset_density=0.7,
        allow_doubles=True,
        double_energy_threshold=0.75,
    ),
    Difficulty.EXPERT: DiffParams(
        min_gap_beats=0.333,      # triplet (3 notes per beat)
        onset_density=0.85,
        allow_doubles=True,
        double_energy_threshold=0.6,
    ),
    Difficulty.EXPERT_PLUS: DiffParams(
        min_gap_beats=0.25,       # 16th note minimum
        onset_density=1.0,
        allow_doubles=True,
        double_energy_threshold=0.5,
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


# ---------------------------------------------------------------------------
# Core placement
# ---------------------------------------------------------------------------

def generate_difficulty(
    analysis: AudioAnalysis,
    difficulty: Difficulty,
    rng: Optional[random.Random] = None,
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

    # Select onsets based on difficulty density
    onsets = _select_onsets(analysis.onset_times, params.onset_density, rng)

    # State per hand — right hand starts at col 2 (center-right)
    hands: dict[NoteType, HandState] = {
        NoteType.LEFT: HandState(last_col=1, last_row=1),
        NoteType.RIGHT: HandState(last_col=2, last_row=1),
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
        place_double = (
            params.allow_doubles
            and energy >= params.double_energy_threshold
            and rng.random() < 0.2  # 20% chance when energy is sufficient
        )

        if place_double:
            left_note = _place_note(NoteType.LEFT, beat_t, onset_t, energy, hands, min_gap_s, rng)
            right_note = _place_note(NoteType.RIGHT, beat_t, onset_t, energy, hands, min_gap_s, rng)
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
            note = _place_note(current_hand, beat_t, onset_t, energy, hands, min_gap_s, rng)
            if note:
                md.notes.append(note)
                pending_beat_times.add(beat_t)
            current_hand = _other_hand(current_hand)

    md.notes.sort(key=lambda n: n.time)
    md.events = _generate_lighting(analysis, rng)
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
    return [generate_difficulty(analysis, d, rng) for d in difficulties]


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


def _other_hand(h: NoteType) -> NoteType:
    return NoteType.RIGHT if h == NoteType.LEFT else NoteType.LEFT


def _direction_position_ok(direction: CutDirection, col: int, row: int) -> bool:
    """
    Return True if this cut direction makes physical sense at this grid position.

    The cut direction defines where the saber ENTERS from (opposite side).
    e.g. DOWN means the saber comes from ABOVE — so at row 2 (top of grid)
    there is nowhere above to start, making a DOWN stroke physically awkward.
    """
    if direction in (CutDirection.DOWN, CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT):
        # Saber approaches from above — needs headroom above the note
        if row >= 2:
            return False
    if direction in (CutDirection.UP, CutDirection.UP_LEFT, CutDirection.UP_RIGHT):
        # Saber approaches from below — needs room below the note
        if row <= 0:
            return False
    return True


def _place_note(
    hand: NoteType,
    beat_t: float,
    onset_t: float,
    energy: float,
    hands: dict[NoteType, HandState],
    min_gap_s: float,
    rng: random.Random,
) -> Optional[Note]:
    """Place a single note for the given hand, respecting flow and reachability."""
    state = hands[hand]

    # Enforce minimum time gap between same-hand notes
    if onset_t - state.last_time_s < min_gap_s:
        return None

    # 1. Pick position first — reachable from last note, filtered by hand zone and energy
    positions = reachable_positions(state.last_col, state.last_row)

    # Strict hand zones: left=cols 0-1, right=cols 2-3
    if hand == NoteType.LEFT:
        strict = [(c, r) for c, r in positions if c <= 1]
        positions = strict or [(c, r) for c, r in positions if c <= 2] or positions
    else:
        strict = [(c, r) for c, r in positions if c >= 2]
        positions = strict or [(c, r) for c, r in positions if c >= 1] or positions

    # Row bias: prefer bottom/middle rows (0-1); top row only on high energy
    if energy < 0.7:
        low_rows = [(c, r) for c, r in positions if r <= 1]
        positions = low_rows or positions

    col, row = rng.choice(positions)

    # 2. Pick direction from flow map, filtered to be physically valid for this position.
    #    This ensures e.g. a DOWN stroke isn't placed at the top row (nowhere to start above).
    flow_candidates = FLOW_MAP.get(state.last_direction, list(CutDirection))
    # Avoid repeating the same direction (no wrist resets)
    flow_candidates = [d for d in flow_candidates if d != state.last_direction] or flow_candidates
    # Filter to directions that are ergonomically legal at (col, row)
    compat = [d for d in flow_candidates if _direction_position_ok(d, col, row)]
    direction = rng.choice((compat or flow_candidates)[:4])

    note = Note(
        time=beat_t,
        line_index=col,
        line_layer=row,
        type=hand,
        cut_direction=direction,
    )

    # Update state
    state.last_col = col
    state.last_row = row
    state.last_direction = direction
    state.last_time_s = onset_t

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
    """Generate beat-synced lighting events (ring lights, lasers, center glow)."""
    events: list[Event] = []
    if not analysis.beat_times:
        return events

    first_onset = analysis.onset_times[0] if analysis.onset_times else 0.0

    def _bt(t: float) -> float:
        return round(time_to_beat(t, analysis.tempo, first_onset), 4)

    # Beat-synced base lighting
    for i, beat_t in enumerate(analysis.beat_times):
        bt = _bt(beat_t)
        energy = rms_at(analysis, beat_t)
        blue_beat = (i % 2 == 0)

        # Ring lights (type 1): every beat, alternate blue/red, flash on energy
        ring_val = (2 if blue_beat else 6) if energy > 0.6 else (1 if blue_beat else 5)
        events.append(Event(time=bt, type=1, value=ring_val))

        # Center/road light (type 4): every beat
        center_val = (2 if blue_beat else 6) if energy > 0.6 else (1 if blue_beat else 5)
        events.append(Event(time=bt, type=4, value=center_val))

        # Back lasers (type 0): every 2 beats
        if i % 2 == 0:
            events.append(Event(time=bt, type=0, value=2 if energy > 0.5 else 1))

        # Rotating lasers (types 2, 3): every 4 beats
        if i % 4 == 0:
            events.append(Event(time=bt, type=2, value=6 if energy > 0.65 else 5))
            events.append(Event(time=bt, type=3, value=2 if energy > 0.65 else 1))

        # Ring rotation (type 8): every 8 beats
        if i % 8 == 0:
            events.append(Event(time=bt, type=8, value=1))

        # Ring zoom (type 9): every 16 beats
        if i % 16 == 0:
            events.append(Event(time=bt, type=9, value=1))

        # Laser speed (types 12, 13): every 8 beats, scaled to energy
        if i % 8 == 0:
            speed = max(1, min(10, int(energy * 12)))
            events.append(Event(time=bt, type=12, value=speed))
            events.append(Event(time=bt, type=13, value=speed))

    # Extra flashes on very strong onsets
    for onset_t in analysis.onset_times:
        bt = _bt(onset_t)
        energy = rms_at(analysis, onset_t)
        if energy > 0.85:
            events.append(Event(time=bt, type=0, value=6))  # back laser red flash
            events.append(Event(time=bt, type=4, value=2))  # center blue flash

    events.sort(key=lambda e: (e.time, e.type))
    return events
