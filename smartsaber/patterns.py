"""Beat Saber pattern templates and flow rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from smartsaber.models import CutDirection, NoteType

# ---------------------------------------------------------------------------
# Flow map: after cutting in direction X, which directions are comfortable next?
# ---------------------------------------------------------------------------
# Values are ordered best-to-worst to allow weighted selection.

FLOW_MAP: dict[CutDirection, list[CutDirection]] = {
    CutDirection.UP: [
        CutDirection.DOWN,
        CutDirection.DOWN_LEFT,
        CutDirection.DOWN_RIGHT,
        CutDirection.UP_LEFT,
        CutDirection.UP_RIGHT,
        CutDirection.LEFT,
        CutDirection.RIGHT,
        CutDirection.DOT,
    ],
    CutDirection.DOWN: [
        CutDirection.UP,
        CutDirection.UP_LEFT,
        CutDirection.UP_RIGHT,
        CutDirection.DOWN_LEFT,
        CutDirection.DOWN_RIGHT,
        CutDirection.LEFT,
        CutDirection.RIGHT,
        CutDirection.DOT,
    ],
    CutDirection.LEFT: [
        CutDirection.RIGHT,
        CutDirection.UP_RIGHT,
        CutDirection.DOWN_RIGHT,
        CutDirection.UP,
        CutDirection.DOWN,
        CutDirection.DOT,
    ],
    CutDirection.RIGHT: [
        CutDirection.LEFT,
        CutDirection.UP_LEFT,
        CutDirection.DOWN_LEFT,
        CutDirection.UP,
        CutDirection.DOWN,
        CutDirection.DOT,
    ],
    CutDirection.UP_LEFT: [
        CutDirection.DOWN,
        CutDirection.DOWN_RIGHT,
        CutDirection.DOWN_LEFT,
        CutDirection.UP,
        CutDirection.RIGHT,
        CutDirection.DOT,
    ],
    CutDirection.UP_RIGHT: [
        CutDirection.DOWN,
        CutDirection.DOWN_LEFT,
        CutDirection.DOWN_RIGHT,
        CutDirection.UP,
        CutDirection.LEFT,
        CutDirection.DOT,
    ],
    CutDirection.DOWN_LEFT: [
        CutDirection.UP,
        CutDirection.UP_RIGHT,
        CutDirection.DOWN,
        CutDirection.RIGHT,
        CutDirection.DOT,
    ],
    CutDirection.DOWN_RIGHT: [
        CutDirection.UP,
        CutDirection.UP_LEFT,
        CutDirection.DOWN,
        CutDirection.LEFT,
        CutDirection.DOT,
    ],
    CutDirection.DOT: list(CutDirection),
}


def is_good_flow(prev: CutDirection, next_: CutDirection) -> bool:
    """Return True if next_ is a legal follow-up to prev."""
    if prev == CutDirection.DOT or next_ == CutDirection.DOT:
        return True
    return next_ in FLOW_MAP.get(prev, [])


def next_direction(prev: CutDirection, preferred: Optional[list[CutDirection]] = None) -> CutDirection:
    """Choose the best-flow next direction (optionally biased by preferred list)."""
    candidates = FLOW_MAP.get(prev, list(CutDirection))
    if preferred:
        for d in preferred:
            if d in candidates:
                return d
    return candidates[0]


# ---------------------------------------------------------------------------
# Position reachability
# ---------------------------------------------------------------------------
# Grid is 4 wide × 3 tall.  line_index 0=left…3=right, line_layer 0=bot…2=top.

# Comfortable column transitions for each hand given the previous column.
# Center columns (1, 2) are always reachable; outer columns (0, 3) only
# from adjacent positions.
_COL_REACH: dict[int, list[int]] = {
    0: [0, 1, 2],
    1: [0, 1, 2, 3],
    2: [0, 1, 2, 3],
    3: [1, 2, 3],
}

# Comfortable row transitions — only adjacent rows.  Jumping bottom→top
# (row 0→2) in a single beat forces a huge arm motion that's uncomfortable
# at speed.  The player should move through row 1 to get from 0→2.
_ROW_REACH: dict[int, list[int]] = {
    0: [0, 1],
    1: [0, 1, 2],
    2: [1, 2],
}


def reachable_positions(
    last_col: int, last_row: int
) -> list[tuple[int, int]]:
    """Return (col, row) pairs reachable from the last note position."""
    cols = _COL_REACH.get(last_col, [0, 1, 2, 3])
    rows = _ROW_REACH.get(last_row, [0, 1, 2])
    return [(c, r) for c in cols for r in rows]


# ---------------------------------------------------------------------------
# Pattern templates
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PatternNote:
    """Relative-time note within a pattern (beat offset from pattern start)."""

    beat_offset: float
    col: int                  # 0-3
    row: int                  # 0-2
    hand: NoteType            # LEFT (red) or RIGHT (blue)
    direction: CutDirection


@dataclass(frozen=True)
class Pattern:
    name: str
    notes: tuple[PatternNote, ...]
    energy_min: float         # 0-1 minimum RMS energy level
    energy_max: float         # 0-1 maximum RMS energy level
    min_difficulty: int       # 1=Easy…9=ExpertPlus (rank)
    complexity: int           # 1 (simple) to 5 (very complex)


def _p(*notes: PatternNote, name: str, emin: float, emax: float, diff: int, complexity: int) -> Pattern:
    return Pattern(name=name, notes=notes, energy_min=emin, energy_max=emax, min_difficulty=diff, complexity=complexity)


L = NoteType.LEFT
R = NoteType.RIGHT
U = CutDirection.UP
D = CutDirection.DOWN
LE = CutDirection.LEFT
RI = CutDirection.RIGHT
UL = CutDirection.UP_LEFT
UR = CutDirection.UP_RIGHT
DL = CutDirection.DOWN_LEFT
DR = CutDirection.DOWN_RIGHT
DOT = CutDirection.DOT

# fmt: off
PATTERNS: list[Pattern] = [
    # --- Basic alternating (low energy, Easy+) ---
    _p(
        PatternNote(0.0, 1, 1, L, D),
        PatternNote(0.5, 2, 1, R, D),
        name="alt_down", emin=0.0, emax=0.5, diff=1, complexity=1,
    ),
    _p(
        PatternNote(0.0, 1, 1, L, U),
        PatternNote(0.5, 2, 1, R, U),
        name="alt_up", emin=0.0, emax=0.5, diff=1, complexity=1,
    ),

    # --- Down-up stream ---
    _p(
        PatternNote(0.0, 1, 1, L, D),
        PatternNote(0.25, 2, 1, R, D),
        PatternNote(0.5, 1, 1, L, U),
        PatternNote(0.75, 2, 1, R, U),
        name="stream_du", emin=0.3, emax=0.7, diff=3, complexity=2,
    ),

    # --- Zigzag ---
    _p(
        PatternNote(0.0, 0, 1, L, UR),
        PatternNote(0.25, 3, 1, R, UL),
        PatternNote(0.5, 0, 1, L, DR),
        PatternNote(0.75, 3, 1, R, DL),
        name="zigzag", emin=0.4, emax=0.8, diff=3, complexity=3,
    ),

    # --- Cross-lane single cross ---
    _p(
        PatternNote(0.0, 2, 1, L, D),   # red goes right of centre
        PatternNote(0.5, 1, 1, R, D),   # blue goes left of centre
        name="cross_lane", emin=0.4, emax=1.0, diff=5, complexity=2,
    ),

    # --- Double (both hands same time) ---
    _p(
        PatternNote(0.0, 1, 1, L, D),
        PatternNote(0.0, 2, 1, R, D),
        name="double_down", emin=0.6, emax=1.0, diff=3, complexity=2,
    ),
    _p(
        PatternNote(0.0, 1, 2, L, U),
        PatternNote(0.0, 2, 2, R, U),
        name="double_up", emin=0.6, emax=1.0, diff=3, complexity=2,
    ),

    # --- Pinwheel (was spiral) ---
    # Outer columns sweep up, inner columns sweep down — stays in mid row.
    # L: UR→DR flow; R: UL→DL flow — both ergonomically correct.
    _p(
        PatternNote(0.0,  0, 1, L, UR),
        PatternNote(0.25, 3, 1, R, UL),
        PatternNote(0.5,  1, 1, L, DR),
        PatternNote(0.75, 2, 1, R, DL),
        name="spiral_cw", emin=0.5, emax=1.0, diff=5, complexity=4,
    ),

    # --- Staircase up — ascending columns and row, UP strokes only ---
    # UP at rows 1 and 2 is fine (saber enters from the row below).
    _p(
        PatternNote(0.0,  0, 1, L, U),
        PatternNote(0.25, 2, 1, R, U),
        PatternNote(0.5,  1, 2, L, UL),
        name="stair_up", emin=0.3, emax=0.7, diff=3, complexity=2,
    ),

    # --- Staircase down — starts high, flows into downward strokes ---
    # First note at top with UL (saber enters from below); next two use DOWN.
    _p(
        PatternNote(0.0,  1, 2, L, UL),
        PatternNote(0.25, 2, 1, R, D),
        PatternNote(0.5,  0, 1, L, DL),
        name="stair_down", emin=0.3, emax=0.7, diff=3, complexity=2,
    ),

    # --- Tower (rapid same-hand flick, mid row only) ---
    # All notes at row 1 — saber oscillates up/down without leaving the grid.
    _p(
        PatternNote(0.0,   1, 1, L, U),
        PatternNote(0.167, 1, 1, L, D),
        PatternNote(0.333, 1, 1, L, U),
        name="tower_left", emin=0.6, emax=1.0, diff=7, complexity=4,
    ),

    # --- Side-swipe left to right ---
    _p(
        PatternNote(0.0, 0, 1, L, RI),
        PatternNote(0.5, 3, 1, R, LE),
        name="sideswipe", emin=0.4, emax=0.9, diff=5, complexity=3,
    ),

    # --- High-density stream (ExpertPlus) ---
    _p(
        PatternNote(0.0, 1, 1, L, D),
        PatternNote(0.125, 2, 1, R, D),
        PatternNote(0.25, 1, 2, L, U),
        PatternNote(0.375, 2, 2, R, U),
        PatternNote(0.5, 1, 0, L, D),
        PatternNote(0.625, 2, 0, R, D),
        PatternNote(0.75, 1, 1, L, U),
        PatternNote(0.875, 2, 1, R, U),
        name="stream_dense", emin=0.7, emax=1.0, diff=9, complexity=5,
    ),

    # --- Dot note rest (any energy) ---
    _p(
        PatternNote(0.0, 1, 1, L, DOT),
        PatternNote(0.5, 2, 1, R, DOT),
        name="dots_alt", emin=0.0, emax=0.4, diff=1, complexity=1,
    ),

    # --- Diagonal alternating — outer columns, mid row, swapping diagonal angle ---
    # DL at row 1: saber enters from above-right. UR at row 1: from below-left. Both valid.
    _p(
        PatternNote(0.0,  0, 1, L, DL),
        PatternNote(0.5,  3, 1, R, UR),
        PatternNote(1.0,  0, 1, L, UR),
        PatternNote(1.5,  3, 1, R, DL),
        name="diagonal_alt", emin=0.5, emax=1.0, diff=5, complexity=3,
    ),

    # --- Simple left-right alternating (Easy) ---
    _p(
        PatternNote(0.0, 1, 1, L, LE),
        PatternNote(1.0, 2, 1, R, RI),
        name="lr_alt", emin=0.0, emax=0.5, diff=1, complexity=1,
    ),

    # --- Burst (3 fast notes) ---
    _p(
        PatternNote(0.0, 1, 1, L, D),
        PatternNote(0.167, 2, 1, R, D),
        PatternNote(0.333, 1, 2, L, U),
        name="burst_3", emin=0.55, emax=1.0, diff=5, complexity=3,
    ),

    # --- Up-down single hand stream (mid row flick) ---
    # All at row 1 so the saber never needs to enter from outside the grid.
    _p(
        PatternNote(0.0,  1, 1, L, U),
        PatternNote(0.25, 1, 1, L, D),
        PatternNote(0.5,  1, 1, L, U),
        PatternNote(0.75, 1, 1, L, D),
        name="ud_stream_left", emin=0.6, emax=1.0, diff=7, complexity=4,
    ),

    # --- Wrist roll (alternating diagonals) ---
    _p(
        PatternNote(0.0, 1, 1, L, UL),
        PatternNote(0.25, 1, 1, L, DR),
        PatternNote(0.5, 2, 1, R, UR),
        PatternNote(0.75, 2, 1, R, DL),
        name="wrist_roll", emin=0.5, emax=1.0, diff=7, complexity=4,
    ),
]
# fmt: on


def patterns_for(energy: float, difficulty_rank: int) -> list[Pattern]:
    """Return patterns suitable for the given energy level and difficulty rank."""
    return [
        p for p in PATTERNS
        if p.energy_min <= energy <= p.energy_max
        and p.min_difficulty <= difficulty_rank
    ]
