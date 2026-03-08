"""Tests for pattern templates and flow rules."""

import pytest

from smartsaber.models import CutDirection, NoteType
from smartsaber.patterns import (
    FLOW_MAP,
    PATTERNS,
    Pattern,
    PatternNote,
    is_good_flow,
    next_direction,
    patterns_for,
    reachable_positions,
)


# ---------------------------------------------------------------------------
# FLOW_MAP completeness
# ---------------------------------------------------------------------------

def test_flow_map_no_self_loops():
    """A cut direction should never lead back to itself (wrist reset)."""
    for direction, follow_ups in FLOW_MAP.items():
        if direction == CutDirection.DOT:
            continue  # dot is special
        assert direction not in follow_ups, (
            f"Direction {direction} has itself as a follow-up (wrist reset)"
        )


def test_flow_map_all_directions_covered():
    for d in CutDirection:
        if d == CutDirection.DOT:
            continue
        assert d in FLOW_MAP


# ---------------------------------------------------------------------------
# Pattern validity
# ---------------------------------------------------------------------------

def test_patterns_non_empty():
    assert len(PATTERNS) >= 15


def test_patterns_have_valid_columns():
    for p in PATTERNS:
        for note in p.notes:
            assert 0 <= note.col <= 3, f"Pattern {p.name}: invalid col {note.col}"
            assert 0 <= note.row <= 2, f"Pattern {p.name}: invalid row {note.row}"


def test_patterns_have_valid_energy_range():
    for p in PATTERNS:
        assert 0.0 <= p.energy_min <= 1.0, f"Pattern {p.name}: energy_min out of range"
        assert 0.0 <= p.energy_max <= 1.0, f"Pattern {p.name}: energy_max out of range"
        assert p.energy_min <= p.energy_max, f"Pattern {p.name}: energy_min > energy_max"


def test_patterns_have_valid_difficulty():
    valid_ranks = {1, 3, 5, 7, 9}
    for p in PATTERNS:
        assert p.min_difficulty in valid_ranks, (
            f"Pattern {p.name}: invalid min_difficulty {p.min_difficulty}"
        )


def test_patterns_note_flow():
    """Each consecutive pair of notes for the same hand should be good flow."""
    for p in PATTERNS:
        # Group notes by hand
        from collections import defaultdict
        by_hand: dict = defaultdict(list)
        for note in sorted(p.notes, key=lambda n: n.beat_offset):
            by_hand[note.hand].append(note)

        for hand, notes in by_hand.items():
            for i in range(1, len(notes)):
                prev_dir = notes[i - 1].direction
                next_dir = notes[i].direction
                assert is_good_flow(prev_dir, next_dir), (
                    f"Pattern {p.name}, hand {hand}: "
                    f"bad flow {prev_dir} → {next_dir}"
                )


def test_patterns_no_simultaneous_same_col_same_hand():
    """Two notes for the same hand at the same beat must not be in the same column."""
    for p in PATTERNS:
        from collections import defaultdict
        by_time_hand: dict = defaultdict(list)
        for note in p.notes:
            by_time_hand[(note.beat_offset, note.hand)].append(note)
        for (t, hand), notes in by_time_hand.items():
            cols = [n.col for n in notes]
            assert len(cols) == len(set(cols)), (
                f"Pattern {p.name}: same-hand col collision at beat {t}"
            )


# ---------------------------------------------------------------------------
# patterns_for
# ---------------------------------------------------------------------------

def test_patterns_for_easy():
    result = patterns_for(0.2, 1)
    assert len(result) > 0
    for p in result:
        assert p.energy_min <= 0.2 <= p.energy_max
        assert p.min_difficulty <= 1


def test_patterns_for_expert_plus():
    result = patterns_for(0.9, 9)
    assert len(result) > 0


def test_patterns_for_low_energy_excludes_hard():
    result = patterns_for(0.05, 1)
    for p in result:
        assert p.energy_min <= 0.05


# ---------------------------------------------------------------------------
# reachable_positions
# ---------------------------------------------------------------------------

def test_reachable_includes_current():
    positions = reachable_positions(1, 1)
    assert (1, 1) in positions


def test_reachable_all_in_grid():
    for col in range(4):
        for row in range(3):
            for c, r in reachable_positions(col, row):
                assert 0 <= c <= 3
                assert 0 <= r <= 2


# ---------------------------------------------------------------------------
# next_direction
# ---------------------------------------------------------------------------

def test_next_direction_avoids_same():
    d = next_direction(CutDirection.UP)
    assert d != CutDirection.UP


def test_next_direction_honors_preferred():
    d = next_direction(CutDirection.DOWN, preferred=[CutDirection.UP])
    assert d == CutDirection.UP
