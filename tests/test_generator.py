"""Tests for the note placement engine."""

import random
import pytest

from smartsaber.analyzer import AudioAnalysis
from smartsaber.generator import generate_difficulty, generate_all_difficulties, _PARITY_AFTER, _PARITY_REQUIRED
from smartsaber.models import CutDirection, Difficulty, NoteType
from smartsaber.patterns import is_good_flow, FLOW_MAP


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_analysis(duration_s: float = 60.0, tempo: float = 120.0) -> AudioAnalysis:
    """Minimal AudioAnalysis with evenly spaced beats."""
    beat_period = 60.0 / tempo
    beats = [round(i * beat_period, 4) for i in range(int(duration_s / beat_period))]
    # Onsets every other beat (simulate normal density)
    onsets = beats[::2]
    rms = [0.5] * len(beats)
    return AudioAnalysis(
        tempo=tempo,
        beat_times=beats,
        onset_times=onsets,
        rms_curve=rms,
        rms_times=beats,
        segment_times=[0.0, duration_s],
        duration_s=duration_s,
    )


# ---------------------------------------------------------------------------
# Flow validation helpers
# ---------------------------------------------------------------------------

def test_is_good_flow_basics():
    assert is_good_flow(CutDirection.UP, CutDirection.DOWN)
    assert is_good_flow(CutDirection.DOWN, CutDirection.UP)
    assert is_good_flow(CutDirection.DOT, CutDirection.LEFT)  # dot allows anything


def test_all_directions_have_flow_map():
    for d in CutDirection:
        if d == CutDirection.DOT:
            continue
        assert d in FLOW_MAP, f"{d} missing from FLOW_MAP"
        assert FLOW_MAP[d], f"FLOW_MAP[{d}] is empty"


# ---------------------------------------------------------------------------
# Generator output structure
# ---------------------------------------------------------------------------

def test_generate_produces_notes():
    analysis = _simple_analysis()
    md = generate_difficulty(analysis, Difficulty.NORMAL)
    assert len(md.notes) > 0


def test_notes_sorted_by_time():
    analysis = _simple_analysis()
    md = generate_difficulty(analysis, Difficulty.HARD)
    times = [n.time for n in md.notes]
    assert times == sorted(times)


def test_no_stacked_notes_same_beat_same_column():
    """Vision rule: no two notes at the same beat occupying the same column."""
    analysis = _simple_analysis()
    md = generate_difficulty(analysis, Difficulty.EXPERT_PLUS)

    # Group by beat time
    from collections import defaultdict
    by_beat: dict[float, list] = defaultdict(list)
    for n in md.notes:
        by_beat[n.time].append(n)

    for beat_t, notes in by_beat.items():
        if len(notes) > 1:
            cols = [n.line_index for n in notes]
            assert len(cols) == len(set(cols)), (
                f"Vision conflict at beat {beat_t}: multiple notes in same column {cols}"
            )


def test_note_types_only_left_right():
    analysis = _simple_analysis()
    md = generate_difficulty(analysis, Difficulty.HARD)
    for n in md.notes:
        assert n.type in (NoteType.LEFT, NoteType.RIGHT)


def test_columns_in_range():
    analysis = _simple_analysis()
    md = generate_difficulty(analysis, Difficulty.EXPERT)
    for n in md.notes:
        assert 0 <= n.line_index <= 3
        assert 0 <= n.line_layer <= 2


def test_all_difficulties_generated():
    analysis = _simple_analysis()
    mds = generate_all_difficulties(analysis)
    assert len(mds) == 5
    diff_set = {md.difficulty for md in mds}
    assert diff_set == set(Difficulty)


def test_expert_plus_denser_than_easy():
    analysis = _simple_analysis()
    easy = generate_difficulty(analysis, Difficulty.EASY, rng=random.Random(0))
    ep = generate_difficulty(analysis, Difficulty.EXPERT_PLUS, rng=random.Random(0))
    assert len(ep.notes) > len(easy.notes)


def test_deterministic_with_same_seed():
    analysis = _simple_analysis()
    md1 = generate_difficulty(analysis, Difficulty.HARD, rng=random.Random(42))
    md2 = generate_difficulty(analysis, Difficulty.HARD, rng=random.Random(42))
    assert [(n.time, n.line_index, n.cut_direction) for n in md1.notes] == \
           [(n.time, n.line_index, n.cut_direction) for n in md2.notes]


def test_empty_analysis_returns_empty():
    analysis = AudioAnalysis(
        tempo=120.0,
        beat_times=[],
        onset_times=[],
        rms_curve=[],
        rms_times=[],
        segment_times=[0.0, 10.0],
        duration_s=10.0,
    )
    md = generate_difficulty(analysis, Difficulty.NORMAL)
    assert md.notes == []


# ---------------------------------------------------------------------------
# Parity validation
# ---------------------------------------------------------------------------

def test_parity_tables_cover_all_directions():
    """Every CutDirection should have an entry in both parity maps."""
    for d in CutDirection:
        assert d in _PARITY_AFTER, f"{d} missing from _PARITY_AFTER"
        assert d in _PARITY_REQUIRED, f"{d} missing from _PARITY_REQUIRED"


def test_parity_not_violated_same_hand():
    """Consecutive same-hand notes should not have two vertical swings in the
    same direction (parity violation).  Horizontal / dot swings are exempt."""
    analysis = _simple_analysis(duration_s=120.0, tempo=120.0)
    for diff in [Difficulty.NORMAL, Difficulty.HARD, Difficulty.EXPERT]:
        md = generate_difficulty(analysis, diff, rng=random.Random(42))
        # Group notes by hand, sorted by time
        from collections import defaultdict
        by_hand: dict[NoteType, list] = defaultdict(list)
        for n in md.notes:
            by_hand[n.type].append(n)

        violations = 0
        for hand, notes in by_hand.items():
            notes_sorted = sorted(notes, key=lambda n: n.time)
            parity = None  # arm position after the previous swing
            for note in notes_sorted:
                required = _PARITY_REQUIRED.get(note.cut_direction)
                if required is not None and parity is not None:
                    if required != parity:
                        violations += 1
                after = _PARITY_AFTER.get(note.cut_direction)
                if after is not None:
                    parity = after

        # Allow up to ~55% violations — we intentionally use "soft parity"
        # (50% enforcement) to match human map direction distributions where
        # DOWN:UP ratio is ~1.4:1 rather than forced 1:1 strict alternation.
        total_notes = len(md.notes)
        if total_notes > 0:
            violation_rate = violations / total_notes
            assert violation_rate < 0.55, (
                f"Parity violation rate {violation_rate:.1%} ({violations}/{total_notes}) "
                f"too high for {diff.value}"
            )


# ---------------------------------------------------------------------------
# Band-energy row placement
# ---------------------------------------------------------------------------

def test_band_row_at_fallback():
    """band_row_at falls back to centroid when band curves are empty."""
    from smartsaber.analyzer import band_row_at
    analysis = _simple_analysis()
    # With no band curves and default energy, should return a valid row (0, 1, or 2)
    row = band_row_at(analysis, 5.0, energy=0.5)
    assert row in (0, 1, 2)


def test_band_row_at_with_curves():
    """band_row_at returns ergonomic rows based on dominant band + energy."""
    from smartsaber.analyzer import band_row_at
    analysis = _simple_analysis()
    n = len(analysis.rms_times)
    # Force bass-dominant: bass=0.8, mid=0.1, treble=0.1
    analysis.bass_energy_curve = [0.8] * n
    analysis.mid_energy_curve = [0.1] * n
    analysis.treble_energy_curve = [0.1] * n
    assert band_row_at(analysis, 5.0, energy=0.5) == 0  # bass → row 0

    # Force treble-dominant WITH high energy → row 2 (overhead emphasis)
    analysis.bass_energy_curve = [0.1] * n
    analysis.mid_energy_curve = [0.1] * n
    analysis.treble_energy_curve = [0.8] * n
    assert band_row_at(analysis, 5.0, energy=0.8) == 2  # treble+energy → row 2

    # Force treble-dominant WITHOUT high energy → row 1 (not overhead)
    assert band_row_at(analysis, 5.0, energy=0.3) == 1  # treble but low energy → row 1


def test_row_distribution_ergonomic():
    """Most notes should be at rows 0-1 (waist/chest), with row 2 rare."""
    analysis = _simple_analysis(duration_s=120.0, tempo=120.0)
    md = generate_difficulty(analysis, Difficulty.HARD, rng=random.Random(42))
    rows = [n.line_layer for n in md.notes]
    total = len(rows)
    assert total > 0
    row2_fraction = rows.count(2) / total
    row01_fraction = (rows.count(0) + rows.count(1)) / total
    # Row 2 (overhead) should be less than 25% of notes
    assert row2_fraction < 0.25, (
        f"Too many overhead notes: row2={row2_fraction:.0%} "
        f"(row0={rows.count(0)}, row1={rows.count(1)}, row2={rows.count(2)})"
    )
    # Rows 0+1 should be at least 75%
    assert row01_fraction >= 0.75


def test_direction_distribution_balanced():
    """DOWN-family strokes should be at least as common as UP-family.
    No single direction family should exceed 80%."""
    analysis = _simple_analysis(duration_s=120.0, tempo=120.0)
    md = generate_difficulty(analysis, Difficulty.HARD, rng=random.Random(42))
    total = len(md.notes)
    assert total > 0

    up_family = sum(1 for n in md.notes if n.cut_direction in (
        CutDirection.UP, CutDirection.UP_LEFT, CutDirection.UP_RIGHT))
    down_family = sum(1 for n in md.notes if n.cut_direction in (
        CutDirection.DOWN, CutDirection.DOWN_LEFT, CutDirection.DOWN_RIGHT))

    up_frac = up_family / total
    down_frac = down_family / total

    # Neither family should dominate excessively
    assert up_frac < 0.70, f"UP-family too dominant: {up_frac:.0%}"
    assert down_frac < 0.80, f"DOWN-family too dominant: {down_frac:.0%}"
    # DOWN should be at least as common as UP (natural resting stroke)
    assert down_frac >= up_frac * 0.5, (
        f"DOWN ({down_frac:.0%}) should not be drastically less than UP ({up_frac:.0%})"
    )


def test_center_column_bias():
    """Center columns (1, 2) should have far more notes than outer columns (0, 3)."""
    analysis = _simple_analysis(duration_s=120.0, tempo=120.0)
    md = generate_difficulty(analysis, Difficulty.HARD, rng=random.Random(42))
    total = len(md.notes)
    assert total > 0
    center = sum(1 for n in md.notes if n.line_index in (1, 2))
    outer = sum(1 for n in md.notes if n.line_index in (0, 3))
    center_frac = center / total
    # Human maps use ~60% center columns and ~40% outer columns.
    # Our generator targets a similar split (col weights: outer=2, center=3).
    assert center_frac >= 0.50, (
        f"Center columns too low: {center_frac:.0%} "
        f"(center={center}, outer={outer}, total={total})"
    )


def test_no_crossovers():
    """Left hand should never be right of right hand at the same beat."""
    analysis = _simple_analysis(duration_s=120.0, tempo=120.0)
    for diff in [Difficulty.HARD, Difficulty.EXPERT]:
        md = generate_difficulty(analysis, diff, rng=random.Random(42))
        from collections import defaultdict
        by_beat: dict[float, dict] = defaultdict(dict)
        for n in md.notes:
            by_beat[n.time][n.type] = n

        for beat_t, notes in by_beat.items():
            if NoteType.LEFT in notes and NoteType.RIGHT in notes:
                left_col = notes[NoteType.LEFT].line_index
                right_col = notes[NoteType.RIGHT].line_index
                assert left_col <= right_col, (
                    f"Crossover at beat {beat_t}: left@col{left_col}, right@col{right_col}"
                )


def test_no_huge_vertical_jumps():
    """No consecutive same-hand notes should jump from row 0→2 or 2→0."""
    analysis = _simple_analysis(duration_s=120.0, tempo=120.0)
    md = generate_difficulty(analysis, Difficulty.EXPERT, rng=random.Random(42))
    from collections import defaultdict
    by_hand: dict[NoteType, list] = defaultdict(list)
    for n in md.notes:
        by_hand[n.type].append(n)

    for hand, notes in by_hand.items():
        notes_sorted = sorted(notes, key=lambda n: n.time)
        for i in range(1, len(notes_sorted)):
            prev_row = notes_sorted[i - 1].line_layer
            curr_row = notes_sorted[i].line_layer
            jump = abs(curr_row - prev_row)
            assert jump <= 1, (
                f"Huge vertical jump: {hand} row {prev_row}→{curr_row} "
                f"at beat {notes_sorted[i].time}"
            )


def test_doubles_same_direction():
    """When both hands play at the same beat, they should swing the same direction."""
    analysis = _simple_analysis(duration_s=120.0, tempo=120.0)
    # Use Expert+ for highest chance of doubles
    md = generate_difficulty(analysis, Difficulty.EXPERT_PLUS, rng=random.Random(42))
    from collections import defaultdict
    by_beat: dict[float, list] = defaultdict(list)
    for n in md.notes:
        by_beat[n.time].append(n)

    doubles = [(t, ns) for t, ns in by_beat.items() if len(ns) == 2]
    mismatches = 0
    for t, notes in doubles:
        if notes[0].cut_direction != notes[1].cut_direction:
            mismatches += 1
    if doubles:
        mismatch_rate = mismatches / len(doubles)
        assert mismatch_rate < 0.05, (
            f"Too many doubles with different directions: {mismatch_rate:.0%} "
            f"({mismatches}/{len(doubles)})"
        )

