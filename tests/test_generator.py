"""Tests for the note placement engine."""

import random
import pytest

from smartsaber.analyzer import AudioAnalysis
from smartsaber.generator import generate_difficulty, generate_all_difficulties
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
