"""Tests for fuzzy matching and string normalization."""

import pytest

from smartsaber.models import BeatSaverMap, Track
from smartsaber.matcher import find_best_match, score_match
from smartsaber.utils import normalize_string


# ---------------------------------------------------------------------------
# normalize_string
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    ("Hello World", "hello world"),
    ("Song (feat. Someone)", "song"),
    ("Track (ft. Artist)", "track"),
    ("Album (Remastered)", "album"),
    ("Album (2023 Remastered)", "album"),
    ("Song (Deluxe Edition)", "song"),
    ("Song [Explicit]", "song"),
    ("Song (Radio Edit)", "song"),
    ("Café", "cafe"),
    ("Naïve  (Live)", "naive"),
    ("  hello   world  ", "hello world"),
])
def test_normalize(raw, expected):
    assert normalize_string(raw) == expected


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track(title: str, artist: str) -> Track:
    return Track(
        title=title,
        artist=artist,
        artists_all=[artist],
        album="",
        duration_ms=180_000,
        album_art_url=None,
        source_id="test",
        source="test",
    )


def _map(name: str, artist: str, upvotes: int = 100, downvotes: int = 5, diffs: list[str] | None = None) -> BeatSaverMap:
    return BeatSaverMap(
        id="abc1",
        name=name,
        artist=artist,
        bpm=120.0,
        duration_s=180.0,
        upvotes=upvotes,
        downvotes=downvotes,
        download_url="https://example.com/map.zip",
        difficulties=diffs or ["Easy", "Normal", "Hard"],
        hash="ABCDEF1234567890" * 2,
    )


# ---------------------------------------------------------------------------
# score_match
# ---------------------------------------------------------------------------

def test_exact_match_high_scores():
    track = _track("Bohemian Rhapsody", "Queen")
    m = _map("Bohemian Rhapsody", "Queen")
    ts, as_ = score_match(track, m)
    assert ts >= 95
    assert as_ >= 95


def test_feat_suffix_ignored():
    track = _track("Song (feat. Artist B)", "Artist A")
    m = _map("Song", "Artist A")
    ts, _ = score_match(track, m)
    assert ts >= 85


def test_remastered_ignored():
    track = _track("Yesterday (Remastered 2009)", "The Beatles")
    m = _map("Yesterday", "The Beatles")
    ts, _ = score_match(track, m)
    assert ts >= 85


def test_case_insensitive():
    track = _track("HELLO WORLD", "ARTIST")
    m = _map("hello world", "artist")
    ts, as_ = score_match(track, m)
    assert ts >= 95
    assert as_ >= 95


# ---------------------------------------------------------------------------
# find_best_match
# ---------------------------------------------------------------------------

def test_finds_best_map():
    track = _track("Shape of You", "Ed Sheeran")
    candidates = [
        _map("Shape of You", "Ed Sheeran", upvotes=200, diffs=["Easy", "Normal", "Hard", "Expert"]),
        _map("Shape Of You", "ed sheeran", upvotes=50, diffs=["Hard"]),
    ]
    match = find_best_match(track, candidates)
    assert match is not None
    assert match.title_score >= 85
    # Should prefer more difficulties
    assert len(match.map.difficulties) >= 3


def test_no_match_below_threshold():
    track = _track("Never Gonna Give You Up", "Rick Astley")
    candidates = [_map("Totally Different Song", "Someone Else")]
    match = find_best_match(track, candidates)
    assert match is None


def test_low_upvote_ratio_rejected():
    track = _track("Song", "Artist")
    candidates = [_map("Song", "Artist", upvotes=1, downvotes=99)]
    match = find_best_match(track, candidates, min_upvote_ratio=0.6)
    assert match is None


def test_empty_candidates():
    track = _track("Song", "Artist")
    match = find_best_match(track, [])
    assert match is None


def test_unicode_normalization():
    track = _track("Naïve", "The Kooks")
    m = _map("Naive", "The Kooks")
    ts, _ = score_match(track, m)
    assert ts >= 90
