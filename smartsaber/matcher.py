"""Fuzzy-match BeatSaver search results against Track objects."""

from __future__ import annotations

from typing import Optional

from rapidfuzz import fuzz

from smartsaber.models import Track, BeatSaverMap, BeatSaverMatch
from smartsaber.utils import normalize_string


def score_match(track: Track, bs_map: BeatSaverMap) -> tuple[float, float]:
    """Return (title_score, artist_score) using token_sort_ratio."""
    t_title = normalize_string(track.title)
    t_artist = normalize_string(track.artist)
    m_title = normalize_string(bs_map.name)
    m_artist = normalize_string(bs_map.artist)

    title_score = fuzz.token_sort_ratio(t_title, m_title)
    artist_score = fuzz.token_sort_ratio(t_artist, m_artist)
    return title_score, artist_score


def find_best_match(
    track: Track,
    candidates: list[BeatSaverMap],
    title_threshold: float = 85.0,
    artist_threshold: float = 75.0,
    min_upvote_ratio: float = 0.6,
) -> Optional[BeatSaverMatch]:
    """
    Given a list of BeatSaver search results pick the best community map.

    Criteria (in order):
    1. Title score >= title_threshold
    2. Artist score >= artist_threshold
    3. Upvote ratio >= min_upvote_ratio
    4. Prefer more difficulty levels, then higher upvote ratio
    """
    qualified: list[tuple[BeatSaverMap, float, float]] = []

    for m in candidates:
        ts, as_ = score_match(track, m)
        if ts < title_threshold:
            continue
        if as_ < artist_threshold:
            continue
        if m.upvote_ratio < min_upvote_ratio:
            continue
        qualified.append((m, ts, as_))

    if not qualified:
        return None

    # Sort: more difficulties first, then upvote ratio
    qualified.sort(key=lambda x: (len(x[0].difficulties), x[0].upvote_ratio), reverse=True)
    best_map, ts, as_ = qualified[0]
    return BeatSaverMatch(track=track, map=best_map, title_score=ts, artist_score=as_)
