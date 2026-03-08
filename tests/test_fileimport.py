"""Tests for fileimport — load_tracks and count_tracks."""

import json
import tempfile
from pathlib import Path

import pytest

from smartsaber.fileimport import count_tracks, load_tracks


# ---------------------------------------------------------------------------
# count_tracks — CSV
# ---------------------------------------------------------------------------

def _write_csv(path: Path, header: str, rows: list[str]) -> None:
    lines = [header] + rows
    path.write_text("\n".join(lines), encoding="utf-8")


def test_count_tracks_csv_basic():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        _write_csv(p, "Spotify ID,Artist Name(s),Track Name,Album Name,Track Duration (ms)", [
            "id1,Artist A,Song One,Album,200000",
            "id2,Artist B,Song Two,Album,300000",
            "id3,Artist C,Song Three,Album,250000",
        ])
        assert count_tracks(p) == 3


def test_count_tracks_csv_skips_empty_rows():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        _write_csv(p, "Spotify ID,Artist Name(s),Track Name", [
            "id1,Artist,Song",
            ",,",          # empty row — should not count
            "",            # blank line
            "id2,Artist,Song2",
        ])
        assert count_tracks(p) == 2


def test_count_tracks_csv_header_only():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        _write_csv(p, "Spotify ID,Artist Name(s),Track Name", [])
        assert count_tracks(p) == 0


def test_count_tracks_csv_empty_file():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        p.write_text("", encoding="utf-8")
        assert count_tracks(p) == 0


# ---------------------------------------------------------------------------
# count_tracks — JSON
# ---------------------------------------------------------------------------

def test_count_tracks_json_basic():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.json"
        data = [
            {"title": "Song A", "artist": "Art A"},
            {"title": "Song B", "artist": "Art B"},
        ]
        p.write_text(json.dumps(data), encoding="utf-8")
        assert count_tracks(p) == 2


def test_count_tracks_json_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.json"
        p.write_text("[]", encoding="utf-8")
        assert count_tracks(p) == 0


def test_count_tracks_json_not_a_list():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.json"
        p.write_text('{"key": "value"}', encoding="utf-8")
        assert count_tracks(p) == 0


# ---------------------------------------------------------------------------
# count_tracks — error handling
# ---------------------------------------------------------------------------

def test_count_tracks_nonexistent_file():
    p = Path("/tmp/nonexistent_file_abc123.csv")
    assert count_tracks(p) == 0


def test_count_tracks_unsupported_extension():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.txt"
        p.write_text("hello", encoding="utf-8")
        assert count_tracks(p) == 0


# ---------------------------------------------------------------------------
# count_tracks matches load_tracks count
# ---------------------------------------------------------------------------

def test_count_tracks_matches_load_tracks_csv():
    """count_tracks and load_tracks should agree on number of tracks."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.csv"
        _write_csv(p, "Spotify ID,Artist Name(s),Track Name,Album Name,Track Duration (ms)", [
            "id1,Artist A,Song One,Album,200000",
            "id2,Artist B,Song Two,Album,300000",
            "id3,,Song Three,Album,250000",  # no artist but has title
        ])
        count = count_tracks(p)
        loaded = load_tracks(p)
        assert count == len(loaded)


def test_count_tracks_matches_load_tracks_json():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "test.json"
        data = [
            {"title": "Song A", "artist": "Art A", "duration_ms": 200000},
            {"title": "", "artist": "Art B"},  # empty title — load_tracks skips
            {"title": "Song C", "artist": "Art C"},
        ]
        p.write_text(json.dumps(data), encoding="utf-8")
        # count_tracks counts raw list length (3), but load_tracks skips
        # entries without a title (2).  count_tracks is intentionally a fast
        # approximation — it doesn't parse individual fields.
        count = count_tracks(p)
        loaded = load_tracks(p)
        assert count >= len(loaded)  # count may be >= since it doesn't filter

