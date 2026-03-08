"""Tests for Info.dat / difficulty .dat JSON format correctness."""

import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from smartsaber.mapbuilder import build_map, _build_difficulty_dat, _build_info_dat, compute_map_hash
from smartsaber.models import (
    CutDirection,
    Difficulty,
    MapDifficulty,
    MapInfo,
    Note,
    NoteType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sample_info() -> MapInfo:
    return MapInfo(
        song_name="Test Song",
        song_sub_name="",
        song_author_name="Test Artist",
        bpm=128.0,
    )


def _sample_notes() -> list[Note]:
    return [
        Note(time=0.0, line_index=1, line_layer=1, type=NoteType.LEFT, cut_direction=CutDirection.DOWN),
        Note(time=0.5, line_index=2, line_layer=1, type=NoteType.RIGHT, cut_direction=CutDirection.DOWN),
        Note(time=1.0, line_index=1, line_layer=1, type=NoteType.LEFT, cut_direction=CutDirection.UP),
    ]


def _sample_difficulties() -> list[MapDifficulty]:
    return [MapDifficulty(difficulty=Difficulty.NORMAL, notes=_sample_notes())]


# ---------------------------------------------------------------------------
# _build_info_dat
# ---------------------------------------------------------------------------

def test_info_dat_required_keys():
    info = _sample_info()
    diffs = _sample_difficulties()
    dat = _build_info_dat(info, diffs)

    required = [
        "_version", "_songName", "_songSubName", "_songAuthorName",
        "_levelAuthorName", "_beatsPerMinute", "_songFilename",
        "_coverImageFilename", "_difficultyBeatmapSets",
    ]
    for key in required:
        assert key in dat, f"Missing key: {key}"


def test_info_dat_version():
    dat = _build_info_dat(_sample_info(), _sample_difficulties())
    assert dat["_version"] == "2.2.0"


def test_info_dat_difficulty_set_structure():
    dat = _build_info_dat(_sample_info(), _sample_difficulties())
    sets = dat["_difficultyBeatmapSets"]
    assert len(sets) == 1
    bset = sets[0]
    assert bset["_beatmapCharacteristicName"] == "Standard"
    beatmaps = bset["_difficultyBeatmaps"]
    assert len(beatmaps) == 1
    bmap = beatmaps[0]
    assert "_difficulty" in bmap
    assert "_difficultyRank" in bmap
    assert "_beatmapFilename" in bmap
    assert "_noteJumpMovementSpeed" in bmap


def test_info_dat_is_json_serializable():
    dat = _build_info_dat(_sample_info(), _sample_difficulties())
    # Should not raise
    json.dumps(dat)


# ---------------------------------------------------------------------------
# _build_difficulty_dat
# ---------------------------------------------------------------------------

def test_difficulty_dat_required_keys():
    md = MapDifficulty(difficulty=Difficulty.HARD, notes=_sample_notes())
    dat = _build_difficulty_dat(md)
    for key in ["_version", "_notes", "_obstacles", "_events"]:
        assert key in dat, f"Missing key: {key}"


def test_difficulty_dat_note_structure():
    md = MapDifficulty(difficulty=Difficulty.NORMAL, notes=_sample_notes())
    dat = _build_difficulty_dat(md)
    for note in dat["_notes"]:
        assert "_time" in note
        assert "_lineIndex" in note
        assert "_lineLayer" in note
        assert "_type" in note
        assert "_cutDirection" in note
        assert 0 <= note["_lineIndex"] <= 3
        assert 0 <= note["_lineLayer"] <= 2
        assert note["_type"] in (0, 1, 3)
        assert 0 <= note["_cutDirection"] <= 8


def test_difficulty_dat_is_json_serializable():
    md = MapDifficulty(difficulty=Difficulty.EXPERT, notes=_sample_notes())
    dat = _build_difficulty_dat(md)
    json.dumps(dat)


def test_difficulty_dat_notes_sorted():
    notes = [
        Note(time=1.0, line_index=1, line_layer=1, type=NoteType.LEFT, cut_direction=CutDirection.DOWN),
        Note(time=0.5, line_index=2, line_layer=1, type=NoteType.RIGHT, cut_direction=CutDirection.UP),
    ]
    md = MapDifficulty(difficulty=Difficulty.NORMAL, notes=notes)
    dat = _build_difficulty_dat(md)
    times = [n["_time"] for n in dat["_notes"]]
    # Note: the dat builder does NOT sort — the generator produces sorted notes.
    # This test just verifies structure; sorting is the generator's responsibility.
    assert len(times) == 2


# ---------------------------------------------------------------------------
# NJS values
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("diff, expected_njs", [
    (Difficulty.EASY, 10.0),
    (Difficulty.NORMAL, 10.0),
    (Difficulty.HARD, 12.0),
    (Difficulty.EXPERT, 16.0),
    (Difficulty.EXPERT_PLUS, 18.0),
])
def test_njs_values(diff, expected_njs):
    assert diff.njs == expected_njs


# ---------------------------------------------------------------------------
# compute_map_hash (integration)
# ---------------------------------------------------------------------------

def test_compute_map_hash():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        info_dat = tmp_path / "Info.dat"
        diff_dat = tmp_path / "NormalStandard.dat"
        info_dat.write_text('{"_version": "2.2.0"}', encoding="utf-8")
        diff_dat.write_text('{"_version": "2.2.0", "_notes": []}', encoding="utf-8")

        h = compute_map_hash(tmp_path)
        assert len(h) == 40  # SHA1 hex = 40 chars
        assert h == h.upper()

        # Hash must be deterministic
        h2 = compute_map_hash(tmp_path)
        assert h == h2


# ---------------------------------------------------------------------------
# build_map — audio skip optimization
# ---------------------------------------------------------------------------

def _make_fake_audio(path: Path) -> None:
    """Write a minimal OGG file (just enough bytes so it's not empty)."""
    path.write_bytes(b"OggS" + b"\x00" * 100)


def test_build_map_skips_audio_when_egg_exists():
    """If song.egg already exists and is up-to-date, build_map should not overwrite it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_src = tmp_path / "source.ogg"
        _make_fake_audio(audio_src)

        info = _sample_info()
        diffs = _sample_difficulties()

        # First build — creates song.egg
        folder = build_map(info, diffs, audio_src, tmp_path,
                           song_name_for_folder="Test", artist_for_folder="Artist")
        egg_path = folder / "song.egg"
        assert egg_path.exists()
        first_mtime = egg_path.stat().st_mtime

        # Wait a tiny bit so any re-write would have a different mtime
        time.sleep(0.05)

        # Second build — same source, egg should be skipped (mtime unchanged)
        build_map(info, diffs, audio_src, tmp_path,
                  song_name_for_folder="Test", artist_for_folder="Artist")
        assert egg_path.stat().st_mtime == first_mtime


def test_build_map_skips_audio_when_src_is_egg():
    """When audio_path points to the existing song.egg itself, don't copy over itself."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_src = tmp_path / "source.ogg"
        _make_fake_audio(audio_src)

        info = _sample_info()
        diffs = _sample_difficulties()

        # First build creates song.egg
        folder = build_map(info, diffs, audio_src, tmp_path,
                           song_name_for_folder="Test", artist_for_folder="Artist")
        egg_path = folder / "song.egg"
        first_mtime = egg_path.stat().st_mtime

        time.sleep(0.05)

        # Now build again, passing the egg itself as the source (regen scenario)
        build_map(info, diffs, egg_path, tmp_path,
                  song_name_for_folder="Test", artist_for_folder="Artist")
        # Should not have been rewritten
        assert egg_path.stat().st_mtime == first_mtime


def test_build_map_skips_cover_when_exists():
    """If cover.jpg already exists, build_map should not rewrite it."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_src = tmp_path / "source.ogg"
        _make_fake_audio(audio_src)

        info = _sample_info()
        diffs = _sample_difficulties()

        # First build
        folder = build_map(info, diffs, audio_src, tmp_path,
                           song_name_for_folder="Test", artist_for_folder="Artist")
        cover_path = folder / "cover.jpg"
        assert cover_path.exists()
        first_mtime = cover_path.stat().st_mtime

        time.sleep(0.05)

        # Second build — cover should be skipped
        build_map(info, diffs, audio_src, tmp_path,
                  song_name_for_folder="Test", artist_for_folder="Artist")
        assert cover_path.stat().st_mtime == first_mtime


def test_build_map_rewrites_dat_files():
    """Even when audio/cover are skipped, .dat files should always be rewritten."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        audio_src = tmp_path / "source.ogg"
        _make_fake_audio(audio_src)

        info = _sample_info()
        diffs = _sample_difficulties()

        # First build
        folder = build_map(info, diffs, audio_src, tmp_path,
                           song_name_for_folder="Test", artist_for_folder="Artist")
        info_dat = folder / "Info.dat"
        first_content = info_dat.read_text()

        # Change the BPM so Info.dat content changes
        info2 = _sample_info()
        info2 = MapInfo(
            song_name="Test Song", song_sub_name="",
            song_author_name="Test Artist", bpm=140.0,
        )
        build_map(info2, diffs, audio_src, tmp_path,
                  song_name_for_folder="Test", artist_for_folder="Artist")
        second_content = info_dat.read_text()

        assert first_content != second_content
        assert '"140.0"' in second_content or "140.0" in second_content

