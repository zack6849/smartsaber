"""Tests for .bplist file generation."""

import json
import tempfile
from pathlib import Path

import pytest

from smartsaber.models import GenerationResult, PlaylistInfo, Track
from smartsaber.playlist import build_playlist


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track(title: str = "Song", artist: str = "Artist") -> Track:
    return Track(
        title=title,
        artist=artist,
        artists_all=[artist],
        album="Album",
        duration_ms=200_000,
        album_art_url=None,
        source_id="id1",
        source="test",
    )


def _playlist_info(name: str = "My Playlist") -> PlaylistInfo:
    return PlaylistInfo(name=name, description="desc", cover_url=None, track_count=1)


def _success_result(track: Track, folder: Path) -> GenerationResult:
    # Write minimal Info.dat and a diff dat so hash can be computed
    (folder / "Info.dat").write_text(
        json.dumps({
            "_version": "2.2.0",
            "_difficultyBeatmapSets": [
                {
                    "_beatmapCharacteristicName": "Standard",
                    "_difficultyBeatmaps": [
                        {
                            "_difficulty": "Normal",
                            "_difficultyRank": 3,
                            "_beatmapFilename": "NormalStandard.dat",
                            "_noteJumpMovementSpeed": 10.0,
                            "_noteJumpStartBeatOffset": 0.0,
                            "_customData": {},
                        }
                    ],
                }
            ],
        }),
        encoding="utf-8",
    )
    (folder / "NormalStandard.dat").write_text(
        '{"_version":"2.2.0","_notes":[],"_obstacles":[],"_events":[]}',
        encoding="utf-8",
    )
    from smartsaber.mapbuilder import compute_map_hash
    h = compute_map_hash(folder)
    return GenerationResult(
        track=track,
        success=True,
        output_path=str(folder),
        map_hash=h,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_build_playlist_creates_file():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        map_folder = tmp_path / "map1"
        map_folder.mkdir()
        track = _track()
        result = _success_result(track, map_folder)
        plist_path = build_playlist([result], _playlist_info(), tmp_path)
        assert plist_path.exists()
        assert plist_path.suffix == ".bplist"


def test_bplist_is_valid_json():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        map_folder = tmp_path / "map1"
        map_folder.mkdir()
        result = _success_result(_track(), map_folder)
        plist_path = build_playlist([result], _playlist_info(), tmp_path)
        data = json.loads(plist_path.read_text(encoding="utf-8"))
        assert "songs" in data
        assert "playlistTitle" in data


def test_bplist_contains_all_songs():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        results = []
        for i in range(3):
            folder = tmp_path / f"map{i}"
            folder.mkdir()
            results.append(_success_result(_track(title=f"Song {i}"), folder))

        plist_path = build_playlist(results, _playlist_info(), tmp_path)
        data = json.loads(plist_path.read_text(encoding="utf-8"))
        assert len(data["songs"]) == 3


def test_bplist_excludes_failed_results():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        folder = tmp_path / "map1"
        folder.mkdir()
        ok = _success_result(_track(title="Good"), folder)
        fail = GenerationResult(
            track=_track(title="Bad"),
            success=False,
            output_path=None,
            map_hash=None,
            error="download failed",
        )
        plist_path = build_playlist([ok, fail], _playlist_info(), tmp_path)
        data = json.loads(plist_path.read_text(encoding="utf-8"))
        assert len(data["songs"]) == 1


def test_bplist_song_has_hash():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        folder = tmp_path / "map1"
        folder.mkdir()
        result = _success_result(_track(), folder)
        plist_path = build_playlist([result], _playlist_info(), tmp_path)
        data = json.loads(plist_path.read_text(encoding="utf-8"))
        song = data["songs"][0]
        assert "hash" in song
        assert len(song["hash"]) == 40
        assert song["hash"] == song["hash"].upper()


def test_bplist_level_id_format():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        folder = tmp_path / "map1"
        folder.mkdir()
        result = _success_result(_track(), folder)
        plist_path = build_playlist([result], _playlist_info(), tmp_path)
        data = json.loads(plist_path.read_text(encoding="utf-8"))
        song = data["songs"][0]
        assert song["levelid"].startswith("custom_level_")


def test_bplist_playlist_title():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        folder = tmp_path / "map1"
        folder.mkdir()
        result = _success_result(_track(), folder)
        info = _playlist_info(name="Weekend Vibes")
        plist_path = build_playlist([result], info, tmp_path)
        data = json.loads(plist_path.read_text(encoding="utf-8"))
        assert data["playlistTitle"] == "Weekend Vibes"
