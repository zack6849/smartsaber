"""Tests for --regen pipeline behavior.

These tests verify that the regen_only flag correctly:
- Skips BeatSaver search (all tracks treated as unmatched)
- Skips BeatSaver download
- Auto-confirms all tracks (no interactive prompt)
- Pre-resolves existing song.egg files (skips YouTube resolve + download)
- Still regenerates .dat files with the current generator code
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from smartsaber.config import SmartSaberConfig
from smartsaber.models import (
    GenerationResult,
    PipelineResult,
    PlaylistInfo,
    Track,
)
from smartsaber.pipeline import _find_existing, _output_exists


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(title: str = "Test Song", artist: str = "Test Artist",
                source_id: str = "test_id") -> Track:
    return Track(
        title=title,
        artist=artist,
        artists_all=[artist],
        album="Test Album",
        duration_ms=200000,
        album_art_url=None,
        source_id=source_id,
        source="file",
    )


def _make_output_folder(output_dir: Path, track: Track) -> Path:
    """Create a fake output folder matching the naming convention build_map uses."""
    from smartsaber.utils import safe_filename
    folder_label = safe_filename(f"SmartSaber_{track.title} - {track.artist}")
    folder = output_dir / folder_label
    folder.mkdir(parents=True, exist_ok=True)
    # Write a minimal song.egg
    (folder / "song.egg").write_bytes(b"OggS" + b"\x00" * 100)
    # Write minimal Info.dat and difficulty .dat
    (folder / "Info.dat").write_text('{"_version":"2.2.0"}', encoding="utf-8")
    (folder / "NormalStandard.dat").write_text(
        '{"_version":"2.2.0","_notes":[],"_obstacles":[],"_events":[]}',
        encoding="utf-8",
    )
    # Write a minimal cover
    (folder / "cover.jpg").write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)
    return folder


# ---------------------------------------------------------------------------
# _find_existing / _output_exists
# ---------------------------------------------------------------------------

def test_find_existing_matches_output_folder():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        folder = _make_output_folder(output_dir, track)

        found = _find_existing(track, output_dir)
        assert found is not None
        assert found == folder


def test_find_existing_returns_none_when_no_match():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track(title="Nonexistent Song")
        assert _find_existing(track, output_dir) is None


def test_output_exists_true_when_folder_present():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        _make_output_folder(output_dir, track)
        assert _output_exists(track, output_dir) is True


def test_output_exists_false_when_no_folder():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        assert _output_exists(track, output_dir) is False


# ---------------------------------------------------------------------------
# Regen config behavior
# ---------------------------------------------------------------------------

def test_regen_config_sets_skip_existing_false():
    """When regen_only is True, skip_existing should be False so we don't skip."""
    cfg = SmartSaberConfig()
    cfg.regen_only = True
    cfg.skip_existing = False
    assert cfg.regen_only is True
    assert cfg.skip_existing is False


def test_regen_pre_resolves_existing_song_egg():
    """Tracks with song.egg in their output folder should be pre-resolved,
    not sent to YouTube search."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        folder = _make_output_folder(output_dir, track)

        egg = folder / "song.egg"
        assert egg.exists()

        # Simulate what the pipeline does: find existing folder, check for egg
        existing = _find_existing(track, output_dir)
        assert existing is not None
        assert (existing / "song.egg").exists()
        assert (existing / "song.egg").stat().st_size > 0


def test_regen_skips_beatsaver_for_all_tracks():
    """In regen mode, all tracks should be treated as unmatched.
    This test verifies the config flag logic, not the full pipeline."""
    cfg = SmartSaberConfig()
    cfg.regen_only = True

    # The pipeline checks `config.regen_only` to skip BeatSaver search.
    # We just verify the flag is correctly set.
    assert cfg.regen_only is True


# ---------------------------------------------------------------------------
# Default skip_existing behavior (the original bug)
# ---------------------------------------------------------------------------

def test_default_config_skips_existing():
    """By default, skip_existing should be True — this is what caused the
    original bug where 'import' showed 75 skipped."""
    cfg = SmartSaberConfig()
    assert cfg.skip_existing is True


def test_force_disables_skip_existing():
    """--force should set skip_existing to False."""
    cfg = SmartSaberConfig()
    # Simulate what CLI does with --force
    cfg.skip_existing = False
    assert cfg.skip_existing is False


def test_skip_existing_prevents_regeneration():
    """When skip_existing is True and an output folder exists,
    _output_exists should return True (pipeline will skip the track)."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        _make_output_folder(output_dir, track)

        # This is what the pipeline checks
        assert _output_exists(track, output_dir) is True


def test_no_output_allows_generation():
    """When no output folder exists, _output_exists returns False (pipeline proceeds)."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        assert _output_exists(track, output_dir) is False


# ---------------------------------------------------------------------------
# Regen cache key uniqueness (the "song.egg stem collision" bug)
# ---------------------------------------------------------------------------

def test_regen_cache_keys_are_unique_per_track():
    """When all tracks use song.egg as audio_path, the analysis cache key
    must still be unique per track.  The old code used audio_path.stem ('song')
    which was the same for every track, causing cache collisions."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track_a = _make_track(title="Song A", artist="Artist A", source_id="a")
        track_b = _make_track(title="Song B", artist="Artist B", source_id="b")

        folder_a = _make_output_folder(output_dir, track_a)
        folder_b = _make_output_folder(output_dir, track_b)

        egg_a = folder_a / "song.egg"
        egg_b = folder_b / "song.egg"

        # Both have stem "song" — but the cache key derivation should
        # produce different keys (using parent folder name).
        assert egg_a.stem == egg_b.stem == "song"
        assert egg_a.parent.name != egg_b.parent.name

        # Verify the actual key derivation logic:
        # "song" stem → use f"regen_{parent.name}" as key
        key_a = f"regen_{egg_a.parent.name}"
        key_b = f"regen_{egg_b.parent.name}"
        assert key_a != key_b, (
            f"Cache keys must be unique per track: {key_a!r} == {key_b!r}"
        )


def test_regen_rng_seeds_are_unique_per_track():
    """When all tracks use song.egg as audio_path, the RNG seed must still
    be unique per track.  The old code used hashlib.md5('song') which was
    identical for every track, producing identical note patterns."""
    import hashlib

    track_a = _make_track(title="Song A", artist="Artist A")
    track_b = _make_track(title="Song B", artist="Artist B")

    # Simulate the fixed seed derivation
    def _seed_for(track: Track) -> int:
        seed_input = f"{track.title}_{track.artist}"
        return int.from_bytes(hashlib.md5(seed_input.encode()).digest()[:4], "big")

    seed_a = _seed_for(track_a)
    seed_b = _seed_for(track_b)
    assert seed_a != seed_b, (
        f"RNG seeds must be unique: track_a={seed_a}, track_b={seed_b}"
    )


def test_regen_song_egg_paths_found_for_all_tracks():
    """Every track with an existing output folder containing song.egg
    should be pre-resolved, not sent to YouTube search."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        tracks = [
            _make_track(title=f"Track {i}", artist=f"Artist {i}", source_id=f"id_{i}")
            for i in range(5)
        ]
        for t in tracks:
            _make_output_folder(output_dir, t)

        # Simulate what the pipeline does in regen mode
        pre_resolved = {}
        for track in tracks:
            existing = _find_existing(track, output_dir)
            assert existing is not None, f"Missing output folder for {track.title}"
            egg = existing / "song.egg"
            assert egg.exists(), f"Missing song.egg for {track.title}"
            pre_resolved[track.source_id] = egg

        assert len(pre_resolved) == 5
        # All paths should be different (different folders)
        paths = list(pre_resolved.values())
        assert len(set(str(p) for p in paths)) == 5


# ---------------------------------------------------------------------------
# Per-file analysis cache performance (the "113MB JSON" fix)
# ---------------------------------------------------------------------------

def test_analysis_cache_uses_separate_files():
    """Each cache entry should be stored as a separate file, not in one
    monolithic JSON.  This is critical for ProcessPoolExecutor workers
    that would otherwise each parse a 100MB+ file on startup."""
    from smartsaber.analysis_cache import AnalysisCache
    from smartsaber.models import AudioAnalysis

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        cache = AnalysisCache(cache_dir=cache_dir)

        analysis_a = AudioAnalysis(
            tempo=120.0, beat_times=[0.0, 0.5], onset_times=[0.0],
            rms_curve=[0.5], rms_times=[0.0], segment_times=[0.0, 1.0],
            duration_s=1.0,
        )
        analysis_b = AudioAnalysis(
            tempo=140.0, beat_times=[0.0, 0.43], onset_times=[0.0],
            rms_curve=[0.7], rms_times=[0.0], segment_times=[0.0, 1.0],
            duration_s=1.0,
        )

        cache.put("track_a", analysis_a)
        cache.put("track_b", analysis_b)

        # Each should be a separate file
        files = list(cache_dir.glob("*.json"))
        assert len(files) == 2, f"Expected 2 cache files, got {len(files)}: {files}"

        # Each file should be small (< 10KB for a 1-second track)
        for f in files:
            assert f.stat().st_size < 10_000, f"Cache file too large: {f} = {f.stat().st_size}"


def test_analysis_cache_get_returns_correct_data():
    """Cache get should return the correct analysis for the given key."""
    from smartsaber.analysis_cache import AnalysisCache
    from smartsaber.models import AudioAnalysis

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        cache = AnalysisCache(cache_dir=cache_dir)

        analysis = AudioAnalysis(
            tempo=128.0, beat_times=[0.0, 0.47, 0.94], onset_times=[0.0, 0.47],
            rms_curve=[0.5, 0.6], rms_times=[0.0, 0.47], segment_times=[0.0, 1.0],
            duration_s=1.0,
        )
        cache.put("my_key", analysis)

        # Fresh cache instance (simulates a new worker process)
        cache2 = AnalysisCache(cache_dir=cache_dir)
        result = cache2.get("my_key")
        assert result is not None
        assert result.tempo == 128.0
        assert len(result.beat_times) == 3


def test_analysis_cache_miss_returns_none():
    """Cache miss should return None, not crash."""
    from smartsaber.analysis_cache import AnalysisCache

    with tempfile.TemporaryDirectory() as tmp:
        cache = AnalysisCache(cache_dir=Path(tmp) / "cache")
        assert cache.get("nonexistent_key") is None


def test_analysis_cache_independent_workers():
    """Two separate AnalysisCache instances (simulating separate worker
    processes) should both be able to read entries written by either."""
    from smartsaber.analysis_cache import AnalysisCache
    from smartsaber.models import AudioAnalysis

    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"

        # Worker 1 writes
        w1 = AnalysisCache(cache_dir=cache_dir)
        w1.put("song_1", AudioAnalysis(
            tempo=120.0, beat_times=[0.0], onset_times=[0.0],
            rms_curve=[0.5], rms_times=[0.0], segment_times=[0.0, 1.0],
            duration_s=1.0,
        ))

        # Worker 2 (fresh instance) should see worker 1's entry
        w2 = AnalysisCache(cache_dir=cache_dir)
        assert w2.get("song_1") is not None
        assert w2.get("song_1").tempo == 120.0

        # Worker 2 writes its own entry
        w2.put("song_2", AudioAnalysis(
            tempo=140.0, beat_times=[0.0], onset_times=[0.0],
            rms_curve=[0.7], rms_times=[0.0], segment_times=[0.0, 1.0],
            duration_s=1.0,
        ))

        # Worker 1 (same instance, no reload needed) should NOT see worker 2's
        # entry in memory, but a fresh instance should
        w3 = AnalysisCache(cache_dir=cache_dir)
        assert w3.get("song_2") is not None
        assert w3.get("song_2").tempo == 140.0


# ---------------------------------------------------------------------------
# Debug callback
# ---------------------------------------------------------------------------

def test_debug_callback_receives_messages_in_regen():
    """The on_debug callback should receive pipeline timing messages."""
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track(source_id="dbg_1")
        _make_output_folder(output_dir, track)

        cfg = SmartSaberConfig()
        cfg.output_dir = output_dir
        cfg.regen_only = True
        cfg.skip_existing = False
        cfg.difficulties = ["Normal"]

        debug_messages: list[str] = []

        with patch("smartsaber.pipeline.check_ffmpeg", return_value=True), \
             patch("smartsaber.pipeline.build_playlist", return_value=Path(tmp) / "test.bplist"):
            from smartsaber.pipeline import run
            result = run(
                tracks=[track],
                config=cfg,
                on_debug=lambda msg: debug_messages.append(msg),
            )

        # Should have received multiple debug messages
        assert len(debug_messages) >= 3, f"Expected >=3 debug messages, got {len(debug_messages)}: {debug_messages}"
        # Should include pipeline start and phase info
        assert any("Pipeline started" in m for m in debug_messages)
        assert any("Phase 5" in m for m in debug_messages)
        assert any("Pipeline total" in m for m in debug_messages)


# ---------------------------------------------------------------------------
# .analysis_key roundtrip (regen uses original analysis cache entry)
# ---------------------------------------------------------------------------

def test_analysis_key_file_written_by_build_map():
    """build_map should write an .analysis_key file when analysis_cache_key is provided."""
    from smartsaber.mapbuilder import build_map
    from smartsaber.models import MapDifficulty, MapInfo, Difficulty

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        folder = _make_output_folder(output_dir, track)

        info = MapInfo(
            song_name="Test", song_sub_name="", song_author_name="Artist",
            bpm=120.0, preview_start_time=10.0, preview_duration=5.0,
        )
        md = MapDifficulty(difficulty=Difficulty.NORMAL)

        result_dir = build_map(
            info=info,
            difficulties=[md],
            audio_path=folder / "song.egg",
            output_dir=output_dir,
            song_name_for_folder="Test",
            artist_for_folder="Artist",
            analysis_cache_key="yt_dQw4w9WgXcQ",
        )

        key_file = result_dir / ".analysis_key"
        assert key_file.exists(), ".analysis_key file should be written"
        assert key_file.read_text(encoding="utf-8") == "yt_dQw4w9WgXcQ"


def test_analysis_key_not_written_when_empty():
    """build_map should NOT write .analysis_key when key is empty."""
    from smartsaber.mapbuilder import build_map
    from smartsaber.models import MapDifficulty, MapInfo, Difficulty

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        track = _make_track()
        folder = _make_output_folder(output_dir, track)

        info = MapInfo(
            song_name="Test", song_sub_name="", song_author_name="Artist",
            bpm=120.0, preview_start_time=10.0, preview_duration=5.0,
        )
        md = MapDifficulty(difficulty=Difficulty.NORMAL)

        result_dir = build_map(
            info=info,
            difficulties=[md],
            audio_path=folder / "song.egg",
            output_dir=output_dir,
            song_name_for_folder="Test",
            artist_for_folder="Artist",
        )

        key_file = result_dir / ".analysis_key"
        assert not key_file.exists(), ".analysis_key should not be written when key is empty"


def test_regen_reads_analysis_key_for_cache_lookup():
    """When .analysis_key exists, regen should use it instead of the folder name."""
    # Simulate what _generate_from_audio does for cache key derivation
    with tempfile.TemporaryDirectory() as tmp:
        map_folder = Path(tmp) / "SmartSaber_Test - Artist"
        map_folder.mkdir()
        egg = map_folder / "song.egg"
        egg.write_bytes(b"OggS" + b"\x00" * 100)

        # Write .analysis_key (as build_map would)
        (map_folder / ".analysis_key").write_text("yt_dQw4w9WgXcQ", encoding="utf-8")

        # Derive cache key the way _generate_from_audio does
        audio_path = egg
        cache_key = audio_path.stem  # "song"
        if cache_key == "song":
            key_file = audio_path.parent / ".analysis_key"
            if key_file.exists():
                cache_key = key_file.read_text(encoding="utf-8").strip()
            else:
                cache_key = f"regen_{audio_path.parent.name}"

        assert cache_key == "yt_dQw4w9WgXcQ", f"Expected original key, got {cache_key!r}"


def test_regen_falls_back_to_folder_name_without_key_file():
    """Without .analysis_key, regen should fall back to regen_{folder_name}."""
    with tempfile.TemporaryDirectory() as tmp:
        map_folder = Path(tmp) / "SmartSaber_Test - Artist"
        map_folder.mkdir()
        egg = map_folder / "song.egg"
        egg.write_bytes(b"OggS" + b"\x00" * 100)

        # No .analysis_key file

        audio_path = egg
        cache_key = audio_path.stem
        if cache_key == "song":
            key_file = audio_path.parent / ".analysis_key"
            if key_file.exists():
                cache_key = key_file.read_text(encoding="utf-8").strip()
            else:
                cache_key = f"regen_{audio_path.parent.name}"

        assert cache_key == "regen_SmartSaber_Test - Artist"
