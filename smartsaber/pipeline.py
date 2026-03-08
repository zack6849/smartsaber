"""
Core pipeline orchestrator — no terminal I/O.

The pipeline:
1. Searches BeatSaver for each track.
2. Calls `on_batch_confirm` with unmatched tracks; caller returns which to generate.
3. Downloads audio for confirmed tracks.
4. Analyses + generates maps.
5. Assembles map folders.
6. Builds .bplist playlist.
7. Returns PipelineResult.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import shutil
import subprocess
import threading
import time as _time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import httpx

from smartsaber.analysis_cache import AnalysisCache
from smartsaber.analyzer import analyze
from smartsaber.beatsaver import find_map, download_map
from smartsaber.bs_cache import BSCache
from smartsaber.config import SmartSaberConfig
from smartsaber.generator import generate_all_difficulties
from smartsaber.mapbuilder import build_map, compute_map_hash
from smartsaber.models import (
    Difficulty,
    GenerationResult,
    MapInfo,
    PipelineResult,
    PlaylistInfo,
    Track,
)
from smartsaber.playlist import build_playlist
from smartsaber.utils import normalize_string
from smartsaber.yt_cache import YTCache
from smartsaber.youtube import fetch_audio, resolve_url, YouTubeResolution

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Callback type aliases (for documentation purposes)
# ---------------------------------------------------------------------------

OnProgress = Callable[[str, int, int], None]
OnBatchConfirm = Callable[[list[Track]], list[Track]]
OnUrlReview = Callable[[list[YouTubeResolution]], list[YouTubeResolution]]
OnTrackComplete = Callable[[Track, GenerationResult], None]
OnTrackStage = Callable[[Track, str], None]
OnError = Callable[[Track, Exception], None]
OnDebug = Callable[[str], None]


def _noop_progress(stage: str, current: int, total: int) -> None:
    pass


def _noop_batch_confirm(tracks: list[Track]) -> list[Track]:
    return tracks


def _noop_url_review(resolutions: list[YouTubeResolution]) -> list[YouTubeResolution]:
    return resolutions


def _noop_track_complete(track: Track, result: GenerationResult) -> None:
    pass


def _noop_track_stage(track: Track, stage: str) -> None:
    pass


def _noop_error(track: Track, exc: Exception) -> None:
    pass


def _noop_debug(msg: str) -> None:
    pass


# ---------------------------------------------------------------------------
# ffmpeg check
# ---------------------------------------------------------------------------

def check_ffmpeg() -> bool:
    """Return True if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run(
    tracks: list[Track],
    config: SmartSaberConfig,
    playlist_info: Optional[PlaylistInfo] = None,
    on_progress: OnProgress = _noop_progress,
    on_batch_confirm: OnBatchConfirm = _noop_batch_confirm,
    on_url_review: OnUrlReview = _noop_url_review,
    on_track_complete: OnTrackComplete = _noop_track_complete,
    on_track_stage: OnTrackStage = _noop_track_stage,
    on_error: OnError = _noop_error,
    on_debug: OnDebug = _noop_debug,
) -> PipelineResult:
    """
    Run the full SmartSaber pipeline.

    Parameters
    ----------
    tracks:
        List of provider-agnostic Track objects to process.
    config:
        Runtime configuration (output dir, thresholds, etc.).
    playlist_info:
        Metadata for the .bplist file (name, cover URL, etc.).
    on_progress:
        Called as (stage_name, current_index, total).
    on_batch_confirm:
        Called with unmatched tracks; returns subset to proceed with.
    on_track_complete:
        Called after each track is fully processed (success or failure).
    on_error:
        Called on unexpected exceptions for a track.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _pipeline_t0 = _time.perf_counter()

    # Debug: dump configuration
    on_debug(f"Pipeline started — {len(tracks)} tracks")
    on_debug(f"  output_dir       = {output_dir}")
    on_debug(f"  regen_only       = {config.regen_only}")
    on_debug(f"  skip_existing    = {config.skip_existing}")
    on_debug(f"  skip_generate    = {config.skip_generate}")
    on_debug(f"  download_workers = {config.download_workers}")
    on_debug(f"  generate_workers = {config.generate_workers}")
    on_debug(f"  difficulties     = {config.difficulties}")
    on_debug(f"  keep_audio       = {config.keep_audio}")


    # Load persistent caches
    bs_cache = BSCache()
    yt_cache = YTCache()
    overrides = _load_overrides(config.overrides_file)
    if overrides:
        logger.info("Loaded %d YouTube URL override(s)", len(overrides))

    if not check_ffmpeg() and not config.skip_generate:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install ffmpeg to generate maps from audio. "
            "Or use --skip-generate to only download BeatSaver maps."
        )

    results: list[GenerationResult] = []
    beatsaver_count = 0
    generated_count = 0
    skipped_count = 0
    error_count = 0

    total = len(tracks)
    client = httpx.Client()

    # ------------------------------------------------------------------
    # Phase 1: BeatSaver search (skipped in --regen mode)
    # ------------------------------------------------------------------
    unmatched: list[Track] = []
    bs_matches: dict[str, "from smartsaber.models import BeatSaverMatch"] = {}  # type: ignore

    # Build a directory index ONCE — avoids re-scanning the output folder
    # (which may contain hundreds of folders) for every single track.
    _dir_index_t0 = _time.perf_counter()
    _output_index = _build_output_index(output_dir)
    _dir_index_elapsed = _time.perf_counter() - _dir_index_t0
    on_debug(f"Output dir index: {len(_output_index)} folders scanned in {_dir_index_elapsed:.3f}s")

    if config.regen_only:
        # --regen: skip BeatSaver entirely, treat all tracks as unmatched
        # so they go through the generation path.
        unmatched = list(tracks)
        on_debug(f"Phase 1: BeatSaver search SKIPPED (--regen mode), {len(unmatched)} tracks to generate")
    else:
        _phase1_t0 = _time.perf_counter()
        on_progress("beatsaver_search", 0, total)
        _bs_cache_hits = 0
        _bs_cache_misses = 0
        for i, track in enumerate(tracks, 1):
            on_progress("beatsaver_search", i, total)

            # Skip-existing check (look for a folder containing the track)
            if config.skip_existing:
                existing = _find_existing_indexed(track, _output_index)
                if existing is not None:
                    skipped_count += 1
                    result = GenerationResult(
                        track=track,
                        success=True,
                        output_path=str(existing),
                        map_hash=None,
                        was_beatsaver=False,
                    )
                    results.append(result)
                    on_track_complete(track, result)
                    continue

            cached, match = bs_cache.get(track)
            if not cached:
                _bs_cache_misses += 1
                try:
                    match = find_map(
                        track,
                        client=client,
                        title_threshold=config.title_threshold,
                        artist_threshold=config.artist_threshold,
                        min_upvote_ratio=config.min_upvote_ratio,
                        delay=config.beatsaver_delay,
                    )
                except Exception as exc:
                    logger.warning("BeatSaver search error for '%s': %s", track.title, exc)
                    match = None
                if match:
                    bs_cache.put_match(track, match)
                else:
                    bs_cache.put_miss(track)
            else:
                _bs_cache_hits += 1

            if match:
                bs_matches[track.source_id] = match
            else:
                unmatched.append(track)

        _phase1_elapsed = _time.perf_counter() - _phase1_t0
        on_debug(f"Phase 1: BeatSaver search done in {_phase1_elapsed:.2f}s — "
                 f"{_bs_cache_hits} cache hits, {_bs_cache_misses} cache misses, "
                 f"{len(bs_matches)} matches, {len(unmatched)} unmatched, "
                 f"{skipped_count} skipped (existing)")

    # ------------------------------------------------------------------
    # Phase 2: Download BeatSaver maps (skipped in --regen mode)
    # ------------------------------------------------------------------
    if not config.regen_only:
        _phase2_t0 = _time.perf_counter()
        on_progress("beatsaver_download", 0, len(bs_matches))
        for j, (src_id, match) in enumerate(bs_matches.items(), 1):
            on_progress("beatsaver_download", j, len(bs_matches))
            try:
                folder = download_map(match, output_dir, client=client)
                h = compute_map_hash(folder)
                result = GenerationResult(
                    track=match.track,
                    success=True,
                    output_path=str(folder),
                    map_hash=h,
                    was_beatsaver=True,
                )
                beatsaver_count += 1
            except Exception as exc:
                on_error(match.track, exc)
                result = GenerationResult(
                    track=match.track,
                    success=False,
                    output_path=None,
                    map_hash=None,
                    error=str(exc),
                    was_beatsaver=True,
                )
                error_count += 1
            results.append(result)
            on_track_complete(match.track, result)

        _phase2_elapsed = _time.perf_counter() - _phase2_t0
        on_debug(f"Phase 2: BeatSaver download done in {_phase2_elapsed:.2f}s — "
                 f"{beatsaver_count} downloaded, {error_count} errors")

    # ------------------------------------------------------------------
    # Phase 3: Batch confirm for generation
    # ------------------------------------------------------------------
    if config.skip_generate or not unmatched:
        to_generate: list[Track] = []
    elif config.regen_only:
        # --regen: auto-confirm all tracks, no interactive prompt
        to_generate = list(unmatched)
    else:
        to_generate = on_batch_confirm(unmatched)
        if config.max_generate > 0:
            to_generate = to_generate[: config.max_generate]

    on_debug(f"Phase 3: {len(to_generate)} tracks confirmed for generation")

    # ------------------------------------------------------------------
    # Phase 4: Resolve YouTube URLs + interactive review
    #
    # In --regen mode, tracks with existing song.egg in their output
    # folder skip YouTube resolve + download entirely.  Only tracks
    # whose audio is missing go through the normal path.
    # ------------------------------------------------------------------
    resolution_map: dict[str, YouTubeResolution] = {}
    # Pre-resolved audio paths: source_id → Path to existing audio file.
    # Tracks in this dict skip the download executor entirely.
    pre_resolved_audio: dict[str, Path] = {}

    if config.regen_only and to_generate:
        _phase4_t0 = _time.perf_counter()
        # Check which tracks already have song.egg in their output folder
        need_resolve: list[Track] = []
        for track in to_generate:
            existing_folder = _find_existing_indexed(track, _output_index)
            if existing_folder:
                egg = existing_folder / "song.egg"
                if egg.exists() and egg.stat().st_size > 0:
                    pre_resolved_audio[track.source_id] = egg
                    continue
            need_resolve.append(track)

        on_debug(f"Phase 4: --regen mode — {len(pre_resolved_audio)} tracks have existing song.egg, "
                 f"{len(need_resolve)} need YouTube resolve")

        # Only resolve YouTube for tracks without existing audio
        if need_resolve:
            on_progress("youtube_resolve", 0, len(need_resolve))
            for idx, track in enumerate(need_resolve, 1):
                on_progress("youtube_resolve", idx, len(need_resolve))
                res = resolve_url(
                    track,
                    cache=yt_cache,
                    override_url=_find_override(track, overrides),
                )
                resolution_map[track.source_id] = res
        on_progress("youtube_resolve", len(to_generate), len(to_generate))
        _phase4_elapsed = _time.perf_counter() - _phase4_t0
        on_debug(f"Phase 4: YouTube resolve done in {_phase4_elapsed:.2f}s")

    elif to_generate:
        _phase4_t0 = _time.perf_counter()
        on_progress("youtube_resolve", 0, len(to_generate))
        resolutions: list[YouTubeResolution] = []
        for idx, track in enumerate(to_generate, 1):
            on_progress("youtube_resolve", idx, len(to_generate))
            res = resolve_url(
                track,
                cache=yt_cache,
                override_url=_find_override(track, overrides),
            )
            resolutions.append(res)

        # Let the caller (CLI) present results and collect any overrides
        resolutions = on_url_review(resolutions)
        resolution_map = {r.track.source_id: r for r in resolutions}

        # Persist URLs to cache immediately — before any download starts.
        # This way a re-run after an interrupted generation skips the YouTube
        # search entirely, even for tracks that never finished downloading.
        for r in resolutions:
            if r.url and r.source in ("search", "override"):
                yt_cache.put_url(r.track, r.url)

        _phase4_elapsed = _time.perf_counter() - _phase4_t0
        on_debug(f"Phase 4: YouTube resolve done in {_phase4_elapsed:.2f}s — "
                 f"{len(resolution_map)} URLs resolved")

    # ------------------------------------------------------------------
    # Phase 5: Download + Generate (fully overlapped pipeline)
    #
    # dl_executor  — ThreadPoolExecutor  (I/O-bound, many workers)
    # gen_executor — ProcessPoolExecutor (CPU-bound, one per core)
    #
    # Architecture:
    #   • A *feeder thread* iterates as_completed(dl_futures) and submits
    #     gen jobs the instant each download finishes.  It puts completed
    #     gen futures into a thread-safe queue.
    #   • The *main thread* pulls from that queue and records results,
    #     firing progress callbacks immediately.
    #
    # This means downloads, generation, AND result collection all happen
    # concurrently — no stage ever blocks another.
    #
    # FAST PATH (--regen with all audio pre-resolved):
    #   When no downloads are needed, skip the heavy ProcessPoolExecutor
    #   entirely.  Note generation + file writing is trivially fast when
    #   the analysis cache is primed, so running in-process avoids the
    #   overhead of spawning N Python processes (~200MB each).
    # ------------------------------------------------------------------
    import queue as _queue

    on_progress("generate", 0, len(to_generate))
    difficulties_enum = [Difficulty(d) for d in config.difficulties]
    completed_count = 0
    counter_lock = threading.Lock()
    total_to_generate = len(to_generate)

    # Tracks with pre-resolved audio (--regen with existing song.egg) go
    # directly to the gen executor, skipping download entirely.
    tracks_needing_download = [t for t in to_generate if t.source_id not in pre_resolved_audio]

    # --- Fast path: process all pre-resolved tracks in parallel ---
    # These tracks already have audio on disk (song.egg), so no download is
    # needed.  We still use ProcessPoolExecutor because note generation is
    # pure-Python CPU work (GIL-bound).  The analysis cache is primed, so
    # each worker just reads a small JSON file — no heavy librosa import.
    if pre_resolved_audio:
        _fast_t0 = _time.perf_counter()
        _fast_count = len(pre_resolved_audio)
        _fast_workers = min(config.generate_workers, _fast_count)
        on_debug(f"Phase 5a: FAST PATH — {_fast_count} pre-resolved tracks, "
                 f"{_fast_workers} workers (ProcessPoolExecutor)")

        fast_tracks = [t for t in to_generate if t.source_id in pre_resolved_audio]

        with ProcessPoolExecutor(max_workers=_fast_workers) as fast_executor:
            future_to_track = {
                fast_executor.submit(
                    _generate_worker, track, pre_resolved_audio[track.source_id],
                    output_dir, difficulties_enum, config,
                ): track
                for track in fast_tracks
            }
            for future in as_completed(future_to_track):
                track = future_to_track[future]
                try:
                    result = future.result()
                except Exception as exc:
                    on_error(track, exc)
                    result = GenerationResult(
                        track=track, success=False,
                        output_path=None, map_hash=None, error=str(exc),
                    )
                _track_elapsed = _time.perf_counter() - _fast_t0
                on_debug(f"  [{completed_count+1}/{total_to_generate}] "
                         f"{track.artist} – {track.title}: "
                         f"{'OK' if result.success else 'FAIL: ' + str(result.error)} "
                         f"(+{_track_elapsed:.1f}s)")
                results.append(result)
                if result.success:
                    generated_count += 1
                else:
                    error_count += 1
                completed_count += 1
                on_progress("generate", completed_count, total_to_generate)
                on_track_complete(track, result)

        _fast_elapsed = _time.perf_counter() - _fast_t0
        on_debug(f"Phase 5a: Fast path done in {_fast_elapsed:.2f}s "
                 f"({_fast_count} tracks"
                 + (f", {_fast_count / _fast_elapsed:.1f} tracks/sec)" if _fast_elapsed > 0 else ")"))

    # --- Normal path: only for tracks that actually need downloading ---
    if tracks_needing_download:
        _phase5_t0 = _time.perf_counter()
        on_debug(f"Phase 5b: DOWNLOAD PATH — {len(tracks_needing_download)} tracks need download+generate, "
                 f"dl_workers={config.download_workers}, gen_workers={config.generate_workers}")
        # Per-track timing dict (thread-safe writes, main-thread reads)
        _track_timings: dict[str, dict] = {}
        _track_timings_lock = threading.Lock()
        # Queue where finished generation futures are placed for the main
        # thread to collect.  None sentinel signals "feeder is done".
        result_q: _queue.Queue = _queue.Queue()

        def _do_download(track: Track) -> tuple[Track, Optional[Path]]:
            on_track_stage(track, "Downloading audio")
            _dl_t0 = _time.perf_counter()
            audio_path = fetch_audio(
                track,
                output_dir,
                prefer_ogg=True,
                cache=yt_cache,
                resolution=resolution_map.get(track.source_id),
            )
            _dl_elapsed = _time.perf_counter() - _dl_t0
            with _track_timings_lock:
                _track_timings[track.source_id] = {"download": _dl_elapsed}
            on_debug(f"  DL {track.artist} – {track.title}: {_dl_elapsed:.2f}s")
            return track, audio_path

        def _record(track: Track, result: GenerationResult) -> None:
            nonlocal generated_count, error_count, completed_count
            results.append(result)
            on_track_complete(track, result)
            with counter_lock:
                if result.success:
                    generated_count += 1
                else:
                    error_count += 1
                completed_count += 1
                on_progress("generate", completed_count, total_to_generate)
                on_debug(f"  [{completed_count}/{total_to_generate}] "
                         f"{track.artist} – {track.title}: "
                         f"{'OK' if result.success else 'FAIL: ' + str(result.error)}")

        dl_executor = ThreadPoolExecutor(max_workers=config.download_workers)
        gen_executor = ProcessPoolExecutor(max_workers=config.generate_workers)

        dl_futures = {
            dl_executor.submit(_do_download, track): track
            for track in tracks_needing_download
        }

        # Number of gen futures we expect the main thread to drain
        expected_gen = 0
        expected_gen_lock = threading.Lock()

        def _feeder() -> None:
            """Runs in a background thread: feeds gen pool from completed downloads."""
            nonlocal expected_gen

            # Process downloads as they complete
            for dl_future in as_completed(dl_futures):
                try:
                    track, audio_path = dl_future.result()
                except Exception as exc:
                    track = dl_futures[dl_future]
                    on_error(track, exc)
                    # Record download failure directly (no gen job needed)
                    _record(track, GenerationResult(
                        track=track, success=False,
                        output_path=None, map_hash=None, error=str(exc),
                    ))
                    continue

                if audio_path is None:
                    _record(track, GenerationResult(
                        track=track, success=False,
                        output_path=None, map_hash=None, error="Audio download failed",
                    ))
                    continue

                # Update UI: download done, now queued/running generation
                on_track_stage(track, "Generating map")

                # Submit generation — runs in a separate process immediately
                gen_fut = gen_executor.submit(
                    _generate_worker, track, audio_path, output_dir,
                    difficulties_enum, config,
                )
                with expected_gen_lock:
                    expected_gen += 1
                # Attach a done-callback that pushes to the queue the instant
                # the gen future completes — the main thread picks it up.
                gen_fut.add_done_callback(lambda f, t=track: result_q.put((t, f)))

            # Sentinel: tell the main thread no more gen jobs are coming.
            # We still need to wait for all in-flight gen futures to finish,
            # so the main thread keeps draining until it has collected
            # expected_gen results.
            result_q.put(None)

        feeder_thread = threading.Thread(target=_feeder, daemon=True)
        feeder_thread.start()

        try:
            feeder_done = False
            collected = 0
            _dl_track_count = len(tracks_needing_download)
            on_debug(f"  Main loop: waiting for {_dl_track_count} download tracks")
            while True:
                try:
                    item = result_q.get(timeout=5.0)
                except _queue.Empty:
                    # Periodic heartbeat so we can tell if we're stuck
                    with expected_gen_lock:
                        _eg = expected_gen
                    on_debug(f"  Main loop heartbeat: collected={collected}/{_eg} "
                             f"feeder_done={feeder_done} feeder_alive={feeder_thread.is_alive()}")
                    # Safety valve: if feeder is dead and we've collected
                    # everything, break even if we missed the sentinel.
                    if not feeder_thread.is_alive():
                        with expected_gen_lock:
                            if collected >= expected_gen:
                                on_debug("  Main loop: feeder dead, all collected — exiting")
                                break
                    continue

                if item is None:
                    feeder_done = True
                    on_debug(f"  Main loop: feeder sentinel received, expected_gen={expected_gen}, collected={collected}")
                    # Check if all gen results already collected
                    with expected_gen_lock:
                        if collected >= expected_gen:
                            break
                    continue

                track, gen_future = item
                try:
                    result = gen_future.result()
                except Exception as exc:
                    on_error(track, exc)
                    result = GenerationResult(
                        track=track, success=False,
                        output_path=None, map_hash=None, error=str(exc),
                    )
                _record(track, result)
                collected += 1

                if feeder_done:
                    with expected_gen_lock:
                        if collected >= expected_gen:
                            break

        except KeyboardInterrupt:
            dl_executor.shutdown(wait=False, cancel_futures=True)
            gen_executor.shutdown(wait=False, cancel_futures=True)
            raise
        else:
            on_debug("  Joining feeder thread…")
            feeder_thread.join(timeout=10)
            if feeder_thread.is_alive():
                on_debug("  WARNING: feeder thread still alive after 10s!")
            on_debug("  Shutting down download executor…")
            dl_executor.shutdown(wait=True)
            on_debug("  Shutting down generation executor…")
            gen_executor.shutdown(wait=True)
            on_debug("  Executors shut down.")

        _phase5_elapsed = _time.perf_counter() - _phase5_t0
        on_debug(f"Phase 5b: Download path done in {_phase5_elapsed:.2f}s — "
                 f"{generated_count} generated, {error_count} errors")

    client.close()

    # ------------------------------------------------------------------
    # Phase 5: Build playlist
    # ------------------------------------------------------------------
    _phase6_t0 = _time.perf_counter()
    playlist_path: Optional[str] = None
    if playlist_info:
        try:
            plist = build_playlist(
                results,
                playlist_info,
                output_dir,
                playlist_cover_url=playlist_info.cover_url,
            )
            playlist_path = str(plist)
        except Exception as exc:
            logger.warning("Playlist generation failed: %s", exc)
    _phase6_elapsed = _time.perf_counter() - _phase6_t0
    on_debug(f"Phase 6: Playlist build done in {_phase6_elapsed:.2f}s")

    _pipeline_elapsed = _time.perf_counter() - _pipeline_t0
    on_debug(f"Pipeline total: {_pipeline_elapsed:.2f}s — "
             f"{beatsaver_count} beatsaver, {generated_count} generated, "
             f"{skipped_count} skipped, {error_count} errors")

    return PipelineResult(
        total=total,
        beatsaver_matches=beatsaver_count,
        generated=generated_count,
        skipped=skipped_count,
        errors=error_count,
        results=results,
        playlist_path=playlist_path,
    )


# ---------------------------------------------------------------------------
# Per-track worker functions
# ---------------------------------------------------------------------------

# Per-process analysis cache — created lazily in each worker process so it
# doesn't need to be pickled across the process boundary.
_worker_analysis_cache: Optional[AnalysisCache] = None


def _get_worker_cache() -> AnalysisCache:
    """Return a per-process AnalysisCache (created once, reused)."""
    global _worker_analysis_cache
    if _worker_analysis_cache is None:
        _worker_analysis_cache = AnalysisCache()
    return _worker_analysis_cache


def _generate_worker(
    track: Track,
    audio_path: Path,
    output_dir: Path,
    difficulties: list[Difficulty],
    config: SmartSaberConfig,
) -> GenerationResult:
    """Top-level picklable entry point for ProcessPoolExecutor workers.

    Each worker process lazily creates its own AnalysisCache instance (the
    cache object contains a threading.Lock which is not picklable, so we
    can't pass it from the parent process).
    """
    return _generate_from_audio(
        track,
        audio_path,
        output_dir,
        difficulties,
        config,
        analysis_cache=_get_worker_cache(),
    )


def _generate_from_audio(
    track: Track,
    audio_path: Path,
    output_dir: Path,
    difficulties: list[Difficulty],
    config: SmartSaberConfig,
    analysis_cache: Optional[AnalysisCache] = None,
    on_stage: Callable[[str], None] = lambda _: None,
    _debug_callback: Callable[[str], None] = _noop_debug,
) -> GenerationResult:
    """CPU-bound half: analyse audio, generate notes, write map files."""
    _gen_t0 = _time.perf_counter()

    # Stable RNG seed derived from the audio filename (video ID or file stem).
    # This ensures the same track always generates identical notes regardless
    # of its position in the playlist.
    # Special case for --regen: audio_path is song.egg for every track, so
    # we use the track identity instead to get unique seeds.
    seed_input = audio_path.stem
    if seed_input == "song":
        seed_input = f"{track.title}_{track.artist}"
    seed = int.from_bytes(hashlib.md5(seed_input.encode()).digest()[:4], "big")
    rng = random.Random(seed)

    on_stage("Analysing BPM")
    # Cache key: use the audio filename stem (e.g. "yt_dQw4w9WgXcQ").
    # Special case for --regen: audio_path is song.egg inside the map folder,
    # so all tracks would get stem "song" — causing cache collisions.
    # In that case, read the .analysis_key file written during the original
    # generation, which records the original cache key.  Fall back to the
    # folder name if the file doesn't exist (old maps before this feature).
    cache_key = audio_path.stem
    if cache_key == "song":
        key_file = audio_path.parent / ".analysis_key"
        if key_file.exists():
            cache_key = key_file.read_text(encoding="utf-8").strip()
        else:
            cache_key = f"regen_{audio_path.parent.name}"

    _analysis_t0 = _time.perf_counter()
    analysis = analysis_cache.get(cache_key) if analysis_cache else None
    _cache_hit = analysis is not None
    if analysis is None:
        analysis = analyze(audio_path)
        if analysis_cache:
            analysis_cache.put(cache_key, analysis)
    _analysis_elapsed = _time.perf_counter() - _analysis_t0

    map_info = MapInfo(
        song_name=track.title,
        song_sub_name="",
        song_author_name=track.artist,
        bpm=analysis.tempo,
        preview_start_time=min(30.0, analysis.duration_s * 0.3),
        preview_duration=min(10.0, analysis.duration_s * 0.1),
    )

    on_stage("Generating notes")
    _notes_t0 = _time.perf_counter()
    map_diffs = generate_all_difficulties(analysis, difficulties, rng)
    _notes_elapsed = _time.perf_counter() - _notes_t0

    _total_notes = sum(len(d.notes) for d in map_diffs)
    _total_events = sum(len(d.events) for d in map_diffs)

    on_stage("Writing files")
    _write_t0 = _time.perf_counter()
    folder = build_map(
        info=map_info,
        difficulties=map_diffs,
        audio_path=audio_path,
        output_dir=output_dir,
        cover_url=track.album_art_url,
        song_name_for_folder=track.title,
        artist_for_folder=track.artist,
        analysis_cache_key=cache_key,
    )
    _write_elapsed = _time.perf_counter() - _write_t0

    if not config.keep_audio:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    _hash_t0 = _time.perf_counter()
    h = compute_map_hash(folder)
    _hash_elapsed = _time.perf_counter() - _hash_t0

    _gen_elapsed = _time.perf_counter() - _gen_t0
    _debug_callback(
        f"    GEN {track.artist} – {track.title}: "
        f"total={_gen_elapsed:.3f}s  "
        f"analysis={_analysis_elapsed:.3f}s({'HIT' if _cache_hit else 'MISS'})  "
        f"notes={_notes_elapsed:.3f}s({_total_notes}n/{_total_events}ev)  "
        f"write={_write_elapsed:.3f}s  hash={_hash_elapsed:.3f}s  "
        f"bpm={analysis.tempo:.1f}  dur={analysis.duration_s:.1f}s  "
        f"onsets={len(analysis.onset_times)}  beats={len(analysis.beat_times)}"
    )

    return GenerationResult(
        track=track,
        success=True,
        output_path=str(folder),
        map_hash=h,
    )


# ---------------------------------------------------------------------------
# Skip-existing helpers
# ---------------------------------------------------------------------------

def _output_exists(track: Track, output_dir: Path) -> bool:
    return _find_existing(track, output_dir) is not None


def _find_existing(track: Track, output_dir: Path) -> Optional[Path]:
    from smartsaber.utils import safe_filename
    partial = safe_filename(f"{track.title} - {track.artist}")[:30].lower()
    for folder in output_dir.iterdir():
        if folder.is_dir() and partial in folder.name.lower():
            return folder
    return None


def _build_output_index(output_dir: Path) -> list[tuple[str, Path]]:
    """Scan the output directory ONCE and return (lowered_name, path) pairs.

    Callers use this to avoid re-scanning the directory for every track.
    """
    try:
        return [
            (entry.name.lower(), entry)
            for entry in output_dir.iterdir()
            if entry.is_dir()
        ]
    except FileNotFoundError:
        return []


def _find_existing_indexed(
    track: Track,
    index: list[tuple[str, Path]],
) -> Optional[Path]:
    """Like _find_existing but uses a pre-built index instead of scanning."""
    from smartsaber.utils import safe_filename
    partial = safe_filename(f"{track.title} - {track.artist}")[:30].lower()
    for name_lower, folder in index:
        if partial in name_lower:
            return folder
    return None


def _load_overrides(overrides_file: Optional[Path]) -> dict[str, str]:
    """Load a JSON overrides file mapping track title/ID → YouTube URL."""
    if not overrides_file:
        return {}
    try:
        return json.loads(Path(overrides_file).read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load overrides file '%s': %s", overrides_file, exc)
        return {}


def _find_override(track: Track, overrides: dict[str, str]) -> Optional[str]:
    """Look up a YouTube URL override for a track by source_id or title."""
    if not overrides:
        return None
    # Exact source_id match (most specific)
    if track.source_id in overrides:
        return overrides[track.source_id]
    # Case-insensitive title match
    title_lower = track.title.lower()
    for key, url in overrides.items():
        if key.lower() == title_lower:
            return url
    # Normalised title match (strips feat./remaster/etc.)
    title_norm = normalize_string(track.title)
    for key, url in overrides.items():
        if normalize_string(key) == title_norm:
            return url
    return None
