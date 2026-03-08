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

import json
import logging
import random
import shutil
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

import httpx

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
    # Phase 1: BeatSaver search
    # ------------------------------------------------------------------
    on_progress("beatsaver_search", 0, total)
    unmatched: list[Track] = []
    bs_matches: dict[str, "from smartsaber.models import BeatSaverMatch"] = {}  # type: ignore

    for i, track in enumerate(tracks, 1):
        on_progress("beatsaver_search", i, total)

        # Skip-existing check (look for a folder containing the track)
        if config.skip_existing and _output_exists(track, output_dir):
            skipped_count += 1
            result = GenerationResult(
                track=track,
                success=True,
                output_path=str(_find_existing(track, output_dir)),
                map_hash=None,
                was_beatsaver=False,
            )
            results.append(result)
            on_track_complete(track, result)
            continue

        cached, match = bs_cache.get(track)
        if not cached:
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

        if match:
            bs_matches[track.source_id] = match
        else:
            unmatched.append(track)

    # ------------------------------------------------------------------
    # Phase 2: Download BeatSaver maps
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Phase 3: Batch confirm for generation
    # ------------------------------------------------------------------
    if config.skip_generate or not unmatched:
        to_generate: list[Track] = []
    else:
        to_generate = on_batch_confirm(unmatched)
        if config.max_generate > 0:
            to_generate = to_generate[: config.max_generate]

    # ------------------------------------------------------------------
    # Phase 4: Resolve YouTube URLs + interactive review
    # ------------------------------------------------------------------
    resolution_map: dict[str, YouTubeResolution] = {}
    if to_generate:
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

    # ------------------------------------------------------------------
    # Phase 5: Download + Generate (producer-consumer pipeline)
    #
    # dl_executor  — I/O-bound, many workers: fetches audio in parallel.
    # gen_executor — CPU-bound, fewer workers: analyse + generate + write.
    #
    # As each download completes it is immediately handed to the gen pool,
    # so generation starts on early tracks while later ones are still
    # downloading.
    # ------------------------------------------------------------------
    on_progress("generate", 0, len(to_generate))
    difficulties_enum = [Difficulty(d) for d in config.difficulties]
    completed_count = 0
    counter_lock = threading.Lock()

    def _do_download(i: int, track: Track) -> tuple[int, Track, Optional[Path]]:
        on_track_stage(track, "Downloading audio")
        audio_path = fetch_audio(
            track,
            output_dir,
            prefer_ogg=True,
            cache=yt_cache,
            resolution=resolution_map.get(track.source_id),
        )
        return i, track, audio_path

    def _do_generate(i: int, track: Track, audio_path: Path) -> GenerationResult:
        return _generate_from_audio(
            track,
            audio_path,
            output_dir,
            difficulties_enum,
            random.Random(1337 + i),
            config,
            on_stage=lambda stage: on_track_stage(track, stage),
        )

    dl_executor = ThreadPoolExecutor(max_workers=config.download_workers)
    gen_executor = ThreadPoolExecutor(max_workers=config.generate_workers)

    dl_futures = {
        dl_executor.submit(_do_download, i, track): track
        for i, track in enumerate(to_generate)
    }
    gen_futures: dict = {}

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
            on_progress("generate", completed_count, len(to_generate))

    try:
        # Feed completed downloads straight into the generation pool
        for dl_future in as_completed(dl_futures):
            try:
                i, track, audio_path = dl_future.result()
            except Exception as exc:
                track = dl_futures[dl_future]
                on_error(track, exc)
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

            gen_futures[gen_executor.submit(_do_generate, i, track, audio_path)] = track

        # Collect generation results
        for gen_future in as_completed(gen_futures):
            track = gen_futures[gen_future]
            try:
                result = gen_future.result()
            except Exception as exc:
                on_error(track, exc)
                result = GenerationResult(
                    track=track, success=False,
                    output_path=None, map_hash=None, error=str(exc),
                )
            _record(track, result)

    except KeyboardInterrupt:
        dl_executor.shutdown(wait=False, cancel_futures=True)
        gen_executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        dl_executor.shutdown(wait=True)
        gen_executor.shutdown(wait=True)

    client.close()

    # ------------------------------------------------------------------
    # Phase 5: Build playlist
    # ------------------------------------------------------------------
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

def _generate_from_audio(
    track: Track,
    audio_path: Path,
    output_dir: Path,
    difficulties: list[Difficulty],
    rng: random.Random,
    config: SmartSaberConfig,
    on_stage: Callable[[str], None] = lambda _: None,
) -> GenerationResult:
    """CPU-bound half: analyse audio, generate notes, write map files."""
    on_stage("Analysing BPM")
    analysis = analyze(audio_path)

    map_info = MapInfo(
        song_name=track.title,
        song_sub_name="",
        song_author_name=track.artist,
        bpm=analysis.tempo,
        preview_start_time=min(30.0, analysis.duration_s * 0.3),
        preview_duration=min(10.0, analysis.duration_s * 0.1),
    )

    on_stage("Generating notes")
    map_diffs = generate_all_difficulties(analysis, difficulties, rng)

    on_stage("Writing files")
    folder = build_map(
        info=map_info,
        difficulties=map_diffs,
        audio_path=audio_path,
        output_dir=output_dir,
        cover_url=track.album_art_url,
        song_name_for_folder=track.title,
        artist_for_folder=track.artist,
    )

    if not config.keep_audio:
        try:
            audio_path.unlink(missing_ok=True)
        except Exception:
            pass

    h = compute_map_hash(folder)
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
