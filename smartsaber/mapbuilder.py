"""Assemble Beat Saber map files (Info.dat, difficulty .dat, song.egg, cover.jpg)."""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import httpx

from smartsaber.models import (
    Difficulty,
    MapDifficulty,
    MapInfo,
    Note,
    NoteType,
    Obstacle,
    Event,
)
from smartsaber.utils import sha1_files, safe_filename

logger = logging.getLogger(__name__)

# Beat Saber map format version
_FORMAT_VERSION = "2.2.0"

# Difficulty filename suffixes
_DIFF_FILENAMES: dict[Difficulty, str] = {
    Difficulty.EASY: "EasyStandard.dat",
    Difficulty.NORMAL: "NormalStandard.dat",
    Difficulty.HARD: "HardStandard.dat",
    Difficulty.EXPERT: "ExpertStandard.dat",
    Difficulty.EXPERT_PLUS: "ExpertPlusStandard.dat",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_map(
    info: MapInfo,
    difficulties: list[MapDifficulty],
    audio_path: Path,
    output_dir: Path,
    cover_url: Optional[str] = None,
    song_name_for_folder: str = "",
    artist_for_folder: str = "",
) -> Path:
    """
    Write all map files to a new folder under output_dir.
    Returns the path to the created folder.
    Raises on audio conversion failure.
    """
    # Create folder — stable name so re-runs overwrite instead of duplicating
    folder_label = safe_filename(
        f"SmartSaber_{song_name_for_folder} - {artist_for_folder}"
    ) if song_name_for_folder else "SmartSaber_map"
    dest = output_dir / folder_label
    dest.mkdir(parents=True, exist_ok=True)

    # Convert / copy audio → song.egg (OGG Vorbis)
    egg_path = dest / "song.egg"
    _convert_audio(audio_path, egg_path)

    # Download cover art (fall back to a minimal valid JPEG if unavailable)
    cover_path = dest / "cover.jpg"
    if cover_url:
        _download_cover(cover_url, cover_path)
    else:
        _write_default_cover(cover_path)

    # Write difficulty .dat files
    written_diffs: list[MapDifficulty] = []
    for md in difficulties:
        if not md.notes:
            continue
        dat_path = dest / _DIFF_FILENAMES[md.difficulty]
        dat_path.write_text(
            json.dumps(_build_difficulty_dat(md), separators=(",", ":")),
            encoding="utf-8",
        )
        written_diffs.append(md)

    # Write Info.dat
    info_dat_path = dest / "Info.dat"
    info_dat_path.write_text(
        json.dumps(_build_info_dat(info, written_diffs), indent=2),
        encoding="utf-8",
    )

    return dest


def compute_map_hash(map_folder: Path) -> str:
    """Compute the Beat Saber map hash for an existing map folder."""
    info_path = map_folder / "Info.dat"
    diff_paths = sorted(map_folder.glob("*.dat"))
    diff_paths = [p for p in diff_paths if p.name != "Info.dat"]
    return sha1_files(info_path, *diff_paths)


# ---------------------------------------------------------------------------
# Info.dat builder
# ---------------------------------------------------------------------------

def _build_info_dat(info: MapInfo, difficulties: list[MapDifficulty]) -> dict:
    diff_sets = []
    for md in difficulties:
        d = md.difficulty
        diff_sets.append(
            {
                "_difficulty": d.value,
                "_difficultyRank": d.rank,
                "_beatmapFilename": _DIFF_FILENAMES[d],
                "_noteJumpMovementSpeed": d.njs,
                "_noteJumpStartBeatOffset": 0.0,
                "_customData": {},
            }
        )

    return {
        "_version": _FORMAT_VERSION,
        "_songName": info.song_name,
        "_songSubName": info.song_sub_name,
        "_songAuthorName": info.song_author_name,
        "_levelAuthorName": info.level_author_name,
        "_beatsPerMinute": info.bpm,
        "_songTimeOffset": 0,
        "_shuffle": 0,
        "_shufflePeriod": 0.5,
        "_previewStartTime": info.preview_start_time,
        "_previewDuration": info.preview_duration,
        "_songFilename": info.song_filename,
        "_coverImageFilename": info.cover_image_filename,
        "_environmentName": "DefaultEnvironment",
        "_allDirectionsEnvironmentName": "GlassDesertEnvironment",
        "_customData": {},
        "_difficultyBeatmapSets": [
            {
                "_beatmapCharacteristicName": "Standard",
                "_difficultyBeatmaps": diff_sets,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Difficulty .dat builder
# ---------------------------------------------------------------------------

def _build_difficulty_dat(md: MapDifficulty) -> dict:
    notes_json = []
    for n in md.notes:
        notes_json.append(
            {
                "_time": round(n.time, 4),
                "_lineIndex": n.line_index,
                "_lineLayer": n.line_layer,
                "_type": n.type.value,
                "_cutDirection": n.cut_direction.value,
            }
        )

    obstacles_json = []
    for o in md.obstacles:
        obstacles_json.append(
            {
                "_time": round(o.time, 4),
                "_lineIndex": o.line_index,
                "_type": o.line_layer,
                "_duration": round(o.duration, 4),
                "_width": o.width,
            }
        )

    events_json = []
    for e in md.events:
        events_json.append(
            {
                "_time": round(e.time, 4),
                "_type": e.type,
                "_value": e.value,
            }
        )

    return {
        "_version": _FORMAT_VERSION,
        "_notes": notes_json,
        "_obstacles": obstacles_json,
        "_events": events_json,
    }


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def _convert_audio(src: Path, dest: Path) -> None:
    """
    Convert src to OGG Vorbis and write to dest (song.egg).
    If src is already OGG, just copy it.
    Requires ffmpeg on PATH.
    """
    if src.suffix.lower() in (".ogg", ".oga"):
        shutil.copy2(src, dest)
        return

    import subprocess
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(src),
            "-c:a", "libvorbis",
            "-q:a", "6",
            str(dest),
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg audio conversion failed:\n{result.stderr.decode(errors='replace')}"
        )


# ---------------------------------------------------------------------------
# Cover art
# ---------------------------------------------------------------------------

def _download_cover(url: str, dest: Path, size: int = 256) -> None:
    """Download album art and save as JPEG. Falls back silently on error."""
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        img = img.resize((size, size), Image.LANCZOS)
        img.save(dest, "JPEG", quality=85)
    except Exception as exc:
        logger.warning("Could not download cover art from %s: %s", url, exc)
        _write_default_cover(dest)


def _write_default_cover(dest: Path) -> None:
    """Write a 256×256 dark placeholder JPEG as cover art."""
    from PIL import Image
    img = Image.new("RGB", (256, 256), color=(20, 20, 20))
    img.save(dest, "JPEG", quality=85)
