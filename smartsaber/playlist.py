"""Generate Beat Saber .bplist playlist files."""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Optional

import httpx

from smartsaber.models import GenerationResult, PlaylistInfo
from smartsaber.mapbuilder import compute_map_hash

logger = logging.getLogger(__name__)


def build_playlist(
    results: list[GenerationResult],
    playlist_info: PlaylistInfo,
    output_dir: Path,
    playlist_cover_url: Optional[str] = None,
) -> Path:
    """
    Build a .bplist file for all successfully processed maps.
    Returns the path to the created file.
    """
    songs = []
    for result in results:
        if not result.success or not result.output_path:
            continue
        map_folder = Path(result.output_path)
        if not map_folder.exists():
            continue

        # Determine hash
        map_hash = result.map_hash or _safe_hash(map_folder)

        songs.append(
            {
                "hash": map_hash.upper() if map_hash else "",
                "levelid": f"custom_level_{map_hash.upper()}" if map_hash else "",
                "songName": result.track.title,
                "levelAuthorName": "SmartSaber",
                "difficulties": [],  # populated below if we can read Info.dat
            }
        )
        # Try to populate difficulties from Info.dat
        info_dat = map_folder / "Info.dat"
        if info_dat.exists():
            try:
                info = json.loads(info_dat.read_text(encoding="utf-8"))
                diffs = []
                for bset in info.get("_difficultyBeatmapSets", []):
                    char = bset.get("_beatmapCharacteristicName", "Standard")
                    for bmap in bset.get("_difficultyBeatmaps", []):
                        diffs.append(
                            {
                                "characteristic": char,
                                "name": bmap.get("_difficulty", ""),
                            }
                        )
                songs[-1]["difficulties"] = diffs
            except Exception:
                pass

    cover_b64 = ""
    if playlist_cover_url:
        cover_b64 = _fetch_cover_b64(playlist_cover_url)

    playlist = {
        "playlistTitle": playlist_info.name or "SmartSaber Playlist",
        "playlistAuthor": "SmartSaber",
        "playlistDescription": playlist_info.description or "",
        "image": f"data:image/jpeg;base64,{cover_b64}" if cover_b64 else "",
        "songs": songs,
    }

    dest = output_dir / _safe_filename(playlist_info.name or "smartsaber_playlist")
    dest = dest.with_suffix(".bplist")
    dest.write_text(json.dumps(playlist, indent=2), encoding="utf-8")
    return dest


def _safe_hash(map_folder: Path) -> Optional[str]:
    try:
        return compute_map_hash(map_folder)
    except Exception:
        return None


def _fetch_cover_b64(url: str) -> str:
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("ascii")
    except Exception as exc:
        logger.warning("Failed to fetch playlist cover: %s", exc)
        return ""


def _safe_filename(s: str) -> str:
    import re
    s = re.sub(r'[\\/:*?"<>|]', "_", s)
    return s[:80].strip()
