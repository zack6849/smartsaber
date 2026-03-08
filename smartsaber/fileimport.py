"""Import playlist tracks from exported files (Exportify CSV, JSON)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from smartsaber.models import Track


def load_tracks(path: Path) -> list[Track]:
    """
    Load tracks from a file.  Supports:
    - Exportify CSV  (.csv)
    - Simple JSON    (.json) — list of {title, artist, duration_ms?, album?}
    """
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _load_exportify_csv(path)
    if suffix == ".json":
        return _load_json(path)
    raise ValueError(f"Unsupported file type '{suffix}'. Use .csv (Exportify) or .json.")


def count_tracks(path: Path) -> int:
    """Quickly count the number of tracks in a file without fully parsing.

    For CSV: counts non-empty data rows (subtracts the header).
    For JSON: counts items in the top-level list.
    Returns 0 on any error.
    """
    try:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            with path.open(newline="", encoding="utf-8-sig") as f:
                # Count rows with at least one non-empty field, minus the header
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return 0
                return sum(1 for row in reader if any(cell.strip() for cell in row))
        if suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return len(data) if isinstance(data, list) else 0
    except Exception:
        return 0
    return 0


# ---------------------------------------------------------------------------
# Exportify CSV
# ---------------------------------------------------------------------------
# Columns (as of 2024):
#   Spotify ID, Artist Name(s), Track Name, Album Name, Disc Number,
#   Track Number, Track Duration (ms), Added By, Added At, Genres,
#   Record Label, Release Date, ISRC, Track Preview URL, Track URI

def _load_exportify_csv(path: Path) -> list[Track]:
    tracks: list[Track] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            # Normalise column names to lowercase for case-insensitive lookup
            row = {k.strip().lower(): v.strip() for k, v in row.items() if k}

            def _get(*keys: str) -> str:
                for k in keys:
                    v = row.get(k.lower())
                    if v:
                        return v
                return ""

            track_name = _get("Track Name", "Title", "Name")
            artist_raw = _get("Artist Name(s)", "Artist Names", "Artists", "Artist")
            album = _get("Album Name", "Album")
            duration_ms_str = _get(
                "Track Duration (ms)", "Duration (ms)", "Duration", "Length (ms)"
            ) or "0"
            spotify_id = _get("Spotify ID", "ID") or f"file_{i}"

            if not track_name:
                continue

            # Exportify uses "; " or ", " to separate multiple artists
            if ";" in artist_raw:
                artists = [a.strip() for a in artist_raw.split(";") if a.strip()]
            else:
                artists = [a.strip() for a in artist_raw.split(",") if a.strip()]

            try:
                duration_ms = int(float(duration_ms_str))
            except (ValueError, TypeError):
                duration_ms = 0

            tracks.append(Track(
                title=track_name,
                artist=artists[0] if artists else "",
                artists_all=artists,
                album=album,
                duration_ms=duration_ms,
                album_art_url=None,
                source_id=spotify_id,
                source="file",
            ))
    return tracks


# ---------------------------------------------------------------------------
# Simple JSON
# ---------------------------------------------------------------------------
# Expected format: list of objects, e.g.
# [{"title": "...", "artist": "...", "duration_ms": 210000, "album": "..."}]

def _load_json(path: Path) -> list[Track]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("JSON file must be a list of track objects.")

    tracks: list[Track] = []
    for i, item in enumerate(data):
        title = item.get("title") or item.get("name") or ""
        artist = item.get("artist") or item.get("artists", [""])[0] or ""
        artists_all = item.get("artists") or ([artist] if artist else [])
        if isinstance(artists_all, str):
            artists_all = [artists_all]
        album = item.get("album") or ""
        duration_ms = int(item.get("duration_ms") or 0)
        source_id = item.get("id") or f"json_{i}"

        if not title:
            continue

        tracks.append(Track(
            title=title,
            artist=artist,
            artists_all=artists_all,
            album=album,
            duration_ms=duration_ms,
            album_art_url=item.get("album_art_url"),
            source_id=source_id,
            source="file",
        ))
    return tracks
