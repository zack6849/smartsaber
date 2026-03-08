"""BeatSaver API search and map download."""

from __future__ import annotations

import time
import zipfile
from pathlib import Path
from typing import Optional

import httpx

from smartsaber.models import Track, BeatSaverMap, BeatSaverMatch
from smartsaber.matcher import find_best_match
from smartsaber.utils import normalize_string, safe_filename


_BASE = "https://api.beatsaver.com"
_HEADERS = {"User-Agent": "SmartSaber/1.0.0"}
_REQUEST_DELAY = 0.25   # seconds between requests
_MAX_RETRIES = 4


def _get(client: httpx.Client, url: str, params: dict | None = None) -> dict:
    """GET with exponential backoff on 429."""
    delay = 1.0
    for attempt in range(_MAX_RETRIES):
        resp = client.get(url, params=params, headers=_HEADERS, timeout=15)
        if resp.status_code == 429:
            time.sleep(delay)
            delay *= 2
            continue
        resp.raise_for_status()
        return resp.json()
    raise RuntimeError(f"BeatSaver API kept returning 429 for {url}")


def _parse_maps(data: dict) -> list[BeatSaverMap]:
    maps = []
    for doc in data.get("docs", []):
        meta = doc.get("metadata", {})
        stats = doc.get("stats", {})
        versions = doc.get("versions", [])
        if not versions:
            continue
        latest = versions[0]
        diffs = [d["difficulty"] for d in latest.get("diffs", [])]
        maps.append(
            BeatSaverMap(
                id=doc["id"],
                name=doc.get("name", ""),
                artist=meta.get("songAuthorName", ""),
                bpm=meta.get("bpm", 0),
                duration_s=meta.get("duration", 0),
                upvotes=stats.get("upvotes", 0),
                downvotes=stats.get("downvotes", 0),
                download_url=latest.get("downloadURL", ""),
                difficulties=diffs,
                hash=latest.get("hash", ""),
            )
        )
    return maps


def search_track(
    client: httpx.Client,
    track: Track,
    delay: float = _REQUEST_DELAY,
) -> list[BeatSaverMap]:
    """Three-tier search for a track on BeatSaver."""
    title_norm = normalize_string(track.title)
    artist_norm = normalize_string(track.artist)
    results: list[BeatSaverMap] = []

    def _search(q: str) -> list[BeatSaverMap]:
        time.sleep(delay)
        try:
            data = _get(client, f"{_BASE}/search/text/0", {"q": q, "pageSize": 10})
            return _parse_maps(data)
        except Exception:
            return []

    # Tier 1: exact quoted search
    results = _search(f'"{title_norm}" "{artist_norm}"')

    # Tier 2: relaxed — just title + artist unquoted
    if not results:
        results = _search(f"{title_norm} {artist_norm}")

    # Tier 3: title only
    if not results:
        results = _search(title_norm)

    return results


def find_map(
    track: Track,
    client: Optional[httpx.Client] = None,
    title_threshold: float = 85.0,
    artist_threshold: float = 75.0,
    min_upvote_ratio: float = 0.6,
    delay: float = _REQUEST_DELAY,
) -> Optional[BeatSaverMatch]:
    """Search BeatSaver and return the best matching map, or None."""
    own_client = client is None
    if own_client:
        client = httpx.Client()
    try:
        candidates = search_track(client, track, delay=delay)
        return find_best_match(
            track,
            candidates,
            title_threshold=title_threshold,
            artist_threshold=artist_threshold,
            min_upvote_ratio=min_upvote_ratio,
        )
    finally:
        if own_client:
            client.close()


def download_map(
    match: BeatSaverMatch,
    output_dir: Path,
    client: Optional[httpx.Client] = None,
) -> Path:
    """Download and extract a BeatSaver map zip. Returns the extraction folder."""
    own_client = client is None
    if own_client:
        client = httpx.Client()
    try:
        folder_name = safe_filename(
            f"BeatSaver_{match.map.id} ({match.track.title} - {match.track.artist})"
        )
        dest = output_dir / folder_name
        dest.mkdir(parents=True, exist_ok=True)

        zip_path = dest / "map.zip"
        resp = client.get(match.map.download_url, headers=_HEADERS, timeout=60, follow_redirects=True)
        resp.raise_for_status()
        zip_path.write_bytes(resp.content)

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(dest)
        zip_path.unlink()

        return dest
    finally:
        if own_client:
            client.close()
