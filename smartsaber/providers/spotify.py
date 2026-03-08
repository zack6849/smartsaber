"""Spotify PlaylistProvider implementation using spotipy."""

from __future__ import annotations

import re
import time
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOAuth

from smartsaber.models import Track, PlaylistInfo


_PLAYLIST_RE = re.compile(
    r"(?:https?://)?open\.spotify\.com/playlist/([A-Za-z0-9]+)"
)

# Scopes needed to read the user's private playlists
_SCOPES = ["playlist-read-private", "playlist-read-collaborative"]
_DEFAULT_REDIRECT_URI = "https://127.0.0.1:8888/callback"


class SpotifyProvider:
    """
    Fetch tracks from a Spotify playlist (public or private).

    Uses OAuth Authorization Code flow.  Tokens are cached in
    ~/.smartsaber/.spotify_token_cache and silently reused on subsequent runs.

    First-time auth:
    - If a local HTTP server can be started on the redirect URI port, the
      browser callback is caught automatically.
    - Otherwise (or when the redirect URI is HTTPS), falls back to asking the
      user to paste the full redirect URL from their browser's address bar.
    """

    def __init__(self, client_id: str = "", client_secret: str = "") -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._sp: Optional[spotipy.Spotify] = None

    # ------------------------------------------------------------------
    # PlaylistProvider protocol
    # ------------------------------------------------------------------

    def parse_url(self, url: str) -> Optional[str]:
        m = _PLAYLIST_RE.search(url)
        return m.group(1) if m else None

    def get_playlist_info(self, playlist_id: str) -> PlaylistInfo:
        sp = self._client()
        data = sp.playlist(playlist_id, fields="name,description,images,tracks.total")
        images = data.get("images") or []
        cover = images[0]["url"] if images else None
        return PlaylistInfo(
            name=data.get("name", ""),
            description=data.get("description", ""),
            cover_url=cover,
            track_count=data["tracks"]["total"],
        )

    def get_tracks(self, playlist_id: str) -> list[Track]:
        sp = self._client()
        tracks: list[Track] = []
        offset = 0
        limit = 100

        while True:
            resp = sp.playlist_tracks(
                playlist_id,
                fields=(
                    "items(track(id,name,artists,album(name,images),duration_ms)),"
                    "next"
                ),
                limit=limit,
                offset=offset,
            )
            items = resp.get("items") or []
            for item in items:
                t = item.get("track")
                if not t or not t.get("id"):
                    continue  # skip local / unavailable tracks
                artists = [a["name"] for a in (t.get("artists") or [])]
                album_images = (t.get("album") or {}).get("images") or []
                art_url = album_images[0]["url"] if album_images else None
                tracks.append(
                    Track(
                        title=t["name"],
                        artist=artists[0] if artists else "",
                        artists_all=artists,
                        album=(t.get("album") or {}).get("name", ""),
                        duration_ms=t.get("duration_ms", 0),
                        album_art_url=art_url,
                        source_id=t["id"],
                        source="spotify",
                    )
                )
            if not resp.get("next"):
                break
            offset += limit
            time.sleep(0.1)

        return tracks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _client(self) -> spotipy.Spotify:
        if self._sp is None:
            self._sp = spotipy.Spotify(auth_manager=self._make_auth())
        return self._sp

    def _make_auth(self) -> SpotifyOAuth:
        import os
        from smartsaber.config import load_config
        from pathlib import Path

        cfg = load_config()
        client_id = self._client_id or cfg.spotify_client_id
        client_secret = self._client_secret or cfg.spotify_client_secret

        # Allow overriding redirect URI via env var so the user can register
        # whatever URI Spotify's dashboard accepts for them
        redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", _DEFAULT_REDIRECT_URI)

        cache_dir = Path.home() / ".smartsaber"
        cache_dir.mkdir(exist_ok=True)
        cache_path = str(cache_dir / ".spotify_token_cache")

        return SpotifyOAuth(
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            scope=_SCOPES,
            cache_path=cache_path,
            open_browser=False,  # we handle browser + paste flow ourselves in CLI
        )

    def get_auth_url(self) -> str:
        """Return the Spotify authorization URL for the login flow."""
        return self._make_auth().get_authorize_url()

    def exchange_code(self, redirected_url: str) -> None:
        """Exchange the code from a redirected URL and cache the token."""
        auth = self._make_auth()
        code = auth.parse_response_code(redirected_url)
        auth.get_access_token(code, as_dict=False, check_cache=False)
        # Now build the client using the cached token
        self._sp = spotipy.Spotify(auth_manager=auth)
