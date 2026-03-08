"""Abstract base / protocol for playlist providers."""

from __future__ import annotations

from typing import Protocol, Optional

from smartsaber.models import Track, PlaylistInfo


class PlaylistProvider(Protocol):
    """
    Protocol that every playlist source (Spotify, Apple Music, …) must satisfy.

    Implementations should be stateless — call `parse_url` to check ownership
    of a URL, then the two fetch methods.
    """

    def parse_url(self, url: str) -> Optional[str]:
        """
        Return the playlist ID if this provider owns the URL, else None.
        """
        ...

    def get_playlist_info(self, playlist_id: str) -> PlaylistInfo:
        """Return high-level info about the playlist (name, cover, etc.)."""
        ...

    def get_tracks(self, playlist_id: str) -> list[Track]:
        """Return all tracks in the playlist as generic Track objects."""
        ...
