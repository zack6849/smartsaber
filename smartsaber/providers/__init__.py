"""Playlist provider implementations."""

from __future__ import annotations

from smartsaber.providers.base import PlaylistProvider, PlaylistInfo

__all__ = ["PlaylistProvider", "PlaylistInfo", "SpotifyProvider", "get_provider"]


def get_provider(url: str) -> PlaylistProvider | None:
    """Return the first provider that claims the URL, or None."""
    # Import lazily so spotipy is not loaded until a Spotify URL is actually used.
    from smartsaber.providers.spotify import SpotifyProvider
    for cls in [SpotifyProvider]:
        provider = cls()
        if provider.parse_url(url) is not None:
            return provider
    return None
