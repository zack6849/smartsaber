"""Playlist provider implementations."""

from smartsaber.providers.base import PlaylistProvider, PlaylistInfo
from smartsaber.providers.spotify import SpotifyProvider

__all__ = ["PlaylistProvider", "PlaylistInfo", "SpotifyProvider"]


def get_provider(url: str) -> "PlaylistProvider | None":
    """Return the first provider that claims the URL, or None."""
    for cls in [SpotifyProvider]:
        provider = cls()
        if provider.parse_url(url) is not None:
            return provider
    return None
