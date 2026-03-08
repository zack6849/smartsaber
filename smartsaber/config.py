"""Configuration loading: env vars → .env file → ~/.smartsaber/config.toml."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load .env from current dir (if present)
load_dotenv()

# Sensible default for CPU-bound workers — os.cpu_count() returns logical
# cores (including hyperthreads).  Using all of them works well for the pure-
# Python + librosa workload since NumPy releases the GIL internally.
_DEFAULT_GEN_WORKERS = os.cpu_count() or 4


@dataclass
class SmartSaberConfig:
    # Spotify
    spotify_client_id: str = ""
    spotify_client_secret: str = ""

    # Output
    output_dir: Path = Path("./output")
    keep_audio: bool = True

    # Matching
    title_threshold: float = 85.0
    artist_threshold: float = 75.0
    min_upvote_ratio: float = 0.6

    # Generation
    skip_generate: bool = False
    skip_existing: bool = True
    regen_only: bool = False       # regenerate notes for existing maps (no download)
    max_generate: int = 0          # 0 = unlimited
    difficulties: list[str] = field(
        default_factory=lambda: ["Easy", "Normal", "Hard", "Expert", "ExpertPlus"]
    )

    # Worker pools (download is I/O-bound; generate is CPU-bound)
    download_workers: int = 16      # parallel audio downloads (threads, I/O-bound)
    generate_workers: int = _DEFAULT_GEN_WORKERS  # parallel map generators (processes, CPU-bound)

    # BeatSaver
    beatsaver_delay: float = 0.25  # seconds between requests

    # YouTube overrides: path to a JSON file mapping title/ID → YouTube URL
    overrides_file: Optional[Path] = None

    # Debug mode — disables progress bars and prints detailed timing info
    debug: bool = False


def load_config() -> SmartSaberConfig:
    """Read config from environment variables and optional TOML config file."""
    cfg = SmartSaberConfig()

    # 1) TOML config file (~/.smartsaber/config.toml)
    toml_path = Path.home() / ".smartsaber" / "config.toml"
    if toml_path.exists():
        _load_toml(cfg, toml_path)

    # 2) Environment variables (override TOML)
    cfg.spotify_client_id = os.getenv("SPOTIPY_CLIENT_ID", cfg.spotify_client_id)
    cfg.spotify_client_secret = os.getenv(
        "SPOTIPY_CLIENT_SECRET", cfg.spotify_client_secret
    )

    out_env = os.getenv("SMARTSABER_OUTPUT_DIR")
    if out_env:
        cfg.output_dir = Path(out_env)

    return cfg


def _load_toml(cfg: SmartSaberConfig, path: Path) -> None:
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomllib  # type: ignore[no-redef]
        except ImportError:
            import tomli as tomllib  # type: ignore[no-redef]

    data = tomllib.loads(path.read_text())

    spotify = data.get("spotify", {})
    cfg.spotify_client_id = spotify.get("client_id", cfg.spotify_client_id)
    cfg.spotify_client_secret = spotify.get("client_secret", cfg.spotify_client_secret)

    output = data.get("output", {})
    if "dir" in output:
        cfg.output_dir = Path(output["dir"])
    cfg.keep_audio = output.get("keep_audio", cfg.keep_audio)

    matching = data.get("matching", {})
    cfg.title_threshold = matching.get("title_threshold", cfg.title_threshold)
    cfg.artist_threshold = matching.get("artist_threshold", cfg.artist_threshold)
    cfg.min_upvote_ratio = matching.get("min_upvote_ratio", cfg.min_upvote_ratio)
