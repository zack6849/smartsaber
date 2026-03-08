"""CLI layer — Click + Rich, calls pipeline.run() with UI callbacks."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from smartsaber import __version__
from smartsaber.config import load_config, SmartSaberConfig
from smartsaber.models import GenerationResult, Track
from smartsaber.providers import get_provider
from smartsaber.providers.spotify import SpotifyProvider
from smartsaber.youtube import YouTubeResolution
import smartsaber.pipeline as pipeline_mod

console = Console()
err_console = Console(stderr=True)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(__version__, prog_name="smartsaber")
def main() -> None:
    """SmartSaber — Spotify playlists → Beat Saber custom maps."""
    pass


# ---------------------------------------------------------------------------
# login command
# ---------------------------------------------------------------------------

@main.command("login")
def cmd_login() -> None:
    """Authenticate with Spotify (one-time setup for private playlists).

    \b
    Requires a hosted callback page — see callback.html in the project root.
    Host it anywhere with free HTTPS (GitHub Pages recommended) and register
    that URL in your Spotify app's Redirect URIs settings.
    Set SPOTIPY_REDIRECT_URI in your .env to match.
    """
    import os

    provider = SpotifyProvider()
    redirect_uri = os.getenv("SPOTIPY_REDIRECT_URI", "")

    if not redirect_uri:
        err_console.print(
            "[red]SPOTIPY_REDIRECT_URI is not set.[/red]\n"
            "Set it in your .env to the URL where you've hosted callback.html.\n"
            "Example: SPOTIPY_REDIRECT_URI=https://yourname.github.io/smartsaber/callback.html"
        )
        raise SystemExit(1)

    auth_url = provider.get_auth_url()

    console.print("\n[bold cyan]Step 1[/bold cyan] — Open this URL in your browser:\n")
    console.print(f"  {auth_url}\n")

    console.print(
        "[bold cyan]Step 2[/bold cyan] — Approve access in the browser.\n\n"
        f"Spotify will redirect to your callback page at:\n"
        f"  [cyan]{redirect_uri}[/cyan]\n\n"
        "The page will display a code. Copy it and paste it below.\n"
        "[dim](You can also paste the full redirect URL — either works.)[/dim]\n"
    )

    code_or_url = click.prompt("Paste the code (or full redirect URL)").strip()

    try:
        provider.exchange_code(code_or_url)
        console.print(
            "\n[bold green]✓ Login successful![/bold green] "
            "Token cached at [cyan]~/.smartsaber/.spotify_token_cache[/cyan]\n"
            "You won't need to do this again unless you revoke access."
        )
    except Exception as exc:
        err_console.print(f"\n[red]Login failed:[/red] {exc}")
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# import command (the main one)
# ---------------------------------------------------------------------------

@main.command("import")
@click.argument("url", required=False, default=None)
@click.option(
    "--from-file", "-f",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Import from an Exportify CSV or JSON file instead of fetching from Spotify.",
)
@click.option(
    "--output", "-o",
    default=None,
    type=click.Path(path_type=Path),
    help="Output directory (default: ./output)",
)
@click.option(
    "--title-threshold",
    default=None, type=float,
    help="Minimum fuzzy title match score 0-100 (default 85)",
)
@click.option(
    "--artist-threshold",
    default=None, type=float,
    help="Minimum fuzzy artist match score 0-100 (default 75)",
)
@click.option(
    "--min-score",
    default=None, type=float,
    help="Minimum BeatSaver upvote ratio 0-1 (default 0.6)",
)
@click.option(
    "--difficulties", "-d",
    default=None,
    help="Comma-separated difficulties to generate (default: all 5)",
)
@click.option(
    "--skip-generate",
    is_flag=True, default=False,
    help="Only download BeatSaver maps; skip local generation",
)
@click.option(
    "--skip-existing",
    is_flag=True, default=False,
    help="Skip tracks whose output folder already exists",
)
@click.option(
    "--dry-run",
    is_flag=True, default=False,
    help="Show what would be done without downloading or generating",
)
@click.option(
    "--max-generate",
    default=0, type=int,
    help="Limit number of locally generated maps (0 = unlimited)",
)
@click.option(
    "--keep-audio",
    is_flag=True, default=False,
    help="Keep downloaded audio files after processing",
)
@click.option(
    "--overrides",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help=(
        "JSON file mapping track titles or Spotify IDs to YouTube URLs. "
        "Use this to force a specific video for a track. "
        'Example: {"Get Back": "https://youtube.com/watch?v=..."}'
    ),
)
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Enable verbose logging",
)
def cmd_import(
    url: Optional[str],
    from_file: Optional[Path],
    output: Optional[Path],
    title_threshold: Optional[float],
    artist_threshold: Optional[float],
    min_score: Optional[float],
    difficulties: Optional[str],
    skip_generate: bool,
    skip_existing: bool,
    dry_run: bool,
    max_generate: int,
    keep_audio: bool,
    overrides: Optional[Path],
    verbose: bool,
) -> None:
    """Import a Spotify playlist into Beat Saber custom maps.

    \b
    Either pass a Spotify playlist URL:
      smartsaber import https://open.spotify.com/playlist/...

    \b
    Or import from an Exportify CSV / JSON file (no Spotify API needed):
      smartsaber import --from-file my_playlist.csv
    """
    # --- Logging ---
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- Config ---
    cfg = load_config()
    if output:
        cfg.output_dir = output
    if title_threshold is not None:
        cfg.title_threshold = title_threshold
    if artist_threshold is not None:
        cfg.artist_threshold = artist_threshold
    if min_score is not None:
        cfg.min_upvote_ratio = min_score
    if difficulties:
        cfg.difficulties = [d.strip() for d in difficulties.split(",")]
    cfg.skip_generate = skip_generate
    cfg.skip_existing = skip_existing
    cfg.max_generate = max_generate
    cfg.keep_audio = keep_audio
    if overrides:
        cfg.overrides_file = overrides

    # --- Load tracks ---
    if from_file:
        from smartsaber.fileimport import load_tracks
        from smartsaber.models import PlaylistInfo as _PlaylistInfo
        try:
            tracks = load_tracks(from_file)
        except Exception as exc:
            err_console.print(f"[red]Failed to read file:[/red] {exc}")
            sys.exit(1)
        playlist_info = _PlaylistInfo(
            name=from_file.stem,
            description="",
            cover_url=None,
            track_count=len(tracks),
        )
        console.print(f"[bold]{from_file.name}[/bold] — {len(tracks)} tracks")
    elif url:
        provider = get_provider(url)
        if provider is None:
            err_console.print(f"[red]Unsupported URL:[/red] {url}")
            sys.exit(1)

        playlist_id = provider.parse_url(url)
        if playlist_id is None:
            err_console.print(f"[red]Could not parse playlist ID from URL:[/red] {url}")
            sys.exit(1)

        console.print("[cyan]Fetching playlist info…[/cyan]")
        try:
            playlist_info = provider.get_playlist_info(playlist_id)
            tracks = provider.get_tracks(playlist_id)
        except Exception as exc:
            err_console.print(f"[red]Failed to fetch playlist:[/red] {exc}")
            sys.exit(1)

        console.print(f"[bold]{playlist_info.name}[/bold] — {len(tracks)} tracks")
    else:
        err_console.print(
            "[red]Provide a Spotify URL or use --from-file.[/red]\n"
            "Example: smartsaber import --from-file playlist.csv"
        )
        sys.exit(1)

    if dry_run:
        console.print("\n[yellow]Dry-run mode — no files will be written.[/yellow]\n")
        for t in tracks:
            console.print(f"  • {t.artist} – {t.title}")
        return

    # --- Run pipeline with Rich callbacks ---
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    ) as progress:
        task_id = progress.add_task("Starting…", total=len(tracks))

        def on_progress(stage: str, current: int, total: int) -> None:
            label = {
                "beatsaver_search": "Searching BeatSaver",
                "beatsaver_download": "Downloading BeatSaver maps",
                "youtube_resolve": "Resolving YouTube URLs",
                "generate": "Generating maps",
            }.get(stage, stage)
            progress.update(task_id, description=label, completed=current, total=total or 1)

        def on_batch_confirm(unmatched: list[Track]) -> list[Track]:
            progress.stop()
            _show_unmatched_table(unmatched)
            if not click.confirm(
                f"\nGenerate {len(unmatched)} maps locally from YouTube audio?",
                default=True,
            ):
                console.print("[yellow]Skipping local generation.[/yellow]")
                progress.start()
                return []
            progress.start()
            return unmatched

        def on_url_review(
            resolutions: list[YouTubeResolution],
        ) -> list[YouTubeResolution]:
            import questionary

            progress.stop()
            _show_url_review_table(resolutions)

            choices = [
                questionary.Choice(
                    title=_resolution_choice_label(r),
                    value=i,
                )
                for i, r in enumerate(resolutions)
            ]

            selected = questionary.checkbox(
                "Select tracks to override (↑↓ navigate, space select, enter when done):",
                choices=choices,
            ).ask()  # returns None if the user hits Ctrl+C

            if selected:
                for idx in selected:
                    r = resolutions[idx]
                    new_url = questionary.text(
                        f"YouTube URL for '{r.track.artist} – {r.track.title}':",
                    ).ask()
                    if new_url and new_url.strip():
                        resolutions[idx] = YouTubeResolution(
                            track=r.track,
                            url=new_url.strip(),
                            duration_diff_s=0.0,
                            source="override",
                        )
                        console.print(f"  [green]✓[/green] Override set for '{r.track.title}'")

            progress.start()
            return resolutions

        def on_track_complete(track: Track, result: GenerationResult) -> None:
            if result.success:
                status = "[green]✓[/green]"
                src = "BeatSaver" if result.was_beatsaver else "Generated"
            else:
                status = "[red]✗[/red]"
                src = f"Error: {result.error}"
            console.log(f"{status} {track.artist} – {track.title} [{src}]")

        def on_error(track: Track, exc: Exception) -> None:
            console.log(f"[red]Error[/red] processing '{track.title}': {exc}")

        result = pipeline_mod.run(
            tracks=tracks,
            config=cfg,
            playlist_info=playlist_info,
            on_progress=on_progress,
            on_batch_confirm=on_batch_confirm,
            on_url_review=on_url_review,
            on_track_complete=on_track_complete,
            on_error=on_error,
        )

    # --- Summary ---
    console.print()
    console.print("[bold green]Done![/bold green]")
    console.print(f"  Total tracks   : {result.total}")
    console.print(f"  BeatSaver maps : {result.beatsaver_matches}")
    console.print(f"  Generated      : {result.generated}")
    console.print(f"  Skipped        : {result.skipped}")
    console.print(f"  Errors         : {result.errors}")
    console.print(f"  Output dir     : {cfg.output_dir}")
    if result.playlist_path:
        console.print(f"  Playlist       : {result.playlist_path}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _show_unmatched_table(tracks: list[Track]) -> None:
    table = Table(title="Unmatched tracks (no BeatSaver community map found)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Artist")
    table.add_column("Title")
    table.add_column("Duration", justify="right")
    for i, t in enumerate(tracks, 1):
        mins, secs = divmod(int(t.duration_s), 60)
        table.add_row(str(i), t.artist, t.title, f"{mins}:{secs:02d}")
    console.print(table)


def _show_url_review_table(resolutions: list["YouTubeResolution"]) -> None:
    table = Table(title="Resolved YouTube URLs — review before generation")
    table.add_column("#", style="dim", width=4)
    table.add_column("Artist")
    table.add_column("Title")
    table.add_column("Status", width=10)
    table.add_column("URL")
    for i, r in enumerate(resolutions, 1):
        if r.url is None:
            status = "[red]not found[/red]"
            url_str = "[dim]—[/dim]"
        elif r.source == "override":
            status = "[magenta]override[/magenta]"
            url_str = r.url[:70] + ("…" if len(r.url) > 70 else "")
        elif r.source in ("cache_file", "cache_url"):
            status = "[cyan]cached[/cyan]"
            url_str = r.url[:70] + ("…" if len(r.url) > 70 else "")
        else:
            status = "[green]search[/green]"
            url_str = r.url[:70] + ("…" if len(r.url) > 70 else "")
        table.add_row(str(i), r.track.artist, r.track.title, status, url_str)
    console.print(table)


def _resolution_choice_label(r: "YouTubeResolution") -> str:
    """One-line label shown next to each checkbox entry."""
    tag = {
        "search": "search",
        "cache_file": "cached",
        "cache_url": "cached",
        "override": "override",
        "not_found": "NOT FOUND",
    }.get(r.source, r.source)
    base = f"{r.track.artist} – {r.track.title}  [{tag}]"
    if r.url:
        base += f"  {r.url[:60]}{'…' if len(r.url) > 60 else ''}"
    return base
