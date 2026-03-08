"""CLI layer — Click + Rich, calls pipeline.run() with UI callbacks."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click

from smartsaber import __version__
from smartsaber.config import load_config, SmartSaberConfig
from smartsaber.models import GenerationResult, Track

# Heavy packages (yt_dlp, librosa, spotipy, numpy, rich) are imported lazily
# inside the commands that need them.  Type annotations only here.
if TYPE_CHECKING:
    from smartsaber.youtube import YouTubeResolution
    import smartsaber.pipeline as pipeline_mod


class _LazyConsole:
    """Defers `rich.Console` import until the first method call."""

    def __init__(self, **kwargs: object) -> None:
        self._kwargs = kwargs
        self._inner: object = None

    def _get(self):  # type: ignore[return]
        if self._inner is None:
            from rich.console import Console
            self._inner = Console(**self._kwargs)
        return self._inner

    def __getattr__(self, name: str):  # type: ignore[return]
        return getattr(self._get(), name)

    def __enter__(self):
        return self._get().__enter__()

    def __exit__(self, *args: object) -> None:
        self._get().__exit__(*args)


console = _LazyConsole()
err_console = _LazyConsole(stderr=True)


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
    from smartsaber.providers.spotify import SpotifyProvider

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
    "--force", "-F",
    is_flag=True, default=False,
    help="Regenerate even if output folder already exists (analysis is still cached)",
)
@click.option(
    "--regen",
    is_flag=True, default=False,
    help="Re-generate notes & lighting for existing maps using cached analysis. "
         "Skips BeatSaver search, audio download, and audio analysis. "
         "Use this after updating the generator to apply changes to existing maps.",
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
    "--no-keep-audio",
    is_flag=True, default=False,
    help="Delete downloaded audio files after processing (default: keep them)",
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
    "--download-workers",
    default=None, type=int,
    help="Parallel audio download threads — I/O bound, can be high (default: 4)",
)
@click.option(
    "--generate-workers", "-w",
    default=None, type=int,
    help="Parallel map generation threads — CPU bound (default: 2)",
)
@click.option(
    "--verbose", "-v",
    is_flag=True, default=False,
    help="Enable verbose logging",
)
@click.option(
    "--debug",
    is_flag=True, default=False,
    help="Disable progress bars and print detailed per-track timing, cache stats, "
         "and pipeline phase durations for troubleshooting performance.",
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
    force: bool,
    regen: bool,
    dry_run: bool,
    max_generate: int,
    no_keep_audio: bool,
    overrides: Optional[Path],
    download_workers: Optional[int],
    generate_workers: Optional[int],
    verbose: bool,
    debug: bool,
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
        level=logging.DEBUG if (verbose or debug) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # --- Config ---
    cfg = load_config()
    if debug:
        cfg.debug = True
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
    if force:
        cfg.skip_existing = False
    if regen:
        cfg.regen_only = True
        cfg.skip_existing = False
    cfg.max_generate = max_generate
    if no_keep_audio:
        cfg.keep_audio = False
    if overrides:
        cfg.overrides_file = overrides
    if download_workers is not None:
        cfg.download_workers = max(1, download_workers)
    if generate_workers is not None:
        cfg.generate_workers = max(1, generate_workers)

    # --- Load tracks ---
    # No args at all — offer a file picker from the import/ folder
    if not from_file and not url:
        from_file = _pick_import_file()
        if from_file is None:
            sys.exit(1)

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
        from smartsaber.providers import get_provider
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

    if dry_run:
        console.print("\n[yellow]Dry-run mode — no files will be written.[/yellow]\n")
        for t in tracks:
            console.print(f"  • {t.artist} – {t.title}")
        return

    # --- Run pipeline with Rich callbacks ---
    import time as _cli_time
    import threading as _threading
    import smartsaber.pipeline as pipeline_mod

    if debug:
        # ---------------------------------------------------------------
        # DEBUG MODE — no Rich progress bars, plain timestamped output
        # ---------------------------------------------------------------
        _t0 = _cli_time.perf_counter()

        def _dbg(msg: str) -> None:
            elapsed = _cli_time.perf_counter() - _t0
            print(f"[DEBUG +{elapsed:7.2f}s] {msg}", flush=True)

        _dbg(f"Debug mode enabled — Rich progress bars disabled")
        _dbg(f"Config: output_dir={cfg.output_dir}")
        _dbg(f"Config: regen_only={cfg.regen_only} skip_existing={cfg.skip_existing}")
        _dbg(f"Config: download_workers={cfg.download_workers} generate_workers={cfg.generate_workers}")
        _dbg(f"Config: difficulties={cfg.difficulties}")
        _dbg(f"Tracks: {len(tracks)}")

        def on_progress_debug(stage: str, current: int, total: int) -> None:
            _dbg(f"Progress: {stage} {current}/{total}")

        def on_batch_confirm_debug(unmatched: list[Track]) -> list[Track]:
            _show_unmatched_table(unmatched)
            if not click.confirm(
                f"\nGenerate {len(unmatched)} maps locally from YouTube audio?",
                default=True,
            ):
                console.print("[yellow]Skipping local generation.[/yellow]")
                return []
            return unmatched

        def on_url_review_debug(
            resolutions: list["YouTubeResolution"],
        ) -> list["YouTubeResolution"]:
            from smartsaber import _tui
            _show_url_review_table(resolutions)
            labels = [_resolution_choice_label(r) for r in resolutions]
            chosen_labels = _tui.checkbox(
                "Select tracks to override (space toggle, enter when done):",
                choices=labels,
            )
            label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
            selected = [label_to_idx[lbl] for lbl in chosen_labels if lbl in label_to_idx]
            if selected:
                for idx in selected:
                    r = resolutions[idx]
                    new_url = _tui.text(
                        f"YouTube URL for '{r.track.artist} \u2013 {r.track.title}'",
                    )
                    if new_url and new_url.strip():
                        resolutions[idx] = YouTubeResolution(
                            track=r.track,
                            url=new_url.strip(),
                            duration_diff_s=0.0,
                            source="override",
                        )
                        print(f"  ✓ Override set for '{r.track.title}'")
            return resolutions

        def on_track_stage_debug(track: Track, stage: str) -> None:
            _dbg(f"Stage: {track.artist} – {track.title} → {stage}")

        def on_track_complete_debug(track: Track, result: GenerationResult) -> None:
            if result.success:
                src = "BeatSaver" if result.was_beatsaver else "Generated"
                _dbg(f"DONE: ✓ {track.artist} – {track.title} [{src}]")
            else:
                _dbg(f"DONE: ✗ {track.artist} – {track.title} [Error: {result.error}]")

        def on_error_debug(track: Track, exc: Exception) -> None:
            _dbg(f"ERROR: {track.artist} – {track.title}: {exc}")

        try:
            result = pipeline_mod.run(
                tracks=tracks,
                config=cfg,
                playlist_info=playlist_info,
                on_progress=on_progress_debug,
                on_batch_confirm=on_batch_confirm_debug,
                on_url_review=on_url_review_debug,
                on_track_complete=on_track_complete_debug,
                on_track_stage=on_track_stage_debug,
                on_error=on_error_debug,
                on_debug=_dbg,
            )
        except KeyboardInterrupt:
            print("\nInterrupted.")
            os._exit(130)

        _total_elapsed = _cli_time.perf_counter() - _t0
        _dbg(f"CLI total elapsed: {_total_elapsed:.2f}s")

    else:
        # ---------------------------------------------------------------
        # NORMAL MODE — Rich progress bars
        # ---------------------------------------------------------------
        from rich.progress import (
            BarColumn, MofNCompleteColumn, Progress,
            SpinnerColumn, TextColumn, TimeElapsedColumn,
        )
        _track_tasks: dict[str, int] = {}   # source_id → Rich task ID
        _track_tasks_lock = _threading.Lock()
        # Show one spinner row per concurrent worker so the UI reflects actual
        # parallelism.  Downloads and generation overlap, so both pools count.
        _MAX_VISIBLE_TASKS = cfg.download_workers + cfg.generate_workers

        try:
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
                    from smartsaber import _tui

                    progress.stop()
                    _show_url_review_table(resolutions)

                    labels = [_resolution_choice_label(r) for r in resolutions]
                    chosen_labels = _tui.checkbox(
                        "Select tracks to override (space toggle, enter when done):",
                        choices=labels,
                    )

                    # Map chosen labels back to indices
                    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
                    selected = [label_to_idx[lbl] for lbl in chosen_labels if lbl in label_to_idx]

                    if selected:
                        for idx in selected:
                            r = resolutions[idx]
                            new_url = _tui.text(
                                f"YouTube URL for '{r.track.artist} \u2013 {r.track.title}'",
                            )
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

                def on_track_stage(track: Track, stage: str) -> None:
                    label = f"  [dim]{track.artist} – {track.title}[/dim]  [cyan]{stage}[/cyan]"
                    with _track_tasks_lock:
                        if track.source_id in _track_tasks:
                            # Already has a spinner row — just update the label
                            progress.update(_track_tasks[track.source_id], description=label)
                        elif len(_track_tasks) < _MAX_VISIBLE_TASKS:
                            # Room for a new spinner row
                            tid = progress.add_task(label, total=None)
                            _track_tasks[track.source_id] = tid
                        # else: too many rows already visible, skip adding a new one

                def on_track_complete(track: Track, result: GenerationResult) -> None:
                    # Hide the per-track spinner row
                    with _track_tasks_lock:
                        tid = _track_tasks.pop(track.source_id, None)
                        if tid is not None:
                            progress.update(tid, visible=False)
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
                    on_track_stage=on_track_stage,
                    on_error=on_error,
                )

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
            os._exit(130)

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
    from rich.table import Table
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
    from rich.table import Table
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


def _pick_import_file() -> Optional[Path]:
    """Scan the ./import folder and let the user pick a file interactively."""
    from smartsaber import _tui
    from smartsaber.fileimport import count_tracks

    import_dir = Path("import")
    if not import_dir.is_dir():
        err_console.print(
            "[red]No import/ folder found and no file or URL provided.[/red]\n"
            "Create an import/ folder and drop your Exportify CSV files in it,\n"
            "or pass a file directly with --from-file."
        )
        return None

    files = sorted(
        p for p in import_dir.iterdir()
        if p.is_file() and p.suffix.lower() in (".csv", ".json")
    )

    if not files:
        err_console.print(
            "[red]No CSV or JSON files found in import/.[/red]\n"
            "Export your playlist at exportify.net, save it to the import/ folder,\n"
            "then run [bold]smartsaber import[/bold] again."
        )
        return None

    # Build display labels with track counts
    labels: list[str] = []
    for p in files:
        n = count_tracks(p)
        label = f"{p.name}  ({n} track{'s' if n != 1 else ''})" if n > 0 else p.name
        labels.append(label)

    chosen = _tui.select(
        "Which playlist would you like to import?",
        choices=labels,
    )
    if chosen is None:
        return None
    # Map the chosen label back to the file path
    idx = labels.index(chosen)
    return files[idx]
