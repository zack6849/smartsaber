# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest

# Run a single test file
pytest tests/test_generator.py

# Run a single test by name
pytest tests/test_generator.py::test_no_crossovers

# Run the CLI
smartsaber import                          # pick from import/ folder
smartsaber import --from-file playlist.csv
smartsaber import https://open.spotify.com/playlist/...
smartsaber import --regen                  # regenerate notes for existing maps
smartsaber import --debug                  # plain output + per-track timing (no progress bars)
```

## Architecture

### Data flow

```
Provider (Spotify URL / CSV file)
    └─► list[Track]
            └─► pipeline.run()
                    ├─► Phase 1: BeatSaver search (beatsaver.py + bs_cache.py)
                    │       └─► matched: download .zip → map folder
                    ├─► Phase 2-3: YouTube resolve (youtube.py + yt_cache.py)
                    │       └─► interactive URL review via on_url_review callback
                    ├─► Phase 4: Download audio (youtube.fetch_audio)
                    │       ThreadPoolExecutor (I/O-bound)
                    └─► Phase 5: Generate map (analyzer.py → generator.py → mapbuilder.py)
                            ProcessPoolExecutor (CPU-bound, one process per core)
```

### Key files

| File | Role |
|---|---|
| `smartsaber/pipeline.py` | Orchestrator — the only file that touches all other modules. Read this first to understand data flow. |
| `smartsaber/models.py` | All shared dataclasses: `Track`, `AudioAnalysis`, `MapDifficulty`, `Note`, `Difficulty`, etc. |
| `smartsaber/cli.py` | Click commands + Rich progress callbacks. Thin layer over `pipeline.run()`. |
| `smartsaber/config.py` | `SmartSaberConfig` dataclass; loads `.env` → `~/.smartsaber/config.toml` → env vars. |
| `smartsaber/analyzer.py` | librosa-based audio analysis → `AudioAnalysis` (BPM, onsets, beat times, band energy curves). |
| `smartsaber/generator.py` | `AudioAnalysis` → `MapDifficulty` (note placement, parity, flow rules). Core generation logic. |
| `smartsaber/patterns.py` | `FLOW_MAP` and `is_good_flow()` — flow/parity rule definitions used by the generator. |
| `smartsaber/mapbuilder.py` | Writes Beat Saber map folder: `Info.dat`, difficulty `.dat` files, `cover.jpg`, `song.egg`. |
| `smartsaber/beatsaver.py` | BeatSaver API search + map download. |
| `smartsaber/matcher.py` | Fuzzy title/artist matching (rapidfuzz) for BeatSaver results. |
| `smartsaber/youtube.py` | yt-dlp wrapper: `resolve_url()` searches YouTube; `fetch_audio()` downloads + converts. |
| `smartsaber/fileimport.py` | Parses Exportify CSV / JSON into `list[Track]`. |
| `smartsaber/analysis_cache.py` | Disk-backed cache (`~/.smartsaber/analysis_cache/`) — avoids re-running librosa on re-imports. |
| `smartsaber/bs_cache.py` | Disk-backed BeatSaver search cache. |
| `smartsaber/yt_cache.py` | Disk-backed YouTube URL cache. |
| `smartsaber/providers/` | Provider abstraction: `base.py` defines the interface; `spotify.py` implements it. |
| `smartsaber/_tui.py` | Minimal interactive terminal widgets (select, checkbox, text prompt). |

### Pipeline concurrency model

The pipeline uses two overlapped executor pools:
- **`ThreadPoolExecutor`** (default 16 workers) for audio downloads (I/O-bound)
- **`ProcessPoolExecutor`** (default: `os.cpu_count()` workers) for note generation (CPU-bound, releases GIL via NumPy/librosa)

A background *feeder thread* watches download futures and immediately submits generation jobs as each download completes. Results flow back to the main thread via a `queue.Queue`. Per-process `AnalysisCache` instances are created lazily (the cache contains a `threading.Lock` and cannot be pickled across processes).

### Map generation rules (generator.py + patterns.py)

- Left saber (red, `NoteType.LEFT`) uses columns 0–1; right saber (blue, `NoteType.RIGHT`) uses columns 2–3
- No crossovers: left column must always ≤ right column on the same beat
- Soft parity enforcement (~50%): consecutive same-hand notes should alternate arm direction
- Flow map (`patterns.FLOW_MAP`): each cut direction has preferred follow-on directions
- Row distribution: rows 0–1 (waist/chest) ≥ 75% of notes; row 2 (overhead) < 25%
- Doubles (both hands same beat) should share the same cut direction

### Configuration priority

1. `~/.smartsaber/config.toml` (TOML file)
2. `.env` in project root / environment variables
3. CLI flags (highest priority)

Key env vars: `SPOTIPY_CLIENT_ID`, `SPOTIPY_CLIENT_SECRET`, `SPOTIPY_REDIRECT_URI`, `SMARTSABER_OUTPUT_DIR`.

### Caches

All persistent caches live in `~/.smartsaber/`:
- `yt_cache.json` — YouTube URL cache
- `bs_cache/` — BeatSaver search results
- `analysis_cache/` — librosa audio analysis results (keyed by YouTube video ID)
- `.spotify_token_cache` — Spotify OAuth token

The `--regen` flag skips BeatSaver search and YouTube download entirely, re-generating notes for existing maps using cached analysis. If `song.egg` already exists in the map folder, audio analysis is also skipped.
