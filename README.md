# SmartSaber

Convert Spotify playlists into playable Beat Saber custom maps — automatically.

SmartSaber looks up each track on [BeatSaver](https://beatsaver.com) first. If a community map exists, it downloads it. For anything that doesn't have a community map, it downloads the audio from YouTube and generates map files from scratch using beat detection and difficulty-aware note placement.

---

## How it works

```
Spotify playlist URL  ─┐
Exportify CSV / JSON  ─┴─▶  Track list
                               │
                               ▼
                       BeatSaver search
                      ╱               ╲
                 found                not found
                   │                     │
            Download .zip          YouTube URL resolve
                   │                     │
                   │              Interactive review ◀── override URLs here
                   │                     │
                   │              Download audio + analyse BPM
                   │                     │
                   │              Generate note patterns per difficulty
                   │                     │
                   └──────────┬──────────┘
                              ▼
                      Beat Saber map folder
                      (.bplist playlist file)
```

---

## System requirements

| Requirement                      | Notes                                                                         |
|----------------------------------|-------------------------------------------------------------------------------|
| **Python 3.10+**                 | 3.11+ recommended (built-in TOML support)                                     |
| **ffmpeg**                       | Required for audio conversion. [ffmpeg.org](https://ffmpeg.org/download.html) |
| **Beat Saber**                   | PC (Steam or Meta). Tested with 1.42.x                                        |
| **Some way to play custom maps** | Required to load custom maps in-game. I used [BSManager](https://github.com/Zagrios/bs-manager)                    |
| **yt-dlp**                       | Installed automatically as a Python dependency                                |

> **Windows users:** WSL (Windows Subsystem for Linux) is recommended. Install ffmpeg inside WSL with `sudo apt install ffmpeg`.

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/yourname/smartsaber.git
cd smartsaber
```

### 2. Create a virtual environment and install

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -e .
```

### 3. Install ffmpeg

**Ubuntu / Debian / WSL:**
```bash
sudo apt install ffmpeg
```

**macOS:**
```bash
brew install ffmpeg
```

**Windows (native):** Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to `PATH`.

---

## Configuration

Copy the example below into a `.env` file in the project root (or set the variables in your shell):

```env
# Required only for live Spotify playlist fetching (not needed for CSV import)
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=https://yourname.github.io/smartsaber/callback.html

# Where to write map folders (default: ./output)
# Point this at your Beat Saber CustomLevels directory:
SMARTSABER_OUTPUT_DIR="/path/to/Beat Saber_Data/CustomLevels"
```

**BSManager users (Windows/WSL):**
```env
SMARTSABER_OUTPUT_DIR="/mnt/c/Users/YourName/BSManager/BSInstances/1.42.1/Beat Saber_Data/CustomLevels"
```

You can also configure matching thresholds and generation options in `~/.smartsaber/config.toml`:

```toml
[matching]
title_threshold = 85.0   # fuzzy match score 0-100 (default 85)
artist_threshold = 75.0  # fuzzy match score 0-100 (default 75)
min_upvote_ratio = 0.6   # BeatSaver map quality filter 0-1 (default 0.6)

[output]
keep_audio = false        # keep downloaded audio files after map generation
```

---

## Usage

### Import from a Spotify playlist URL

Requires Spotify API credentials (see [Spotify setup](#spotify-api-setup)).

```bash
smartsaber import https://open.spotify.com/playlist/YOUR_PLAYLIST_ID
```

### Import from an Exportify CSV (no API needed)

Export your playlist at [exportify.net](https://exportify.net) and drop the file into the `import/` folder, then run with no arguments:

```bash
smartsaber import
```

SmartSaber will show a list of files in `import/` and let you pick one. You can also point directly at any file:

```bash
smartsaber import --from-file my_playlist.csv
```

Supports both **Exportify CSV** and a simple **JSON** format (list of `{title, artist, duration_ms}`).

### Common options

| Option | Description |
|---|---|
| `-o, --output PATH` | Output directory (overrides config / env) |
| `--skip-generate` | Only download BeatSaver maps; skip local generation |
| `--skip-existing` | Skip tracks whose output folder already exists |
| `--dry-run` | Print what would be processed without downloading anything |
| `--max-generate N` | Limit locally generated maps to N (0 = unlimited) |
| `--keep-audio` | Keep downloaded audio files after processing |
| `--difficulties` | Comma-separated list: `Easy,Normal,Hard,Expert,ExpertPlus` |
| `--overrides PATH` | JSON file mapping track title or Spotify ID → YouTube URL |
| `--title-threshold F` | Fuzzy title match threshold 0–100 (default 85) |
| `--artist-threshold F` | Fuzzy artist match threshold 0–100 (default 75) |
| `--min-score F` | Minimum BeatSaver upvote ratio 0–1 (default 0.6) |
| `-v, --verbose` | Enable debug logging |

### Interactive URL review

Before generation starts, SmartSaber shows a table of the resolved YouTube URLs and lets you override any of them:

```
  #  Artist          Title              Status    Δ sec  URL
  1  The Beatles     Get Back           search    2      https://youtube.com/watch?v=...
  2  David Bowie     Heroes             cached         https://youtube.com/watch?v=...
  3  Unknown Band    Rare Track         not found       —

Enter numbers to override (e.g. 1,3) or press Enter to continue:
```

Enter track numbers to paste in a different YouTube URL before downloading.

### Pre-set YouTube URL overrides

For tracks you always want to pull from a specific video, create an overrides JSON file:

```json
{
  "Get Back": "https://www.youtube.com/watch?v=YOUR_VIDEO_ID",
  "spotify:track:ABC123": "https://www.youtube.com/watch?v=ANOTHER_ID"
}
```

Pass it with `--overrides overrides.json`.

---

## Spotify API setup

Only needed if you want to import directly from a Spotify URL (not required for CSV import).

1. Go to [developer.spotify.com/dashboard](https://developer.spotify.com/dashboard) and create an app.
2. Set the **Redirect URI** in the app settings to a URL where you've hosted `callback.html` (GitHub Pages works well).
3. Add your credentials to `.env`:
   ```env
   SPOTIPY_CLIENT_ID=...
   SPOTIPY_CLIENT_SECRET=...
   SPOTIPY_REDIRECT_URI=https://yourname.github.io/smartsaber/callback.html
   ```
4. Run the one-time login:
   ```bash
   smartsaber login
   ```
   Follow the prompts — the token is cached at `~/.smartsaber/.spotify_token_cache` and reused on future runs.

---

## Generated map details

- **5 difficulties** generated: Easy, Normal, Hard, Expert, ExpertPlus
- Notes placed on detected beat onsets using librosa beat tracking
- Hand zones respected: left saber (red) uses columns 0–1, right saber (blue) uses columns 2–3
- Cut directions are chosen to create flowing movement patterns
- 3-second intro buffer — no notes in the first 3 seconds
- Lighting events synced to beats
- Album art fetched from Spotify/YouTube and embedded as `cover.jpg`
- Output is a standard Beat Saber map folder loadable by SongCore

---

## YouTube cache

Resolved YouTube URLs are cached in `~/.smartsaber/yt_cache.json`. On subsequent imports of the same playlist, SmartSaber reuses cached URLs and skips re-searching. If the audio file is still on disk (`--keep-audio`), it skips the download entirely.

---

## Development

```bash
pip install -e ".[dev]"
pytest
```

---

## License

[Polyform Noncommercial 1.0.0](LICENSE) — free for personal, educational, and non-commercial use. Commercial use is not permitted.
