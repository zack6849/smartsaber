"""Service-agnostic data models for SmartSaber."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Playlist / Track
# ---------------------------------------------------------------------------


@dataclass
class PlaylistInfo:
    name: str
    description: str
    cover_url: Optional[str]
    track_count: int


@dataclass
class Track:
    """Provider-agnostic track representation."""

    title: str
    artist: str                        # Primary artist display string
    artists_all: list[str]             # All contributing artists
    album: str
    duration_ms: int
    album_art_url: Optional[str]
    source_id: str                     # Opaque ID from the provider
    source: str                        # e.g. "spotify"

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000.0


# ---------------------------------------------------------------------------
# BeatSaver
# ---------------------------------------------------------------------------


@dataclass
class BeatSaverMap:
    id: str                            # BeatSaver map key (e.g. "1a2b3")
    name: str
    artist: str
    bpm: float
    duration_s: float
    upvotes: int
    downvotes: int
    download_url: str
    difficulties: list[str]           # e.g. ["Easy", "Normal", "Hard", "Expert"]
    hash: str                          # SHA1 hash of the map

    @property
    def upvote_ratio(self) -> float:
        total = self.upvotes + self.downvotes
        return self.upvotes / total if total else 0.0


@dataclass
class BeatSaverMatch:
    track: Track
    map: BeatSaverMap
    title_score: float                 # 0-100 rapidfuzz ratio
    artist_score: float
    output_path: Optional[str] = None  # Populated after download


# ---------------------------------------------------------------------------
# Audio analysis
# ---------------------------------------------------------------------------


@dataclass
class AudioAnalysis:
    tempo: float                       # BPM estimated by librosa
    beat_times: list[float]            # Seconds of each detected beat
    onset_times: list[float]           # Merged onset times (perc + harm, quantized)
    rms_curve: list[float]             # Per-frame normalised RMS (0-1)
    rms_times: list[float]             # Timestamps for rms_curve / centroid frames
    segment_times: list[float]         # Structural segment boundaries (seconds)
    duration_s: float
    # Per-stream onset sets (empty list if not yet computed)
    perc_onset_times: list[float] = field(default_factory=list)  # drum / transient onsets
    harm_onset_times: list[float] = field(default_factory=list)  # melodic / harmonic onsets
    # Spectral centroid normalised to [0,1] per song (0=bass, 1=treble/vocals).
    # Same frame timestamps as rms_times.  Empty list if not yet computed.
    spectral_centroid_curve: list[float] = field(default_factory=list)
    # Per-frame band energy ratios — three lists (bass/mid/treble), same timestamps
    # as rms_times.  Each value is the fraction of total energy in that band for
    # that frame (sums to ~1.0).  Empty lists if not yet computed.
    bass_energy_curve: list[float] = field(default_factory=list)
    mid_energy_curve: list[float] = field(default_factory=list)
    treble_energy_curve: list[float] = field(default_factory=list)
    # --- Salience features (v4+) ---
    # Per-onset strength: how strong is each onset in onset_times?
    # Normalised 0-1 within each song.  Same length as onset_times.
    onset_strengths: list[float] = field(default_factory=list)
    # Per-onset metrical weight: how strong is the beat position?
    # Downbeat=1.0, backbeat=0.8, off-beat=0.5, sixteenth=0.25.
    # Same length as onset_times.
    onset_metrical_weights: list[float] = field(default_factory=list)
    # Per-segment mean RMS energy (same length as number of segments,
    # i.e. len(segment_times) - 1).  Used to scale note density by section.
    segment_energies: list[float] = field(default_factory=list)
    # Spectral novelty curve — per-frame cosine distance between consecutive
    # mel spectrogram frames, normalised 0-1.  Same timestamps as rms_times.
    spectral_novelty_curve: list[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Map notes / difficulties
# ---------------------------------------------------------------------------


class CutDirection(int, Enum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    UP_LEFT = 4
    UP_RIGHT = 5
    DOWN_LEFT = 6
    DOWN_RIGHT = 7
    DOT = 8                            # Any direction (dot note)


class NoteType(int, Enum):
    LEFT = 0   # Red saber
    RIGHT = 1  # Blue saber
    BOMB = 3


@dataclass
class Note:
    time: float                        # Beat time
    line_index: int                    # Column 0-3 (left to right)
    line_layer: int                    # Row 0-2 (bottom to top)
    type: NoteType
    cut_direction: CutDirection


@dataclass
class Obstacle:
    time: float
    duration: float
    line_index: int
    line_layer: int
    width: int
    height: int


@dataclass
class Event:
    time: float
    type: int
    value: int


class Difficulty(str, Enum):
    EASY = "Easy"
    NORMAL = "Normal"
    HARD = "Hard"
    EXPERT = "Expert"
    EXPERT_PLUS = "ExpertPlus"

    @property
    def njs(self) -> float:
        return {
            Difficulty.EASY: 10.0,
            Difficulty.NORMAL: 10.0,
            Difficulty.HARD: 12.0,
            Difficulty.EXPERT: 16.0,
            Difficulty.EXPERT_PLUS: 18.0,
        }[self]

    @property
    def rank(self) -> int:
        return {
            Difficulty.EASY: 1,
            Difficulty.NORMAL: 3,
            Difficulty.HARD: 5,
            Difficulty.EXPERT: 7,
            Difficulty.EXPERT_PLUS: 9,
        }[self]


@dataclass
class MapDifficulty:
    difficulty: Difficulty
    notes: list[Note] = field(default_factory=list)
    obstacles: list[Obstacle] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)


@dataclass
class MapInfo:
    """Everything needed to write Info.dat + difficulty .dat files."""

    song_name: str
    song_sub_name: str
    song_author_name: str
    level_author_name: str = "SmartSaber"
    bpm: float = 120.0
    preview_start_time: float = 12.0
    preview_duration: float = 10.0
    difficulties: list[MapDifficulty] = field(default_factory=list)
    cover_image_filename: str = "cover.jpg"
    song_filename: str = "song.egg"


# ---------------------------------------------------------------------------
# Generation results
# ---------------------------------------------------------------------------


@dataclass
class GenerationResult:
    track: Track
    success: bool
    output_path: Optional[str]         # Absolute path to the map folder
    map_hash: Optional[str]
    error: Optional[str] = None
    was_beatsaver: bool = False        # True if downloaded from BeatSaver


@dataclass
class PipelineResult:
    total: int
    beatsaver_matches: int
    generated: int
    skipped: int
    errors: int
    results: list[GenerationResult] = field(default_factory=list)
    playlist_path: Optional[str] = None
