"""String normalization, hashing helpers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from pathlib import Path


# Parenthetical suffixes to strip before fuzzy matching
_STRIP_PATTERNS = [
    r"\(feat\.?[^)]*\)",
    r"\(ft\.?[^)]*\)",
    r"\(with[^)]*\)",
    r"\(radio edit\)",
    r"\(album version\)",
    r"\(single version\)",
    r"\(mono\)",
    r"\(stereo\)",
    r"\(remaster(?:ed)?(?:\s+\d{4})?\)",
    r"\(\d{4}\s+remaster(?:ed)?\)",
    r"\(deluxe(?:\s+edition)?\)",
    r"\(bonus track\)",
    r"\(explicit\)",
    r"\(clean\)",
    r"\(live[^)]*\)",
    r"\(acoustic[^)]*\)",
    r"\(instrumental[^)]*\)",
    r"\(extended[^)]*\)",
    r"\(remix[^)]*\)",
    r"\[[^\]]*\]",   # strip square-bracket annotations too
]
_STRIP_RE = re.compile("|".join(_STRIP_PATTERNS), re.IGNORECASE)


def normalize_string(s: str) -> str:
    """Lowercase, NFKD-normalize, strip parenthetical annotations, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = _STRIP_RE.sub("", s)
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    # Strip trailing punctuation
    s = s.rstrip(".,;:-")
    return s.strip()


def safe_filename(s: str, max_length: int = 80) -> str:
    """Convert a string into a safe filesystem name."""
    s = re.sub(r'[\\/:*?"<>|]', "", s)
    s = s.strip(". ")
    return s[:max_length]


def sha1_files(*paths: Path) -> str:
    """SHA-1 hash of the concatenated contents of the given files."""
    h = hashlib.sha1()
    for p in paths:
        h.update(p.read_bytes())
    return h.hexdigest().upper()
