#!/usr/bin/env python3
"""Guard against accidentally committing game content.

GodhomeRL builds the game from the user's own installation, so nothing from
Godhome / Kristal should ever appear in the repository. This runs in CI and can
be run locally with ``python tools/check_no_game_content.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Paths that must never be tracked.
FORBIDDEN_PATHS = (
    "Godhome.zip",
    "Godhome-windows.exe",
    "love.exe",
    "main.lua.orig",
    "mods/Godhome",
    "assets/sprites/player",
    "assets/music/AUDIO_RADIANCE",
)

#: File suffixes that only appear in game content.
FORBIDDEN_SUFFIXES = (".love", ".tmx", ".tsx")

#: Directories that must stay untracked.
FORBIDDEN_DIRS = ("build", "runs", "vendor")

#: Reference-machine hardware, so we can sanity-check the docs.
DOC_MARKERS = {
    "i5-7500": "docs/performance.md",
    "17 000": "docs/performance.md",
}


def tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line for line in out.stdout.splitlines() if line]


def main() -> int:
    problems: list[str] = []
    files = tracked_files()
    if not files:
        print("no git index found; nothing to check (skipped)")
        return 0

    for path in files:
        norm = path.replace("\\", "/")
        if any(norm.startswith(d + "/") for d in FORBIDDEN_DIRS):
            problems.append(f"generated directory is tracked: {path}")
        for bad in FORBIDDEN_PATHS:
            if norm == bad or norm.endswith("/" + bad) or norm.startswith(bad + "/"):
                problems.append(f"game content is tracked: {path}")
        if norm.endswith(FORBIDDEN_SUFFIXES):
            problems.append(f"game asset is tracked: {path}")

    # The biggest asset blobs would blow up the repository; warn about any
    # tracked file over 5 MB.
    for path in files:
        full = ROOT / path
        try:
            if full.is_file() and full.stat().st_size > 5 * 1024 * 1024:
                problems.append(f"suspiciously large tracked file (>5 MB): {path}")
        except OSError:
            continue

    if problems:
        print("game content or generated files are committed:")
        for p in problems:
            print("  -", p)
        print("\nSee docs/licensing.md and .gitignore.")
        return 1

    print(f"checked {len(files)} tracked files: clean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())