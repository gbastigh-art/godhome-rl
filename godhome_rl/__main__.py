"""Command line entry point: ``python -m godhome_rl``."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from godhome_rl.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())