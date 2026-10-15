"""Filesystem layout + build/cache management for the Godhome RL agent.

The agent never runs the shipped ``Godhome-windows.exe``.  It builds its *own*
copy of the game into ``build/`` with a small Lua harness injected, so the
original installation is never touched and rebuilding is a single command.

Layout produced by :func:`ensure_build`::

    build/
      game/                 <- extracted + patched LÖVE source tree (run with love.exe)
      runtime/              <- self-contained directory: Godhome-rl.exe + its DLLs
      manifest.json         <- versions/hashes used for cache invalidation
      ports/                <- unix-domain sockets / port files, one per env
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import struct
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
#  Repository layout
# --------------------------------------------------------------------------- #

ROOT = Path(__file__).resolve().parent.parent

LUA_DIR = ROOT / "lua"
BUILD_DIR = ROOT / "build"
GAME_DIR = BUILD_DIR / "game"
RUNTIME_DIR = BUILD_DIR / "runtime"
MANIFEST = BUILD_DIR / "manifest.json"

#: Files copied out of the standalone folder next to the patched executable.
RUNTIME_DLLS = (
    "love.dll",
    "lua51.dll",
    "mpg123.dll",
    "msvcp120.dll",
    "msvcr120.dll",
    "OpenAL32.dll",
    "SDL2.dll",
)

#: Lua files from ``lua/`` that are copied into the game tree verbatim.
LUA_MODULES = (
    "rl_boot.lua",
    "rl_config.lua",
    "rl_proto.lua",
    "rl_net.lua",
    "rl_obs.lua",
    "rl_battle.lua",
    "rl_main.lua",
)

BOOT_BEGIN = "-- >>> GodhomeRL harness BEGIN >>>"
BOOT_END = "-- <<< GodhomeRL harness END <<<"

#: Anchor line that must exist in ``main.lua`` (sanity check on engine layout).
BOOT_ANCHOR = 'require("src.hotswapper")'

#: Bump whenever the injection/patch logic below changes so existing build dirs
#: get re-patched instead of silently keeping stale output.
PATCH_VERSION = 3


class BuildError(RuntimeError):
    """Raised when the game cannot be located or patched."""


# --------------------------------------------------------------------------- #
#  Locating the user's Godhome installation
# --------------------------------------------------------------------------- #


@dataclass
class GodhomeInstall:
    """Where the vanilla game lives."""

    root: Path
    zip_path: Path | None
    exe_path: Path | None
    love_exe: Path | None

    @property
    def has_zip(self) -> bool:
        return self.zip_path is not None

    @property
    def has_exe(self) -> bool:
        return self.exe_path is not None


def _is_godhome_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            return "main.lua" in names and "src/kristal.lua" in names
    except (zipfile.BadZipFile, OSError):
        return False


def _inspect_dir(root: Path) -> GodhomeInstall | None:
    """Return an install descriptor if ``root`` looks like the standalone."""
    if not root.is_dir():
        return None

    zip_path: Path | None = None
    preferred = root / "Godhome.zip"
    if preferred.is_file() and _is_godhome_zip(preferred):
        zip_path = preferred
    else:
        for candidate in sorted(root.glob("*.zip")):
            if _is_godhome_zip(candidate):
                zip_path = candidate
                break

    exes = sorted(p for p in root.glob("*.exe") if "godhome" in p.name.lower())
    exe_path: Path | None = exes[0] if exes else None
    if exe_path is None:
        love = root / "love.exe"
        exe_path = love if love.is_file() else None

    if zip_path is None and exe_path is None:
        return None
    return GodhomeInstall(
        root=root,
        zip_path=zip_path,
        exe_path=exe_path,
        love_exe=find_love_exe(),
    )


def find_install(explicit: str | os.PathLike[str] | None = None) -> GodhomeInstall:
    """Find the Godhome standalone folder.

    Search order:

    1. ``--install`` / ``GODHOME_RL_INSTALL``
    2. ``vendor/*``
    3. ``./godhome-standalone*`` / ``./Godhome*`` next to the repository
    4. any single folder beside the repository that looks like a Godhome install
    """
    roots: list[Path] = []

    if explicit:
        roots.append(Path(explicit))
    env = os.environ.get("GODHOME_RL_INSTALL")
    if env:
        roots.append(Path(env))

    vendor = ROOT / "vendor"
    if vendor.is_dir():
        roots.extend(sorted(p for p in vendor.iterdir() if p.is_dir()))

    for base in (ROOT, ROOT.parent):
        roots.append(base)
        if base.is_dir():
            for child in sorted(base.iterdir()):
                if child.is_dir() and "godhome" in child.name.lower():
                    roots.append(child)

    seen: set[Path] = set()
    for root in roots:
        try:
            root = root.resolve()
        except OSError:
            continue
        if root in seen:
            continue
        seen.add(root)
        found = _inspect_dir(root)
        if found is not None:
            return found

    raise BuildError(
        "Could not find the Godhome standalone installation.\n"
        "Expected a folder containing Godhome.zip (and Godhome-windows.exe).\n"
        "Do one of:\n"
        "  * set GODHOME_RL_INSTALL=C:\\path\\to\\godhome-standalone_e1c66\n"
        "  * copy that folder into ./vendor/\n"
        "  * pass --install C:\\path\\to\\godhome-standalone_e1c66\n"
        f"(searched: {', '.join(str(r) for r in roots) or '<nothing>'})"
    )


def find_love_exe() -> Path | None:
    """Locate a stock LÖVE 11 ``love.exe``.

    Used for the *source-tree* launch mode.  Not required if the self-contained
    runtime could be built (the fused executable carries its own LÖVE).
    """
    env = os.environ.get("LOVE_EXE") or os.environ.get("GODHOME_RL_LOVE_EXE")
    if env and Path(env).is_file():
        return Path(env)

    candidates = [
        Path("C:/Program Files/LOVE/love.exe"),
        Path("C:/Program Files (x86)/LOVE/love.exe"),
        Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "LOVE" / "love.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "LOVE" / "love.exe",
        Path("/usr/bin/love"),
        Path("/usr/local/bin/love"),
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


# --------------------------------------------------------------------------- #
#  Hashing / manifest
# --------------------------------------------------------------------------- #


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _hash_lua_sources() -> str:
    h = hashlib.sha256()
    for name in LUA_MODULES:
        p = LUA_DIR / name
        h.update(name.encode())
        h.update(p.read_bytes() if p.is_file() else b"<missing>")
    return h.hexdigest()


def _read_manifest() -> dict:
    if not MANIFEST.is_file():
        return {}
    try:
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


# --------------------------------------------------------------------------- #
#  Build steps
# --------------------------------------------------------------------------- #


def _extract_game(install: GodhomeInstall) -> None:
    if not install.zip_path:
        raise BuildError(
            f"No Godhome.zip found in {install.root}.  The standalone download "
            "folder must contain Godhome.zip."
        )
    if GAME_DIR.exists():
        shutil.rmtree(GAME_DIR, ignore_errors=True)
    GAME_DIR.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(install.zip_path) as zf:
        # Reject path traversal (zip-slip) before writing anything.
        for member in zf.namelist():
            target = (GAME_DIR / member).resolve()
            if not str(target).startswith(str(GAME_DIR.resolve())):
                raise BuildError(f"Refusing to extract unsafe zip member: {member!r}")
        zf.extractall(GAME_DIR)


def _copy_lua_modules() -> None:
    target_dir = GAME_DIR / "rl"
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in LUA_MODULES:
        src = LUA_DIR / name
        if not src.is_file():
            raise BuildError(f"Missing Lua harness module: {src}")
        shutil.copyfile(src, target_dir / name)


def _patch_main() -> None:
    """Append the RL harness bootstrap to the very end of ``main.lua``.

    It must come *after* Kristal's own ``function love.run()`` definition so
    that our replacement is the one LÖVE actually calls, and after every engine
    require so every global exists.  The function is idempotent: a previous
    injection is stripped first.
    """
    path = GAME_DIR / "main.lua"
    text = path.read_text(encoding="utf-8")

    # Strip any previous injection, line-based so we can never eat real code.
    lines = text.splitlines()
    kept: list[str] = []
    inside = False
    for line in lines:
        if line.strip() == BOOT_BEGIN:
            inside = True
            continue
        if line.strip() == BOOT_END:
            inside = False
            continue
        if not inside:
            kept.append(line)
    text = "\n".join(kept) + "\n"

    if BOOT_ANCHOR not in text:
        raise BuildError(
            f"Could not find the bootstrap anchor {BOOT_ANCHOR!r} in {path}. "
            "The engine layout changed; update tools/build.py."
        )
    if "function love.run" not in text:
        raise BuildError(
            f"Could not find Kristal's love.run override in {path}; "
            "the harness replaces it and must be required afterwards."
        )

    text = (
        text.rstrip()
        + f"\n\n{BOOT_BEGIN}\n"
        + 'require("rl/rl_boot")\n'
        + f"{BOOT_END}\n"
    )
    path.write_text(text, encoding="utf-8", newline="\n")


def _patch_conf() -> None:
    """Make the window tiny/invisible-friendly and default the config to 30 FPS."""
    path = GAME_DIR / "conf.lua"
    if not path.is_file():
        return
    text = path.read_text(encoding="utf-8")
    if "GODHOME_RL_CONF" in text:
        return
    text = text.replace(
        "function love.conf(t)",
        "-- >>> GodhomeRL harness (injected by tools/build.py) >>>\n"
        "local GODHOME_RL_CONF = os.getenv('GODHOME_RL_CONF')\n"
        "function love.conf(t)\n"
        "    if GODHOME_RL_CONF then\n"
        "        t.window.vsync = 0\n"
        "        t.window.msaa = 0\n"
        "        t.window.depth = 0\n"
        "        t.modules.audio = false\n"
        "        t.modules.sound = false\n"
        "        t.modules.video = false\n"
        "        t.modules.thread = true\n"
        "        return\n"
        "    end",
        1,
    )
    path.write_text(text, encoding="utf-8", newline="\n")


# --------------------------------------------------------------------------- #
#  Self-contained fused executable
# --------------------------------------------------------------------------- #


def _find_fused_zip_offset(data: bytes) -> int | None:
    """Offset at which LÖVE's fused ``.zip`` payload starts.

    LÖVE >= 11.1 writes a trailing ``int64`` offset; older fusers simply
    concatenate.  Handle both.
    """
    size = len(data)
    if size >= 8:
        (trailer,) = struct.unpack("<q", data[-8:])
        if 0 < trailer < size - 8 and data[trailer : trailer + 4] == b"PK\x03\x04":
            return trailer
    idx = data.find(b"PK\x03\x04")
    if idx == -1:
        return None
    # The stub must itself look like a PE image.
    if not data.startswith(b"MZ"):
        return None
    return idx


def _build_runtime(install: GodhomeInstall) -> Path | None:
    """Create ``build/runtime/Godhome-rl.exe`` = LÖVE stub + our patched game.

    Returns the executable path, or ``None`` when it cannot be produced.
    """
    if not install.exe_path or not install.has_zip:
        return None

    exe_bytes = install.exe_path.read_bytes()
    stub_len = _find_fused_zip_offset(exe_bytes)
    if stub_len is None:
        return None

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    stub = exe_bytes[:stub_len]

    out_exe = RUNTIME_DIR / "Godhome-rl.exe"
    # Build to a scratch name and swap in, so a stale running harness can never
    # leave us with a half-written executable.
    tmp_exe = RUNTIME_DIR / "Godhome-rl.exe.tmp"
    with tmp_exe.open("wb") as fh:
        fh.write(stub)
        with tempfile.TemporaryDirectory(dir=BUILD_DIR) as tmp:
            archive = Path(tmp) / "game.zip"
            # ZIP_STORED: the payload is ~4k already-compressed asset files, so
            # deflate costs ~90 s of CPU to save ~20 % of a file nothing reads
            # at runtime.  Storing keeps a full rebuild under a couple of seconds.
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as zf:
                for path in sorted(GAME_DIR.rglob("*")):
                    if path.is_file():
                        zf.write(path, path.relative_to(GAME_DIR).as_posix())
            with archive.open("rb") as src:
                shutil.copyfileobj(src, fh, length=4 << 20)

    if out_exe.exists():
        out_exe.unlink()
    tmp_exe.replace(out_exe)

    for dll in RUNTIME_DLLS:
        src = install.root / dll
        if src.is_file():
            shutil.copyfile(src, RUNTIME_DIR / dll)

    return out_exe


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


def ensure_build(
    install: GodhomeInstall | None = None,
    *,
    force: bool = False,
    with_runtime: bool = True,
    quiet: bool = False,
) -> dict:
    """Build (or reuse) the patched game.  Returns the manifest dict."""

    def log(msg: str) -> None:
        if not quiet:
            print(f"[build] {msg}", flush=True)

    if install is None:
        install = find_install()
    log(f"install: {install.root}")

    manifest = _read_manifest()
    lua_hash = _hash_lua_sources()
    zip_hash = _sha256(install.zip_path) if install.zip_path else None

    need_extract = (
        force
        or not GAME_DIR.is_dir()
        or not (GAME_DIR / "main.lua").is_file()
        or manifest.get("zip_hash") != zip_hash
    )
    need_lua = (
        force
        or manifest.get("lua_hash") != lua_hash
        or manifest.get("patch_version") != PATCH_VERSION
        or not (GAME_DIR / "rl" / "rl_boot.lua").is_file()
    )

    touched = False
    if need_extract:
        log("extracting Godhome.zip (one-off, ~90 s) ...")
        _extract_game(install)
        _patch_conf()
        touched = True
    if need_extract or need_lua:
        _patch_main()
        _copy_lua_modules()
        touched = True
    if not touched:
        log("cached build is up to date")

    runtime_exe = RUNTIME_DIR / "Godhome-rl.exe"
    if with_runtime and (touched or force or not runtime_exe.is_file()):
        built = _build_runtime(install)
        if built is None and not quiet:
            log(
                "warning: could not build the self-contained executable; "
                "falling back to `love.exe` + source tree."
            )
        else:
            log("repacked self-contained runtime")

    love_exe = str(install.love_exe) if install.love_exe else None
    new_manifest = {
        "version": 3,
        "patch_version": PATCH_VERSION,
        "zip_hash": zip_hash,
        "lua_hash": lua_hash,
        "install_root": str(install.root),
        "love_exe": love_exe,
        "runtime_exe": str(runtime_exe) if runtime_exe.is_file() else None,
        "game_dir": str(GAME_DIR),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(new_manifest, indent=2), encoding="utf-8")
    log(f"game dir: {GAME_DIR}")
    if new_manifest["runtime_exe"]:
        log(f"runtime: {runtime_exe}")
    else:
        log(f"runtime: (none) using love.exe = {love_exe}")
    return new_manifest


def load_manifest() -> dict:
    """Return a *fresh* manifest, rebuilding only what actually changed."""
    return ensure_build(quiet=True)


def launcher_command(manifest: dict, headless: bool = True) -> list[str]:
    """Return the command used to launch one environment process."""
    runtime_exe = manifest.get("runtime_exe")
    if runtime_exe and Path(runtime_exe).is_file():
        return [runtime_exe]
    love_exe = manifest.get("love_exe")
    if not love_exe or not Path(love_exe).is_file():
        raise BuildError(
            "No launcher available.  Either build the self-contained runtime "
            "(needs Godhome-windows.exe) or install LÖVE 11.x and point "
            "LOVE_EXE at love.exe."
        )
    return [love_exe, str(GAME_DIR)]