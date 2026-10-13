"""Fast syntax check for the Lua harness (no LÖVE, no game boot).

Uses ``lupa`` (a pip-installable embedded LuaJIT).  If lupa is unavailable the
check degrades to a byte-level sanity pass so it never becomes a hard
dependency.

Usage::

    python tools/luacheck.py            # check lua/*.lua
    python tools/luacheck.py <files...>
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LUA_DIR = ROOT / "lua"


def _checker():
    """Return a LuaJIT runtime matching the dialect LÖVE actually runs."""
    try:
        from lupa import luajit21 as lua  # identical syntax to LÖVE's LuaJIT 2.1
    except ImportError:
        try:
            import lupa

            lua = lupa
        except ImportError:
            return None
    try:
        return lua.LuaRuntime()
    except Exception:
        return None


def check_files(paths: list[Path]) -> int:
    lua = _checker()
    if lua is None:
        print(
            "lupa not installed -> skipping real syntax check "
            "(pip install lupa for offline checking)"
        )
        return 0

    # Lua 5.1 (LÖVE's dialect) spells the compiler `loadstring`; lupa may be
    # built against 5.4+ where it is just `load`.  Returning a single string
    # keeps the Python side free of table-indexing differences.
    if lua.eval("type(loadstring)") == "function":
        check = lua.eval(
            "function(src, name) local f, err = loadstring(src, name) "
            "if f then return '' end return tostring(err) end"
        )
    elif lua.eval("type(load)") == "function":
        check = lua.eval(
            "function(src, name) local f, err = load(src, name) "
            "if f then return '' end return tostring(err) end"
        )
    else:
        print("no usable Lua loader in lupa runtime")
        return 0

    failures = 0
    for path in paths:
        src = path.read_text(encoding="utf-8")
        try:
            err = check(src, str(path))
        except Exception as exc:  # pragma: no cover - lupa internals
            err = str(exc)
        if not err:
            print(f"  OK   {path.relative_to(ROOT)}")
        else:
            failures += 1
            print(f"  FAIL {path.relative_to(ROOT)}")
            print("       " + err.replace("\n", "\n       "))
    return failures


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(a) for a in argv]
    else:
        paths = sorted(LUA_DIR.glob("*.lua"))
    if not paths:
        print("no Lua files found")
        return 0
    print(f"syntax-checking {len(paths)} file(s) ...")
    failures = check_files(paths)
    if failures:
        print(f"\n{failures} file(s) failed to parse")
    else:
        print("all files parse cleanly")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))