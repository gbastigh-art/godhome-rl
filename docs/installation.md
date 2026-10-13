# Installation

GodhomeRL needs exactly one thing from you: **your own copy of the Godhome
standalone download**. Everything else is built or downloaded by the setup
command. Nothing is vendored into this repository.

---

## 1. Prerequisites

| Requirement | Version | Notes |
| --- | --- | --- |
| Python | 3.10 or newer | 3.11–3.13 tested |
| PyTorch | 2.0 or newer | CPU build is enough; see [Performance](performance.md) |
| NumPy | 1.24 or newer | |
| Disk | ~1.5 GB | extracted game + self-contained executable |
| RAM | 2 GB per harness | 4 GB recommended for 3 environments |

There is **no** requirement for a GPU, Vulkan, DirectX, or a separate LÖVE
installation. The build step re-uses the LÖVE runtime that ships inside
`Godhome-windows.exe`.

---

## 2. Get the Godhome standalone

Download Godhome from its official source and unzip it somewhere permanent.
You will have a folder like this:

```
godhome-standalone_e1c66/
├── Godhome-windows.exe     <- the LÖVE runtime + the fused game
├── Godhome.zip             <- the game itself (this is what we patch)
├── love.dll, SDL2.dll, ... <- runtime dependencies
└── license.txt
```

`Godhome.zip` is the only file the build strictly requires; `Godhome-windows.exe`
lets us produce a self-contained executable. If you only have the zip, GodhomeRL
falls back to launching LÖVE from a source tree (see below).

> **Do not** modify the originals. GodhomeRL extracts to `build/` and never
> writes to your installation.

---

## 3. Point GodhomeRL at it

Any of these work; the first one found wins.

```console
# a) environment variable (recommended)
#    Windows PowerShell
$env:GODHOME_RL_INSTALL = "C:\Games\godhome-standalone_e1c66"
#    Windows cmd.exe
set GODHOME_RL_INSTALL=C:\Games\godhome-standalone_e1c66
#    Linux / macOS
export GODHOME_RL_INSTALL=~/games/godhome-standalone_e1c66

# b) copy the folder into vendor/
mkdir vendor
cp -r /path/to/godhome-standalone_e1c66 vendor/

# c) a per-invocation flag
python -m godhome_rl build --install /path/to/godhome-standalone_e1c66
```

GodhomeRL searches, in order: `--install`, `$GODHOME_RL_INSTALL`, `vendor/*`,
the repository root and its immediate subdirectories, then the parent directory.

---

## 4. Install and build

```console
python -m godhome_rl setup
```

This:

1. installs the Python dependencies (`pip install -e .`),
2. extracts `Godhome.zip` into `build/game/`,
3. copies the Lua harness from `lua/` into `build/game/rl/`,
4. appends a two-line bootstrap to `build/game/main.lua`,
5. repacks a self-contained `build/runtime/Godhome-rl.exe`.

Steps 2–5 take about two minutes on a cold cache and a few seconds afterwards.
Subsequent runs reuse the cache: only changed Lua files are re-copied.

### Into a virtual environment

```console
python -m godhome_rl setup --venv .venv
# Windows:  .venv\Scripts\activate
# Linux:    source .venv/bin/activate
python -m godhome_rl build
```

---

## 5. Verify

```console
python -m godhome_rl doctor
```

This checks Python, NumPy, PyTorch and Lua availability, locates your install,
rebuilds if needed, then boots one harness and runs a short random-policy
episode. A healthy machine prints:

```
running a 60-step smoke episode ...
[ok]   boot 8.6s, 485 steps/s, obs_dim=890
```

If anything fails, [docs/troubleshooting.md](troubleshooting.md) lists every
diagnostic we have collected.

---

## 6. Optional: offline Lua checks

Installing the `dev` extra gives you a LuaJIT-backed syntax checker, so you can
validate harness changes without booting the game:

```console
pip install -e ".[dev]"
python tools/luacheck.py
# or through the CLI
python -m godhome_rl doctor
```

---

## 7. Launch modes

GodhomeRL can launch the game two ways. Both speak the same protocol.

### Self-contained executable (default)

`build/runtime/Godhome-rl.exe` is the LÖVE stub from `Godhome-windows.exe` with
our patched game appended. It needs no LÖVE installation and is what every
command uses unless it fails to build.

### Source tree

If `Godhome-windows.exe` is missing, GodhomeRL looks for a stock LÖVE 11
`love.exe` and runs `build/game/` as a directory. Override with `LOVE_EXE`:

```console
set LOVE_EXE=C:\Program Files\LOVE\love.exe
```

---

## 8. Uninstalling

```console
# remove everything GodhomeRL generated
rmdir /s /q build        # Windows
rm -rf build             # Linux / macOS
rm -rf runs
pip uninstall godhome-rl
```

Your Godhome installation is untouched throughout.

---

## Next

* [Quickstart](quickstart.md) — your first training run, end to end
* [Architecture](architecture.md) — how the pieces fit
* [Troubleshooting](troubleshooting.md) — when something goes wrong