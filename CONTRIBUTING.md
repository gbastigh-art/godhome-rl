# Contributing

Thanks for looking at this. The project is small enough that a focused patch
is more useful than a large refactor.

## Ground rules

1. **Never commit game content.** No files from Godhome, Kristal, Hollow Knight
   or Deltarune — not assets, not Lua, not the zip, not the executable. If a
   change would put one of those in the diff, it is the wrong change.
2. **Never commit `build/`, `runs/` or `vendor/`.** They are generated and
   git-ignored. See `.gitignore`.
3. **Keep both sides of the protocol in sync.** `lua/rl_proto.lua` and
   `godhome_rl/env/proto.py` describe the same wire format. Changing one means
   changing the other, and bumping `PATCH_VERSION` in `tools/build.py` if the
   patch logic changes.

## Setting up

```console
git clone <your fork>
cd GodhomeRLagent
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e ".[dev]"

# point at your Godhome standalone folder
set GODHOME_RL_INSTALL=C:\path\to\godhome-standalone_e1c66   # PowerShell on Windows

python -m godhome_rl build
python -m godhome_rl doctor
```

## Before you open a pull request

```console
python tools/luacheck.py          # Lua harness parses under LuaJIT
python -m pytest                  # offline tests (no game install needed)
python -m godhome_rl doctor       # end-to-end smoke test
python -m godhome_rl bench --boss "p1/falseknight" --episodes 2
```

If you touched the trainer:

```console
python -m godhome_rl train --name pr-check --bosses "p1/falseknight" \
    --n-envs 2 --updates 20 --n-steps 64 --no-resume
```

## Style

**Python** — type hints on public APIs, `dataclass(slots=True)` for configs,
docstrings that explain *why* rather than restating the signature. Keep to the
existing shape: modules are small and single-purpose.

**Lua** — two-space indent, `--` comments, locals at declaration. The harness
runs under LuaJIT (Lua 5.1 dialect): **no** `//`, no `goto`, no integer
division. `tools/luacheck.py` enforces this.

**Performance** — the harness runs millions of times. Prefer a lookup table over
a division in a per-frame path, reuse the FFI scratch buffers, and do not
allocate in `Obs.build`. If you add something to the inner loop, measure with
`python -m godhome_rl bench` before and after.

## Good first contributions

* **New observation channels** — a directional threat raster, per-bullet damage
  magnitude, spell state. See [docs/extending.md](docs/extending.md).
* **Per-boss reward tweaks** in `godhome_rl/curriculum.py`.
* **Encounter notes** — which boss taught you something is genuinely useful.
* **Documentation fixes** — especially anything that turned out to be wrong.

## Reporting bugs

Please include the output of:

```console
python -m godhome_rl doctor
python -m godhome_rl bench --boss "<the boss>" --episodes 3
```

plus the last ~20 lines of `runs/<name>/metrics.csv` and, if you used
`--verbose`, the matching `build/ports/harness-*.log`. **Do not attach your
Godhome installation.**

A bug in a specific Godhome encounter is especially welcome: attach the boss id
and the harness trace.

## Licence

By contributing you agree that your contribution is licensed under the MIT
Licence, as stated in [LICENSE](LICENSE).