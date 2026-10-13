# Troubleshooting

Every symptom we have seen, what it means, and what to do about it.

---

## Setup

### `Could not find the Godhome standalone installation`

GodhomeRL searched `--install`, `$GODHOME_RL_INSTALL`, `vendor/`, the repository
root and its subdirectories, and the parent directory.

```console
python -m godhome_rl build --install "C:\path\to\godhome-standalone_e1c66"
```

The folder must contain `Godhome.zip`. If you only have an `.exe`, unpack it —
the zip is inside the standalone folder. If you have a `.love` file, rename it
to `.zip`.

### `Refusing to extract unsafe zip member`

A zip-slip check in `tools/build.py` fired. Either the archive is corrupt or it
is not actually Godhome. Do not bypass the check.

### `No launcher available`

The self-contained executable could not be built **and** no LÖVE was found.
Either `Godhome-windows.exe` is missing from your install folder, or LÖVE 11 is
not installed. Fix with:

```console
set LOVE_EXE=C:\Program Files\LOVE\love.exe
```

### `Permission denied: build\runtime\Godhome-rl.exe`

A harness from a previous run is still running and holding the executable.

```console
# Windows
taskkill /IM Godhome-rl.exe /F
# PowerShell
Get-Process Godhome-rl | Stop-Process -Force
```

The build writes to a scratch name and swaps atomically, so this only happens
while a process has the file open.

### The build takes forever

The first `build` unpacks ~4 000 asset files (~90 s) and repacks a ~70 MB
executable (~5 s). Later builds are incremental: only changed Lua files are
re-copied, ~6 s. If you are re-extracting on every run, check that
`build/manifest.json` is writable and that `tools/build.py` has not been edited
so that `PATCH_VERSION` mismatches the stored manifest.

---

## `doctor` failures

### `torch is not installed`

```console
python -m pip install -e .
```

### `harness closed the connection during boot`

The error above it in the same message is the reason. The usual causes:

* `Unable to create OpenGL window` — your LÖVE build needs a GL 2.1 driver. If
  you are on a headless machine, install Mesa (`opengl32.dll` drop-in) or use
  `SDL_VIDEODRIVER=offscreen` with a software GL.
* `LuaSocket is unavailable in this LÖVE build` — LÖVE older than 11.1. Use a
  newer Godhome build or a standalone LÖVE 11.3+.

### `[FAIL] harness: ...` with a Lua error

Read the `GODHOME_RL_DEBUG` section of the diagnostic. It contains the harness's
boot trace and the Lua traceback.

---

## Training

### `sps` is very low (< 100)

1. **Harnesses are restarting.** Check the `restarts` column in `metrics.csv`.
   Non-zero and climbing means crashes — see the next section.
2. **The policy forward pass is the bottleneck.** Confirm with `wait=[...]`:
   values near 1–3 ms are healthy; values in the hundreds mean the harnesses are
   busy, not the trainer.
3. **Too many environments.** On 4 cores use `--n-envs 2` or `3`.

### `restarts` climbs during training

The harness died or wedged. GodhomeRL restarts it automatically and prints

```
[vec_env] restarted environment 2: harness error: <lua message>
```

* **A specific Lua error** — narrow it down:
  `python -m godhome_rl bench --boss "<the boss>" --episodes 10`. If one
  encounter is responsible, remove it from the curriculum stage and note it in
  an issue.
* **No message, just `restarted environment N: ConnectionError`** — the process
  faulted inside LÖVE. Add `--verbose` to get the breadcrumb log.
* **A few restarts over hours is normal.** Tens per hour is not.

### `entropy` stays at 3.58

The policy is not moving off uniform. In order of likelihood:

1. `target_kl` is stopping the update every iteration — raise `--n-steps` so the
   rollout is less noisy, or lower `--lr`.
2. The reward is flat — e.g. every episode ends the same way immediately. Check
   `episode_length` in the CSV; if it is tiny, the stall breaker in
   `VecGodhome` will log restarts.
3. Learning rate too high: `--lr 1e-4`.

### `explained_variance` stays negative

The critic cannot fit the returns. Usually the rewards are dominated by one
terminal event. Try:

* raise `--n-epochs` to 6,
* lower `--gamma` to `0.99`,
* shrink the terminal bonuses (`RewardWeights.win`, `.death`) so the dense
  terms carry more of the signal.

### Training got much slower partway through

Almost always a curriculum promotion: the new stage has harder bosses, so
episodes last longer and `sps` (decision steps per second) drops because
auto-confirm frames are not counted. `fps` in the CSV is the honest measure of
simulated throughput.

### The run never promotes out of a stage

Promotion needs **both** a rolling win rate over the last 100 episodes and a
minimum episode count. If you are evaluating far too often you will see
`episodes` climb but never reach the win rate. More training resolves it;
if you want to skip a stage, train that boss list directly.

---

## Playback

### The window is black / nothing renders

* The game needs an OpenGL context. Check that your graphics driver provides
  GL 2.1.
* `settings.json` may have a corrupted `windowScale`. GodhomeRL writes it at
  boot; deleting the file makes the harness rewrite it.
* `SDL_VIDEODRIVER` is forced to `dummy` for audio only; the harness does not
  force a video driver. If you exported `SDL_VIDEODRIVER=dummy` yourself,
  unset it for `play`.

### Playback is choppy

The harness paces itself to wall-clock time, so it drops frames rather than
slowing the simulation. `--render-every 2` halves the rendering load. The
simulation itself always runs at a fixed 1/30 s step.

### `play` disagrees with `eval`

`play` is greedy by default (`argmax`) and `eval` is greedy too; the difference
should be the RNG. For a fair comparison use the same `--seed`, and remember
that a handful of episodes is not a measurement.

---

## Lua harness development

### `python tools/luacheck.py` reports a syntax error

The checker uses lupa's LuaJIT, so it accepts exactly the dialect LÖVE runs —
including rejecting `//` (a Lua 5.3 operator that LuaJIT does not have). This
catches the most common harness mistake.

If lupa is not installed the checker is a no-op; `pip install -e ".[dev]"` to
get it.

### My Lua change had no effect

Rebuild — the game tree is a copy:

```console
python -m godhome_rl build
```

`build` is incremental and takes a few seconds when only Lua changed. If it
says `cached build is up to date` when it should not, delete
`build/manifest.json` and rebuild.

### A harness change makes the game crash on startup

Run with `GODHOME_RL_DEBUG` pointed at a file; `rl_boot` writes a trace through
boot and traps errors:

```console
$env:GODHOME_RL_DEBUG = "C:\temp\rl.log"
python -m godhome_rl doctor
Get-Content C:\temp\rl.log
```

The harness also sends errors to the trainer over the socket, so
`--verbose` during training shows the same message in the terminal.

### Impossible Lua errors from inside the engine

Errors like `attempt to index global '('`, `attempt to perform arithmetic on
local 'dt' (a nil value)` for a value the harness always passes, or a call to a
method that does not exist (`setVolum0`) mean **memory corruption**, not a logic
bug. Historically this was an observation-buffer overflow in `rl_obs.lua`.

The harness now bounds-checks every write into the observation FFI buffer and
allocates slack past the end of the vector, so this class of failure should not
recur. If you see it anyway:

1. run `python tools/luacheck.py` — a syntax error is the usual cause of a
   partially-executed module;
2. try `set GODHOME_RL_JIT=off` to rule the LuaJIT compiler in or out;
3. re-run with `--verbose` and attach the `build/ports/harness-*.log`.

---

## Performance

### `fps` in the CSV looks wrong

`fps` is computed from a per-frame delta of the harness's cumulative frame
counter and resets at episode boundaries. During a restart it can dip. It is a
directional metric, not a benchmark — use `bench` for numbers.

### The trainer is faster than the simulators

That is the goal: if `wait=[...]` is ~1 ms per environment, the LÖVE processes
are keeping up. Raise `--n-envs` until `wait` starts to grow, then stop.

---

## Getting help

Include, from a run that misbehaves:

```console
python -m godhome_rl doctor
python -m godhome_rl bench --boss "<the boss>" --episodes 3
```

plus the last ~20 lines of `runs/<name>/metrics.csv` and, if `--verbose` was
used, the corresponding `build/ports/harness-*.log`. Please **not** attach your
Godhome installation or anything under `build/`.