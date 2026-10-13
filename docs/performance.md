# Performance

Everything below was measured on the reference machine used during development:

```
CPU      Intel Core i5-7500, 4 cores / 4 threads @ 3.4 GHz
GPU      Intel HD Graphics 630 (not used for training)
RAM      16 GB
OS       Windows 11, Python 3.13, PyTorch 2.10 CPU, NumPy 2.3
Godhome  0.6.0 / mod v1.1.7 (standalone build)
```

Reproduce any of it with `python -m godhome_rl bench` or by reading the `sps`
column of `runs/<name>/metrics.csv`.

---

## Where the time goes

A single decision step costs:

| Component | Cost | Runs in |
| --- | --- | --- |
| Lua engine update (battle, bullets, collision, particles) | 0.9–1.4 ms | the LÖVE process |
| Observation encoder (bullet scan + threat raster) | 0.05–0.2 ms | the LÖVE process |
| Observation serialisation (3.5 KiB) | ~0.02 ms | the LÖVE process |
| TCP round trip | ~0.05 ms | kernel |
| Python bookkeeping (`frombuffer`, stats, dict) | ~0.05 ms | trainer |
| Policy forward pass | ~1.0 ms | trainer |

The engine dominates, and because each environment is a separate OS process the
trainer and the simulators overlap — that is the whole reason for the process
split.

---

## Single environment

```
python -m godhome_rl bench --boss "p1/falseknight" --episodes 5
```

| Boss | boot | decision steps/s | game frames/s |
| --- | --- | --- | --- |
| False Knight | ~8 s | 460–520 | ~1 250 |
| Grimm | ~8 s | ~310 | ~1 430 |
| Radiance | ~8 s | ~330 | ~1 650 |

`decision steps/s` counts only the frames the policy controls. `game frames/s`
counts every simulated frame, including the auto-confirmed menus, so it is the
honest measure of how much game time the simulator actually produced.

A fight is ~1 800 engine frames of which roughly a quarter are decisions, so a
sustained 1 250 frames/s is about **25× real time** while playing *and* a
human-speed fight takes about a second of wall clock instead of a minute.

---

## Several environments

Three harnesses on three cores, one reserved for the trainer:

| `--n-envs` | `sps` in `metrics.csv` | notes |
| --- | --- | --- |
| 1 | ~460 | baseline |
| 2 | ~380 | already trainer-bound |
| 3 | ~285 | the default; the policy forward pass is batched across envs |
| 4 | *no better* on 4 cores | contention |

In practice the trainer sustains **150–400 decision steps/s** during training:
the rollout loop spends ~1 ms per round on a batched policy forward plus
~5–9 ms stepping the harnesses, and the PPO update costs 150–300 ms per
iteration. That is ≈ **0.5–1.4 M decisions per hour**, or ≈ 10 M per day.

---

## Scaling rules of thumb

* `n_envs = max(1, cores - 2)`. On 4 cores that is 2; on 16, fourteen.
* More environments improves gradient quality more than it improves throughput
  once you are past the point where the trainer is the bottleneck.
* `--n-steps` trades memory for quality, not time. 256 is comfortable.
* With a GPU, `--device cuda` speeds up both inference and the update, but the
  simulators still do not — raise `--n-envs` to give the GPU work.

---

## Boot time

Booting a harness loads every Godhome asset into GPU memory, once per process.

| Stage | Cost |
| --- | --- |
| Extract `Godhome.zip` (first build only) | ~90 s |
| Repack the self-contained executable | ~5 s |
| Per-harness boot at training start | 7–11 s × n_envs |

Because a training run creates environments once and then keeps them alive for
hours, boot cost is amortised to nothing. `VecGodhome` restarts individual
environments on crash, which costs one boot each time — visible in the
`restarts` column of `metrics.csv`.

---

## Build cost

| Action | Cost |
| --- | --- |
| `build` (cached) | < 1 s |
| `build` (Lua changed) | ~6 s — copies 8 files and repacks |
| `build` (Godhome changed) | ~95 s — full re-extract |

The incremental path matters while developing the harness: editing a Lua file
and re-running `build` is a few seconds, not two minutes.

---

## Why the harness drops frames

A boss fight at 30 FPS is roughly:

| Phase | Frames | Share |
| --- | --- | --- |
| bullet patterns (`DEFENDING`) | 300–900 | 20–35 % |
| attack animation + hit reactions | ~120 per turn | ~15 % |
| dialogue text boxes | ~90 per turn | ~20 % |
| arena spawn animation | 15 per turn | ~5 % |
| menu navigation | ~10 per turn | ~2 % |

Only the first row can be influenced by a decision. The harness therefore
auto-confirms the rest **inside a single network step**, so the trainer's
sample budget is spent only on frames where the action matters — while the
reward for the skipped frames (survival, damage) is still accumulated, so the
value estimates are not distorted.

Three timers are also zeroed during auto-confirm (dialogue, arena spawn, intro)
so those cosmetic animations cost one frame instead of ninety. No game state is
skipped: the fight plays out exactly as it would for a player mashing confirm.

---

## Things that made it faster

For the record, these were measured:

| Change | Effect |
| --- | --- |
| Skipping rendering, `love.timer.sleep` and the accumulator loop | 60 FPS → 300+ FPS |
| Dropping the overworld map and neutering `World:update` | ~15 % |
| Auto-confirming menus inside one network step | ~4× more useful samples |
| Replacing `torch.distributions.Categorical` with hand-rolled math | 560 → 950 policy calls/s |
| Bounds-checking every observation write | fixed a heap corruption that crashed ~1 fight in 50 |
| Dispatching actions to all harnesses before reading replies | ~2.5× with 3 environments |
| Caching the observation normaliser's float32 view | ~20 % of the forward pass |
| Moving the normaliser update out of the PPO epoch loop | ~15 % of update time |

`torch.distributions.Categorical` costs ~370 µs per call on CPU for a 36-way
distribution — more than the entire game step. That is why `ActorCritic` does
the softmax/logsumexp/gather by hand.

---

## Benchmarking your own machine

```console
python -m godhome_rl bench --boss "p1/falseknight,p3/grimm,p5/radiance" --episodes 3
python -m godhome_rl train --name bench --bosses "p1/falseknight" --n-envs 3 --updates 30 --n-steps 64
```

The second command prints `sps` and per-environment `wait` every update; a
`wait` above ~20 ms means that environment is spending its time in
auto-confirm rather than in the fight.