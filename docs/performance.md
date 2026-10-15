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
| False Knight | 12.0 s | 310 | 751 |
| Grimm | 8.0 s | 310 | 1 427 |

`decision steps/s` counts only the frames the policy controls. `game frames/s`
counts every simulated frame, including the auto-confirmed menus, so it is the
honest measure of how much game time the simulator actually produced.

A fight is ~1 800 engine frames of which roughly a quarter are decisions, so a
sustained 750–1 400 frames/s is about **25× real time** while playing — a fight
that takes a human 60 seconds costs about 2–3 seconds of wall clock.

---

## Several environments

Measured by running `--bosses p1/falseknight --updates 120` at each setting and
reading `sps` off update 119:

| `--n-envs` | `sps` | vs 1 env |
| --- | --- | --- |
| 1 | 135 | 1.0x |
| 3 | 231 | 1.7x |
| 5 | 290 | 2.1x |

Scaling is **strongly sublinear**, and that is the single most useful fact on
this page. Five environments do not give you five times the throughput; they
give you a little over twice.

The reason is that the trainer is **latency-bound, not compute-bound**. Each
decision step is a TCP round trip — write 4 bytes, the harness runs one
`love.update`, ship back 3.5 KiB of observation — and the trainer spends most of
its time blocked on a socket rather than computing. You can see it directly:
during a 3-environment run the whole job occupies only about **21 % of four
cores**, leaving roughly three cores idle. Adding environments fills that idle
time, but every new one also adds more round trips for the single-threaded
trainer to absorb.

Practical consequence: `--n-envs` buys you **gradient quality** (less variance
per update, more diverse experience) far more than it buys raw speed.

### Memory is the real ceiling on parallelism

Each harness process settles at roughly **2.5 GB** because every one loads the
full Godhome asset set independently. On a 16 GB machine that caps you at about
five environments before the system starts swapping. This, not the CPU, is what
stops you scaling further.

A single shared process running several independent battle states would remove
that limit, but Kristal only ever has one global `Game.battle`, so this is a
harness rewrite rather than a patch. See [README](README.md#status).

---

## Scaling rules of thumb

* `--n-envs 5` is a sensible ceiling on 16 GB. Beyond that you are trading
  throughput for swapping.
* More environments improves gradient quality more than throughput. If your
  policy is plateauing, this is *not* the fix — see
  [README](README.md#status).
* `--n-steps` trades memory for quality, not time. 256 is comfortable.
* With a GPU, `--device cuda` speeds up both inference and the update, but the
  simulators still do not — raise `--n-envs` to give the GPU work.

---

## Boot time

Booting a harness loads every Godhome asset into memory, once per process.

| Stage | Cost |
| --- | --- |
| Extract `Godhome.zip` (first build only) | ~90 s |
| Repack the self-contained executable | ~5 s |
| Per-harness boot at training start | 8–12 s × n_envs |

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