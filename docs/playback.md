# Watching and evaluating the agent

---

## `play` — watch it in the real game

```console
python -m godhome_rl play --model runs/godhome/checkpoints/latest.pt --boss radiance
```

A normal Godhome window opens at 640×480 (scaled by your `settings.json`) and
the policy plays. This is the *same* build the trainer used, the same fixed
timestep, the same physics — the only difference is that the loop renders and
paces itself to wall-clock time instead of running flat out.

```
[play] LOSS Radiance (p5=The Abyss)  frames= 900 boss_hp=0.71 hp=12/90 grazes= 37 hits=  8
[play] WIN  Radiance (p5=The Abyss)  frames= 741 boss_hp=0.00 hp=61/90 grazes=118 hits=  2
[play] LOSS Radiance (p5=The Abyss)  frames= 900 boss_hp=0.68 hp=20/90 grazes= 44 hits=  6
```

`frames` is the whole fight including the auto-confirmed parts; `boss_hp` is the
pooled health+armour fraction; `hp` and `grazes` are the run's cumulative
values for that fight.

### Flags

| Flag | Default | Effect |
| --- | --- | --- |
| `--boss` | `all` | encounter id, name, area prefix, or `all`; cycles through the matches |
| `--episodes` | `0` | 0 = play until you close the window |
| `--max-steps` | `1800` | frame budget per fight |
| `--render-every` | `1` | render every Nth frame; `2` halves the rendering cost |
| `--stochastic` | off | sample from the policy instead of taking the argmax |
| `--seed` | `0` | seeds the harness RNG |

### Window control

* **Close the window** to quit (the harness treats `quit` as a shutdown).
* **Ctrl-C** in the terminal also works.
* If the window is choppy, `--render-every 2` or `3`. The policy is unaffected;
  only the picture rate changes.
* `settings.json` in the LÖVE save directory controls `windowScale`; set it to
  `2` or `3` for a bigger window.

---

## `eval` — headless scoring

```console
python -m godhome_rl eval --model runs/godhome/checkpoints/latest.pt --boss all --episodes 5
```

```
boss                                 win%   return  frames  hits  graze
------------------------------------------------------------------------
Vengeful King (p1=Greenpath)            0%   -0.31    900.0   7.0   21.3
Gruz Mother (p1=Greenpath)             40%    2.10    801.0   3.2   44.7
False Knight (p1=Greenpath)           100%    5.40    702.0   1.4   61.2
...
overall: 37/225 wins (16%)
```

| Column | Meaning |
| --- | --- |
| `win%` | fraction of fights beaten |
| `return` | mean undiscounted episode return |
| `frames` | mean fight length in engine frames |
| `hits` | mean hits taken |
| `graze` | mean grazes |

`--difficulty 3` evaluates against ascended bosses, `--difficulty 4` radiant,
which is how you check whether the policy learned dodging or just memorised
attack timings.

Use a large `--episodes` for anything you intend to report: Godhome patterns
have real randomness, and five episodes of a boss is not a measurement.

---

## Scoring policies across checkpoints

```console
for f in runs/godhome/checkpoints/update_000*.pt; do
  python -m godhome_rl eval --model "$f" --boss all --episodes 10 | tail -1
done
```

---

## Recording

There is no built-in video capture; the game renders at whatever rate the
machine sustains. Two easy options:

1. **Screen recorder** — any OS tool works. `--render-every 1` keeps the window
   at its natural rate.
2. **Per-frame PNG dump** — set `KEYFRAMES = true` in the harness render branch
   and add a `love.graphics.captureScreenshot` call after `do_draw` in
   `lua/rl_main.lua`. This is deliberately not enabled by default because it
   costs a full framebuffer read per frame.

For analysis rather than video, the CSV the trainer writes is the better
artefact: it contains everything needed to plot learning curves without
re-running anything.

---

## Debugging what the agent sees

With `--verbose` the harness writes a breadcrumb log next to its port file:

```
[build/ports/harness-<pid>-<stamp>.log]
```

It records boot progress, timing marks, a per-state frame census during
auto-confirm, and any error that killed the loop. `GodhomeEnv(verbose=True)`
enables it, and `python -m godhome_rl train --verbose` turns it on for every
environment.

To see the observation directly, from Python:

```python
import numpy as np
from godhome_rl.env.godhome_env import GodhomeEnv
from godhome_rl.env.layout import Layout

layout = Layout()
with GodhomeEnv(bosses=["p3/grimm"], max_steps=900) as env:
    obs, _ = env.reset()
    for _ in range(120):
        obs, *_ = env.step(np.random.default_rng().integers(0, 36))
    global_block = obs[layout.global_slice]
    grid = obs[layout.density_slice].reshape(layout.grid_h, layout.grid_w)
    print("soul:", global_block[:4], "hp:", global_block[4])
    print((grid > 0.5).astype(int))          # coarse ASCII picture of the arena
```

`godhome_rl.env.layout.Layout` exposes `global_slice`, `density_slice`,
`threat_slice` and `bullet_slice` so you never have to hard-code offsets.

---

## Watching a specific moment

Because episodes are seeded, you can reproduce one:

```python
from godhome_rl.env.proto import EpisodeSpec

env.harness.start_episode(EpisodeSpec(boss_id="p3/grimm", seed=12345))
```

Replay it with `play --stochastic --seed 12345` for a comparable, though not
bit-identical, run (GPU rasterisation is not part of the simulation, but the
harness does not assert determinism across LÖVE builds).