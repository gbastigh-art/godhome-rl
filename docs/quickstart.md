# Quickstart

Ten minutes from a fresh clone to a training run, with the expected output at
each step so you can tell whether things are working.

---

## Step 0 — prerequisites

* Python 3.10+
* Your Godhome standalone folder (see [Installation](installation.md))

```console
python --version
```

---

## Step 1 — build

```console
python -m godhome_rl build
```

Expected:

```
[build] install: C:\Games\godhome-standalone_e1c66
[build] extracting Godhome.zip (one-off, ~90 s) ...
[build] repacked self-contained runtime
[build] game dir: C:\...\GodhomeRLagent\build\game
[build] runtime: C:\...\GodhomeRLagent\build\runtime\Godhome-rl.exe
```

The first build unpacks ~4 000 asset files and takes a minute or two. Every
later invocation says `cached build is up to date` and finishes in well under a
second.

---

## Step 2 — doctor

```console
python -m godhome_rl doctor
```

Expected: `[ok]` on every line and `all good` at the end. This boots a real
harness and plays 60 frames of False Knight.

---

## Step 3 — see the boss list

```console
python -m godhome_rl bosses
```

45 encounters across the five Godhome pantheon areas. Any of these selectors
work anywhere a boss is accepted:

| Selector | Meaning |
| --- | --- |
| `p5/radiance` | an encounter id |
| `Radiance` | a display name |
| `p3` | every boss in Crystal Peak |
| `all` | every boss |

---

## Step 4 — measure your machine

```console
python -m godhome_rl bench --boss "p1/falseknight,p3/grimm,p5/radiance"
```

This boots each boss once, plays a few random-policy episodes and reports
throughput. A rough baseline on the reference machine (4-core i5-7500, no GPU):

```
boss                                 boot  steps/s  game-s/s  ep  win%
--------------------------------------------------------------------------
False Knight (p1=Greenpath)          8.1s     462      1241  4     0%
Grimm (p3=Crystal Peak)              8.5s     306      1430  4     0%
Radiance (p5=The Abyss)              8.2s     326      1657  4     0%
```

`steps/s` counts *decision* steps — frames where the soul is under the policy's
control. `game-s/s` counts every simulated frame, including the auto-confirmed
menus.

---

## Step 5 — train

```console
python -m godhome_rl train --name my-agent --bosses all --n-envs 3
```

Expected:

```
[train] started 3 harness(es) | stage 1/6 'dodge-basics': 4 bosses, max_steps=900, win_rate=0.00 (promote at 0.35 after 40 episodes)
[    0] dodge-basics       steps=      192 sps=  220 ret=  0.000 win=0.00 ent=3.58 kl=0.0068 ev=-0.52
[    1] dodge-basics       steps=      384 sps=  234 ret=  0.000 win=0.00 ent=3.58 kl=0.0033 ev=+0.48
...
```

Things to watch:

* **`sps`** should settle at 150–400 for three environments. Much lower means
  the harnesses are restarting (check the `restarts` column in `metrics.csv`).
* **`ent`** starts near `ln(36) = 3.58` (a uniform policy) and should fall
  steadily. If it pins at 3.58 nothing is being learned.
* **`ret`** is the mean episode return. Random play sits near 0; a competent
  agent on the first stage reaches 2–4.
* **`win=0.00`** is normal for the first few hundred episodes.

Training writes to `runs/my-agent/`:

```
runs/my-agent/
├── config.json            # exact configuration, for reproducibility
├── metrics.csv            # one row per PPO iteration
└── checkpoints/
    ├── update_000050.pt
    └── latest.pt
```

Stop with `Ctrl-C` at any time; the loop closes the harnesses cleanly and the
last checkpoint stays usable.

---

## Step 6 — watch it

```console
python -m godhome_rl play --model runs/my-agent/checkpoints/latest.pt --boss radiance
```

A real Godhome window opens and the trained policy plays it, one per encounter
in the rotation. You get a summary line after each fight:

```
[play] LOSS Radiance (p5=The Abyss)       frames= 900 boss_hp=0.71 hp=12/90 grazes= 37 hits=  8
```

Useful flags:

* `--stochastic` — sample from the policy instead of taking the argmax
* `--episodes 5` — quit after five fights
* `--render-every 2` — halve the rendering cost if the window is choppy

---

## Step 7 — evaluate

```console
python -m godhome_rl eval --model runs/my-agent/checkpoints/latest.pt --boss all --episodes 3
```

```
boss                                 win%   return  frames  hits  graze
------------------------------------------------------------------------
Vengeful King (p1=Greenpath)            0%   -0.31    900.0   7.0   21.3
...
overall: 0/135 wins (0%)
```

Use this to decide which curriculum stage the policy has actually mastered.

---

## Step 8 — resume

```console
python -m godhome_rl train --name my-agent --bosses all --n-envs 3
```

Same command, no flags: the run picks up the newest checkpoint whose
observation layout matches. Use `--no-resume` to start over.

---

## Next

* [Training](training.md) — what to expect, how to tune, how long it takes
* [Environment](environment.md) — the exact action/observation/reward definition
* [Troubleshooting](troubleshooting.md) — when a step does not go as above