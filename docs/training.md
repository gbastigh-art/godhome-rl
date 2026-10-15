# Training guide

---

## The command

```console
python -m godhome_rl train --name godhome --bosses all --n-envs 3
```

Everything else has a default that works. `--help` lists the rest.

---

## Why PPO

* **The simulator is the bottleneck.** At ~200–400 decision steps per second
  aggregate on a four-core CPU, the gradient step is not the constraint. A
  method that squeezes more out of each transition helps; one that needs a
  bigger network or longer rollouts does not.
* **Credit assignment is short.** A hit costs roughly a second of progress and
  a win pays out immediately, so GAE with a short `gamma` is well matched.
* **Robustness matters more than asymptotics.** PPO's clipped objective is
  extremely stable with a single reward scale and no privileged critic target —
  which matters when the reward has to be hand-shaped per boss family.

## Why this network

The observation is three very different kinds of data: a handful of important
scalars, a spatial raster, and a list of entities. Feeding them into one flat
layer lets the 890-float grid drown out "am I about to be hit". So they get
separate encoders:

```
global (48)  ──► Linear(64)  ──► tanh ─┐
grid  (600) ──► Linear(64)  ──► tanh ──┼──► concat(256) ──► 2×[Linear(256)+tanh]
bullet(240) ──► Linear(128) ──► tanh ─┘                            │
                                                          ┌─────────┴────────┐
                                                        Linear(36)      Linear(1)
```

~150 k parameters, orthonormal initialisation, `0.01` gain on the policy head so
initial log-probabilities are near-uniform, and a running observation
normaliser (Welford, float64 statistics, float32 application).

Torch is pinned to a single thread inside the trainer: the LÖVE processes need
the cores, and oversubscription costs more than the matmuls gain.

---

## The curriculum

`--bosses all` uses a staged ladder in `godhome_rl/curriculum.py`. Each stage
lists bosses, a frame budget, and a promotion rule (rolling win rate and minimum
episodes).

| # | Stage | Bosses | max frames | Promote at |
| --- | --- | --- | --- | --- |
| 1 | `dodge-basics` | False Knight, Mossy Charger, Soul Warrior, Marma | 900 | 35 % over 40 episodes |
| 2 | `greenpath-fogcanyon` | 12 bosses across Greenpath + Fog Canyon | 1500 | 45 % over 80 |
| 3 | `crystal-peak` | 8 Crystal Peak bosses | 1800 | 45 % over 80 |
| 4 | `kingdom` | 13 Kingdom + Abyss bosses | 1800 | 40 % over 100 |
| 5 | `nightmare` | Grimm, Grey Prince, Broken Vessel, Sheo, Nail Master | 1800 | 30 % over 120 |
| 6 | `radiance` | Radiance | 2400 | 25 % over 120 |

Why start small: a uniform policy loses to False Knight in a few seconds and
never sees the interesting part of the fight, so the gradient signal is mostly
"you died". Short, sparse-pattern fights give a dense learning problem first;
dodging skill then transfers to everything else.

To train one boss instead, pass any selector:

```console
python -m godhome_rl train --name radiance-only --bosses "p5/radiance"
python -m godhome_rl train --name p3 --bosses p3
python -m godhome_rl train --name mixed --bosses "p3/grimm,p5/radiance,p1/falseknight"
```

---

## Reading the output

```
[   41] dodge-basics  steps=  7,872 sps= 310 ret= 1.842 win=0.21 ent=3.104 kl=0.0041 ev=+0.31 wait=[1.3,1.4,1.2]
```

| Column | Meaning | Healthy |
| --- | --- | --- |
| `steps` | decision steps since the run started | monotonic |
| `sps` | decision steps per wall-clock second | 150–400 on 3 envs |
| `ret` | mean episode return over the last 200 episodes | rising |
| `win` | win rate over all episodes so far | rising |
| `ent` | policy entropy | falling from 3.58 toward ~2 |
| `kl` | approximate KL between old and new policy | `< target_kl` |
| `ev` | explained variance of the critic | rising above 0 |
| `wait` | per-environment step latency in ms | ~1–3 |
| `restarts` | column in `metrics.csv` | 0, or a handful over hours |

### Failure signatures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `ent` pinned at 3.58 | learning rate too high for the batch, or `kl` guard firing every update | lower `--lr`, raise `--n-steps` |
| `ev` stays negative | critic cannot fit the return; usually because rewards are dominated by one huge terminal | normalise reward magnitudes, raise `--n-epochs` |
| `ret` rises then collapses | a curriculum promotion made the task too hard | rewind to the previous checkpoint; add bosses to the earlier stage |
| `sps` collapses, `restarts` climbs | harness crashes (see [troubleshooting](troubleshooting.md)) | usually one specific encounter; narrow `--bosses` |
| `kl` always ≈ `target_kl` | normalising observations that are mostly zeros | leave it; the early-stop is doing its job |
| `ret` ≈ 0 with `ent` falling | the policy learned to stand still (focus + no move) | check `--ent-coef`; raise the `alive` term |

---

## Hyperparameters

Defaults, and when to change them.

| Flag | Default | Notes |
| --- | --- | --- |
| `--n-envs` | 3 | More environments = better gradient quality, but the LÖVE processes compete for cores. Leave at 2–3 on 4 cores; on 8+ use `cores - 2`. |
| `--n-steps` | 128 | Rollout length per environment. 64–256 is a sane band. |
| `--n-epochs` | 4 | PPO passes per rollout. |
| `--n-minibatches` | 4 | More minibatches = less overfitting per update on a CPU. |
| `--lr` | `3e-4` | Linearly annealed to 10 % unless `--no-anneal`. |
| `--ent-coef` | `0.012` | Entropy bonus. Raise for exploration on hard bosses. |
| `--clip-coef` | `0.2` | Standard. |
| `--gamma` | `0.995` | Shorter (0.99) makes the critic easier to fit. |
| `--gae-lambda` | `0.95` | |
| `--max-grad-norm` | `0.5` | |
| `--updates` | 4000 | Iterations, not steps. 4000 × 3 × 128 ≈ 1.5 M decisions. |
| `--max-steps` | 0 | 0 = follow the curriculum's frame budget. |
| `--difficulty` | 1 | 1 attuned, 2 fake-ascended, 3 ascended, 4 radiant. |
| `--device` | `auto` | `cuda` if available. |
| `--seed` | 0 | Also seeds the harnesses' episode RNG. |

### A stronger configuration

If you have a GPU or more cores and want the policy to be more reliable:

```console
python -m godhome_rl train --name strong --bosses all --n-envs 8 \
    --n-steps 256 --n-minibatches 8 --lr 2.5e-4 --ent-coef 0.01
```

### Run length expectations

**We do not have measured promotion times, because nothing has been promoted
yet.** Every run so far has stalled in stage 1. Publishing a table of "typical
updates to promote" here would be inventing numbers, so instead this is what
was actually observed on the reference machine (4-core i5-7500, no GPU):

| Run | Bosses | Updates | Wall clock | Outcome |
| --- | --- | --- | --- | --- |
| A | stage 1 (4 bosses) | 15 902 | 4.3 h | `win_rate` 0.000, plateau at 71.7 % boss health |
| B | stage 1 (4 bosses) | 14 232 | 2.8 h | same plateau, `win_rate` 0.000 |
| C | `p1/falseknight` only, 5 envs | — | 9 h | still dies, but clears one full moveset with zero hits |

Run C is the encouraging one: surviving a complete moveset untouched is not
something a random policy does. But run C still did not win, so it is
progress, not success.

Two practical lessons that did hold up:

* **Single-boss runs are dramatically faster to get signal from.** Interleaving
  four bosses splits the gradient four ways; run C on one boss showed visible
  behavioural change in hours where runs A and B were flat after 1 000 updates.
* **Runs plateau early.** In A and B, `boss_progress` froze at 0.717 by roughly
  update 1 000 and never moved again across 13 000 further updates. If your
  `boss_progress` column has been flat for a long time, more hours at the same
  settings will not fix it.

When a run does eventually promote, expect the later stages to be considerably
harder than stage 1 — stage 5 is Nightmarish King and Radiance — and treat any
projected multi-day schedule as a guess until someone measures it.

---

## Artefacts

```
runs/<name>/
├── config.json     exact TrainConfig + PPOConfig
├── metrics.csv     one row per update
└── checkpoints/
    ├── update_000050.pt
    └── latest.pt
```

`metrics.csv` is plain CSV with a stable header:

```
update,steps,frames,sps,fps,episode_return,episode_length,win_rate,stage,
boss_progress,policy_loss,value_loss,entropy,approx_kl,clipfrac,
explained_variance,lr,elapsed_s,restarts,episodes
```

```python
import pandas as pd
df = pd.read_csv("runs/godhome/metrics.csv")
df.plot(x="update", y=["episode_return", "win_rate"], secondary_y="win_rate")
```

### Resuming

`python -m godhome_rl train --name godhome` picks up `latest.pt` if its
observation layout matches. Curriculum position and update index are stored in
the checkpoint, so a resumed run continues the same stage.

---

## Plotting

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("runs/godhome/metrics.csv")
fig, ax = plt.subplots(2, 1, sharex=True)
ax[0].plot(df.update, df.episode_return, label="return")
ax[0].plot(df.update, df.win_rate * 5, label="win rate × 5")
ax[0].legend()
ax[1].plot(df.update, df.entropy, label="entropy")
ax[1].plot(df.update, df.explained_variance, label="explained var")
ax[1].legend()
plt.show()
```

---

## Scaling out

The trainer is synchronous: it collects a rollout, then updates. That keeps the
code small and the debugging sane, and on a CPU-bound simulator it loses very
little. To use more hardware:

* **More cores** → raise `--n-envs`. Beyond `cores - 2` the trainer becomes the
  bottleneck rather than the simulators.
* **A GPU** → `--device cuda`. Inference and the update both speed up; the
  simulators still do not, so the gain is modest until `--n-envs` is raised to
  feed it.
* **Multiple machines** → each can run its own `train --name` with a different
  `--bosses` selector; the checkpoint format is the interchange format.

---

## Reproducibility

* `--seed` seeds the torch RNG, the NumPy RNGs and each harness's episode RNG.
* `config.json` records everything, including the reward weights.
* Boss behaviour uses the game's own RNG; the harness passes a per-episode seed
  through `EpisodeSpec` and applies it, so an episode can be replayed.