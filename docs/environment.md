# Environment reference

The agent controls one thing: the red heart in a Godhome boss fight. This page
is the exact contract between the game and the policy.

---

## Episode

One episode = **one boss fight**, start to finish.

| Property | Value |
| --- | --- |
| Starts | `Game:encounter("<boss id>", false)` — no battle transition, no overworld |
| Success | every boss in the encounter is defeated (`encounter.bosses` empty) |
| Failure | the Knight goes down (`Game.state == "GAMEOVER"`) |
| Truncation | `max_steps` **engine frames** elapse (default 1800 = 60 s of game time) |
| Reset | HP back to max, tension to 0, flags cleared, stage wiped |

The player is **The Knight** with Godhome's default stats: 90 HP, 10 attack,
3 defence, 13 magic, and the Void Heart / Fury armour pair from `mod.json`.
Difficulty defaults to **attuned** (`encounter.difficulty == 1`); `--difficulty 3`
gives ascended health/damage and `--difficulty 4` radiant damage.

---

## Action space

`Discrete(36)`, encoded as

```
action = direction + 9 * attack + 18 * focus
```

| Field | Values | Maps to |
| --- | --- | --- |
| `direction` | 0–8 | 0 = stand still; 1 left, 2 right, 3 up, 4 down, 5 up-left, 6 up-right, 7 down-left, 8 down-right |
| `attack` | 0/1 | `Z` pressed this frame → nail swing |
| `focus` | 0/1 | `X` held → half speed, drains tension, converts it to healing at 16 TP |

`move(0)` is not a no-op — it keeps whatever velocity and knockback the soul
already has, which is how you ride the momentum out of a fast pattern.

Outside `Battle.state == "DEFENDING"` the harness ignores the action and
auto-confirms instead, so **every action the agent sees was applied to a frame
where the soul was live**.

---

## Observation space

`Box(-inf, inf, shape=(obs_dim,))`, `float32`, where

```
obs_dim = 50 + 2 * grid_w * grid_h + max_bullets * 10
```

with the defaults `grid_w = 20`, `grid_h = 15`, `max_bullets = 24` giving
**890**. The harness reports its real layout in the handshake and Python asserts
it matches, so a mismatch can never silently train on a mis-shaped vector.

### Block 1 — global scalars (50)

| Index | Field | Scale |
| --- | --- | --- |
| 0–1 | soul position | `x / 640`, `y / 480` |
| 2–3 | soul velocity | per-frame delta / 160, / 120 (smoothed) |
| 4 | player HP fraction | `(hp + 45) / 135` — Godhome's HP floors at −45 |
| 5 | player HP | `hp / 100` |
| 6 | tension | `tp / 100` |
| 7 | invulnerability timer | `inv_timer / 2` |
| 8 | focus available | 0/1 |
| 9 | focus banked | `focus_used / 16` |
| 10 | nail off cooldown | 0/1 |
| 11 | nail active | 0/1 (the swing is live) |
| 12 | knockback | `knockback / 4` |
| 13–18 | battle state | one-hot over DEFENDING / DEFENDINGBEGIN / dialogue / menu / INTRO / end |
| 19 | wave progress | `wave_timer / wave_length` |
| 20 | wave time remaining | seconds / 5 |
| 21–24 | boss 0 | HP fraction, armour fraction (−1 = no armour), relative x, y |
| 25–28 | boss 1 | HP fraction, armour fraction, placeholders |
| 29–32 | boss 2 | HP fraction, armour fraction, placeholders |
| 33–40 | arena | left/right/top/bottom (screen fraction), width, height, is-circle, exists |
| 41 | episode progress | frames / max_frames |
| 42 | frames since hit | / 30 |
| 43 | hits taken | / 10 |
| 44 | grazes | / 200 |
| 45–48 | active spells | fury, unnamed (void heart), quickslash, quickfocus |
| 49 | previous action | index / 100 — lets the policy keep a rhythm |

Boss HP and armour are pooled into a single monotone progress bar, so
"break the armour, then the boss" reads as one number going down.

### Block 2 — spatial grid (2 × 20 × 15)

A screen-sized raster with two channels:

* **density** — bullets per cell, scaled so 4 bullets saturates at 1.
* **threat** — for each bullet, the closest approach to the soul over the next
  9 frames is computed analytically from its position and velocity; the cell it
  would occupy is stamped with `1 - distance/120`. The max wins.

The threat channel is what lets a small MLP reason about "something is about to
cross my path" instead of only "something is near me right now".

### Block 3 — bullet list (24 × 10)

The 24 bullets nearest the soul (partial selection sort, O(24·n)):

| Feature | Meaning |
| --- | --- |
| 0–1 | offset from the soul, / 160 and / 120 |
| 2–3 | velocity per frame, / 6 |
| 4 | conservative bounding radius, / 20 |
| 5 | damages the player |
| 6 | harmless (0 damage) |
| 7 | grazeable (`tp ≠ 0`) |
| 8 | time to closest approach over the lookahead, in [−1, 1] |
| 9 | already grazed |

Unused slots are zero. Velocity is measured exactly by differencing positions
between frames and cached per object; the physics vector is used for objects
first seen this frame.

Bullet radii are derived conservatively from whatever collider the bullet uses
(circle, hitbox, line or polygon), so oversized hitboxes read as more dangerous
than they look — which is the safe direction for a policy.

### Info vector (16 floats)

Alongside every observation the harness sends diagnostics, consumed for logging
and evaluation only:

| Index | Field |
| --- | --- |
| 0 | frames elapsed in the episode |
| 1 | boss progress (1 = untouched, 0 = defeated) |
| 2, 3 | player HP / max HP |
| 4, 5 | damage dealt / taken |
| 6 | grazes |
| 7 | hits taken |
| 8 | outcome: `+1` win, `−1` death, `0` otherwise |
| 9 | boss total health pool |
| 10 | engine-reported graze count |
| 11 | termination reason code |
| 12 | tension |
| 13 | wave timer |
| 14 | bullets on stage |
| 15 | agent-controlled frames this episode |

---

## Reward

Every term is normalised by the fight's own scale, so the same weights transfer
from Mossy Charger to Radiance.

| Term | Weight | When |
| --- | --- | --- |
| `alive` | `+0.006` per frame | while a bullet pattern is running |
| `damage` | `−1.2` per point of **max-HP** lost | on being hit |
| `damage_dealt` | `+1.0` per unit of the boss health pool removed | nail hits, armour included |
| `graze` | `+0.012` per new graze | first time a grazeable bullet enters the graze ring |
| `focus` | `−0.0008` per frame | while focusing, to discourage stalling |
| `win` | `+4.0` | boss defeated |
| `death` | `−3.0` | Knight down |
| `timeout` | `−0.75` | frame budget exhausted |

Design notes:

* `alive` at 0.006/frame is 0.18 per second — meaningful but never dominant
  against a 4.0 win bonus, so the agent has a reason to finish fights.
* Losing one full health bar costs 1.2, which is comparable to removing 12 % of
  a boss. That ratio is what makes hitless play the optimal strategy rather
  than merely a stylish one.
* Grazes are credited geometrically (a damaging, grazeable bullet whose hull
  enters the graze ring), because some Godhome bullets opt out of tension and a
  per-bullet engine flag would make the reward inconsistent across bosses.

Change any of these with `RewardWeights` in `godhome_rl/env/rewards.py`; they
are passed to the harness at launch and recorded in `runs/<name>/config.json`.

---

## Termination

| Reason | Flags sent | `info[8]` |
| --- | --- | --- |
| boss defeated | `TERMINAL | EPISODE` | `+1` |
| Knight down | `TERMINAL | EPISODE` | `−1` |
| frame budget exhausted | `TERMINAL | TRUNCATED | EPISODE` | `0` |

The first frame of an episode carries `RESET` and a reward of `0`: nothing has
happened yet, so there is nothing to learn from.

---

## Python API

```python
from godhome_rl.env.godhome_env import GodhomeEnv
from godhome_rl.env.layout import Layout

with GodhomeEnv(bosses=["p1/falseknight"], max_steps=1800,
                layout=Layout(grid_w=20, grid_h=15, max_bullets=24)) as env:
    obs, info = env.reset()
    done = False
    while not done:
        action = int(env.rng.integers(0, 36))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
```

And for several at once:

```python
from godhome_rl.env.vec_env import VecGodhome

with VecGodhome(3, bosses=["p1/falseknight"], max_steps=1800) as venv:
    obs = venv.reset()
    while True:
        obs, rewards, dones, infos = venv.step(venv.n_actions and
                                                __import__("numpy").zeros(
                                                    venv.n_envs, dtype=int))
        if all(dones):
            break
```

`VecGodhome` auto-resets on `dones`; the terminal observation of a finished
episode is preserved in `infos[i]["final_obs"]`.