# Extending GodhomeRL

Four things people reasonably want to change, with the smallest edit that does
it.

---

## 1. Add a boss

GodhomeRL reads the boss catalog from the mod, so a new encounter is available
immediately. Only the friendly name and difficulty tier need adding:

```python
# godhome_rl/bosses.py
BOSSES = (
    ...
    Boss("p2/mynewboss", "My New Boss", "p2", 3),
)
```

Then:

```console
python -m godhome_rl train --name newboss --bosses "p2/mynewboss"
python -m godhome_rl boss            # confirm it is listed
```

`resolve()` also matches display names, so `--bosses "My New Boss"` works.

If the mod adds an encounter whose soul is *not* the Knight, or which never
enters `Battle.state == "DEFENDING"`, the harness will burn its auto-confirm
budget and end the episode with the timeout reward. `VecGodhome` notices after
eight such episodes and restarts the harness; add a trace with `--verbose` to
see which states it is cycling through.

---

## 2. Change the observation

The observation encoder is `lua/rl_obs.lua`. The vector is three concatenated
blocks:

```lua
Obs.GLOBAL_SIZE     = 50
Obs.GRID_CHANNELS   = 2
Obs.BULLET_FEATURES = 10

function Obs.configure(grid_w, grid_h, max_bullets) ... end
function Obs.build(extra) ... end   -- fills Obs.buffer()
```

To add a global scalar:

1. Add a `put(...)` in the global block of `Obs.build`.
2. Bump `Obs.GLOBAL_SIZE` — it sizes the FFI buffer. Every write is bounds
   checked, so a forgotten bump degrades into a truncated vector rather than a
   heap corruption, but the handshake check will still fail.
3. Mirror the new constant in `godhome_rl/env/layout.py`.
4. `python tools/luacheck.py && python -m godhome_rl build && python -m godhome_rl doctor`

The handshake check will refuse to start if Python and Lua disagree, so a
partial edit fails loudly rather than training on garbage.

**Worth adding:**

| Idea | Where |
| --- | --- |
| arena *shape* (not just bounds) | `Arena.processed_shape` → rasterise into a third grid channel |
| per-bullet damage | bullet descriptor, index 5 already has a damage flag |
| tension gain rate | global block, next to `frames since hit` |
| arena corner teleports (Maltes, Borgo) | global block, a small one-hot |
| boss attack animation phase | per-boss: many bosses set `enemy.phase` or an animation name |

**Probably not worth adding:** raw pixels. At 640×480 downsampled, a frame-based
CNN would need orders of magnitude more compute per step and would still have
to rediscover the bullet velocities the structured vector already gives
explicitly.

### Size trade-off

`obs_dim = 50 + 2·grid_w·grid_h + max_bullets·10`. The defaults give 890. On a
CPU:

| Layout | dim | relative forward cost |
| --- | --- | --- |
| `12×9` grid, 16 bullets | 456 | ~0.7× |
| defaults (`20×15`, 24) | 890 | 1.0× |
| `28×21` grid, 32 bullets | 1 830 | ~1.8× |

```console
python -m godhome_rl train --grid-w 28 --grid-h 21 --max-bullets 32
```

Checkpoints record the layout, so `train --resume` will refuse to continue a run
whose observation size changed — which is correct, since the policy's input
layer no longer matches.

---

## 3. Change the reward

`godhome_rl/env/rewards.py`:

```python
@dataclass(slots=True)
class RewardWeights:
    alive: float = 0.006
    damage: float = 1.2
    damage_dealt: float = 1.0
    graze: float = 0.012
    focus: float = 0.0008
    win: float = 4.0
    death: float = -3.0
    timeout: float = -0.75
```

These are forwarded to the Lua harness through the `GODHOME_RL` environment
string and applied there. To change the *shape* of a term (not just its weight),
edit `Battle.engine_frame` in `lua/rl_battle.lua`.

Things that tend to help:

* **Hitless bonus** — a term proportional to `frames_since_hit` makes no-hit
  clears explicitly better than lucky ones.
* **Damage dealt as a fraction of max HP** rather than as a fraction of the
  current pool, so fighting a boss at full health is not worth more than
  finishing it.
* **Spells** — Godhome's Fury (red aura below 15 HP) and Unbreakable spells make
  low-health play strong. If you want the agent to find them, reward the
  transition (spell becomes active), not the state.
* **Faster deaths** — raising `death`'s magnitude relative to `win` makes the
  agent more risk-averse, which is usually right for early stages and wrong for
  Radiance.

After changing weights, **start a new run**: a policy trained under one reward
is not meaningful under another.

---

## 4. Swap the algorithm

`godhome_rl/rl/ppo.py` is self-contained. To add, say, IMPALA or a DQN-style
replay baseline, implement the same four methods and pass the object to
`PPO.collect`/`update` call sites in `godhome_rl/train.py`:

```python
class Learner:
    def begin_episode(self) -> RolloutBuffer: ...
    def collect(self, rollout, env, next_obs) -> (rollout, next_obs, meta): ...
    def compute_returns(self, rollout, next_obs) -> (advantages, returns): ...
    def update(self, rollout, advantages, returns) -> dict: ...
    def anneal(self, update, total) -> float: ...
    def save(self, path) -> None: ...
```

`train.py` only uses those, plus `ppo.ac` for checkpoints. The observation
normaliser lives on the model (`ActorCritic.norm`), so a different
architecture should keep it.

Because the action space is a single categorical, almost any on-policy
discrete-action algorithm drops straight in.

---

## 5. Add a harness mode

The Lua side is small and layered:

| To add… | Touch |
| --- | --- |
| a new config key | `lua/rl_config.lua` (defaults + numeric set) and `HarnessConfig.env_string` |
| a new observation channel | `lua/rl_obs.lua` |
| a new reward term | `Battle.engine_frame` in `lua/rl_battle.lua` |
| a new protocol message | `lua/rl_proto.lua` and `godhome_rl/env/proto.py` |
| a new pacing mode | `Main.serve` in `lua/rl_main.lua` |

Then `python tools/luacheck.py` and `python -m godhome_rl build`.

An easy first experiment: a `render`-mode where the harness plays itself with a
hard-coded policy and no Python at all. Add a branch in `Main.serve` that
generates actions locally when `config.offline_policy` is set; the loop,
observation encoder and rewards are all reusable as-is.

---

## 6. Port to a different LÖVE game

The pieces that are Godhome/Kristal-specific are:

* `lua/rl_battle.lua` — `Battle.start` / `Battle.tick` / `in_defend_phase`
* the global block of `lua/rl_obs.lua`
* `godhome_rl/bosses.py` and `godhome_rl/curriculum.py`

The parts that are generic to LÖVE games are `rl_boot`, `rl_main`, `rl_net`,
`rl_proto`, the grid/bullet blocks of `rl_obs`, the Python protocol client, the
network and PPO. A different game needs a new boot sequence in `Battle.boot`
and a new "is the player in control" predicate, but the transport, the trainer
and the algorithm come across unchanged.