# Architecture

There are three pieces: a **Lua harness** that lives inside the real game, a
**Python trainer** that speaks to it, and a **build step** that stitches the
harness into the game.

```
┌──────────────────────── your Godhome install ─────────────────────────┐
│  Godhome.zip          Godhome-windows.exe      love.dll, SDL2.dll, …  │
└───────────────┬──────────────────────────────────┬───────────────────┘
                │ tools/build.py                   │  re-used verbatim
                ▼                                  ▼
┌───────────────────────────────────────────────────────────────────────┐
│  build/game/            an *unmodified* copy of the game plus:       │
│    ├── main.lua         + require("rl/rl_boot")   (2 lines appended) │
│    ├── conf.lua         + headless window/module tweaks              │
│    └── rl/*.lua         the harness (copied from lua/)                │
│                                                                       │
│  build/runtime/Godhome-rl.exe   love.exe stub + build/game re-zipped  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ launched once per environment
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  LÖVE process  (one per environment)                                  │
│    love.run  ──replaced──►  rl_main.serve()                           │
│                              │  TCP on 127.0.0.1                      │
│                              ▼                                        │
│    Battle:update  ◄──  rl_battle  ──►  rl_obs (observation vector)     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │  binary frames, float32 arrays
                               ▼
┌───────────────────────────────────────────────────────────────────────┐
│  Python trainer                                                      │
│    env/vec_env.py   N harnesses, pipelined                           │
│    rl/ppo.py        rollout → GAE → clipped policy gradient           │
│    train.py         loop, curriculum, checkpoints, metrics           │
└───────────────────────────────────────────────────────────────────────┘
```

---

## Why a harness inside the game at all

The obvious alternative — re-implementing the boss patterns in Python — fails
for Godhome specifically. There are ~45 encounters and several hundred bullet
scripts, each with its own timing curves, arena shapes, knockback, i-frame
windows, tension rules and spell interactions. Any re-implementation drifts.

Running the shipped game means:

* **Exactness.** Colliders, `CollisionUtil`, `Object` transforms, the nail
  arc, the arena polygon solver and every boss script are the originals.
* **Free content.** Adding a boss to the mod is immediately available to the
  agent; there is nothing to port.
* **Replayability.** `play` shows you the real game, so what the agent sees is
  what you see.

The cost is that we cannot change the engine's update loop, so the harness has
to live *inside* it. That is what `rl_boot.lua` does: it replaces `love.run`
with a loop that never sleeps, never draws, and uses a fixed timestep.

---

## The Lua harness

| Module | Responsibility |
| --- | --- |
| `rl_boot.lua` | Gated on `$GODHOME_RL`. Parses configuration, checks LuaSocket, installs the loop, wraps everything in a `pcall` that reports failures to the trainer |
| `rl_config.lua` | Defaults + parsing of the `key=value;…` environment string |
| `rl_main.lua` | `serve()`: accept a connection, handshake, then alternate "receive action → step → send observation" |
| `rl_battle.lua` | `boot()`, `start()`, `tick()`, rewards, termination, input synthesis |
| `rl_obs.lua` | Turns a live `Battle` into a flat `float32` vector |
| `rl_proto.lua` | Binary framing via the LuaJIT FFI |
| `rl_net.lua` | LuaSocket wrapper (blocking listen/accept) |

### Booting

`rl_battle.boot()`:

1. writes a `settings.json` with `skipIntro`, 30 FPS and zero volume,
2. calls Kristal's own `love.load`,
3. pumps `love.update` until the engine's loading state reports all assets
   streamed in,
4. loads the `godhome` mod with `Kristal.loadMod`, sets `Mod.info.map = nil`
   so no overworld is ever built, and switches straight into the `Game` state,
5. replaces the overworld `World` with an inert one.

Step 5 matters: Kristal's `World:update` unconditionally indexes `self.map`,
and running an unloaded overworld ticks camera, tilemap and event code for
nothing. The inert `World` keeps a real `Map` object (so anything that merely
*reads* it still works) but has `update`/`draw` replaced with no-ops.

### Episodes

`Battle.start(boss_id, opts)` wipes the stage, re-initialises party HP,
tension and flags, then calls `Game:encounter(boss_id, false)`. It costs about
25 ms — cheap enough to run thousands of times per hour.

`Battle.tick(action)` is one **network** step:

```
while the soul is not in play:        # menus, dialogue, wave transitions
    inject a synthetic "confirm" press and run one engine frame
once the soul is in play:
    apply the agent's action and run exactly one engine frame
```

So every transition Python sees is a frame where the decision changes
something. Auto-confirmed frames still produce reward (the survival bonus and
any damage that lands) but the agent never wastes capacity learning to press
Z in a menu.

### Input synthesis

Godhome reads input through Kristal's `Input` tables, so the harness writes
straight into them:

| Action bit | Keyboard |
| --- | --- |
| direction 0–8 | `left`/`right`/`up`/`down` held |
| attack | `z` **pressed** this frame (nail swing) |
| focus | `x` held (slow movement, drains tension) |

Auto-confirm uses the real `love.keypressed("z")` dispatch rather than poking
the tables, so menu code paths run exactly as they would for a player.

---

## The wire protocol

A deliberately boring framing — `[uint8 tag][uint32 length][payload]` — over
TCP on loopback, with LuaSocket on one side and a plain `socket` on the other.
Full details in [protocol.md](protocol.md).

| Tag | Direction | Meaning |
| --- | --- | --- |
| `0xA1 ACT` | Py→Lu | action index for the next frame |
| `0xA2 EPISODE` | Py→Lu | start a new fight with this spec |
| `0xA3 QUIT` | Py→Lu | shut down cleanly |
| `0xB1 READY` | Lu→Py | handshake: obs dim, action count, layout, dt |
| `0xB2 STEP` | Lu→Py | observation, reward, flags, info vector |
| `0xB3 LOG` | Lu→Py | diagnostics (including crash reports) |

Everything numeric is little-endian `float32`; observations are sent as a raw
block, so a step costs one 3.5 KiB `send` and no parsing on the Python side
beyond `np.frombuffer`.

The Lua side **listens** and Python **connects**. That inverts the usual
arrangement for a good reason: the port file is written before asset loading
starts, so a boot failure reaches Python as a readable `LOG` message instead of
an opaque timeout.

---

## Vectorised environments

`VecGodhome` owns *N* `GodhomeEnv`s. A round dispatches to every harness
*before* reading any reply:

```python
for env in self.envs:
    env.send_action(action)      # all LÖVE processes now run concurrently
for env in self.envs:
    obs, reward, done, info = env.recv()
```

Without this the LÖVE processes serialise behind the Python loop and adding
environments buys nothing. With it, three harnesses on three cores are close to
three times one harness.

Because the policy forward pass is batched across environments, the Python side
costs one ~1 ms forward per round regardless of *N*.

---

## Robustness

Long unattended runs are the point, so failures are contained rather than
fatal:

* **Transport failure** — `GodhomeEnv` closes and relaunches its harness, then
  discards the step. The trainer sees one wasted transition.
* **Desynchronised / wedged harness** — `VecGodhome` restarts an environment
  that ends eight consecutive episodes in under 30 steps.
* **Lua error inside the loop** — the harness sends a `LOG` message describing
  the error before the socket closes, so the reason is in the training log
  rather than in a stack trace on a dead process.
* **Trainer crash** — checkpoints every `save_interval` updates and the loop
  resumes from the newest compatible one.

Restart statistics are tracked and printed as the `restarts` column in
`metrics.csv`.

---

## File map

```
lua/rl_boot.lua        bootstrap; replaces love.run; pcall + error reporting
lua/rl_main.lua        serve(): handshake, episode/step loop, render pacing
lua/rl_battle.lua      boot, start, tick, reward, termination, input synthesis
lua/rl_obs.lua         Battle → float32 observation vector
lua/rl_proto.lua       binary framing (LuaJIT FFI)
lua/rl_net.lua         LuaSocket helpers
lua/rl_config.lua      defaults and environment parsing

godhome_rl/env/proto.py     Python mirror of rl_proto.lua
godhome_rl/env/layout.py   observation layout constants + handshake check
godhome_rl/env/harness.py  process launch + transport
godhome_rl/env/godhome_env.py  single environment, episode stats, self-healing
godhome_rl/env/vec_env.py  pipelined N environments
godhome_rl/env/rewards.py  reward weights

godhome_rl/rl/model.py     structured actor-critic + running observation norm
godhome_rl/rl/ppo.py       GAE, clipped surrogate, value clipping, KL guard
godhome_rl/train.py        training loop, curriculum, checkpoints, CSV metrics
godhome_rl/cli.py          python -m godhome_rl …
tools/build.py             extract → patch → repack
tools/luacheck.py          offline Lua syntax check (lupa)
```