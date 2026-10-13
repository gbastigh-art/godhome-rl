# GodhomeRL

**A reinforcement-learning agent that learns to play [Godhome](https://github.com/vitellaryjr/Godhome) — the Hollow Knight boss-rush mod for [Deltarune](https://deltarune.com) / [Kristal](https://github.com/KristalTeam/Kristal).**

GodhomeRL runs the *actual game* — the real LÖVE build, the real bullet patterns,
the real nail arcs — inside a headless harness, and trains a PPO policy against
it at **hundreds of simulated seconds per wall-clock second**. No part of the
game is re-implemented or approximated.

```console
$ python -m godhome_rl doctor
[ok]   torch            2.10.0+cpu (cuda=False)
[ok]   Godhome install /home/you/godhome-standalone_e1c66
[ok]   game tree        .../build/game
[ok]   runtime          .../build/runtime/Godhome-rl.exe
[ok]   boot 8.6s, 485 steps/s, obs_dim=890
all good

$ python -m godhome_rl train --name godhome --bosses all --n-envs 3
[train] started 3 harness(es) | stage 1/6 'dodge-basics': 4 bosses ...
[    0] dodge-basics       steps=      192 sps=  220 ret=  0.000 ...
```

---

## Contents

| Document | What is in it |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Every supported way to get set up, from scratch, on Windows / Linux / macOS |
| [docs/quickstart.md](docs/quickstart.md) | Ten minutes from zero to a training run |
| [docs/architecture.md](docs/architecture.md) | How the harness, the wire protocol and the trainer fit together |
| [docs/environment.md](docs/environment.md) | Action space, observation layout, reward function, episode lifecycle |
| [docs/training.md](docs/training.md) | The curriculum, PPO hyperparameters, tuning guide, checkpoints |
| [docs/performance.md](docs/performance.md) | Measured throughput, how to scale it, what the bottlenecks are |
| [docs/playback.md](docs/playback.md) | Watching the agent, recording runs, evaluating policies |
| [docs/troubleshooting.md](docs/troubleshooting.md) | Every failure mode we have seen, and the fix |
| [docs/protocol.md](docs/protocol.md) | The binary protocol between LÖVE and Python |
| [docs/extending.md](docs/extending.md) | Adding bosses, observations, rewards, or algorithms |
| [docs/faq.md](docs/faq.md) | Questions people actually ask |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to work on this |
| [LICENSE](LICENSE) | MIT |

---

## Why this exists

Godhome is, mechanically, a bullet-hell dodging game: a red heart moves on a
640×480 screen, the arena has walls, and every boss is a scripted pattern of
projectiles. That is *exactly* the kind of problem modern RL handles well — and
exactly the kind of problem where "the simulator must be the real game"
matters. Agents trained on an approximation learn to beat the approximation.

So GodhomeRL does the opposite of the usual thing:

1. **It runs the shipped game.** `build/` contains the untouched `Godhome.zip`
   with a ~1 500-line Lua harness injected. Bosses, bullets, arenas, collision,
   i-frames, tension and spells are the game's own code.
2. **It removes only what no agent could act on.** Menus, dialogue boxes, the
   attack animation and the arena-spawn animation are auto-confirmed, so the
   policy only ever sees frames where the soul is actually under its control.
   Roughly 25 % of a fight is kept.
3. **It runs it far faster than real time** — no window, no rendering, no
   sleeping, fixed 1/30 s timestep, ~600–1 500 decision steps per second per
   process on a four-year-old laptop CPU.
4. **It is self-contained.** One `Godhome.zip` is the only input. No LÖVE
   install, no Vulkan, no GPU, no network at training time.

---

## Quick start

```console
git clone <this repo>
cd GodhomeRLagent

# 1. Point at your Godhome standalone folder (contains Godhome.zip)
#    -- Windows PowerShell
$env:GODHOME_RL_INSTALL = "C:\Games\godhome-standalone_e1c66"
#    -- Linux / macOS
export GODHOME_RL_INSTALL=~/games/godhome-standalone_e1c66

# 2. Install dependencies and build the patched game
python -m godhome_rl setup

# 3. Train
python -m godhome_rl train --name godhome --bosses all --n-envs 3

# 4. Watch it play
python -m godhome_rl play --model runs/godhome/checkpoints/latest.pt --boss radiance
```

That is the whole workflow. See [docs/quickstart.md](docs/quickstart.md) for a
guided version with expected output at each step.

---

## What "insanely good" means here

The agent plays the real fight under the real rules: it starts with the
Knight's 90 HP, has to break armour phases, dodge patterns that fill the screen,
build tension for Fury, and land nail hits. Winning a Godhome boss fight is the
only way to finish an episode with a positive terminal reward.

`docs/training.md` has the training curve you should expect, the wall-clock
budget for each stage, and how to tell "learning to dodge" from
"learning this specific boss".

---

## Repository layout

```
GodhomeRLagent/
├── godhome_rl/            # the Python package
│   ├── env/               # harness client, protocol, envs, observation layout
│   ├── rl/                # actor-critic network and PPO
│   ├── bosses.py          # the Godhome boss catalog
│   ├── curriculum.py      # staged boss progression
│   ├── train.py           # training loop
│   ├── play.py            # observer mode
│   └── cli.py             # `python -m godhome_rl ...`
├── lua/                   # the in-game harness (injected into the game at build)
│   ├── rl_boot.lua        #   bootstrap, replaces love.run
│   ├── rl_main.lua        #   the serve loop
│   ├── rl_battle.lua      #   episode lifecycle, input synthesis, reward
│   ├── rl_obs.lua         #   observation encoder
│   ├── rl_proto.lua       #   binary protocol
│   ├── rl_net.lua         #   LuaSocket helpers
│   └── rl_config.lua      #   configuration
├── tools/                 # build, Lua syntax checking
├── tests/                 # offline tests that need no game install
├── docs/                  # the documentation you are reading
├── build/                 # generated: patched game + self-contained exe
├── runs/                  # generated: checkpoints and metrics
└── vendor/                # optional: drop the Godhome standalone here
```

`build/`, `runs/` and `vendor/` are generated and git-ignored. The repository
contains **no** game assets — the build step produces them from your own copy of
Godhome.

---

## Requirements

* Python 3.10+
* A copy of the Godhome standalone download (the folder with `Godhome.zip`)
* ~1 GB of free disk for the extracted game and the self-contained executable
* No GPU. A CPU-only PyTorch install is fine and is what this was tuned on
  (4 cores, 16 GB RAM, Intel HD 630).

---

## Licence

The harness and training code in this repository are MIT licensed
(see [LICENSE](LICENSE)). Godhome is © its authors and is distributed under its
own licence; Kristal is GPLv3. This project **does not redistribute either** —
it builds a patched copy from your own installation at setup time. See
[docs/licensing.md](docs/licensing.md).

Godhome is a fan project of *Hollow Knight* by Team Cherry and *Deltarune* by
Toby Fox. Neither is affiliated with or endorsed by this repository.