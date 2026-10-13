# FAQ

---

### Do I need to own Hollow Knight or Deltarune?

No. Godhome is a standalone fan game; you only need its own download. Nothing
from either commercial game is bundled here or required at runtime.

### Do I need a GPU?

No. Everything was developed and measured on a four-year-old i5 with Intel
integrated graphics and CPU-only PyTorch. A GPU makes the policy forward pass
free but does nothing for the simulators, so the gain is modest until you raise
`--n-envs`.

### Do I need LÖVE installed separately?

No. The build re-uses the LÖVE runtime inside `Godhome-windows.exe` and produces
a self-contained `build/runtime/Godhome-rl.exe`. LÖVE 11 is only used if you
delete that executable and set `LOVE_EXE`.

### Is this legal / can I share it?

The code in this repository is MIT. **The game is not redistributed** — the
build step creates a patched copy from *your* installation inside `build/`, which
is git-ignored. Do not commit `vendor/`, `build/` or `runs/`. See
[licensing.md](licensing.md).

### Will it actually beat Godhome bosses?

The setup is built for it: real patterns, real collision, rewards that make
hitless play optimal, and a curriculum that starts where a learning agent can
actually get signal. Whether a given run reaches a given boss depends on the
budget you put in — see the run-length table in [training.md](training.md). We
report what we measured; we do not claim results we have not got.

### How long until it beats anything?

Stage 1 (`dodge-basics`) should show a non-trivial win rate within a few hundred
thousand decisions — a couple of hours on the reference machine. Full bosses
take on the order of days of wall clock.

### How much does it cost in electricity / cloud credits?

A typical run to stage 3 is a few million decisions. At 250 steps/s that is
roughly 4–6 hours on a laptop. A cheap cloud VM with 4 vCPUs does the same in a
day; nothing here needs a GPU instance.

### Can I train on one boss only?

Yes, and you probably should for a first win:

```console
python -m godhome_rl train --name falseknight --bosses "p1/falseknight"
```

---

### What exactly is the agent controlling?

Only the heart. Movement (8 directions), the nail swing and focus. Menus,
dialogue and animations are auto-confirmed by the harness. See
[environment.md](environment.md).

### Why not pixel observations?

Because at 640×480 a frame-based policy would need orders of magnitude more
compute per simulated second, and would still have to rediscover bullet
velocities that the structured vector already provides. Bullet-hell RL
literature is overwhelmingly symbolic for this reason.

### Does the agent get to see the whole screen?

It gets a 20×15 raster of the whole screen (density + a look-ahead threat
channel) plus the 24 nearest bullets in full detail, plus global state. Yes —
it has complete information except for the contents of bullets beyond the
nearest 24, which the raster still summarises.

### What about the arena shapes?

The agent sees the arena's bounding box and whether it is circular, and cannot
leave it. Bounding-box-only information is a small handicap for bosses with
non-rectangular arenas (Markoth's ring, Soul Tyrant's rectangle); adding a
shape raster is the first thing to try if a stage stalls — see
[extending.md](extending.md).

### What are "attuned", "ascended" and "radiant"?

Godhome's own difficulty tiers. `--difficulty 1` (default) is attuned;
`3` gives ascended health and damage, `4` radiant damage. The harness sets
`encounter.difficulty`, which is the game's native knob — no simulation is
altered.

### Why does the harness auto-play the menus?

Because a decision to press Z in a menu carries no information for a dodging
policy, and Godhome's turn loop spends 60–75 % of a fight there. Spending
sample budget on it would cut the useful rate by 3–4×. Auto-confirm still
accumulates reward for those frames, so value estimates stay correct.

### Is the simulation deterministic?

The harness runs at a fixed 1/30 s step with no wall-clock dependence and no
rendering, so it is deterministic given a seed — within a single LÖVE build.
Rendering is not part of the simulation. Do not depend on bit-identical
reproducibility across different LÖVE versions or GPUs.

### Why PPO and not SAC / PPO+λ / Dreamer?

Bullet-hell credit assignment is short-horizon and the reward is dense after
shaping, so GAE with `gamma ≈ 0.995` captures almost all of the signal. SAC's
replay buffer would store millions of transitions for little gain and cost a
lot of RAM. Dreamer needs a world model, which is a research project in itself.
PPO's clipped objective is also famously hard to break — which matters for
unattended overnight runs.

### The trainer keeps restarting a harness. Is that a bug?

Occasional restarts are handled automatically and are expected; the log says
`restarted environment N`. Many per *hour* is not expected — see
[troubleshooting.md](troubleshooting.md).

### Can I watch multiple environments at once?

Not in one window. Each harness has its own LÖVE process; run separate `play`
commands for separate models.

### Can I resume after a crash?

Yes. Checkpoints are written every `--save-interval` updates (50 by default)
and `train` picks up `latest.pt` when the observation layout matches. `Ctrl-C`
during training closes the harnesses cleanly and leaves the checkpoint usable.

### What if I change `--grid-w` mid-run?

The checkpoint's layout no longer matches, so `train` starts fresh rather than
loading an incompatible policy. That is deliberate.

### Does it work on Linux and macOS?

Yes. The build, the protocol and the trainer are cross-platform. What you need
there is a LÖVE 11 runtime: either `Godhome-windows.exe` is absent so set
`LOVE_EXE` to your system `love`, or copy the DLL set next to the executable.
`build/runtime/` is assembled from whatever the standalone folder provides.

### Can I use the Godhome GitHub repo instead of the standalone?

Yes, in principle — GodhomeRL needs a folder containing `Godhome.zip` (the
engine plus the mod). Build one by zipping a Kristal checkout with the mod in
`mods/Godhome`. The standalone download is simply the supported way to get one.

### How do I know the agent is not just memorising?

* Evaluate with `--difficulty 3`; a memoriser falls apart.
* Evaluate with `--stochastic`; memorisation relies on the argmax path.
* Look at `graze` in the `eval` table. High graze counts with low hit counts
  mean the policy is actually reading the pattern.

### Is there a way to get better sample efficiency?

Yes, in rough order of effort:

1. More environments (`--n-envs`) — better gradients from the same simulator cost.
2. Better shaped rewards for the specific boss family.
3. Longer rollouts (`--n-steps 256`) so the critic sees more of the fight.
4. A richer threat channel (directional rather than positional).
5. Frame stacking — not needed here, because velocity and time-to-closest-
   approach are in the observation already.