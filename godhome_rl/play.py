"""Play a trained agent inside the real Godhome window.

``python -m godhome_rl play --model runs/godhome/checkpoints/latest.pt --boss radiance``

The game window is opened by the same harness the trainer uses; the only
difference is that it renders and paces to wall-clock time, so what you see is
exactly what the policy would do in a normal playthrough.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .bosses import display
from .env.harness import Harness, HarnessConfig
from .env.layout import Layout
from .env.proto import EpisodeSpec
from .rl.model import ActorCritic


def play(
    model_path: Path,
    bosses: list[str],
    *,
    episodes: int = 0,
    max_steps: int = 1800,
    seed: int = 0,
    greedy: bool = True,
    render_every: int = 1,
    quiet: bool = False,
) -> dict:
    blob = torch.load(model_path, map_location="cpu", weights_only=False)
    layout = Layout(*blob["layout"])
    model = ActorCritic(layout, hidden=int(blob["hidden"]), use_norm=bool(blob["use_norm"]))
    model.load_state_dict(blob["model"])
    model.eval()

    harness = Harness(
        HarnessConfig(
            grid_w=layout.grid_w,
            grid_h=layout.grid_h,
            max_bullets=layout.max_bullets,
            bosses=bosses,
            max_steps=max_steps,
            render=True,
            render_every=render_every,
        )
    )
    rng = np.random.default_rng(seed)
    totals = {"episodes": 0, "wins": 0, "frames": 0, "steps": 0}

    try:
        harness.start()
        layout.verify(harness.handshake)
        for ep in range(episodes if episodes > 0 else 10**9):
            boss = bosses[ep % len(bosses)]
            step = harness.start_episode(
                EpisodeSpec(boss_id=boss, seed=int(rng.integers(0, 2**31 - 1)),
                            max_steps=max_steps)
            )
            n = 0
            while not step.episode_end:
                obs = torch.as_tensor(
                    np.array(step.obs, dtype=np.float32, copy=True)
                ).unsqueeze(0)
                with torch.no_grad():
                    action, _, _, _ = model.act(obs, deterministic=greedy)
                step = harness.step(int(action.item()))
                n += 1
            won = bool(step.info.size > 8 and step.info[8] > 0)
            totals["episodes"] += 1
            totals["steps"] += n
            totals["frames"] += int(step.info[0]) if step.info.size else 0
            totals["wins"] += int(won)
            if not quiet:
                result = "WIN " if won else ("LOSS" if step.info[0] < max_steps else "TIME")
                print(
                    f"[play] {result} {display(boss):<28} "
                    f"frames={int(step.info[0]):>5} boss_hp={step.info[1]:.2f} "
                    f"hp={step.info[2]:.0f}/{step.info[3]:.0f} "
                    f"grazes={int(step.info[6])} hits={int(step.info[7])}",
                    flush=True,
                )
    except KeyboardInterrupt:
        if not quiet:
            print("\n[play] interrupted")
    finally:
        harness.close()

    if totals["episodes"]:
        print(
            f"[play] {totals['wins']}/{totals['episodes']} wins "
            f"({100 * totals['wins'] / totals['episodes']:.0f}%), "
            f"{totals['frames']:,} game frames"
        )
    return totals


def observe_random(
    bosses: list[str],
    *,
    episodes: int = 3,
    max_steps: int = 1800,
    render_every: int = 2,
    seed: int = 0,
) -> None:
    """Watch the harness play itself with a random policy (sanity check)."""
    from .env.godhome_env import GodhomeEnv

    rng = np.random.default_rng(seed)
    env = GodhomeEnv(bosses=bosses, max_steps=max_steps, render=True)
    try:
        for _ in range(episodes):
            obs, _ = env.reset()
            while True:
                obs, reward, terminated, truncated, info = env.step(int(rng.integers(0, 36)))
                if terminated or truncated:
                    break
    finally:
        env.close()