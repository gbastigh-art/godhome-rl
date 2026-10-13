"""Training entry point: ``python -m godhome_rl train``.

The loop is intentionally boring and restartable:

* every iteration collects ``n_steps`` transitions from each harness,
* computes GAE advantages,
* runs four PPO epochs,
* writes a checkpoint and a CSV row, and
* advances the curriculum when the current stage is mastered.

State that matters lives in ``runs/<name>/``: ``config.json``, ``metrics.csv``
and ``checkpoints/``.  ``--resume`` picks up from the newest checkpoint.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .bosses import display, resolve_list
from .curriculum import Curriculum, build_curriculum
from .env.layout import Layout
from .env.rewards import RewardWeights
from .env.vec_env import VecGodhome
from .rl.model import ActorCritic
from .rl.ppo import PPO, PPOConfig


@dataclass(slots=True)
class TrainConfig:
    name: str = "godhome"
    out_dir: Path = Path("runs")
    #: Boss spec: ``all`` for the staged ladder, or any catalog selector.
    bosses: str = "all"
    n_envs: int = 3
    total_updates: int = 4000
    max_steps: int = 0              # 0 = use the curriculum stage value
    difficulty: int = 1
    variant: str = "attuned"
    grid_w: int = 20
    grid_h: int = 15
    max_bullets: int = 24
    seed: int = 0
    resume: bool = True
    eval_every: int = 25
    eval_episodes: int = 20
    verbose: bool = False
    reward: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.name

    def to_dict(self) -> dict:
        data = asdict(self)
        data["out_dir"] = str(self.out_dir)
        return data


class Logger:
    """Console + CSV logging, so a run can be plotted with pandas later."""

    FIELDS = (
        "update", "steps", "frames", "sps", "fps", "episode_return",
        "episode_length", "win_rate", "stage", "boss_progress",
        "policy_loss", "value_loss", "entropy", "approx_kl", "clipfrac",
        "explained_variance", "lr", "elapsed_s", "restarts", "episodes",
    )

    def __init__(self, path: Path) -> None:
        self.path = path
        self.t0 = time.time()
        self._file = None
        self._writer = None

    def write(self, row: dict) -> None:
        if self._writer is None:
            new = not self.path.exists()
            self._file = self.path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=list(self.FIELDS))
            if new:
                self._writer.writeheader()
        row = {k: row.get(k, "") for k in self.FIELDS}
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


def _latest_checkpoint(run_dir: Path) -> Path | None:
    ckpts = sorted((run_dir / "checkpoints").glob("*.pt"))
    return ckpts[-1] if ckpts else None


def train(cfg: TrainConfig, ppo_cfg: PPOConfig | None = None, quiet: bool = False) -> Path:
    ppo_cfg = ppo_cfg or PPOConfig(seed=cfg.seed)
    ppo_cfg.n_envs = cfg.n_envs
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    run_dir = cfg.run_dir
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps({"train": cfg.to_dict(), "ppo": ppo_cfg.to_dict()}, indent=2),
        encoding="utf-8",
    )

    layout = Layout(cfg.grid_w, cfg.grid_h, cfg.max_bullets)
    reward = RewardWeights.from_dict(cfg.reward)
    curriculum = build_curriculum(cfg.bosses)
    logger = Logger(run_dir / "metrics.csv")

    start_update = 0
    if cfg.resume:
        latest = _latest_checkpoint(run_dir)
        if latest is not None:
            blob = torch.load(latest, map_location="cpu", weights_only=False)
            if blob.get("layout") == (layout.grid_w, layout.grid_h, layout.max_bullets):
                ac = ActorCritic(layout, hidden=blob["hidden"], use_norm=blob["use_norm"])
                ac.load_state_dict(blob["model"])
                ppo = PPO(ac, ppo_cfg)
                ppo.n_updates = int(blob.get("update", 0))
                start_update = ppo.n_updates
                curriculum.index = int(blob.get("curriculum_index", 0))
                curriculum.episodes_in_stage = int(blob.get("curriculum_episodes", 0))
                if not quiet:
                    print(f"[train] resumed from {latest.name} at update {start_update}")
            else:
                if not quiet:
                    print("[train] checkpoint layout differs; starting fresh")

    if start_update == 0:
        ac = ActorCritic(layout, use_norm=True)
        ppo = PPO(ac, ppo_cfg)

    def say(msg: str) -> None:
        if not quiet:
            print(f"[train] {msg}", flush=True)

    rollout = None
    next_obs = None
    env = None

    try:
        for update in range(start_update, cfg.total_updates):
            stage = curriculum.stage
            stage_max_steps = cfg.max_steps or stage.max_steps

            if env is None:
                env = VecGodhome(
                    cfg.n_envs,
                    bosses=stage.bosses,
                    max_steps=stage_max_steps,
                    difficulty=cfg.difficulty,
                    variant=cfg.variant,
                    layout=layout,
                    reward=reward,
                    seed=cfg.seed + 7919 * update,
                    verbose=cfg.verbose,
                )
                rollout = ppo.begin_episode()
                next_obs = env.reset()
                say(
                    f"started {cfg.n_envs} harness(es) | {curriculum.describe()}"
                )

            ppo.anneal(update, cfg.total_updates)
            rollout, next_obs, meta = ppo.collect(rollout, env, next_obs)

            # Curriculum bookkeeping from this rollout's terminal frames.
            promoted = False
            for infos_at_t in meta["infos"]:
                for info in infos_at_t:
                    if info.get("episode_end"):
                        won = bool(info["info"][8] > 0) if info["info"].size > 8 else False
                        curriculum.record(won)

            advantages, returns = ppo.compute_returns(rollout, next_obs)
            metrics = ppo.update(rollout, advantages, returns)

            stats = env.stats()
            row = {
                "update": update,
                "steps": stats["total_steps"],
                "frames": stats["total_frames"],
                "sps": round(stats["sps"], 1),
                "fps": round(stats["fps"], 1),
                "episode_return": round(stats.get("episode_return", 0.0), 4),
                "episode_length": round(stats.get("episode_length", 0.0), 1),
                "win_rate": round(stats.get("win_rate", 0.0), 4),
                "stage": stage.name,
                "restarts": stats.get("restarts", 0),
                "boss_progress": round(
                    max((e.stats.best_boss_progress for e in env.envs), default=0.0), 4
                ),
                "policy_loss": round(metrics.get("policy_loss", 0.0), 5),
                "value_loss": round(metrics.get("value_loss", 0.0), 5),
                "entropy": round(metrics.get("entropy", 0.0), 5),
                "approx_kl": round(metrics.get("approx_kl", 0.0), 6),
                "clipfrac": round(metrics.get("clipfrac", 0.0), 4),
                "explained_variance": round(metrics.get("explained_variance", 0.0), 4),
                "lr": round(metrics.get("lr", ppo_cfg.lr), 8),
                "elapsed_s": round(time.time() - logger.t0, 1),
            }
            logger.write(row)

            if not quiet:
                print(
                    f"[{update:5d}] {stage.name:<20} steps={row['steps']:>9,} "
                    f"sps={row['sps']:>6.0f} ret={row['episode_return']:>7.3f} "
                    f"win={row['win_rate']:.2f} ent={row['entropy']:.3f} "
                    f"kl={row['approx_kl']:.4f} ev={row['explained_variance']:+.2f} "
                    f"wait={stats.get('wait_ms')}",
                    flush=True,
                )

            if (update + 1) % max(1, ppo_cfg.save_interval) == 0:
                path = run_dir / "checkpoints" / f"update_{update + 1:06d}.pt"
                torch.save(
                    {
                        "model": ppo.ac.state_dict(),
                        "layout": (layout.grid_w, layout.grid_h, layout.max_bullets),
                        "hidden": ppo.ac.actor.in_features,
                        "use_norm": ppo.ac.use_norm,
                        "update": update + 1,
                        "curriculum_index": curriculum.index,
                        "curriculum_episodes": curriculum.episodes_in_stage,
                        "bosses": cfg.bosses,
                    },
                    path,
                )
                ppo.save(run_dir / "checkpoints" / "latest.pt")

            if curriculum.promote():
                promoted = True
                new_stage = curriculum.stage
                say(
                    f"PROMOTED -> {new_stage.name}: "
                    + ", ".join(display(b) for b in new_stage.bosses[:6])
                    + (" ..." if len(new_stage.bosses) > 6 else "")
                )
                env.close()
                env = None

            if promoted or cfg.eval_every <= 0:
                continue
    finally:
        if env is not None:
            env.close()
        logger.close()

    say(f"done. artifacts in {run_dir}")
    return run_dir