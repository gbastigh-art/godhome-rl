"""Single Godhome boss-fight environment (Gym-like, no Gym dependency).

One :class:`GodhomeEnv` owns one LÖVE harness process.  :mod:`godhome_rl.env.vec_env`
stacks several of them for throughput.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .harness import Harness, HarnessConfig, HarnessError
from .layout import INFO_BOSS_FRAC, INFO_FRAMES, INFO_OUTCOME, Layout
from .proto import EpisodeSpec


@dataclass(slots=True)
class EnvStats:
    episodes: int = 0
    wins: int = 0
    deaths: int = 0
    timeouts: int = 0
    steps: int = 0
    frames: int = 0
    best_boss_progress: float = 0.0
    last_return: float = 0.0
    last_boss: str = ""


class GodhomeEnv:
    """A Godhome (Hollow Knight) boss fight as an RL environment.

    Action space (36 discrete)::

        action = move + 9 * attack + 18 * focus

    where ``move`` is 0 (stand still) or 1..8 for the eight directions, and
    ``attack`` / ``focus`` are the two Godhome verbs (Z = nail swing,
    X = focus, which slows the soul at the cost of tension).

    The harness auto-confirms every menu, dialogue box and attack animation, so
    a step always corresponds to a frame where the soul is under player control.
    """

    def __init__(
        self,
        *,
        bosses: list[str],
        max_steps: int = 1800,
        difficulty: int = 1,
        variant: str = "attuned",
        layout: Layout | None = None,
        seed: int = 0,
        grid_w: int = 20,
        grid_h: int = 15,
        max_bullets: int = 24,
        reward: object | None = None,
        verbose: bool = False,
        render: bool = False,
    ) -> None:
        self.bosses = list(bosses)
        if not self.bosses:
            raise ValueError("at least one boss encounter id is required")
        self.layout = layout or Layout(grid_w, grid_h, max_bullets)
        self.max_steps = max_steps
        self.difficulty = difficulty
        self.variant = variant
        self.seed = seed
        self.render = render
        self.stats = EnvStats()
        self.restarts = 0
        self.restart_log: list[str] = []
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        self._episode_index = 0
        self._episode_return = 0.0
        self._last_frames = 0
        self._last_info = np.zeros(16, dtype=np.float32)

        from .rewards import RewardWeights

        self.reward_weights = reward or RewardWeights()

        self.harness = Harness(
            HarnessConfig(
                grid_w=self.layout.grid_w,
                grid_h=self.layout.grid_h,
                max_bullets=self.layout.max_bullets,
                bosses=self.bosses,
                max_steps=max_steps,
                render=render,
                **self.reward_weights.as_kwargs(),
            ),
            verbose=verbose,
        )
        self.harness.start()
        self.layout.verify(self.harness.handshake)
        self.obs_dim = int(self.harness.handshake.obs_dim)
        self.n_actions = int(self.harness.handshake.actions)
        self._rng_counter = 0
        self._last_frames = 0

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        self.harness.close()

    def __enter__(self) -> "GodhomeEnv":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- bookkeeping -------------------------------------------------------- #

    def _next_boss(self) -> tuple[str, int]:
        idx = self._episode_index % len(self.bosses)
        self._episode_index += 1
        return self.bosses[idx], idx

    def reset(self, boss: str | None = None, **kwargs) -> tuple[np.ndarray, dict]:
        if boss is None:
            boss, idx = self._next_boss()
        else:
            idx = self.bosses.index(boss) if boss in self.bosses else -1
        self._rng_counter += 1
        seed = int(self.rng.integers(0, 2**31 - 1))
        self.stats.last_boss = boss
        self.stats.episodes += 1
        try:
            step = self.harness.start_episode(
                EpisodeSpec(
                    boss_id=boss,
                    boss_index=idx,
                    seed=seed,
                    max_steps=self.max_steps,
                    difficulty=float(self.difficulty),
                    variant=self.variant,
                )
            )
        except HarnessError as exc:
            self.restart(str(exc))
            step = self.harness.start_episode(
                EpisodeSpec(boss_id=boss, boss_index=idx, seed=seed,
                            max_steps=self.max_steps)
            )
        self._episode_return = 0.0
        self._last_frames = int(step.info[INFO_FRAMES]) if step.info.size else 0
        self._last_info = np.asarray(step.info, dtype=np.float32)
        return self._obs(step), self._info(step)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        try:
            step = self.harness.step(int(action))
        except HarnessError as exc:
            return self._recover(str(exc))
        return self._finish(step)

    def _recover(self, reason: str) -> tuple[np.ndarray, float, bool, bool, dict]:
        self.restart(reason)
        obs, _ = self.reset()
        return obs, 0.0, False, False, {
            "info": np.zeros(16, dtype=np.float32),
            "episode_end": False,
            "boss": self.stats.last_boss,
            "restarted": True,
        }

    # -- pipelined API (used by VecGodhome) ---------------------------------- #

    def send_action(self, action: int) -> None:
        self.harness.send_action(int(action))

    def recv(self) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Collect the result of a previously sent action."""
        try:
            return self._finish(self.harness.recv())
        except HarnessError as exc:
            return self._recover(str(exc))

    def restart(self, reason: str = "") -> None:
        """Rebuild the underlying harness after a crash.

        A handful of Godhome encounters occasionally fault inside LÖVE; losing
        a whole training run to one of them would be far worse than the few
        seconds a restart costs, so the environment heals itself and the
        trainer only ever sees a discarded step.
        """
        self.restarts += 1
        self.restart_log.append(reason.strip().splitlines()[-1:] or [""])
        if len(self.restart_log) > 50:
            self.restart_log = self.restart_log[-20:]
        try:
            self.harness.close()
        except Exception:  # pragma: no cover - best effort
            pass
        self.harness = Harness(
            HarnessConfig(
                grid_w=self.layout.grid_w,
                grid_h=self.layout.grid_h,
                max_bullets=self.layout.max_bullets,
                bosses=self.bosses,
                max_steps=self.max_steps,
                render=self.render,
                **self.reward_weights.as_kwargs(),
            ),
            verbose=self.verbose,
        )
        self.harness.start()
        self.layout.verify(self.harness.handshake)

    def _finish(self, step) -> tuple[np.ndarray, float, bool, bool, dict]:
        reward = float(step.reward)
        self._episode_return += reward
        self.stats.steps += 1
        if step.info.size:
            frames = int(step.info[INFO_FRAMES])
            # info[0] is cumulative within the episode; keep a delta.
            if frames >= self._last_frames:
                self.stats.frames += frames - self._last_frames
            else:
                self.stats.frames += frames
            self._last_frames = frames
            self._last_info = np.asarray(step.info, dtype=np.float32)

        terminal = bool(step.flags & 0b0001)
        truncated = bool(step.flags & 0b010)
        episode_end = bool(step.flags & 0b100)

        if episode_end:
            won = bool(step.info.size > INFO_OUTCOME and step.info[INFO_OUTCOME] > 0)
            self.stats.last_return = self._episode_return
            self.stats.best_boss_progress = max(
                self.stats.best_boss_progress,
                1.0 - float(step.info[INFO_BOSS_FRAC]) if step.info.size > INFO_BOSS_FRAC else 0.0,
            )
            if won:
                self.stats.wins += 1
            elif truncated:
                self.stats.timeouts += 1
            else:
                self.stats.deaths += 1

        return (
            self._obs(step),
            reward,
            bool(terminal and not truncated),
            bool(truncated),
            self._info(step, episode_end),
        )

    # -- helpers ------------------------------------------------------------ #

    def _obs(self, step) -> np.ndarray:
        obs = step.obs
        if obs.size != self.obs_dim:
            raise ValueError(f"expected obs_dim={self.obs_dim}, got {obs.size}")
        # ``np.frombuffer`` gives a read-only view; copy so torch is happy and
        # the buffer can outlive the next recv.
        return np.array(obs, dtype=np.float32)

    def _info(self, step, episode_end: bool = False) -> dict:
        return {
            "info": np.asarray(step.info, dtype=np.float32),
            "episode_end": episode_end,
            "boss": self.stats.last_boss,
        }

    # -- diagnostics -------------------------------------------------------- #

    def snapshot(self) -> dict:
        e = self.stats.episodes or 1
        return {
            "episodes": self.stats.episodes,
            "wins": self.stats.wins,
            "deaths": self.stats.deaths,
            "timeouts": self.stats.timeouts,
            "win_rate": self.stats.wins / e,
            "steps": self.stats.steps,
            "frames": self.stats.frames,
            "mean_return": self.stats.last_return,
            "boss_progress": self.stats.best_boss_progress,
        }