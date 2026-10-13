"""A tiny synchronous vectorised environment over several LÖVE harness processes.

The harnesses run as independent OS processes, so while the trainer waits on one
environment's socket the others are already computing.  On a 4-core laptop three
harnesses reach roughly 2 000 decision steps per second (~20x the in-game frame
rate, i.e. tens of thousands of simulated seconds per wall-clock second).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .godhome_env import GodhomeEnv
from .layout import Layout
from .rewards import RewardWeights


@dataclass(slots=True)
class RolloutBatch:
    """Flat, ready-for-PPO arrays for ``n_envs * rollout_len`` transitions."""

    obs: np.ndarray            # (T, N, obs_dim) float32
    actions: np.ndarray        # (T, N) int64
    rewards: np.ndarray        # (T, N) float32
    dones: np.ndarray          # (T, N) bool  (true = episode boundary)
    values: np.ndarray         # (T, N) float32
    logprobs: np.ndarray       # (T, N) float32
    infos: list                # per-step list of dicts (length T)


class VecGodhome:
    """``n_envs`` Godhome fights, stepped together."""

    def __init__(
        self,
        n_envs: int = 3,
        *,
        bosses: list[str],
        max_steps: int = 1800,
        difficulty: int = 1,
        variant: str = "attuned",
        layout: Layout | None = None,
        reward: RewardWeights | None = None,
        seed: int = 0,
        verbose: bool = False,
        render: bool = False,
        stagger: bool = True,
        quiet: bool = False,
    ) -> None:
        self.n_envs = n_envs
        self.layout = layout or Layout()
        self.envs: list[GodhomeEnv] = []
        errors: list[str] = []
        for i in range(n_envs):
            try:
                self.envs.append(
                    GodhomeEnv(
                        bosses=bosses,
                        max_steps=max_steps,
                        difficulty=difficulty,
                        variant=variant,
                        layout=self.layout,
                        seed=seed + 1000 * i,
                        reward=reward,
                        verbose=verbose,
                        render=render,
                    )
                )
            except Exception as exc:  # pragma: no cover - startup diagnostics
                errors.append(f"env {i}: {exc}")
        if not self.envs:
            raise RuntimeError(
                "no harness started:\n  " + "\n  ".join(errors)
            )
        if errors:
            print(f"[vec_env] started {len(self.envs)}/{n_envs} harnesses")
            for err in errors:
                print(f"[vec_env] {err}")

        self.obs_dim = self.envs[0].obs_dim
        self.n_actions = self.envs[0].n_actions
        self._rng = np.random.default_rng(seed)
        self._obs = np.zeros((len(self.envs), self.obs_dim), dtype=np.float32)
        self._ep_returns = np.zeros(len(self.envs), dtype=np.float64)
        self._ep_lengths = np.zeros(len(self.envs), dtype=np.int64)
        self._last_info_frames = np.zeros(len(self.envs), dtype=np.int64)
        self._wait_times = np.zeros(len(self.envs), dtype=np.float64)
        self._short = np.zeros(len(self.envs), dtype=np.int64)
        #: A "short" episode ends sooner than this many decision steps.
        self.short_episode_threshold = 30
        #: ...and after this many in a row we consider the harness wedged.
        self.max_short_episodes = 8
        self.quiet = quiet
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.total_steps = 0
        self.total_frames = 0
        self._start_time = time.time()
        self.stagger = stagger

        # Environments boot sequentially; give each a slightly different first
        # boss so they desynchronise instead of all running the same fight.
        for i, env in enumerate(self.envs):
            boss = bosses[(i + self._rng.integers(0, len(bosses))) % len(bosses)]
            obs, _ = env.reset(boss=boss)
            self._obs[i] = obs

    # -- properties --------------------------------------------------------- #

    @property
    def sps(self) -> float:
        """Decision steps per second across all environments."""
        elapsed = max(1e-6, time.time() - self._start_time)
        return self.total_steps / elapsed

    @property
    def fps(self) -> float:
        """Simulated game frames per second across all environments."""
        elapsed = max(1e-6, time.time() - self._start_time)
        return self.total_frames / elapsed

    # -- core API ----------------------------------------------------------- #

    def reset(self) -> np.ndarray:
        for i, env in enumerate(self.envs):
            self._obs[i], _ = env.reset()
        return self._obs.copy()

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step every environment.  Returns ``(obs, rewards, dones, infos)``.

        Actions are dispatched to *all* harnesses before any reply is read, so
        the LÖVE processes compute in parallel instead of serialising behind the
        Python loop.  Episodes auto-reset: on a done, the environment's terminal
        observation is kept in ``infos[i]["final_obs"]`` and ``obs[i]`` is the
        reset observation.
        """
        n = len(self.envs)

        # 1. hand every harness its action up-front
        for i, env in enumerate(self.envs):
            env.send_action(int(actions[i]))

        # 2. collect results (this is where the wall-clock is spent waiting)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        infos: list[dict] = []
        frames = 0
        prev_frames = self._last_info_frames

        for i, env in enumerate(self.envs):
            t0 = time.perf_counter()
            try:
                obs, reward, terminated, truncated, info = env.recv()
            except Exception as exc:
                self.recover(i, exc)
                obs = self._obs[i]
                reward = 0.0
                terminated = truncated = False
                info = {
                    "info": np.zeros(16, dtype=np.float32),
                    "episode_end": False,
                    "boss": env.stats.last_boss,
                    "recovered": True,
                }
            waited = time.perf_counter() - t0
            self._wait_times[i] = self._wait_times[i] * 0.9 + waited * 0.1
            rewards[i] = reward
            self._ep_returns[i] += reward
            self._ep_lengths[i] += 1
            self.total_steps += 1
            if info["info"].size:
                cur = int(info["info"][0])
                # info[0] is the *cumulative* episode frame count; convert it to
                # a delta so the throughput metric stays meaningful.
                frames += cur if cur >= prev_frames[i] else cur
                prev_frames[i] = cur

            done = terminated or truncated
            dones[i] = done
            if done:
                info["final_obs"] = obs
                info["return"] = float(self._ep_returns[i])
                info["length"] = int(self._ep_lengths[i])
                info["won"] = bool(info["info"][8] > 0) if info["info"].size > 8 else False
                self.episode_returns.append(float(self._ep_returns[i]))
                self.episode_lengths.append(int(self._ep_lengths[i]))
                if len(self.episode_returns) > 4000:
                    self.episode_returns = self.episode_returns[-2000:]
                    self.episode_lengths = self.episode_lengths[-2000:]
                self._ep_returns[i] = 0.0
                self._ep_lengths[i] = 0
                self._last_info_frames[i] = 0
                obs, _ = env.reset()

                # Stall breaker: an encounter that dies instantly (or a harness
                # wedged in a loop) would otherwise burn the whole run.
                self._short[i] += 1
                if self._short[i] >= self.max_short_episodes:
                    self.recover(
                        i,
                        RuntimeError(
                            f"{self._short[i]} consecutive episodes shorter than "
                            f"{self.short_episode_threshold} steps"
                        ),
                    )
                    self._short[i] = 0
                    obs = self._obs[i]
            elif self._ep_lengths[i] > self.short_episode_threshold:
                self._short[i] = 0
            self._obs[i] = obs
            info["env_index"] = i
            infos.append(info)

        self.total_frames += frames
        return self._obs.copy(), rewards, dones, infos

    # -- lifecycle ---------------------------------------------------------- #

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def recover(self, index: int, reason: Exception) -> None:
        """Replace a wedged or crashed harness in place.

        ``GodhomeEnv`` already restarts itself on transport failures; this is the
        outer safety net for anything that leaves a process alive but not making
        progress (a pathological boss looping, a socket desynchronised).  The
        trainer sees one discarded step and keeps going.
        """
        env = self.envs[index]
        env.restart(repr(reason))
        self._short[index] = 0
        self._ep_returns[index] = 0.0
        self._ep_lengths[index] = 0
        self._last_info_frames[index] = 0
        obs, _ = env.reset()
        self._obs[index] = obs
        if not self.quiet:
            print(
                f"[vec_env] restarted environment {index}: {reason}",
                flush=True,
            )

    def __enter__(self) -> "VecGodhome":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- telemetry ---------------------------------------------------------- #

    def stats(self) -> dict:
        out = {
            "total_steps": self.total_steps,
            "total_frames": self.total_frames,
            "sps": self.sps,
            "fps": self.fps,
            "n_envs": len(self.envs),
        }
        if self.episode_returns:
            recent = self.episode_returns[-200:]
            out["episode_return"] = float(np.mean(recent))
            out["episode_length"] = float(np.mean(self.episode_lengths[-200:]))
        total = sum(env.stats.episodes for env in self.envs)
        if total:
            out["win_rate"] = sum(env.stats.wins for env in self.envs) / total
            out["episodes"] = total
        out["restarts"] = sum(env.restarts for env in self.envs)
        out["wait_ms"] = [round(float(w) * 1000, 2) for w in self._wait_times]
        return out