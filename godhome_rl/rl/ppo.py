"""Proximal Policy Optimization tuned for this environment.

Why PPO and not something fancier:

* The simulator is the bottleneck (~1.5 ms per decision step), not the
  gradient step, so a method that squeezes more sample efficiency out of each
  transition matters -- but only if the update itself stays cheap.
* Bullet-hell credit assignment is short-horizon (a hit costs you ~1 s of
  progress, a win pays out immediately), so GAE with a modest ``gamma`` is a
  good fit and long-horizon methods add nothing.
* PPO's clipped objective is extremely robust with a single reward scale and
  no privileged critic target, which matters when the reward shaping has to be
  hand-tuned per boss family.

Everything is plain PyTorch on the CPU by default so the simulators get the
cores.  ``device="cuda"`` works if you have one.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .model import ActorCritic, check_obs_format


@dataclass(slots=True)
class PPOConfig:
    # optimisation
    lr: float = 3e-4
    anneal_lr: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.012
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03

    # rollout
    n_envs: int = 3
    n_steps: int = 128            # per env, per iteration
    n_epochs: int = 4
    n_minibatches: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    normalize_adv: bool = True

    # bookkeeping
    log_interval: int = 1
    save_interval: int = 50
    seed: int = 0
    device: str = "auto"

    def to_dict(self) -> dict:
        return asdict(self)


class PPO:
    """A single-policy PPO learner."""

    def __init__(self, ac: ActorCritic, cfg: PPOConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(
            cfg.device
            if cfg.device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        if self.device.type == "cpu":
            # One torch thread per process: the LÖVE simulators are the real
            # bottleneck and thread oversubscription costs more than it buys.
            try:
                torch.set_num_threads(1)
            except Exception:  # pragma: no cover - torch build dependent
                pass
        self.ac = ac.to(self.device)
        self.optimizer = torch.optim.Adam(self.ac.parameters(), lr=cfg.lr, eps=1e-5)
        self.n_updates = 0
        self._last_lr = cfg.lr

    # -- rollout storage --------------------------------------------------- #

    def allocate(self, obs_dim: int, n_actions: int) -> dict:
        c = self.cfg
        T, N = c.n_steps, c.n_envs
        dev = self.device
        return {
            "obs": torch.zeros((T, N, obs_dim), dtype=torch.float32, device=dev),
            "actions": torch.zeros((T, N), dtype=torch.long, device=dev),
            "logprobs": torch.zeros((T, N), dtype=torch.float32, device=dev),
            "rewards": torch.zeros((T, N), dtype=torch.float32, device=dev),
            "dones": torch.zeros((T, N), dtype=torch.float32, device=dev),
            "values": torch.zeros((T, N), dtype=torch.float32, device=dev),
        }

    def begin_episode(self) -> dict:
        c = self.cfg
        return self.allocate(self.ac.obs_dim, self.ac.n_actions)

    # -- learning ---------------------------------------------------------- #

    @torch.no_grad()
    def collect(self, rollout: dict, env, next_obs: np.ndarray) -> tuple[dict, np.ndarray, dict]:
        """Run one full rollout.  ``next_obs`` is the observation carried over
        from the previous iteration so episodes are never restarted."""
        c = self.cfg
        T, N = c.n_steps, c.n_envs
        obs_np = next_obs
        infos: list[list[dict]] = [[] for _ in range(T)]

        for t in range(T):
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
            action, value, logp, _ = self.ac.act(obs_t)
            actions_np = action.cpu().numpy()

            obs_np, rewards, dones, step_infos = env.step(actions_np)

            rollout["obs"][t] = obs_t
            rollout["actions"][t] = action
            rollout["logprobs"][t] = logp
            rollout["rewards"][t] = torch.as_tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            rollout["dones"][t] = torch.as_tensor(
                dones.astype(np.float32), dtype=torch.float32, device=self.device
            )
            rollout["values"][t] = value
            infos[t] = step_infos

        return rollout, obs_np, {"infos": infos}

    def compute_returns(self, rollout: dict, last_obs: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        c = self.cfg
        T, N = c.n_steps, c.n_envs
        with torch.no_grad():
            last_value = self.ac(
                torch.as_tensor(last_obs, dtype=torch.float32, device=self.device)
            )[1]

        advantages = torch.zeros_like(rollout["rewards"])
        last_gae = torch.zeros(N, dtype=torch.float32, device=self.device)
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - rollout["dones"][t]
            next_value = last_value if t == T - 1 else rollout["values"][t + 1]
            delta = (
                rollout["rewards"][t]
                + c.gamma * next_value * next_non_terminal
                - rollout["values"][t]
            )
            last_gae = delta + c.gamma * c.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + rollout["values"]
        return advantages, returns

    def update(self, rollout: dict, advantages: torch.Tensor, returns: torch.Tensor) -> dict:
        c = self.cfg
        T, N = c.n_steps, c.n_envs
        B = T * N
        mb_size = max(1, B // c.n_minibatches)

        obs = rollout["obs"].reshape((B, -1))
        actions = rollout["actions"].reshape(B)
        old_logprobs = rollout["logprobs"].reshape(B)
        old_values = rollout["values"].reshape(B)
        adv = advantages.reshape(B)
        ret = returns.reshape(B)

        if c.normalize_adv:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        # Refresh the observation statistics once per update rather than once per
        # minibatch: the float64 pass over the whole batch is what it costs, and
        # doing it 16 times per update showed up in the profile.
        norm = getattr(self.ac, "norm", None)
        if norm is not None:
            with torch.no_grad():
                norm.update(obs)

        idx = np.arange(B)
        clip_fracs: list[float] = []
        losses: dict[str, float] = {}
        approx_kl = 0.0
        stop = False

        for epoch in range(c.n_epochs):
            np.random.shuffle(idx)
            for start in range(0, B, mb_size):
                mb = idx[start : start + mb_size]
                mb_t = torch.as_tensor(mb, dtype=torch.long, device=self.device)

                mb_obs = obs[mb_t]

                values, logp, entropy = self.ac.evaluate_actions(mb_obs, actions[mb_t])

                log_ratio = logp - old_logprobs[mb_t]
                ratio = log_ratio.exp()

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > c.clip_coef).float().mean().item()

                mb_adv = adv[mb_t]
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - c.clip_coef, 1 + c.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (values - ret[mb_t]) ** 2
                v_loss_clipped = old_values[mb_t] + torch.clamp(
                    values - old_values[mb_t], -c.clip_coef, c.clip_coef
                ) - ret[mb_t]
                v_loss = (
                    0.5 * torch.max(v_loss_unclipped, v_loss_clipped ** 2).mean()
                    if c.clip_vloss
                    else 0.5 * v_loss_unclipped.mean()
                )
                ent_loss = entropy.mean()

                loss = pg_loss + c.vf_coef * v_loss - c.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), c.max_grad_norm)
                self.optimizer.step()

                clip_fracs.append(clipfrac)
                losses = {
                    "policy_loss": pg_loss.item(),
                    "value_loss": v_loss.item(),
                    "entropy": ent_loss.item(),
                    "total_loss": loss.item(),
                }

            if c.target_kl is not None and approx_kl > c.target_kl:
                stop = True
                break

        self.n_updates += 1
        losses["approx_kl"] = approx_kl
        losses["clipfrac"] = float(np.mean(clip_fracs)) if clip_fracs else 0.0
        losses["explained_variance"] = explained_variance(rollout["values"], returns)
        losses["stopped_early"] = float(stop)
        losses["lr"] = self._last_lr
        return losses

    # -- schedule ---------------------------------------------------------- #

    def anneal(self, progress: float, total: int) -> float:
        """Linear LR schedule in ``[lr*0.1, lr]`` over ``total`` updates."""
        c = self.cfg
        if not c.anneal_lr:
            self._last_lr = c.lr
        else:
            frac = 1.0 - (progress / max(1, total))
            self._last_lr = frac * (c.lr - c.lr * 0.1) + c.lr * 0.1
        for group in self.optimizer.param_groups:
            group["lr"] = self._last_lr
        return self._last_lr

    # -- io ----------------------------------------------------------------- #

    def save(self, path) -> None:
        self.ac.save(path)

    def load(self, path) -> None:
        blob = torch.load(path, map_location=self.device, weights_only=False)
        check_obs_format(blob, path)
        self.ac.load_state_dict(blob["model"])


def explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    y = returns.cpu().numpy()
    yhat = values.cpu().numpy()
    var_y = float(np.var(y))
    if var_y == 0:
        return 0.0
    return float(1.0 - np.var(y - yhat) / var_y)