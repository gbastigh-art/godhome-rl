"""Actor-critic network.

A deliberately small MLP: bullet-hell dodging is a dense, low-dimensional
control problem once the observation is structured, and on a 4-core CPU a big
network would eat the budget that the simulators need.

The observation is split into three regions (global scalars, spatial grid, bullet
list) which get their own small encoders before being concatenated -- this keeps
the bullet features from drowning the few global scalars that matter.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from ..env.layout import Layout


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias: float = 0.0) -> nn.Linear:
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias)
    return layer


class RunningNorm(nn.Module):
    """Welford-style running mean/std normalisation, kept in float64."""

    def __init__(self, dim: int, clip: float = 10.0, epsilon: float = 1e-8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim, dtype=torch.float64))
        self.register_buffer("var", torch.ones(dim, dtype=torch.float64))
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))
        self.clip = clip
        self.eps = epsilon

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x = x.to(torch.float64)
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), dtype=torch.float64)

        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Cache the float32 view of the statistics: recasting the float64
        # buffers on every call showed up as ~30 % of the forward pass.
        if self._cached is None or self._cache_version != self.count:
            self._cached = (
                self.mean.to(torch.float32),
                (self.var + self.eps).sqrt().to(torch.float32),
            )
            self._cache_version = self.count.clone()
        mean, std = self._cached
        return ((x - mean) / std).clamp_(-self.clip, self.clip)

    _cached = None
    _cache_version = None


class ActorCritic(nn.Module):
    """Categorical policy + value head with structured inputs."""

    def __init__(
        self,
        layout: Layout,
        hidden: int = 256,
        depth: int = 3,
        use_norm: bool = True,
        orthogonal_gain: float = 0.01,
    ) -> None:
        super().__init__()
        self.layout = layout
        self.obs_dim = layout.dim
        self.n_actions = 36
        self.use_norm = use_norm

        g = layout.global_slice
        d = layout.density_slice
        t = layout.threat_slice
        b = layout.bullet_slice
        self._slices = (g, d, t, b)

        self.norm = RunningNorm(layout.dim) if use_norm else None

        n_global = g.stop - g.start
        n_grid = d.stop - d.start + (t.stop - t.start)
        n_bullet = b.stop - b.start

        # 1x1-style projections of the grid/bullet blocks, then a shared MLP.
        self.grid_encoder = nn.Sequential(
            layer_init(nn.Linear(n_grid, 64)),
            nn.Tanh(),
        )
        self.bullet_encoder = nn.Sequential(
            layer_init(nn.Linear(n_bullet, 128)),
            nn.Tanh(),
        )
        self.global_encoder = nn.Sequential(
            layer_init(nn.Linear(n_global, 64)),
            nn.Tanh(),
        )
        feat = 64 + 128 + 64

        body: list[nn.Module] = []
        last = feat
        for _ in range(max(0, depth - 1)):
            body.append(layer_init(nn.Linear(last, hidden)))
            body.append(nn.Tanh())
            last = hidden
        self.body = nn.Sequential(*body) if body else nn.Identity()

        self.actor = layer_init(nn.Linear(last, self.n_actions), std=orthogonal_gain)
        self.critic = layer_init(nn.Linear(last, 1), std=1.0)

    # -- forward ------------------------------------------------------------ #

    def _encode(self, obs: torch.Tensor) -> torch.Tensor:
        g, d, t, b = self._slices
        gf = self.global_encoder(obs[..., g])
        df = self.grid_encoder(torch.cat([obs[..., d], obs[..., t]], dim=-1))
        bf = self.bullet_encoder(obs[..., b])
        return torch.cat([gf, df, bf], dim=-1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.norm is not None:
            obs = self.norm(obs)
        h = self.body(self._encode(obs))
        return self.actor(h), self.critic(h).squeeze(-1)

    # -- distribution helpers ----------------------------------------------- #
    #
    # torch.distributions.Categorical costs ~370 us per call for a 36-way
    # distribution on CPU (it routes through torch.multinomial), which is more
    # than the entire game step.  Doing the softmax / logsumexp / gather by hand
    # is ~15x faster and numerically identical.

    def _logprobs_and_entropy(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logz = torch.logsumexp(logits, dim=-1, keepdim=True)
        logp = logits - logz
        probs = logp.exp()
        entropy = -(probs * logp).sum(-1)
        return logp.squeeze(-1), entropy

    def act(self, obs: torch.Tensor, deterministic: bool = False):
        logits, value = self.forward(obs)
        if deterministic:
            action = torch.argmax(logits, dim=-1)
            return action, value, torch.zeros_like(value), logits
        # Gumbel-max sampling: exact categorical sampling without multinomial.
        u = torch.rand_like(logits).clamp_(1e-9, 1.0 - 1e-9)
        gumbel = logits - torch.log(-torch.log(u))
        action = torch.argmax(gumbel, dim=-1)
        logp = torch.gather(logits, -1, action.unsqueeze(-1)).squeeze(-1)
        logp = logp - torch.logsumexp(logits, dim=-1)
        return action, value, logp, logits

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self.forward(obs)
        logp_all, entropy = self._logprobs_and_entropy(logits)
        logp = torch.gather(logp_all, -1, actions.unsqueeze(-1)).squeeze(-1)
        return value, logp, entropy.mean()

# -- (de)serialisation -------------------------------------------------- #

    def save(self, path) -> None:
        torch.save(
            {
                "model": self.state_dict(),
                "layout": (self.layout.grid_w, self.layout.grid_h, self.layout.max_bullets),
                "hidden": self.actor.in_features,
                "use_norm": self.use_norm,
            },
            path,
        )

    @classmethod
    def load(cls, path, map_location: str = "cpu") -> "ActorCritic":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        layout = Layout(*blob["layout"])
        model = cls(layout, hidden=int(blob["hidden"]), use_norm=bool(blob.get("use_norm", True)))
        model.load_state_dict(blob["model"])
        model.eval()
        return model