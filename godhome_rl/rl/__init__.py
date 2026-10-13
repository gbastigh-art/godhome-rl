"""Algorithms: a structured actor-critic and a PPO learner.

* :class:`ActorCritic` -- categorical policy + value head over the structured
  observation, with a running observation normaliser
* :class:`PPO`         -- clipped-surrogate PPO with GAE, value clipping and a KL
  early-stop guard
"""

from .model import ActorCritic, RunningNorm
from .ppo import PPO, PPOConfig

__all__ = ["ActorCritic", "PPO", "PPOConfig", "RunningNorm"]