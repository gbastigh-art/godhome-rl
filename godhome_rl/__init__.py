"""GodhomeRL -- a reinforcement-learning agent that plays the Godhome mod.

Public surface:

* :func:`godhome_rl.train.train`       -- PPO training loop
* :func:`godhome_rl.play.play`         -- watch a policy in the real game window
* :class:`godhome_rl.env.GodhomeEnv`   -- single-environment Gym-like API
* :class:`godhome_rl.env.VecGodhome`   -- several harnesses stepped together
* :mod:`godhome_rl.bosses`             -- the Godhome boss catalog
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]