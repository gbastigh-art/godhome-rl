"""Environment layer: the LÖVE harness, the wire protocol and the Gym-like APIs.

* :class:`GodhomeEnv` -- one boss fight in one LÖVE process
* :class:`VecGodhome` -- several of them stepped together for throughput
* :class:`Layout`    -- the observation layout, verified against the live harness
"""

from .godhome_env import EnvStats, GodhomeEnv
from .harness import Harness, HarnessConfig, HarnessError
from .layout import ACTION_SPACE, INFO_SIZE, Layout, decode_action
from .rewards import RewardWeights
from .vec_env import VecGodhome

__all__ = [
    "ACTION_SPACE",
    "EnvStats",
    "GodhomeEnv",
    "Harness",
    "HarnessConfig",
    "HarnessError",
    "INFO_SIZE",
    "Layout",
    "RewardWeights",
    "VecGodhome",
    "decode_action",
]