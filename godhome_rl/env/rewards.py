"""Reward shaping for Godhome boss fights.

The shaping is deliberately *dense and bounded*: every term is expressed as a
fraction of the fight's own progress (boss health pool) or the player's max HP,
so the same weights transfer across bosses of very different size.

Defaults are tuned so a typical successful fight returns roughly ``+5`` and a
typical death roughly ``-3``, which keeps the advantage estimates on a sane
scale without any per-boss retuning.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RewardWeights:
    """Weights handed to the Lua harness via ``GODHOME_RL``."""

    #: Per frame survived during a bullet pattern.  0.006 * 30 = 0.18 / second.
    alive: float = 0.006
    #: Per point of *max-HP* lost.  One full health bar costs 1.2.
    damage: float = 1.2
    #: Per unit of the boss's total health pool removed (armour included).
    damage_dealt: float = 1.0
    #: Per graze.  Encourages the aggressive play that builds tension for Fury
    #: and the Unbreakable spells.
    graze: float = 0.012
    #: Per frame spent focusing.  Focusing is excellent for tight dodges but
    #: must not become a way to stall forever.
    focus: float = 0.0008
    #: Terminal bonuses.
    win: float = 4.0
    death: float = -3.0
    timeout: float = -0.75

    def as_kwargs(self) -> dict:
        return {
            "r_alive": self.alive,
            "r_damage": self.damage,
            "r_damage_dealt": self.damage_dealt,
            "r_graze": self.graze,
            "r_focus": self.focus,
            "r_win": self.win,
            "r_death": self.death,
            "r_timeout": self.timeout,
        }

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict | None) -> "RewardWeights":
        if not data:
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__slots__}
        return cls(**known)