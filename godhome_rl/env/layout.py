"""Observation / action layout.

This mirrors ``lua/rl_obs.lua`` exactly.  The harness reports its real layout in
the handshake, and :func:`verify` asserts that the two agree, so a mismatch can
never silently train a policy on a mis-shaped vector.
"""

from __future__ import annotations

from dataclasses import dataclass

GLOBAL_SIZE = 50
GRID_CHANNELS = 2
BULLET_FEATURES = 10
ACTION_SPACE = 36  # 9 movement directions x attack x focus

# Movement encoding used by the Lua side (0 = stand still).
MOVE_NONE = 0
MOVES = {
    1: (-1, 0),   # left
    2: (1, 0),    # right
    3: (0, -1),   # up
    4: (0, 1),    # down
    5: (-1, -1),  # up-left
    6: (1, -1),   # up-right
    7: (-1, 1),   # down-left
    8: (1, 1),    # down-right
}


def decode_action(action: int) -> tuple[int, bool, bool]:
    """``action -> (move_index, attack, focus)`` -- the inverse of Lua's packing."""
    action = int(action)
    move = action % 9
    attack = (action // 9) % 2 == 1
    focus = (action // 18) % 2 == 1
    return move, attack, focus


@dataclass(frozen=True, slots=True)
class Layout:
    """Resolved sizes for one harness configuration."""

    grid_w: int = 20
    grid_h: int = 15
    max_bullets: int = 24

    @property
    def grid_size(self) -> int:
        return GRID_CHANNELS * self.grid_w * self.grid_h

    @property
    def bullet_size(self) -> int:
        return self.max_bullets * BULLET_FEATURES

    @property
    def dim(self) -> int:
        return GLOBAL_SIZE + self.grid_size + self.bullet_size

    # -- named slices ------------------------------------------------------ #

    @property
    def global_slice(self) -> slice:
        return slice(0, GLOBAL_SIZE)

    @property
    def density_slice(self) -> slice:
        return slice(GLOBAL_SIZE, GLOBAL_SIZE + self.grid_w * self.grid_h)

    @property
    def threat_slice(self) -> slice:
        n = GLOBAL_SIZE + self.grid_w * self.grid_h
        return slice(n, n + self.grid_w * self.grid_h)

    @property
    def bullet_slice(self) -> slice:
        return slice(GLOBAL_SIZE + self.grid_size, self.dim)

    @property
    def n_bullet_slots(self) -> int:
        return self.max_bullets

    def verify(self, handshake) -> None:
        """Raise if the live harness disagrees with this layout."""
        problems = []
        checks = {
            "obs_dim": (self.dim, int(handshake.obs_dim)),
            "actions": (ACTION_SPACE, int(handshake.actions)),
            "grid_w": (self.grid_w, int(handshake.grid_w)),
            "grid_h": (self.grid_h, int(handshake.grid_h)),
            "max_bullets": (self.max_bullets, int(handshake.max_bullets)),
            "global_size": (GLOBAL_SIZE, int(handshake.global_size)),
            "grid_channels": (GRID_CHANNELS, int(handshake.grid_channels)),
            "bullet_features": (BULLET_FEATURES, int(handshake.bullet_features)),
        }
        for key, (expected, actual) in checks.items():
            if expected != actual:
                problems.append(f"{key}: python={expected} lua={actual}")
        if problems:
            raise ValueError(
                "observation layout mismatch between Python and the Lua harness:\n  "
                + "\n  ".join(problems)
                + "\nRebuild with `python -m godhome_rl build --force`."
            )


#: Indices into the info vector the harness sends alongside each step.
INFO_FRAMES = 0
INFO_BOSS_FRAC = 1
INFO_HP = 2
INFO_MAX_HP = 3
INFO_DAMAGE_DEALT = 4
INFO_DAMAGE_TAKEN = 5
INFO_GRAZES = 6
INFO_HITS = 7
INFO_OUTCOME = 8
INFO_BOSS_TOTAL = 9
INFO_ENGINE_GRAZES = 10
INFO_OVER_REASON = 11
INFO_SIZE = 16
#: Indices 12 and 13 are reserved (the harness does not populate them).
INFO_TENSION = 12
INFO_WAVE_TIMER = 13
INFO_N_BULLETS = 14
#: Decision frames consumed so far in this episode -- i.e. the frames the
#: policy actually saw, excluding auto-confirmed menus/dialogue.
INFO_AGENT_FRAMES = 15