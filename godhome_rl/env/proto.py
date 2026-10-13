"""Python side of the GodhomeRL wire protocol.

Every field here mirrors ``lua/rl_proto.lua`` exactly.  Messages are framed as::

    [uint8 tag][uint32 length][payload]

The harness never sends a message larger than ~64 KiB, so a single blocking
``recv_exact`` per message is safe and fast.
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# Tags
# --------------------------------------------------------------------------- #

MSG_ACT = 0xA1  # python -> lua : action for the next engine step
MSG_EPISODE = 0xA2  # python -> lua : start a new episode
MSG_QUIT = 0xA3  # python -> lua : shut down
MSG_PONG = 0xA4  # python -> lua : reply

MSG_READY = 0xB1  # lua -> python : handshake complete
MSG_STEP = 0xB2  # lua -> python : observation frame
MSG_LOG = 0xB3  # lua -> python : diagnostics
MSG_PING = 0xB4  # lua -> python : liveness probe

# flags carried in MSG_STEP
FLAG_TERMINAL = 1
FLAG_TRUNCATED = 2
FLAG_EPISODE = 4
FLAG_RESET = 8

_HDR = struct.Struct("<BxxxI")
_HDR_SIZE = _HDR.size  # 8
_CTRL = struct.Struct("<Bxxxiiiifff")  # mirrors rl_msg_t (32 bytes)
_STEP_HEAD = struct.Struct("<IfBBxx")  # count, reward, flags, info_len

assert _CTRL.size == 32, _CTRL.size

ACTION_SPACE = 36
"""9 movement directions x (attack, focus) booleans."""


# --------------------------------------------------------------------------- #
# Framing helpers
# --------------------------------------------------------------------------- #


def recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly ``n`` bytes or raise ``ConnectionError``."""
    chunks = []
    remaining = n
    while remaining > 0:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("harness closed the connection")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks) if len(chunks) > 1 else chunks[0]


def send_frame(sock: socket.socket, tag: int, payload: bytes = b"") -> None:
    sock.sendall(_HDR.pack(tag, len(payload)) + payload)


def recv_frame(sock: socket.socket) -> tuple[int, bytes]:
    tag, length = _HDR.unpack(recv_exact(sock, _HDR_SIZE))
    payload = recv_exact(sock, length) if length else b""
    return tag, payload


# --------------------------------------------------------------------------- #
# Control messages
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class EpisodeSpec:
    """Everything the harness needs to start one boss fight."""

    boss_id: str = ""
    boss_index: int = 0
    seed: int = 0
    max_steps: int = 1800
    difficulty: float = 1.0
    variant: str = "attuned"

    def pack(self) -> bytes:
        return _CTRL.pack(
            MSG_EPISODE,
            0,
            self.boss_index,
            self.seed,
            self.max_steps,
            self.difficulty,
            1.0 if self.variant != "attuned" else 0.0,
            0.0,
        )


def pack_action(action: int) -> bytes:
    return _CTRL.pack(MSG_ACT, int(action), 0, 0, 0, 0.0, 0.0, 0.0)


def pack_quit() -> bytes:
    return _CTRL.pack(MSG_QUIT, 0, 0, 0, 0, 0.0, 0.0, 0.0)


# --------------------------------------------------------------------------- #
# Ready handshake
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Handshake:
    proto: int
    obs_dim: int
    actions: int
    dt: float
    grid_w: int
    grid_h: int
    max_bullets: int
    global_size: int
    grid_channels: int
    bullet_features: int
    info_len: int

    KEYS = (
        "proto",
        "obs_dim",
        "actions",
        "dt",
        "grid_w",
        "grid_h",
        "max_bullets",
        "global_size",
        "grid_channels",
        "bullet_features",
        "info_len",
    )

    @classmethod
    def parse(cls, payload: bytes) -> "Handshake":
        values = np.frombuffer(payload, dtype="<f4", count=len(payload) // 4)
        # The Lua side builds this from a hash table, so match by position using
        # the order it writes them in `rl_main.serve`.
        order = (
            "proto",
            "obs_dim",
            "actions",
            "dt",
            "grid_w",
            "grid_h",
            "max_bullets",
            "global_size",
            "grid_channels",
            "bullet_features",
            "info_len",
        )
        if values.size < len(order):
            raise ValueError("short handshake payload")
        return cls(**{name: float(v) for name, v in zip(order, values[: len(order)])})


# --------------------------------------------------------------------------- #
# Step messages
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class Step:
    obs: np.ndarray
    reward: float
    flags: int
    info: np.ndarray

    @property
    def terminal(self) -> bool:
        return bool(self.flags & FLAG_TERMINAL)

    @property
    def truncated(self) -> bool:
        return bool(self.flags & FLAG_TRUNCATED)

    @property
    def episode_end(self) -> bool:
        return bool(self.flags & FLAG_EPISODE)

    @property
    def reset(self) -> bool:
        return bool(self.flags & FLAG_RESET)

    @property
    def won(self) -> bool:
        """Info slot 8 is +1 for a win, -1 for a death, 0 otherwise."""
        return bool(self.info.size > 8 and self.info[8] > 0)


class HarnessLog(RuntimeError):
    """The harness sent a diagnostic line (crash report, boot failure, ...)."""


def recv_step(sock: socket.socket) -> Step:
    tag, payload = recv_frame(sock)
    if tag != MSG_STEP:
        if tag == MSG_LOG:
            raise HarnessLog(payload.decode("utf-8", errors="replace"))
        raise RuntimeError(f"unexpected tag 0x{tag:02X}")
    count, reward, flags, info_len = _STEP_HEAD.unpack_from(payload, 0)
    off = _STEP_HEAD.size
    obs = np.frombuffer(payload, dtype="<f4", count=count, offset=off)
    off += count * 4
    info = np.frombuffer(payload, dtype="<f4", count=info_len, offset=off)
    return Step(obs=obs, reward=float(reward), flags=flags, info=info)