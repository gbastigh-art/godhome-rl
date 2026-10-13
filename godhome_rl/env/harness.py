"""Launch one LÖVE harness process and speak the GodhomeRL protocol to it.

This module owns *only* transport + process lifetime.  Episode semantics live in
``godhome_rl.env.godhome_env``.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from tools.build import BUILD_DIR, ensure_build, launcher_command, load_manifest

from . import proto
from .proto import EpisodeSpec

READY_PREFIX = "GODHOME_RL_READY port="


class HarnessError(RuntimeError):
    pass


@dataclass(slots=True)
class HarnessConfig:
    """Everything the Lua side reads out of ``GODHOME_RL``."""

    dt: float = 1.0 / 30.0
    grid_w: int = 20
    grid_h: int = 15
    max_bullets: int = 24
    render: bool = False
    render_every: int = 1
    bosses: list[str] = field(default_factory=list)
    max_steps: int = 1800
    seed: int | None = None
    log_level: int = 1
    # reward weights (mirrored in lua/rl_config.lua)
    r_alive: float = 0.006
    r_damage: float = 1.2
    r_damage_dealt: float = 1.0
    r_graze: float = 0.01
    r_focus: float = 0.0008
    r_win: float = 4.0
    r_death: float = -3.0
    r_timeout: float = -0.75

    def env_string(self) -> str:
        parts = [
            f"dt={self.dt:.8f}",
            f"grid_w={self.grid_w}",
            f"grid_h={self.grid_h}",
            f"max_bullets={self.max_bullets}",
            f"render={'1' if self.render else '0'}",
            f"render_every={self.render_every}",
            f"bosses={quote(','.join(self.bosses))}",
            f"max_steps={self.max_steps}",
            f"log_level={self.log_level}",
            f"r_alive={self.r_alive}",
            f"r_damage={self.r_damage}",
            f"r_damage_dealt={self.r_damage_dealt}",
            f"r_graze={self.r_graze}",
            f"r_focus={self.r_focus}",
            f"r_win={self.r_win}",
            f"r_death={self.r_death}",
            f"r_timeout={self.r_timeout}",
        ]
        if self.seed is not None:
            parts.append(f"seed={self.seed}")
        return ";".join(parts)


class Harness:
    """A live LÖVE environment process."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        *,
        render: bool | None = None,
        verbose: bool = False,
    ):
        self.config = config or HarnessConfig()
        if render is not None:
            self.config.render = render
        self.verbose = verbose
        self.manifest = load_manifest()
        self.proc: subprocess.Popen | None = None
        self.sock: socket.socket | None = None
        self.handshake: proto.Handshake | None = None
        self.port: int = 0
        self._tmpdir: Path | None = None
        self._debug_file: Path | None = None

    # -- process lifecycle -------------------------------------------------- #

    def start(self, timeout: float = 240.0, step_timeout: float = 30.0) -> "Harness":
        manifest = self.manifest
        port_dir = BUILD_DIR / "ports"
        port_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"{os.getpid()}-{int(time.time() * 1000) % 1000000}"
        port_file = port_dir / f"harness-{stamp}.port"
        debug_file = port_dir / f"harness-{stamp}.log"
        for p in (port_file, debug_file):
            p.unlink(missing_ok=True)

        env = dict(os.environ)
        env["GODHOME_RL"] = (
            self.config.env_string() + f";port=0;port_file={port_file.as_posix()}"
        )
        if self.verbose:
            env["GODHOME_RL_DEBUG"] = str(debug_file)
        else:
            debug_file.unlink(missing_ok=True)
        env.setdefault("SDL_AUDIODRIVER", "dummy")

        cmd = launcher_command(manifest)
        cwd = Path(cmd[0]).parent if len(cmd) == 1 else manifest["game_dir"]

        creationflags = 0
        if platform.system() == "Windows" and not self.config.render:
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

        self.proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
        )
        self._debug_file = debug_file

        port = self._wait_for_port(port_file, timeout)
        port_file.unlink(missing_ok=True)
        self.port = port

        self.sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        # Boot gets a generous budget (asset loading takes ~10 s); a single
        # decision step normally costs a few milliseconds and at worst a couple
        # of seconds when the harness fast-forwards a whole turn, so a 30 s
        # ceiling is generous and turns a wedged process into a fast restart.
        self.sock.settimeout(step_timeout)

        logs: list[str] = []
        while True:
            try:
                tag, payload = proto.recv_frame(self.sock)
            except ConnectionError:
                self.close()
                raise HarnessError(
                    "harness closed the connection during boot"
                    + ("\n" + "\n".join(logs) if logs else "")
                    + "\n" + self._diagnostics()
                ) from None
            if tag == proto.MSG_READY:
                self.handshake = proto.Handshake.parse(payload)
                break
            if tag == proto.MSG_LOG:
                logs.append(payload.decode("utf-8", errors="replace"))
                continue
            self.close()
            raise HarnessError(
                f"unexpected handshake tag 0x{tag:02X}"
                + ("\n" + "\n".join(logs) if logs else "")
            )
        if logs:
            self.boot_log = "\n".join(logs)
        return self

    def _wait_for_port(self, port_file: Path, timeout: float) -> int:
        deadline = time.time() + timeout
        assert self.proc is not None
        while time.time() < deadline:
            if port_file.is_file():
                text = port_file.read_text(encoding="utf-8").strip()
                if text.isdigit():
                    return int(text)
            if self.proc.poll() is not None:
                raise HarnessError(
                    f"harness exited with code {self.proc.returncode} before "
                    f"listening\n{self._diagnostics()}"
                )
            time.sleep(0.05)
        self.kill()
        raise HarnessError(
            f"harness did not report a port within {timeout:.0f}s\n{self._diagnostics()}"
        )

    def _diagnostics(self) -> str:
        parts = []
        if self.proc is not None:
            code = self.proc.poll()
            parts.append(f"harness exit code: {code}")
            if code is not None and code not in (0,):
                parts.append(
                    "  (0x%08X)" % (code & 0xFFFFFFFF)
                    + "  <- 0xC0000005 is an access violation inside LÖVE/LuaJIT"
                )
        debug_file = getattr(self, "_debug_file", None)
        if debug_file is not None and debug_file.is_file():
            parts.append(
                "--- GODHOME_RL_DEBUG ---\n"
                + debug_file.read_text(encoding="utf-8", errors="replace")
            )
        text = self._drain_stderr()
        if text:
            parts.append("--- stderr ---\n" + text)
        text = self._drain_stdout()
        if text:
            parts.append("--- stdout ---\n" + text)
        return "\n".join(parts) or "(no diagnostics captured)"

    def _drain_stdout(self) -> str:
        if not self.proc or not self.proc.stdout:
            return ""
        try:
            os.set_blocking(self.proc.stdout.fileno(), False)
            data = self.proc.stdout.read() or b""
            return data.decode("utf-8", "replace")
        except Exception:  # pragma: no cover - best effort diagnostics
            return ""

    def _drain_stderr(self) -> str:
        if not self.proc or not self.proc.stderr:
            return ""
        try:
            os.set_blocking(self.proc.stderr.fileno(), False)
            data = self.proc.stderr.read() or b""
            return data.decode("utf-8", "replace")
        except Exception:  # pragma: no cover - best effort diagnostics
            return ""

    def drain_output(self) -> str:
        """Non-blocking read of whatever the process printed (diagnostics)."""
        return self._drain_stdout() + self._drain_stderr()

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.sendall(proto._HDR.pack(proto.MSG_QUIT, 0))
            except OSError:
                pass
            try:
                self.sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self.sock.close()
            self.sock = None
        if self.proc is not None:
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # The harness may be wedged inside the game loop; a clean QUIT
                # will not be read, so stop waiting and kill it.
                self.proc.kill()
                try:
                    self.proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            for stream in (self.proc.stdout, self.proc.stderr):
                if stream:
                    try:
                        stream.close()
                    except OSError:
                        pass
            self.proc = None

    def __enter__(self) -> "Harness":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.close()

    def kill(self) -> None:
        if self.proc is not None:
            self.proc.kill()

    # -- protocol ------------------------------------------------------------ #

    def start_episode(self, spec: EpisodeSpec) -> proto.Step:
        assert self.sock is not None
        self.send_frame(proto.MSG_EPISODE, spec.pack())
        return self.recv()

    def send_action(self, action: int) -> None:
        """Queue an action without waiting for the reply.

        Lets a vectorised environment hand work to every harness at once so the
        LÖVE processes actually run in parallel instead of one after another.
        """
        assert self.sock is not None
        self.send_frame(proto.MSG_ACT, proto.pack_action(int(action)))

    def send_frame(self, tag: int, payload: bytes) -> None:
        assert self.sock is not None
        self.sock.sendall(proto._HDR.pack(tag, len(payload)) + payload)

    def recv(self) -> proto.Step:
        assert self.sock is not None
        try:
            return proto.recv_step(self.sock)
        except proto.HarnessLog as exc:
            # A diagnostic from the harness (boot failure, Lua error, "episode
            # start failed").  Treat it exactly like a crash so the environment
            # heals itself instead of taking the training run down.
            raise HarnessError(str(exc) + "\n" + self._diagnostics()) from None
        except socket.timeout:
            raise HarnessError(
                "harness stopped responding within "
                f"{self.sock.gettimeout():.0f}s (probably wedged inside the game "
                "loop)\n" + self._diagnostics()
            ) from None
        except (ConnectionError, ConnectionResetError, OSError):
            raise HarnessError(
                "harness died while stepping the game\n" + self._diagnostics()
            ) from None

    def step(self, action: int) -> proto.Step:
        self.send_action(action)
        return self.recv()