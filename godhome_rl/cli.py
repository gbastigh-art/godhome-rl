"""Command line interface.

::

    python -m godhome_rl setup            # one-shot bootstrap for this machine
    python -m godhome_rl build            # extract + patch the game (cached)
    python -m godhome_rl doctor           # verify the toolchain end to end
    python -m godhome_rl bosses           # list Godhome's boss encounters
    python -m godhome_rl bench            # measure steps/s for a few bosses
    python -m godhome_rl train ...        # train a policy
    python -m godhome_rl play  ...        # watch a trained policy
    python -m godhome_rl eval  ...        # headless evaluation over N episodes
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from . import bosses as boss_catalog


def _print_bosses() -> None:
    print(f"{'encounter id':<26} {'name':<24} {'area':<16} tier")
    print("-" * 74)
    for boss in boss_catalog.BOSSES:
        print(
            f"{boss.id:<26} {boss.name:<24} "
            f"{boss.area}={boss_catalog.AREAS[boss.area]:<12} {boss.tier}"
        )
    print(f"\n{len(boss_catalog.BOSSES)} encounters. "
          "Selectors: an id, a name ('False Knight'), an area ('p3'), or 'all'.")


# --------------------------------------------------------------------------- #
#  setup
# --------------------------------------------------------------------------- #


def cmd_setup(args: argparse.Namespace) -> int:
    """Install the Python dependencies (if a venv is requested) and build."""
    if args.venv:
        venv = Path(args.venv)
        if not (venv / "Scripts" / "python.exe").is_file() and not (
            venv / "bin" / "python"
        ).is_file():
            print(f"[setup] creating virtual environment at {venv}")
            import subprocess

            subprocess.check_call([sys.executable, "-m", "venv", str(venv)])
        py = venv / ("Scripts" / "python.exe" if sys.platform == "win32" else "bin/python")
        print("[setup] installing dependencies into the venv")
        subprocess.check_call([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.check_call([str(py), "-m", "pip", "install", "-e", "."])
        print(f"\n[setup] done.  Activate with: {venv}\\Scripts\\activate  (Windows)")
        print(f"[setup] then run:  python -m godhome_rl build")
        return 0

    print("[setup] installing dependencies into the current interpreter")
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."])
    return cmd_build(args)


# --------------------------------------------------------------------------- #
#  build / doctor
# --------------------------------------------------------------------------- #


def cmd_build(args: argparse.Namespace) -> int:
    from tools.build import ensure_build, find_install

    try:
        install = find_install(args.install)
    except Exception as exc:
        print(f"[build] {exc}", file=sys.stderr)
        return 2
    manifest = ensure_build(install, force=args.force)
    if args.json:
        print(json.dumps(manifest, indent=2))
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    print("GodhomeRL doctor")
    print("=" * 62)

    import numpy

    print(f"[ok]   python           {sys.version.split()[0]}")
    print(f"[ok]   numpy            {numpy.__version__}")
    try:
        import torch

        print(f"[ok]   torch            {torch.__version__} "
              f"(cuda={torch.cuda.is_available()})")
    except ImportError:
        print("[FAIL] torch is not installed -- run `pip install -e .`")
        ok = False

    try:
        from lupa import luajit21  # noqa: F401

        print("[ok]   lupa (lua check) installed")
    except ImportError:
        print("[warn] lupa not installed (only needed for offline Lua checks)")

    try:
        from tools.build import ensure_build, find_install

        install = find_install(args.install)
        print(f"[ok]   Godhome install {install.root}")
        manifest = ensure_build(install, force=args.force, quiet=True)
        print(f"[ok]   game tree        {manifest['game_dir']}")
        if manifest.get("runtime_exe"):
            print(f"[ok]   runtime          {manifest['runtime_exe']}")
        else:
            print(f"[ok]   launcher         love.exe = {manifest.get('love_exe')}")
    except Exception as exc:
        print(f"[FAIL] build: {exc}")
        return 2

    print("\nrunning a 60-step smoke episode ...")
    try:
        from .env.godhome_env import GodhomeEnv

        t0 = time.time()
        env = GodhomeEnv(bosses=["p1/falseknight"], max_steps=300)
        try:
            boot = time.time() - t0
            import numpy as np

            rng = np.random.default_rng(0)
            obs, _ = env.reset()
            t1 = time.time()
            for _ in range(60):
                obs, *_ = env.step(int(rng.integers(0, 36)))
            rate = 60 / max(1e-6, time.time() - t1)
            print(f"[ok]   boot {boot:.1f}s, {rate:.0f} steps/s, obs_dim={obs.size}")
            if env.restarts:
                print(
                    f"[warn] harness restarted {env.restarts} time(s) during the "
                    "smoke test; it self-healed, but see docs/troubleshooting.md"
                )
                for line in env.restart_log[-3:]:
                    print("       " + line.strip())
        finally:
            env.close()
    except Exception as exc:
        print(f"[FAIL] harness: {exc}")
        ok = False

    print("=" * 62)
    print("all good" if ok else "problems found (see above)")
    return 0 if ok else 1


# --------------------------------------------------------------------------- #
#  bosses / bench
# --------------------------------------------------------------------------- #


def cmd_bosses(args: argparse.Namespace) -> int:
    _print_bosses()
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    import numpy as np

    from .env.godhome_env import GodhomeEnv

    specs = args.boss or "p1/falseknight,p2/mantislords,p3/grimm,p5/radiance"
    ids = boss_catalog.resolve_list(specs)
    print(f"{'boss':<30} {'boot':>7} {'steps/s':>9} {'game-s/s':>10} {'ep':>4} {'win%':>6}")
    print("-" * 74)
    for boss in ids:
        t0 = time.time()
        env = GodhomeEnv(bosses=[boss], max_steps=args.max_steps, verbose=args.verbose)
        try:
            boot = time.time() - t0
            env.reset(boss=boss)
            rng = np.random.default_rng(0)
            frames = 0
            n = 0
            t0 = time.time()
            for _ in range(args.episodes):
                obs, _ = env.reset(boss=boss)
                while True:
                    obs, _, term, trunc, info = env.step(int(rng.integers(0, 36)))
                    n += 1
                    if info["info"].size:
                        frames += 1
                    if term or trunc:
                        break
            dt = max(1e-6, time.time() - t0)
            s = env.snapshot()
            game_frames = env.stats.frames
            print(
                f"{boss_catalog.display(boss):<30} {boot:>6.1f}s {n / dt:>9.0f} "
                f"{game_frames / dt:>10.0f} {s['episodes']:>4} "
                f"{100 * s['win_rate']:>5.0f}%"
                + (f"  restarts={env.restarts}" if env.restarts else "")
            )
        finally:
            env.close()
    print("\nsteps/s counts frames where the soul is under policy control;")
    print("game-s/s counts every simulated frame, including auto-confirmed menus.")
    return 0


# --------------------------------------------------------------------------- #
#  train / play / eval
# --------------------------------------------------------------------------- #


def cmd_train(args: argparse.Namespace) -> int:
    from .rl.ppo import PPOConfig
    from .train import TrainConfig, train

    cfg = TrainConfig(
        name=args.name,
        out_dir=Path(args.out_dir),
        bosses=args.bosses,
        n_envs=args.n_envs,
        total_updates=args.updates,
        max_hours=args.max_hours,
        max_steps=args.max_steps,
        difficulty=args.difficulty,
        variant=args.variant,
        grid_w=args.grid_w,
        grid_h=args.grid_h,
        max_bullets=args.max_bullets,
        seed=args.seed,
        resume=not args.no_resume,
        eval_every=args.eval_every,
        verbose=args.verbose,
    )
    ppo_cfg = PPOConfig(
        lr=args.lr,
        ent_coef=args.ent_coef,
        clip_coef=args.clip_coef,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        n_steps=args.n_steps,
        n_epochs=args.n_epochs,
        n_minibatches=args.n_minibatches,
        max_grad_norm=args.max_grad_norm,
        anneal_lr=not args.no_anneal,
        seed=args.seed,
        device=args.device,
    )
    train(cfg, ppo_cfg)
    return 0


def _resolve_model(args: argparse.Namespace) -> Path:
    model = Path(args.model)
    if model.is_dir():
        model = model / "checkpoints" / "latest.pt"
    if not model.is_file():
        raise SystemExit(f"model not found: {model}")
    return model


def cmd_play(args: argparse.Namespace) -> int:
    from .play import play

    ids = boss_catalog.resolve_list(args.boss)
    play(
        _resolve_model(args),
        ids,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        greedy=not args.stochastic,
        render_every=args.render_every,
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    import numpy as np
    import torch

    from .env.godhome_env import GodhomeEnv
    from .rl.model import load_actor_critic

    model_path = _resolve_model(args)
    model = load_actor_critic(model_path)
    layout = model.layout

    ids = boss_catalog.resolve_list(args.boss)
    print(f"{'boss':<30} {'win%':>6} {'return':>8} {'frames':>8} {'hits':>6} {'graze':>7}")
    print("-" * 72)
    rows = []
    for boss in ids:
        env = GodhomeEnv(bosses=[boss], max_steps=args.max_steps, difficulty=args.difficulty)
        try:
            for _ in range(args.episodes):
                obs, _ = env.reset(boss=boss)
                while True:
                    with torch.no_grad():
                        t = torch.as_tensor(
                            np.array(obs, dtype=np.float32, copy=True)
                        ).unsqueeze(0)
                        action, _, _, _ = model.act(t, deterministic=not args.stochastic)
                    obs, reward, term, trunc, info = env.step(int(action.item()))
                    if term or trunc:
                        break
            s = env.snapshot()
            episodes = max(1, s["episodes"])
            last = env._last_info
            print(
                f"{boss_catalog.display(boss):<30} {100 * s['win_rate']:>5.0f}% "
                f"{s['mean_return']:>8.2f} {last[0]:>8.0f} "
                f"{last[7]:>6.1f} {last[6]:>7.1f}"
            )
            rows.append((boss, s))
        finally:
            env.close()
    if rows:
        total = sum(s["episodes"] for _, s in rows)
        wins = sum(s["wins"] for _, s in rows)
        print("-" * 72)
        print(f"overall: {wins}/{total} wins ({100 * wins / max(1, total):.0f}%)")
    return 0


# --------------------------------------------------------------------------- #
#  argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="godhome-rl",
        description="Train a reinforcement-learning agent that plays the Godhome mod.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--install", default=None,
                   help="path to the Godhome standalone folder (or set GODHOME_RL_INSTALL)")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="install dependencies and build the harness")
    s.add_argument("--venv", default=None, help="create and install into this virtualenv path")
    s.add_argument("--force", action="store_true", help="re-extract the game from scratch")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("build", help="extract + patch the game into build/")
    s.add_argument("--force", action="store_true")
    s.add_argument("--json", action="store_true", help="print the build manifest")
    s.set_defaults(func=cmd_build)

    s = sub.add_parser("doctor", help="check the whole toolchain")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_doctor)

    s = sub.add_parser("bosses", help="list every Godhome boss encounter")
    s.set_defaults(func=cmd_bosses)

    s = sub.add_parser("bench", help="measure simulation throughput")
    s.add_argument("--boss", default=None, help="comma separated encounters")
    s.add_argument("--episodes", type=int, default=2)
    s.add_argument("--max-steps", type=int, default=1200)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("train", help="train a policy with PPO")
    s.add_argument("--name", default="godhome")
    s.add_argument("--out-dir", default="runs")
    s.add_argument("--bosses", default="all",
                   help="'all' for the curriculum ladder, or any boss selector")
    s.add_argument("--n-envs", type=int, default=3)
    s.add_argument("--updates", type=int, default=4000)
    s.add_argument("--max-hours", type=float, default=0.0,
                   help="stop after this many hours (0 = --updates only)")
    s.add_argument("--max-steps", type=int, default=0, help="0 = follow the curriculum")
    s.add_argument("--difficulty", type=int, default=1, choices=(1, 2, 3, 4))
    s.add_argument("--variant", default="attuned", choices=("attuned", "ascended"))
    s.add_argument("--grid-w", type=int, default=20)
    s.add_argument("--grid-h", type=int, default=15)
    s.add_argument("--max-bullets", type=int, default=24)
    s.add_argument("--n-steps", type=int, default=128)
    s.add_argument("--n-epochs", type=int, default=4)
    s.add_argument("--n-minibatches", type=int, default=4)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--ent-coef", type=float, default=0.012)
    s.add_argument("--clip-coef", type=float, default=0.2)
    s.add_argument("--gamma", type=float, default=0.995)
    s.add_argument("--gae-lambda", type=float, default=0.95)
    s.add_argument("--max-grad-norm", type=float, default=0.5)
    s.add_argument("--no-anneal", action="store_true")
    s.add_argument("--no-resume", action="store_true")
    s.add_argument("--eval-every", type=int, default=25)
    s.add_argument("--device", default="auto")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_train)

    s = sub.add_parser("play", help="watch a trained policy play")
    s.add_argument("--model", default="runs/godhome/checkpoints/latest.pt")
    s.add_argument("--boss", default="all")
    s.add_argument("--episodes", type=int, default=0, help="0 = keep playing")
    s.add_argument("--max-steps", type=int, default=1800)
    s.add_argument("--render-every", type=int, default=1)
    s.add_argument("--stochastic", action="store_true", help="sample instead of argmax")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("eval", help="headless evaluation of a trained policy")
    s.add_argument("--model", default="runs/godhome/checkpoints/latest.pt")
    s.add_argument("--boss", default="all")
    s.add_argument("--episodes", type=int, default=5)
    s.add_argument("--max-steps", type=int, default=1800)
    s.add_argument("--difficulty", type=int, default=1)
    s.add_argument("--stochastic", action="store_true")
    s.add_argument("--seed", type=int, default=0)
    s.set_defaults(func=cmd_eval)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\ninterrupted")
        return 130
    except ValueError as exc:
        # Mostly "this checkpoint predates the current observation format" --
        # a user error with an actionable fix, so show the message, not a
        # traceback.
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())