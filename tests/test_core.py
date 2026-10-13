"""Tests that need no copy of the game.

These cover the parts that are pure logic: the observation layout, the action
packing, the boss catalog, the reward weights and the build's patch logic.  The
end-to-end behaviour is covered by ``python -m godhome_rl doctor``, which does
need the game.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from godhome_rl.bosses import BY_ID, BOSSES, display, resolve, resolve_list
from godhome_rl.curriculum import Curriculum, build_curriculum
from godhome_rl.env.layout import ACTION_SPACE, INFO_SIZE, Layout, decode_action
from godhome_rl.env.proto import EpisodeSpec, Handshake, pack_action, pack_quit
from godhome_rl.env.rewards import RewardWeights
from godhome_rl.rl.model import ActorCritic
from godhome_rl.rl.ppo import PPO, PPOConfig


# --------------------------------------------------------------------------- #
#  Layout
# --------------------------------------------------------------------------- #


def test_layout_dim_matches_formula():
    layout = Layout()
    expected = 50 + 2 * layout.grid_w * layout.grid_h + layout.max_bullets * 10
    assert layout.dim == expected == 890


def test_layout_slices_partition_the_vector():
    layout = Layout()
    slices = [
        layout.global_slice,
        layout.density_slice,
        layout.threat_slice,
        layout.bullet_slice,
    ]
    assert slices[0].start == 0
    for a, b in zip(slices, slices[1:]):
        assert a.stop == b.start
    assert slices[-1].stop == layout.dim


def test_layout_verify_accepts_a_matching_handshake():
    layout = Layout()
    handshake = Handshake(
        proto=1.0, obs_dim=float(layout.dim), actions=float(ACTION_SPACE),
        dt=1 / 30, grid_w=float(layout.grid_w), grid_h=float(layout.grid_h),
        max_bullets=float(layout.max_bullets), global_size=50.0,
        grid_channels=2.0, bullet_features=10.0, info_len=float(INFO_SIZE),
    )
    layout.verify(handshake)


def test_layout_verify_rejects_a_stale_harness():
    layout = Layout()
    handshake = Handshake(
        proto=1.0, obs_dim=123.0, actions=float(ACTION_SPACE), dt=1 / 30,
        grid_w=float(layout.grid_w), grid_h=float(layout.grid_h),
        max_bullets=float(layout.max_bullets), global_size=50.0,
        grid_channels=2.0, bullet_features=10.0, info_len=float(INFO_SIZE),
    )
    with pytest.raises(ValueError, match="layout mismatch"):
        layout.verify(handshake)


def test_info_vector_size():
    assert INFO_SIZE == 16


# --------------------------------------------------------------------------- #
#  Actions
# --------------------------------------------------------------------------- #


def test_action_roundtrip():
    for action in range(ACTION_SPACE):
        move, attack, focus = decode_action(action)
        assert 0 <= move <= 8
        assert isinstance(attack, bool)
        assert isinstance(focus, bool)
        assert move + 9 * int(attack) + 18 * int(focus) == action


def test_actions_are_unique_per_behaviour():
    seen = {decode_action(a) for a in range(ACTION_SPACE)}
    assert len(seen) == ACTION_SPACE


# --------------------------------------------------------------------------- #
#  Protocol
# --------------------------------------------------------------------------- #


def test_control_payload_size():
    assert len(pack_action(7)) == 32
    assert len(pack_quit()) == 32
    assert len(EpisodeSpec().pack()) == 32


def test_episode_spec_pack_is_deterministic():
    a = EpisodeSpec(boss_id="p1/falseknight", seed=7, max_steps=900).pack()
    b = EpisodeSpec(boss_id="p1/falseknight", seed=7, max_steps=900).pack()
    assert a == b


def test_handshake_parse_rejects_short_payload():
    with pytest.raises(ValueError):
        Handshake.parse(b"\x00" * 4)


# --------------------------------------------------------------------------- #
#  Bosses
# --------------------------------------------------------------------------- #


def test_every_boss_has_a_unique_id():
    ids = [b.id for b in BOSSES]
    assert len(ids) == len(set(ids))
    assert len(ids) >= 40


def test_resolve_by_id_name_and_area():
    assert resolve("p5/radiance") == "p5/radiance"
    assert resolve("radiance") == "p5/radiance"
    assert resolve("False Knight") == "p1/falseknight"
    assert "p3/grimm" in resolve("p3")
    assert len(resolve_list("all")) == len(BOSSES)


def test_resolve_rejects_nonsense():
    with pytest.raises(ValueError):
        resolve("not-a-boss")


def test_display_is_human_readable():
    assert "Radiance" in display("p5/radiance")


# --------------------------------------------------------------------------- #
#  Curriculum
# --------------------------------------------------------------------------- #


def test_default_curriculum_is_ordered_and_complete():
    cur = build_curriculum("all")
    assert len(cur.stages) >= 5
    seen = set()
    for stage in cur.stages:
        assert stage.bosses
        seen.update(stage.bosses)
    # every boss except the very last stage is reachable from the ladder
    assert len(seen) >= 40


def test_single_boss_curriculum_never_promotes():
    cur = build_curriculum("p1/falseknight")
    assert cur.finished is False
    for _ in range(500):
        cur.record(True)
    assert not cur.promote()


def test_curriculum_promotes_after_the_thresholds():
    cur = Curriculum.single(["p1/falseknight"])
    cur = build_curriculum("all")
    stage = cur.stage
    for _ in range(stage.min_episodes):
        cur.record(True)
    assert cur.should_promote()
    assert cur.promote()
    assert cur.index == 1


def test_curriculum_does_not_promote_when_losing():
    cur = build_curriculum("all")
    for _ in range(cur.stage.min_episodes + 10):
        cur.record(False)
    assert not cur.promote()


# --------------------------------------------------------------------------- #
#  Rewards
# --------------------------------------------------------------------------- #


def test_reward_weights_roundtrip_to_env_kwargs():
    weights = RewardWeights()
    kwargs = weights.as_kwargs()
    assert set(kwargs) == {
        "r_alive", "r_damage", "r_damage_dealt", "r_graze",
        "r_focus", "r_win", "r_death", "r_timeout",
    }
    assert kwargs["r_win"] == weights.win
    assert kwargs["r_death"] < 0 < kwargs["r_win"]


def test_reward_weights_from_dict_ignores_unknown_keys():
    weights = RewardWeights.from_dict({"win": 9.0, "nonsense": 1})
    assert weights.win == 9.0


# --------------------------------------------------------------------------- #
#  Model + PPO
# --------------------------------------------------------------------------- #


def test_actor_critic_forward_and_evaluate():
    layout = Layout()
    model = ActorCritic(layout, hidden=32, use_norm=True)
    obs = torch_batch(layout.dim, 4)
    action, value, logp, logits = model.act(obs)
    assert action.shape == (4,)
    assert value.shape == (4,)
    assert logp.shape == (4,)
    assert logits.shape == (4, ACTION_SPACE)

    values, logp2, entropy = model.evaluate_actions(obs, action)
    assert values.shape == (4,)
    assert logp2.shape == (4,)
    assert entropy.dim() == 0


def test_actor_critic_greedy_action_is_deterministic():
    layout = Layout()
    model = ActorCritic(layout, hidden=32, use_norm=False).eval()
    obs = torch_batch(layout.dim, 3)
    a1, _, _, _ = model.act(obs, deterministic=True)
    a2, _, _, _ = model.act(obs, deterministic=True)
    assert np.array_equal(a1.numpy(), a2.numpy())


def test_model_save_load_roundtrip(tmp_path):
    import torch

    layout = Layout()
    model = ActorCritic(layout, hidden=32, use_norm=True)
    obs = torch_batch(layout.dim, 8)
    model.norm.update(obs)
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = ActorCritic.load(path)

    with torch.no_grad():
        a, v, _, _ = model.act(obs, deterministic=True)
        b, w, _, _ = loaded.act(obs, deterministic=True)
    assert torch.allclose(a, b)
    assert torch.allclose(v, w)


def test_ppo_update_changes_the_policy_and_reports_metrics():
    import torch

    layout = Layout()
    cfg = PPOConfig(n_envs=2, n_steps=8, n_epochs=2, n_minibatches=2, lr=1e-2, device="cpu")
    ppo = PPO(ActorCritic(layout, hidden=32, use_norm=True), cfg)
    rollout = ppo.begin_episode()
    obs = torch.rand((cfg.n_steps, cfg.n_envs, layout.dim))
    for t in range(cfg.n_steps):
        rollout["obs"][t] = obs[t]
        rollout["actions"][t] = torch.randint(0, ACTION_SPACE, (cfg.n_envs,))
        rollout["logprobs"][t] = -2.0
        rollout["rewards"][t] = torch.randn(cfg.n_envs)
        rollout["dones"][t] = 0.0
        rollout["values"][t] = torch.randn(cfg.n_envs)

    before = [p.detach().clone() for p in ppo.ac.parameters()]
    advantages, returns = ppo.compute_returns(rollout, torch.rand((cfg.n_envs, layout.dim)))
    assert advantages.shape == returns.shape == (cfg.n_steps, cfg.n_envs)

    metrics = ppo.update(rollout, advantages, returns)
    assert {"policy_loss", "value_loss", "entropy", "approx_kl"} <= set(metrics)
    after = list(ppo.ac.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after))


def test_ppo_anneal_schedule():
    import torch

    cfg = PPOConfig(lr=1e-3, anneal_lr=True, device="cpu")
    ppo = PPO(ActorCritic(Layout(), hidden=16), cfg)
    start = ppo.anneal(0, 1000)
    end = ppo.anneal(1000, 1000)
    assert start > end
    assert end == pytest.approx(cfg.lr * 0.1, rel=1e-3)


def test_running_norm_normalises():
    import torch

    layout = Layout()
    model = ActorCritic(layout, hidden=16, use_norm=True)
    assert model.norm is not None
    obs = torch.rand((256, layout.dim)) * 5 + 3
    model.norm.update(obs)
    out = model.norm(obs)
    assert out.mean().abs().item() < 0.5


def torch_batch(dim: int, n: int):
    import torch

    return torch.rand((n, dim), dtype=torch.float32)


# --------------------------------------------------------------------------- #
#  Build tooling
# --------------------------------------------------------------------------- #


def test_build_patch_is_idempotent(tmp_path):
    """The main.lua injection must survive being applied twice."""
    from tools.build import BOOT_BEGIN, BOOT_END, BOOT_ANCHOR

    original = (
        f'local Kristal = require("src.kristal")\n'
        f'local Hotswapper = require("src.hotswapper")\n'
        f'\nfunction love.run()\n    return 0\nend\n'
    )

    def patch(text: str) -> str:
        lines = text.splitlines()
        kept, inside = [], False
        for line in lines:
            if line.strip() == BOOT_BEGIN:
                inside = True
                continue
            if line.strip() == BOOT_END:
                inside = False
                continue
            if not inside:
                kept.append(line)
        out = "\n".join(kept).rstrip() + f"\n\n{BOOT_BEGIN}\n" + 'require("rl/rl_boot")\n' + f"{BOOT_END}\n"
        assert BOOT_ANCHOR in out
        return out

    once = patch(original)
    twice = patch(once)
    assert once == twice
    assert once.count(BOOT_BEGIN) == 1


def test_fused_zip_offset_detection():
    from tools.build import _find_fused_zip_offset

    stub = b"MZ" + b"\x00" * 100
    payload = b"PK\x03\x04" + b"payload"
    exe = stub + payload
    assert _find_fused_zip_offset(exe) == len(stub)
    assert _find_fused_zip_offset(b"MZ" + b"\x00" * 100) is None


def test_lua_harness_parses():
    """Every Lua file must load under LuaJIT (the dialect LÖVE uses)."""
    pytest.importorskip("lupa")
    from tools.luacheck import check_files

    files = sorted((ROOT / "lua").glob("*.lua"))
    assert files
    assert check_files(files) == 0


# --------------------------------------------------------------------------- #
#  Repository hygiene
# --------------------------------------------------------------------------- #


def test_no_game_content_tracked():
    from tools.check_no_game_content import main

    assert main() == 0


def test_documentation_links_resolve():
    docs = ROOT / "docs"
    readme = ROOT / "README.md"
    text = readme.read_text(encoding="utf-8")
    for target in ("docs/installation.md", "docs/quickstart.md", "LICENSE"):
        assert target in text
        assert (ROOT / target).is_file(), target
    for path in docs.glob("*.md"):
        assert path.stat().st_size > 200, f"{path} looks like a stub"