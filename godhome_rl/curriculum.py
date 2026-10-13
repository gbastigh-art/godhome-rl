"""Curriculum: which bosses to train on, and when to graduate to harder ones.

Learning to dodge a bullet pattern generalises surprisingly well, but a
from-scratch agent thrown at Nightmarish King will not get a single useful
gradient.  The curriculum therefore

1.  starts with a handful of short, sparse-pattern fights,
2.  expands to everything once the win rate on the current stage clears a
    threshold, and
3.  optionally ramps the Godhome difficulty flag (attuned -> ascended ->
    radiant) for the endgame bosses.

Stages are defined declaratively so a run is fully reproducible from its config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .bosses import DEFAULT_ORDER


@dataclass(slots=True)
class Stage:
    """One rung of the ladder."""

    name: str
    bosses: list[str]
    max_steps: int = 1800
    #: Promote once the rolling win rate clears this.
    promote_win_rate: float = 0.6
    #: ...and at least this many episodes have been finished.
    min_episodes: int = 60
    #: Hard cap on frames per episode at this stage (keeps early stages short).
    difficulty: int = 1


def _default_stages() -> list[Stage]:
    return [
        Stage(
            name="dodge-basics",
            bosses=["p1/falseknight", "p1/mosscharger", "p1/soulwarrior", "p2/marmu"],
            max_steps=900,
            promote_win_rate=0.35,
            min_episodes=40,
        ),
        Stage(
            name="greenpath-fogcanyon",
            bosses=[
                "p1/vengeflyking", "p1/gruzmother", "p1/dungdefender",
                "p1/broodingmawlek", "p1/nailmasters",
                "p2/xero", "p2/crystalguardian", "p2/mantislords",
                "p2/nosk", "p2/flukemarm", "p2/oblobbles", "p2/soulmaster",
            ],
            max_steps=1500,
            promote_win_rate=0.45,
            min_episodes=80,
        ),
        Stage(
            name="crystal-peak",
            bosses=[
                "p3/hiveknight", "p3/elderhu", "p3/collector", "p3/uumuu",
                "p3/galien", "p3/godtamer", "p3/hornet2", "p3/sly",
            ],
            max_steps=1800,
            promote_win_rate=0.45,
            min_episodes=80,
        ),
        Stage(
            name="kingdom",
            bosses=[
                "p4/enragedguardian", "p4/lostkin", "p4/noeyes",
                "p4/whitedefender", "p4/traitorlord", "p4/failedchamp",
                "p4/markoth", "p4/watcherknights", "p4/soultyrant",
                "p4/purevessel",
                "p5/wingednosk", "p5/sisters", "p5/kristalguardian",
            ],
            max_steps=1800,
            promote_win_rate=0.4,
            min_episodes=100,
        ),
        Stage(
            name="nightmare",
            bosses=["p3/grimm", "p3/greyprince", "p2/brokenvessel", "p2/sheo", "p5/nkg"],
            max_steps=1800,
            promote_win_rate=0.3,
            min_episodes=120,
        ),
        Stage(
            name="radiance",
            bosses=["p5/radiance"],
            max_steps=2400,
            promote_win_rate=0.25,
            min_episodes=120,
        ),
    ]


@dataclass(slots=True)
class Curriculum:
    stages: list[Stage] = field(default_factory=_default_stages)
    index: int = 0
    episodes_in_stage: int = 0
    recent_outcomes: list[bool] = field(default_factory=list)

    @classmethod
    def single(cls, bosses: list[str], max_steps: int = 1800, name: str = "custom") -> "Curriculum":
        return cls(stages=[Stage(name=name, bosses=list(bosses), max_steps=max_steps,
                                 promote_win_rate=2.0, min_episodes=10**9)])

    @property
    def stage(self) -> Stage:
        return self.stages[min(self.index, len(self.stages) - 1)]

    @property
    def finished(self) -> bool:
        return self.index >= len(self.stages)

    def record(self, won: bool) -> None:
        self.episodes_in_stage += 1
        self.recent_outcomes.append(bool(won))
        if len(self.recent_outcomes) > 300:
            self.recent_outcomes = self.recent_outcomes[-200:]

    @property
    def win_rate(self) -> float:
        if not self.recent_outcomes:
            return 0.0
        window = self.recent_outcomes[-100:]
        return sum(window) / len(window)

    def should_promote(self) -> bool:
        if self.finished:
            return False
        stage = self.stage
        return (
            self.episodes_in_stage >= stage.min_episodes
            and self.win_rate >= stage.promote_win_rate
        )

    def promote(self) -> bool:
        if not self.should_promote():
            return False
        self.index += 1
        self.episodes_in_stage = 0
        self.recent_outcomes = []
        return True

    def describe(self) -> str:
        stage = self.stage
        return (
            f"stage {self.index + 1}/{len(self.stages)} '{stage.name}': "
            f"{len(stage.bosses)} bosses, max_steps={stage.max_steps}, "
            f"win_rate={self.win_rate:.2f} (promote at "
            f"{stage.promote_win_rate:.2f} after {stage.min_episodes} episodes)"
        )


def build_curriculum(spec: str) -> Curriculum:
    """Build a curriculum from a string spec.

    * ``all``        -> the default staged ladder
    * ``<boss>``     -> a single custom stage (repeatable via commas)
    """
    from .bosses import resolve_list

    if spec.strip().lower() in ("all", "default", "ladder"):
        return Curriculum()
    return Curriculum.single(resolve_list(spec))