"""The Godhome boss catalog.

Encounter ids match ``Mod.encounters`` in the mod's ``mod.lua`` exactly, grouped
by the Godhome pantheon areas (p1..p5).  Used by the CLI (``--boss``),
the curriculum, and the docs so that the list of available fights is never
duplicated by hand.
"""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True, slots=True)
class Boss:
    id: str
    name: str
    area: str
    #: Rough ordering by difficulty (1 = easiest).  Only used as a default
    #: curriculum order; the actual progression is configurable.
    tier: int


AREAS = {
    "p1": "Greenpath",
    "p2": "Fog Canyon",
    "p3": "Crystal Peak",
    "p4": "The Kingdom",
    "p5": "The Abyss",
}

BOSSES: tuple[Boss, ...] = (
    Boss("p1/vengeflyking", "Vengeful King", "p1", 1),
    Boss("p1/gruzmother", "Gruz Mother", "p1", 2),
    Boss("p1/falseknight", "False Knight", "p1", 1),
    Boss("p1/mosscharger", "Mossy Charger", "p1", 1),
    Boss("p1/hornet", "Hornet 1", "p1", 3),
    Boss("p1/gorb", "Gorak Sure the Mighty", "p1", 3),
    Boss("p1/dungdefender", "Dung Defender", "p1", 2),
    Boss("p1/soulwarrior", "Soul Warrior", "p1", 2),
    Boss("p1/broodingmawlek", "Brooding Mawlek", "p1", 2),
    Boss("p1/nailmasters", "The Nailmasters", "p1", 3),
    Boss("p2/xero", "Xero", "p2", 2),
    Boss("p2/crystalguardian", "Crystal Guardian", "p2", 1),
    Boss("p2/soulmaster", "Soul Master", "p2", 3),
    Boss("p2/oblobbles", "Oblobbles", "p2", 3),
    Boss("p2/mantislords", "Mantis Lords", "p2", 3),
    Boss("p2/marmu", "Marmu", "p2", 2),
    Boss("p2/nosk", "Nosk", "p2", 3),
    Boss("p2/flukemarm", "Flukemarm", "p2", 3),
    Boss("p2/brokenvessel", "Broken Vessel", "p2", 4),
    Boss("p2/sheo", "Sheo", "p2", 4),
    Boss("p3/hiveknight", "Hive Knight", "p3", 2),
    Boss("p3/elderhu", "Elder Hu", "p3", 2),
    Boss("p3/collector", "The Collector", "p3", 3),
    Boss("p3/godtamer", "God Tamer", "p3", 4),
    Boss("p3/grimm", "Grimm", "p3", 5),
    Boss("p3/galien", "Galien", "p3", 3),
    Boss("p3/greyprince", "Grey Prince", "p3", 4),
    Boss("p3/uumuu", "Uumuu", "p3", 3),
    Boss("p3/hornet2", "Hornet 2", "p3", 4),
    Boss("p3/sly", "Sly", "p3", 4),
    Boss("p4/enragedguardian", "Enraged Guardian", "p4", 3),
    Boss("p4/lostkin", "Lost Kin", "p4", 3),
    Boss("p4/noeyes", "No Eyes", "p4", 4),
    Boss("p4/traitorlord", "Traitor Lord", "p4", 5),
    Boss("p4/whitedefender", "White Defender", "p4", 4),
    Boss("p4/failedchamp", "Failed Champion", "p4", 5),
    Boss("p4/markoth", "Markoth", "p4", 5),
    Boss("p4/watcherknights", "Watcher Knights", "p4", 6),
    Boss("p4/soultyrant", "Soul Tyrant", "p4", 6),
    Boss("p4/purevessel", "Pure Vessel", "p4", 6),
    Boss("p5/wingednosk", "Winged Nosk", "p5", 4),
    Boss("p5/sisters", "Sisters of Battle", "p5", 5),
    Boss("p5/kristalguardian", "Kristal Guardian", "p5", 4),
    Boss("p5/nkg", "Nail Master", "p5", 6),
    Boss("p5/radiance", "Radiance", "p5", 7),
)

BY_ID = {b.id: b for b in BOSSES}
BY_NAME = {b.name.lower(): b for b in BOSSES}

#: Canonical ordering used by the default curriculum.
DEFAULT_ORDER: tuple[str, ...] = tuple(b.id for b in sorted(BOSSES, key=lambda b: (b.tier, b.id)))


def resolve(spec: str) -> str:
    """Resolve a user-supplied boss spec to an encounter id.

    Accepts an encounter id (``p1/falseknight``), a display name
    (``False Knight``), an area prefix (``p3``), or ``all``.
    """
    key = spec.strip()
    if key.lower() == "all":
        return ",".join(b.id for b in BOSSES)
    if key in BY_ID:
        return key
    low = key.lower().replace(" ", "")
    for name, boss in BY_NAME.items():
        if name.replace(" ", "") == low:
            return boss.id
    if key in AREAS:
        return ",".join(b.id for b in BOSSES if b.area == key)
    if key in ("p1", "p2", "p3", "p4", "p5"):
        return key
    raise ValueError(
        f"unknown boss {spec!r}.  Try an id from the catalog, an area like 'p3', "
        "or 'all'."
    )


def resolve_list(spec: str) -> list[str]:
    """Resolve a comma/space separated boss spec into a list of encounter ids."""
    out: list[str] = []
    for part in spec.replace(",", " ").split():
        out.extend(resolve(part).split(","))
    seen: set[str] = set()
    unique = []
    for boss_id in out:
        if boss_id not in seen:
            seen.add(boss_id)
            unique.append(boss_id)
    if not unique:
        raise ValueError(f"no bosses matched {spec!r}")
    return unique


def display(boss_id: str) -> str:
    boss = BY_ID.get(boss_id)
    return f"{boss.name} ({boss.area}={AREAS.get(boss.area, boss.area)})" if boss else boss_id