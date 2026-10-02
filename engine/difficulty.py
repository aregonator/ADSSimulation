"""The difficulty schedule — the single source of truth for how difficulty
1-100 maps onto simulation parameters.

WHY THIS IS ITS OWN MODULE.  The schedule has seven consumers spread across
``engine.simulation`` (drain multiplier, config interpolation, health
recovery, ambient hazard), ``engine.grid`` (depletion scaling), ``engine.events`` (epidemic
seeding) and ``scenarios.scenario_base`` (auto event schedule).  Open-coding
the knee separately at each site would let two expressions of the same
formula silently disagree.

It cannot live in ``engine.simulation``, because ``simulation`` already imports
``grid`` and ``events`` — having them import back would be a circular import.
So the schedule lives here, in a leaf module with NO intra-package imports, and
everything else depends on it.  That is what makes "one knee, one expression"
structurally enforceable rather than a convention people have to remember.

``engine.simulation`` re-exports every public name below, so
``from engine.simulation import difficulty_multiplier`` keeps working.
"""

from __future__ import annotations

__all__ = [
    "DIFFICULTY_KNEE_LEVEL",
    "DIFFICULTY_TAIL_SLOPE",
    "DIFFICULTY_T_MAX",
    "effective_difficulty",
    "past_knee",
    "difficulty_t",
    "difficulty_multiplier",
    "AMBIENT_HAZARD_FULL",
    "AMBIENT_HAZARD_RAMP_END_LEVEL",
    "ambient_hazard",
    "validate_schedule",
]


#: Difficulty level at which the parameter schedule MAY change slope — "the
#: knee".  Below it the schedule advances one full step per difficulty level;
#: at and above it, it advances :data:`DIFFICULTY_TAIL_SLOPE` of a step per
#: level.
#:
#: At the shipped setting ``DIFFICULTY_TAIL_SLOPE = 1.75`` the knee is ACTIVE:
#: the schedule advances 1.75 steps per level above D90, so D90, D95 and D100
#: are three distinct, increasingly harsh conditions, and D95/D100
#: extrapolate past the D1-D90 lerp endpoints.  Below D90 the knee is always
#: inert, at every slope: D1-D90 are bit-identical regardless of
#: ``DIFFICULTY_TAIL_SLOPE``.  The knee machinery is kept (not deleted) so
#: schedules at slope 1.0 (no extrapolation) and slope 0.0 (hard plateau) can
#: each be reproduced from this codebase by flipping one constant.
#:
#: Archives generated at a different ``DIFFICULTY_TAIL_SLOPE`` are not
#: directly comparable: at slope 0.0, D90, D95 and D100 are
#: parameter-identical replicates (19 distinct conditions on the 21-level
#: grid); at slope 1.0 they are distinct but non-extrapolated.  Check
#: ``manifest.json``'s ``schedule`` block before comparing archives.
DIFFICULTY_KNEE_LEVEL = 90

#: How steeply the schedule keeps rising past the knee, as a fraction of the
#: pre-knee rate.  ``1.75`` extrapolates every lerp built on
#: :func:`difficulty_t` past its documented 1.0 endpoint: D100 reaches
#: ``difficulty_t(100) = 1.0758`` (exactly ``(89 + 10*1.75)/99``), which stays
#: inside the ``DIFFICULTY_T_MAX = 1.25`` domain where every channel remains
#: physically meaningful (see below).  ``1.0`` and ``0.0`` are RETAINED as
#: reproducible values, not deleted:
#:   - ``1.0`` = no knee at all, the schedule linear across the whole 1-100
#:     domain, D100 at the lerp endpoint exactly (drain 2.00, regen 0.25,
#:     density 0.40, warnings 5).
#:   - ``0.0`` = a hard plateau (D90-D100 identical).
#:
#: At 1.75, D95 and D100 map to effective difficulties 98.75 and 107.5
#: (``90 + 1.75 * (d - 90)``; see :func:`effective_difficulty`).
#:
#: Changing this is a BEHAVIOURAL change to D91-D100 and requires a resweep.
DIFFICULTY_TAIL_SLOPE = 1.75


def effective_difficulty(difficulty: int) -> float:
    """Fold *difficulty* through the knee, in difficulty-LEVEL space (1-100).

    This is the single source of truth for the knee.  :func:`difficulty_t` is
    derived from it, so the difficulty knobs cannot drift apart into one knee
    expressed two different ways with two different answers.

    Why level space is primitive and ``t`` space derived, and not the other way
    round.  Three call sites (grid depletion, epidemic seeding, health recovery)
    need an effective difficulty *level*, not an interpolation position.
    Recovering a level from ``t`` via ``1 + t * 99`` is not exact in IEEE-754:
    it disagrees with the integer at 7 of the 100 levels (14, 27, 28, 53, 55,
    58, 60), and two of those — 55 and 60 — are on the swept grid.  The opposite
    direction, ``(level - 1) / 99``, is exact at every level.  So the level form
    must be the primitive for the mapping to stay exact in both directions.

    Out-of-range inputs are clamped to [1, 100], matching every consumer site.
    No caller in this repo supplies one — the only construction path,
    ``SimulationConfig.from_difficulty``, clamps first.
    """
    d = max(1, min(100, int(difficulty)))
    if d <= DIFFICULTY_KNEE_LEVEL:
        return float(d)
    return DIFFICULTY_KNEE_LEVEL + (d - DIFFICULTY_KNEE_LEVEL) * DIFFICULTY_TAIL_SLOPE


def past_knee(difficulty: int) -> bool:
    """True iff the SCHEDULE has advanced beyond the knee at *difficulty*.

    A statement about the schedule, not about the level.  At
    ``DIFFICULTY_TAIL_SLOPE = 0.0`` this is False for EVERY difficulty in
    [1, 100]; at any slope > 0 it is True exactly for
    ``difficulty > DIFFICULTY_KNEE_LEVEL``.

    It lets a consumer clamp scope to the pre-knee range WITHOUT
    re-expressing the knee's position.  Because it derives from
    :func:`effective_difficulty`, a scoped clamp is inert-by-construction at
    slope 0.0.
    """
    return effective_difficulty(difficulty) > DIFFICULTY_KNEE_LEVEL


def difficulty_t(difficulty: int) -> float:
    """Interpolation position of *difficulty* on the 1-100 schedule, in [0, 1].

    Every difficulty-scaled parameter in this package is a lerp against this
    value.  Callers must not re-derive it; that is what this function exists to
    prevent.  The knee's own position is ``difficulty_t(DIFFICULTY_KNEE_LEVEL)``
    — deliberately not a separate module constant, so it cannot go stale.

    Range guarantee: the result is in ``[0, difficulty_t(100)] ⊆
    [0, DIFFICULTY_T_MAX]``.  Values above 1.0 occur only above the knee, when
    ``DIFFICULTY_TAIL_SLOPE > 1``, and mean deliberate extrapolation past the
    documented lerp endpoints.  At the shipped ``DIFFICULTY_TAIL_SLOPE =
    1.75``, ``difficulty_t(100) == 1.0757575757575757`` (``D100 t =
    1.0758``): every downstream clamp expressed as a *physical* bound
    (``max(0.0, regen)``, ``max(1, warnings)``) stays inert inside
    ``DIFFICULTY_T_MAX`` and every emitted parameter stays inside its
    documented physically-valid range.  :func:`validate_schedule` is what
    proves the module-level constants keep this guarantee; it is not
    re-checked here on every call.  At slope ``1.0`` the result never
    exceeds 1.0.
    """
    return (effective_difficulty(difficulty) - 1.0) / 99.0


#: The physical-validity domain of every lerp in this package.
#: ``difficulty_t`` may legitimately exceed 1.0 once
#: ``DIFFICULTY_TAIL_SLOPE > 1``, but every lerp built on it stops being
#: physically meaningful somewhere above 1.0: warnings would round below the
#: 1-cycle minimum at t ≈ 1.267, regen would go negative at t = 4/3, stocks
#: would go negative at t = 1.5, and density would go negative at t = 1.8.
#: 1.25 is the largest round number below all four limits, so it is the
#: shared ceiling for every channel rather than a per-channel bound.
DIFFICULTY_T_MAX = 1.25


#: Ambient hazard at full strength: a uniform per-cycle hazard added to every
#: cell on top of its terrain's own hazard (``engine.grid.TERRAIN_STATS``).
#: Reached at :data:`AMBIENT_HAZARD_RAMP_END_LEVEL` and held constant above it.
#: See :func:`ambient_hazard`.
AMBIENT_HAZARD_FULL = 0.008

#: Difficulty level at which the ambient hazard reaches
#: :data:`AMBIENT_HAZARD_FULL`.  From D1 (zero) up to this level it rises
#: linearly in difficulty-level space; at and above it, it is constant.
AMBIENT_HAZARD_RAMP_END_LEVEL = 20


def validate_schedule() -> None:
    """Raise ValueError unless the CURRENT schedule is valid at every level 1..100.

    The domain is finite (inputs clamp to [1, 100]) and the schedule is a pure
    function of two module constants, so checking all 100 levels once proves
    every possible call valid.  That is why there is NO per-call check.
    Anything that mutates the constants at runtime (tests, pilots) must call
    this again.  ``test_difficulty_schedule.py``'s ``_slope()`` does, and
    ``benchmark_core._schedule_block()`` does, so a sweep cannot write a
    manifest for an invalid schedule.

    Why this lives here rather than inside :func:`difficulty_t` or
    :func:`effective_difficulty`: four of the six consumers
    (``engine.grid``, ``engine.events``, the health-recovery site, and
    :func:`difficulty_t` itself) call :func:`effective_difficulty` directly,
    so a check placed inside :func:`difficulty_t` would not cover them. A
    check inside :func:`effective_difficulty` would have to re-transcribe
    ``(x - 1) / 99`` — a second expression of the same normalisation, which is
    exactly the class of bug this module exists to prevent. The eager,
    whole-domain check expressed through :func:`difficulty_t` is both
    complete and free of a second expression.
    """
    if DIFFICULTY_TAIL_SLOPE < 0:
        raise ValueError(
            f"DIFFICULTY_TAIL_SLOPE={DIFFICULTY_TAIL_SLOPE!r} must be >= 0: "
            "the schedule must be monotone non-decreasing above the knee")
    if not 0.0 <= AMBIENT_HAZARD_FULL < 1.0:
        raise ValueError(
            f"AMBIENT_HAZARD_FULL={AMBIENT_HAZARD_FULL!r} must be in [0, 1): "
            "it is a per-cycle hazard added to every cell")
    if (not isinstance(AMBIENT_HAZARD_RAMP_END_LEVEL, int)
            or not 2 <= AMBIENT_HAZARD_RAMP_END_LEVEL <= DIFFICULTY_KNEE_LEVEL):
        raise ValueError(
            f"AMBIENT_HAZARD_RAMP_END_LEVEL={AMBIENT_HAZARD_RAMP_END_LEVEL!r} "
            f"must be an int in [2, {DIFFICULTY_KNEE_LEVEL}]: the ramp runs in "
            "difficulty-level space below the knee")
    worst = max(difficulty_t(d) for d in range(1, 101))
    if worst > DIFFICULTY_T_MAX:
        raise ValueError(
            f"DIFFICULTY_TAIL_SLOPE={DIFFICULTY_TAIL_SLOPE!r} gives "
            f"difficulty_t(100)={worst!r} > DIFFICULTY_T_MAX={DIFFICULTY_T_MAX!r}: "
            "some lerp built on difficulty_t would extrapolate past its "
            "physical-validity domain")


validate_schedule()   # import time: every spawned worker re-runs this on import


def difficulty_multiplier(difficulty: int) -> float:
    """Map difficulty 1-100 to the health-drain multiplier, starting at 0.50.

    At the shipped ``DIFFICULTY_TAIL_SLOPE = 1.75`` this rises linearly to
    D90 (1.8485, unchanged -- D1-D90 are bit-identical across every slope),
    then EXTRAPOLATES past the 2.00 endpoint to D95 1.9811 and D100 2.1136.
    At slope ``1.0`` it rises to 2.00 at D100 exactly (D90 1.8485, D95
    1.9242) with no extrapolation.  At slope ``0.0``, D90-D100 all return
    1.8485.
    """
    return round(0.50 + difficulty_t(difficulty) * 1.50, 4)


def ambient_hazard(difficulty: int) -> float:
    """Ambient hazard at *difficulty*: the uniform hazard every cell carries on
    top of its terrain hazard.

    ``0`` at D1, rising linearly to :data:`AMBIENT_HAZARD_FULL` (0.008) at
    :data:`AMBIENT_HAZARD_RAMP_END_LEVEL` (D20), constant from there to D100:

        D1 0.0   D5 0.0016842   D10 0.0037895   D15 0.0058947   D>=20 0.008

    The ramp is expressed in difficulty-LEVEL space,
    ``(effective_difficulty(d) - 1) / (AMBIENT_HAZARD_RAMP_END_LEVEL - 1)``,
    for the reason :func:`effective_difficulty` gives: level space is exact at
    every integer, whereas the equivalent ``difficulty_t(d) /
    difficulty_t(20)`` differs by one ULP at D5, D10 and D15.  These are
    exactly the values at D1/5/10/15 and D>=20, and the ramp position at D20
    is exactly 1.0.

    Because the ramp ends below :data:`DIFFICULTY_KNEE_LEVEL` (checked by
    :func:`validate_schedule`), the ambient hazard does not depend on
    :data:`DIFFICULTY_TAIL_SLOPE`.  Out-of-range inputs clamp to [1, 100] via
    :func:`effective_difficulty`.

    Consumed by ``SimulationConfig`` (``engine.simulation``), which carries the
    value to ``Grid`` and every ``Cell``; it is recorded in each scenario plan
    and covered by the manifest's ``schedule_digest``.
    """
    ramp = ((effective_difficulty(difficulty) - 1.0)
            / (AMBIENT_HAZARD_RAMP_END_LEVEL - 1))
    return AMBIENT_HAZARD_FULL * min(1.0, ramp)
