"""
SCB-RL Phase 2 Reward
=====================

Goal:
    Teach the agent to FIND a useful finite SCB and then STOP
    when the current solution is good enough.

Phase 2 deliberately does NOT use the GA/teacher solution.

The reward has four ideas:

1. Zero separation is strongly bad.
2. Any real separation is useful and receives a small positive signal.
3. Every consumed step has a cost.
4. STOP is judged on the CURRENT state:
       - zero separation -> large penalty
       - finite SCB -> reward proportional to SCB quality

This creates the intended "blitz exam" behavior:
    search -> obtain useful separation -> improve enough -> STOP.

Phase 3/4 hooks are left below for future expansion.
"""

from SCB_RL.actions import ActionType


# ==========================================================
# CURRICULUM PHASES
# ==========================================================

PHASE_CONSTRUCTION = 1
PHASE_TERMINATION = 2
PHASE_OPTIMIZATION = 3
PHASE_JOINT = 4

DEFAULT_PHASE = PHASE_TERMINATION


# ==========================================================
# PHASE 2 HYPERPARAMETERS
# ==========================================================

# Small cost for spending another step.
#
# At 50 steps:  -1.00
# At 75 steps:  -1.50
# At 100 steps: -2.00
# At 150 steps: -3.00
#
# This is deliberately not an artificial hard step limit.
TIME_PENALTY_WEIGHT = 0.02


# Strong signal that the current cut has achieved nothing.
ZERO_SEPARATION_PENALTY = -1.50


# Small positive reward for obtaining ANY separation.
#
# This is intentionally much smaller than a good STOP reward.
SEPARATION_REWARD = 0.10


# Extra quality signal while searching.
#
# Lower SCB -> larger quality.
SEARCH_QUALITY_WEIGHT = 0.30


# STOP rewards.
#
# STOP is where the main Phase-2 signal lives.
STOP_QUALITY_WEIGHT = 2.50

# Additional bonus for stopping with at least one separated
# session. This makes a successful STOP clearly preferable
# to an invalid STOP.
VALID_STOP_BONUS = 0.25

# Extra penalty for stopping with zero separation.
INVALID_STOP_PENALTY = -3.00


# Invalid graph-edit action.
INVALID_ACTION_PENALTY = -0.25


# Numerical stability.
EPSILON = 1e-8


# ==========================================================
# PHASE 3 / 4 PLACEHOLDERS
# ==========================================================

PHASE3_SCBOBJ_WEIGHT = 0.0
PHASE3_GLOBAL_BEST_WEIGHT = 0.0

PHASE4_CONSTRUCTION_WEIGHT = 0.0
PHASE4_TERMINATION_WEIGHT = 0.0
PHASE4_OPTIMIZATION_WEIGHT = 0.0


# ==========================================================
# HELPERS
# ==========================================================

def _is_finite_scb(scb):
    if scb is None:
        return False

    try:
        value = float(scb)
    except (TypeError, ValueError):
        return False

    return value > EPSILON and value != float("inf")


def scb_quality(scb):
    """
    Bounded quality score.

        SCB 0.25 -> 0.800
        SCB 0.50 -> 0.667
        SCB 1.00 -> 0.500
        SCB 2.00 -> 0.333
        SCB 5.00 -> 0.167
        SCB 10.0 -> 0.091
        SCB inf  -> 0.000
    """
    if not _is_finite_scb(scb):
        return 0.0

    return 1.0 / (1.0 + float(scb))


def _time_penalty(state):
    step = max(int(getattr(state, "step", 0)), 1)
    return TIME_PENALTY_WEIGHT


# ==========================================================
# PHASE 2
# ==========================================================

def _compute_phase2_reward(
    old_state,
    new_state,
    action,
    is_stop=False,
):
    """
    Compute one Phase-2 transition.

    IMPORTANT:
        No GA/teacher value enters this calculation.
    """

    old_sep = old_state.separated_count
    new_sep = new_state.separated_count
    new_scb = new_state.scb

    separation_component = 0.0
    scb_component = 0.0
    stop_component = 0.0
    time_component = -_time_penalty(new_state)

    # ------------------------------------------------------
    # STOP
    # ------------------------------------------------------

    if is_stop:

        if new_sep <= 0 or new_state.cut_size <= 0:

            stop_component = (
                INVALID_STOP_PENALTY
                + ZERO_SEPARATION_PENALTY
            )

        else:

            stop_component = (
                VALID_STOP_BONUS
                + STOP_QUALITY_WEIGHT
                * scb_quality(new_scb)
            )

        total = (
            separation_component
            + scb_component
            + stop_component
            + time_component
        )

        return {
            "separation": separation_component,
            "scb": scb_component,
            "stop": stop_component,
            "time": time_component,
            "total": total,
        }

    # ------------------------------------------------------
    # Normal action with zero separation
    # ------------------------------------------------------

    if new_sep <= 0:

        separation_component = ZERO_SEPARATION_PENALTY

    # ------------------------------------------------------
    # Normal action with separation
    # ------------------------------------------------------

    else:

        # Reward the transition into useful territory.
        if old_sep <= 0:
            separation_component += SEPARATION_REWARD
        else:
            # Small continuing signal so separation remains useful,
            # but it is not large enough to replace the STOP objective.
            separation_component += SEPARATION_REWARD * 0.25

        scb_component = (
            SEARCH_QUALITY_WEIGHT
            * scb_quality(new_scb)
        )

    total = (
        separation_component
        + scb_component
        + time_component
    )

    return {
        "separation": separation_component,
        "scb": scb_component,
        "stop": 0.0,
        "time": time_component,
        "total": total,
    }


# ==========================================================
# PUBLIC API
# ==========================================================

def compute_reward_components(
    old_state,
    new_state,
    action,
    invalid=False,
    best_scb=None,
    is_stop=False,
    new_best=False,
    phase=DEFAULT_PHASE,
    episode_done=False,
):
    """
    Curriculum-compatible reward API.

    Phase 2 is active by default.

    Parameters kept for compatibility with the existing
    environment/trainer.
    """

    if invalid:

        return {
            "separation": 0.0,
            "scb": 0.0,
            "stop": 0.0,
            "time": INVALID_ACTION_PENALTY,
            "total": INVALID_ACTION_PENALTY,
        }

    if phase == PHASE_TERMINATION:

        return _compute_phase2_reward(
            old_state=old_state,
            new_state=new_state,
            action=action,
            is_stop=is_stop,
        )

    if phase == PHASE_CONSTRUCTION:
        raise NotImplementedError(
            "Phase 1 reward should use the Phase-1 reward file."
        )

    if phase == PHASE_OPTIMIZATION:
        raise NotImplementedError(
            "Phase 3 reward has not been implemented yet."
        )

    if phase == PHASE_JOINT:
        raise NotImplementedError(
            "Phase 4 reward has not been implemented yet."
        )

    raise ValueError(f"Unknown SCB-RL phase: {phase}")


def compute_reward(
    old_state,
    new_state,
    action,
    done=False,
    invalid=False,
    best_scb=None,
    is_stop=False,
    new_best=False,
    phase=DEFAULT_PHASE,
):
    components = compute_reward_components(
        old_state=old_state,
        new_state=new_state,
        action=action,
        invalid=invalid,
        best_scb=best_scb,
        is_stop=is_stop,
        new_best=new_best,
        phase=phase,
        episode_done=done,
    )

    return components["total"]


def phase_name(phase):
    return {
        PHASE_CONSTRUCTION: "CONSTRUCTION",
        PHASE_TERMINATION: "TERMINATION",
        PHASE_OPTIMIZATION: "OPTIMIZATION",
        PHASE_JOINT: "JOINT",
    }.get(phase, "UNKNOWN")
