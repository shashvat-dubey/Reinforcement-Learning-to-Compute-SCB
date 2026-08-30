"""
SCB-RL Phase 3 Reward
=====================

Phase 3 starts from the FINAL cut produced by Phase 2.

Goal:
    Given an existing cut, learn to reduce its SCB and then STOP.

Primary signal:
    lower SCB = better

Important:
    Separation is NOT independently rewarded for increasing.
    A decrease in separation is acceptable if it produces a lower SCB.

GA / teacher values are never used by this reward.
"""

# ==========================================================
# CURRICULUM PHASES
# ==========================================================

PHASE_CONSTRUCTION = 1
PHASE_TERMINATION = 2
PHASE_OPTIMIZATION = 3
PHASE_JOINT = 4

DEFAULT_PHASE = PHASE_OPTIMIZATION


# ==========================================================
# PHASE 3 HYPERPARAMETERS
# ==========================================================

# Invalid ADD / REMOVE / SWAP action.
INVALID_ACTION_PENALTY = -0.25

# Main optimization signal.
#
# Relative improvement:
#     10 -> 8  = +0.20
#     10 -> 12 = -0.20
#
# Lower SCB is always better.
SCB_IMPROVEMENT_WEIGHT = 2.00

# Additional signal when a genuinely new best SCB is found.
NEW_BEST_BONUS = 0.50

# Mild penalty when a finite solution is temporarily destroyed.
#
# We deliberately allow:
#     good -> inf -> better
# because restructuring can require temporary regression.
REGRESSION_PENALTY = 0.15

# STOP with no meaningful finite SCB.
INVALID_STOP_PENALTY = -2.50

# Valid STOP reward.
#
# The improvement term compares the final SCB with the
# Phase-2 starting SCB.  The quality term gives an absolute
# bounded signal as well.
STOP_IMPROVEMENT_WEIGHT = 3.00
STOP_QUALITY_WEIGHT = 1.00
VALID_STOP_BONUS = 0.25

# Small cost for consuming optimization steps.
TIME_PENALTY_WEIGHT = 0.02

EPSILON = 1e-8


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
    Bounded absolute SCB quality.

        SCB 1  -> 0.500
        SCB 2  -> 0.333
        SCB 5  -> 0.167
        SCB 10 -> 0.091

    Invalid / infinite SCB -> 0.
    """
    if not _is_finite_scb(scb):
        return 0.0

    return 1.0 / (1.0 + float(scb))


def relative_scb_improvement(old_scb, new_scb):
    """
    Positive = SCB improved.
    Zero     = unchanged / unavailable.
    Negative = SCB worsened.
    """
    if not (
        _is_finite_scb(old_scb)
        and _is_finite_scb(new_scb)
    ):
        return 0.0

    return (
        float(old_scb) - float(new_scb)
    ) / max(
        abs(float(old_scb)),
        EPSILON,
    )


def get_step_budget(state):
    """
    Graph-scaled soft optimization budget.

    This is NOT a hard episode limit.
    """
    problem = state.problem

    num_edges = len(problem.edges)
    num_sessions = len(problem.sessions)

    budget = (
        0.25 * num_edges
        + 2.0 * num_sessions
    )

    return max(budget, 1.0)


def _time_penalty(state):
    step = max(
        int(getattr(state, "step", 0)),
        1,
    )

    budget = get_step_budget(state)

    penalty = (
        TIME_PENALTY_WEIGHT
        * step
        / budget
    )

    return min(
        penalty,
        TIME_PENALTY_WEIGHT,
    )


# ==========================================================
# PHASE 3 REWARD
# ==========================================================

def _compute_phase3_reward(
    old_state,
    new_state,
    action,
    best_scb=None,
    is_stop=False,
    new_best=False,
    initial_scb=None,
):
    """
    Compute one Phase-3 transition.

    There is deliberately NO standalone separation reward.

    The agent is rewarded for:
        SCB decrease
        new best SCB
        valid stopping

    and penalized for:
        SCB increase
        unnecessary time
        invalid STOP
    """

    old_scb = old_state.scb
    new_scb = new_state.scb

    old_valid = _is_finite_scb(old_scb)
    new_valid = _is_finite_scb(new_scb)

    scb_component = 0.0
    best_component = 0.0
    regression_component = 0.0
    stop_component = 0.0

    # ------------------------------------------------------
    # STOP
    # ------------------------------------------------------

    if is_stop:

        valid_solution = (
            new_state.separated_count > 0
            and new_state.cut_size > 0
            and new_valid
        )

        if not valid_solution:

            stop_component = INVALID_STOP_PENALTY

        else:

            initial_improvement = 0.0

            if _is_finite_scb(initial_scb):

                initial_improvement = (
                    float(initial_scb) - float(new_scb)
                ) / max(
                    abs(float(initial_scb)),
                    EPSILON,
                )

                # Keep the terminal contribution bounded.
                initial_improvement = max(
                    min(initial_improvement, 1.0),
                    -1.0,
                )

            stop_component = (
                VALID_STOP_BONUS
                + STOP_IMPROVEMENT_WEIGHT
                * initial_improvement
                + STOP_QUALITY_WEIGHT
                * scb_quality(new_scb)
            )

        time_component = -_time_penalty(new_state)

        total = (
            scb_component
            + best_component
            + regression_component
            + stop_component
            + time_component
        )

        return {
            "scb": scb_component,
            "best_bonus": best_component,
            "stop": stop_component,
            "regression": regression_component,
            "time": time_component,
            "total": total,
        }

    # ------------------------------------------------------
    # NORMAL GRAPH-EDIT ACTION
    # ------------------------------------------------------

    if old_valid and new_valid:

        improvement = relative_scb_improvement(
            old_scb,
            new_scb,
        )

        scb_component = (
            SCB_IMPROVEMENT_WEIGHT
            * improvement
        )

    elif not old_valid and new_valid:

        # Recovery from an invalid/infinite SCB.
        # Give a bounded signal rather than comparing infinity.
        scb_component = (
            SCB_IMPROVEMENT_WEIGHT
            * scb_quality(new_scb)
        )

    elif old_valid and not new_valid:

        # Mildly discourage destroying a solution, while still
        # allowing temporary restructuring.
        regression_component = -REGRESSION_PENALTY

    # ------------------------------------------------------
    # NEW BEST
    # ------------------------------------------------------

    if new_best and new_valid:

        best_component = (
            NEW_BEST_BONUS
            * scb_quality(new_scb)
        )

    # ------------------------------------------------------
    # TIME
    # ------------------------------------------------------

    time_component = -_time_penalty(new_state)

    total = (
        scb_component
        + best_component
        + regression_component
        + time_component
    )

    return {
        "scb": scb_component,
        "best_bonus": best_component,
        "stop": 0.0,
        "regression": regression_component,
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
    initial_scb=None,
):
    """
    Phase-3-compatible reward API.

    initial_scb is the SCB of the final cut handed over by Phase 2.
    It is used only for the terminal STOP improvement signal.
    """

    if invalid:

        return {
            "scb": 0.0,
            "best_bonus": 0.0,
            "stop": 0.0,
            "regression": 0.0,
            "time": INVALID_ACTION_PENALTY,
            "total": INVALID_ACTION_PENALTY,
        }

    if phase == PHASE_OPTIMIZATION:

        return _compute_phase3_reward(
            old_state=old_state,
            new_state=new_state,
            action=action,
            best_scb=best_scb,
            is_stop=is_stop,
            new_best=new_best,
            initial_scb=initial_scb,
        )

    if phase == PHASE_CONSTRUCTION:
        raise NotImplementedError(
            "Use the Phase-1 reward for construction."
        )

    if phase == PHASE_TERMINATION:
        raise NotImplementedError(
            "Use phase2_reward.py for Phase 2."
        )

    if phase == PHASE_JOINT:
        raise NotImplementedError(
            "Phase 4 reward has not been implemented yet."
        )

    raise ValueError(
        f"Unknown SCB-RL phase: {phase}"
    )


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
    initial_scb=None,
):
    """
    Return only the scalar reward.
    """

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
        initial_scb=initial_scb,
    )

    return components["total"]


def phase_name(phase):
    return {
        PHASE_CONSTRUCTION: "CONSTRUCTION",
        PHASE_TERMINATION: "TERMINATION",
        PHASE_OPTIMIZATION: "OPTIMIZATION",
        PHASE_JOINT: "JOINT",
    }.get(phase, "UNKNOWN")
