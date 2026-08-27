"""
SCB RL Reward
---------------

Curriculum-based reward system for the Sparsest Cut Bound
reinforcement-learning environment.

CURRENTLY IMPLEMENTED:
    Phase 1 - Learn to construct low-SCB cuts.

FUTURE PHASES:
    Phase 2 - Learn when to STOP.
    Phase 3 - Learn aggressive SCB minimization.
    Phase 4 - Joint optimization.

IMPORTANT
---------
The GA is NEVER used by this reward module.

GA values may exist inside the environment for:
    - evaluation
    - logging
    - post-training comparison

but they must never enter PPO's transition reward.
"""


# ==========================================================
# CURRICULUM PHASES
# ==========================================================

PHASE_CONSTRUCTION = 1
PHASE_TERMINATION = 2
PHASE_OPTIMIZATION = 3
PHASE_JOINT = 4


# Default phase.
#
# The environment/trainer can explicitly pass a phase.
DEFAULT_PHASE = PHASE_CONSTRUCTION


# ==========================================================
# GENERAL HYPERPARAMETERS
# ==========================================================

# Invalid ADD / REMOVE / SWAP action.
INVALID_ACTION_PENALTY = -0.10


# Small penalty for consuming a step.
#
# This prevents completely useless wandering but is deliberately
# small so that the agent is free to restructure a cut.
TIME_PENALTY_WEIGHT = 0.01


# Numerical stability.
EPSILON = 1e-8


# ==========================================================
# PHASE 1
# ==========================================================

# ----------------------------------------------------------
# SCB improvement
# ----------------------------------------------------------

# Main Phase-1 learning signal.
#
# Lower SCB = better.
#
# The actual improvement is normalized:
#
#     (old_scb - new_scb) / old_scb
#
# Example:
#
#     10 -> 5
#         = +0.50
#
#     5 -> 10
#         = -1.00
#
SCB_IMPROVEMENT_WEIGHT = 1.00


# ----------------------------------------------------------
# First finite SCB
# ----------------------------------------------------------

# Reward for discovering the first usable SCB.
#
# The reward is based on:
#
#     1 / (1 + SCB)
#
# Therefore smaller first SCB values receive more reward.
FIRST_SCB_WEIGHT = 1.00


# ----------------------------------------------------------
# Best SCB discovery
# ----------------------------------------------------------

# Extra reward whenever the episode discovers a new best SCB.
#
# This encourages the agent to keep searching for better
# solutions instead of merely finding one valid solution.
NEW_BEST_WEIGHT = 0.50


# ----------------------------------------------------------
# Final Phase-1 reward
# ----------------------------------------------------------

# Reward given when the fixed Phase-1 episode reaches its
# maximum number of steps.
#
# It uses the BEST SCB discovered during the episode.
#
#     1 / (1 + best_scb)
#
FINAL_BEST_WEIGHT = 2.00


# ----------------------------------------------------------
# Regression
# ----------------------------------------------------------

# Penalty when a valid SCB is destroyed.
#
# This is intentionally mild.
#
# The agent must be allowed to do:
#
#     good cut
#         ↓
#     destroy / restructure
#         ↓
#     better cut
#
REGRESSION_PENALTY = 0.15


# ==========================================================
# PHASE 2 PLACEHOLDERS
# ==========================================================

# These are intentionally NOT active yet.

PHASE2_STOP_WEIGHT = 0.0

PHASE2_INVALID_STOP_PENALTY = 0.0

PHASE2_TERMINATION_THRESHOLD = None


# ==========================================================
# PHASE 3 PLACEHOLDERS
# ==========================================================

# These will later control stronger minimum-SCB optimization.

PHASE3_SCBOBJ_WEIGHT = 0.0

PHASE3_GLOBAL_BEST_WEIGHT = 0.0


# ==========================================================
# PHASE 4 PLACEHOLDERS
# ==========================================================

# Final combined curriculum weights.

PHASE4_CONSTRUCTION_WEIGHT = 0.0

PHASE4_TERMINATION_WEIGHT = 0.0

PHASE4_OPTIMIZATION_WEIGHT = 0.0


# ==========================================================
# SCB HELPERS
# ==========================================================

def _is_finite_scb(scb):
    """
    Return True if SCB is a meaningful finite value.
    """

    if scb is None:
        return False

    try:

        value = float(scb)

    except (TypeError, ValueError):

        return False

    return (
        value > EPSILON
        and value != float("inf")
    )


def scb_quality(scb):
    """
    Convert SCB into a bounded quality score.

    Lower SCB = better quality.

        SCB = 1   -> 0.500
        SCB = 2   -> 0.333
        SCB = 5   -> 0.167
        SCB = 10  -> 0.091

    Invalid / infinite SCB -> 0.
    """

    if not _is_finite_scb(scb):

        return 0.0

    return (
        1.0
        / (1.0 + float(scb))
    )


# ==========================================================
# STEP BUDGET
# ==========================================================

def get_step_budget(state):
    """
    Estimate a reasonable exploration budget for a graph.

    This is NOT the hard episode limit.

    It is only used to scale the small time penalty.
    """

    problem = state.problem

    num_edges = len(problem.edges)

    num_sessions = len(problem.sessions)

    budget = (
        0.25 * num_edges
        + 2.0 * num_sessions
    )

    return max(
        budget,
        1.0
    )


# ==========================================================
# PHASE 1 REWARD
# ==========================================================

def _compute_phase1_reward(
    old_state,
    new_state,
    best_scb=None,
    episode_done=False,
    new_best=False,
):
    """
    Compute Phase-1 reward.

    Phase 1 objective:

        Find a low-SCB cut.

    STOP is intentionally NOT rewarded here.

    The agent is expected to operate for a fixed horizon.
    """

    old_scb = old_state.scb

    new_scb = new_state.scb

    old_valid = _is_finite_scb(
        old_scb
    )

    new_valid = _is_finite_scb(
        new_scb
    )

    # ======================================================
    # 1. SCB PROGRESS
    # ======================================================

    scb_reward = 0.0

    # ------------------------------------------------------
    # Valid -> Valid
    # ------------------------------------------------------

    if old_valid and new_valid:

        improvement = (
            float(old_scb) - float(new_scb)
        ) / max(
            abs(float(old_scb)),
            EPSILON
        )

        scb_reward = (
            improvement
            * SCB_IMPROVEMENT_WEIGHT
        )

    # ------------------------------------------------------
    # Invalid -> Valid
    # ------------------------------------------------------

    elif (
        not old_valid
        and new_valid
    ):

        # First finite SCB discovered.
        #
        # Smaller SCB = larger reward.
        scb_reward = (
            scb_quality(new_scb)
            * FIRST_SCB_WEIGHT
        )

    # ------------------------------------------------------
    # Valid -> Invalid
    # ------------------------------------------------------

    elif (
        old_valid
        and not new_valid
    ):

        # Mild penalty only.
        #
        # Restructuring is allowed.
        scb_reward = (
            -REGRESSION_PENALTY
        )

    # ======================================================
    # 2. NEW BEST SCB
    # ======================================================

    best_bonus = 0.0

    if (
        new_valid
        and new_best
    ):

        best_bonus = (
            scb_quality(new_scb)
            * NEW_BEST_WEIGHT
        )

    # ======================================================
    # 3. FINAL BEST-SCB REWARD
    # ======================================================

    final_reward = 0.0

    if (
        episode_done
        and best_scb is not None
        and _is_finite_scb(best_scb)
    ):

        final_reward = (
            scb_quality(best_scb)
            * FINAL_BEST_WEIGHT
        )

    # ======================================================
    # 4. TOTAL
    # ======================================================

    total = (
        scb_reward
        + best_bonus
        + final_reward
    )

    return {

        "separation":
            0.0,

        "scb":
            scb_reward,

        "best_bonus":
            best_bonus,

        "stop":
            0.0,

        "final":
            final_reward,

        "time":
            0.0,

        "total":
            total,
    }


# ==========================================================
# REWARD COMPONENTS
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
    Compute curriculum reward components.

    Existing arguments are preserved for compatibility.

    Additional arguments:

    phase :
        Curriculum phase.

    episode_done :
        True when the current transition reaches the
        Phase-1 fixed horizon.

    ----------------------------------------------------------

    Phase 1:

        Learn to construct low-SCB cuts.

        STOP reward is disabled.

    Phase 2:

        Placeholder.

    Phase 3:

        Placeholder.

    Phase 4:

        Placeholder.
    """

    # ======================================================
    # INVALID ACTION
    # ======================================================

    if invalid:

        return {

            "separation":
                0.0,

            "scb":
                0.0,

            "best_bonus":
                0.0,

            "stop":
                0.0,

            "final":
                0.0,

            "time":
                INVALID_ACTION_PENALTY,

            "total":
                INVALID_ACTION_PENALTY,
        }


    # ======================================================
    # PHASE 1
    # ======================================================

    if phase == PHASE_CONSTRUCTION:

        components = _compute_phase1_reward(

            old_state=old_state,

            new_state=new_state,

            best_scb=best_scb,

            episode_done=episode_done,

            new_best=new_best,
        )

        # --------------------------------------------------
        # Small time penalty.
        #
        # IMPORTANT:
        # The actual maximum number of steps is controlled
        # by the environment.
        # --------------------------------------------------

        step = max(
            new_state.step,
            1
        )

        step_budget = get_step_budget(
            new_state
        )

        time_penalty = (
            TIME_PENALTY_WEIGHT
            * (
                step
                / step_budget
            )
        )

        time_penalty = min(
            time_penalty,
            TIME_PENALTY_WEIGHT
        )

        components["time"] = (
            -time_penalty
        )

        components["total"] += (
            -time_penalty
        )

        return components


    # ======================================================
    # PHASE 2
    # ======================================================

    if phase == PHASE_TERMINATION:

        # --------------------------------------------------
        # PLACEHOLDER
        # --------------------------------------------------
        #
        # STOP learning will be implemented later.
        #
        # Do not silently reuse Phase 1 here.
        # --------------------------------------------------

        raise NotImplementedError(
            "Phase 2 reward has not been implemented yet."
        )


    # ======================================================
    # PHASE 3
    # ======================================================

    if phase == PHASE_OPTIMIZATION:

        # --------------------------------------------------
        # PLACEHOLDER
        # --------------------------------------------------

        raise NotImplementedError(
            "Phase 3 reward has not been implemented yet."
        )


    # ======================================================
    # PHASE 4
    # ======================================================

    if phase == PHASE_JOINT:

        # --------------------------------------------------
        # PLACEHOLDER
        # --------------------------------------------------

        raise NotImplementedError(
            "Phase 4 reward has not been implemented yet."
        )


    # ======================================================
    # UNKNOWN PHASE
    # ======================================================

    raise ValueError(
        f"Unknown SCB-RL training phase: {phase}"
    )


# ==========================================================
# MAIN REWARD FUNCTION
# ==========================================================

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
    """
    Compute scalar reward.

    Existing function name/signature is preserved.

    The GA is deliberately absent from this calculation.
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
    )

    return components["total"]


# ==========================================================
# PHASE INFORMATION
# ==========================================================

def phase_name(phase):
    """
    Return a human-readable curriculum phase name.
    """

    names = {

        PHASE_CONSTRUCTION:
            "CONSTRUCTION",

        PHASE_TERMINATION:
            "TERMINATION",

        PHASE_OPTIMIZATION:
            "OPTIMIZATION",

        PHASE_JOINT:
            "JOINT",
    }

    return names.get(
        phase,
        "UNKNOWN"
    )