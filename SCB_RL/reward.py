"""
SCB RL Reward
V2-D

Reward philosophy
-----------------

The agent must learn:

    construct a useful cut
            ↓
    obtain a finite SCB
            ↓
    improve SCB
            ↓
    recognize when to STOP

The GA/teacher is NOT used by this reward.

GA values may be retained by the environment for
evaluation/logging, but they must never enter PPO's
transition reward.

Important design principle
--------------------------

Temporary destruction of a good solution is allowed.

For example:

    SCB 6
      ↓
    SCB inf
      ↓
    SCB 4

The middle transition should not receive a catastrophic
penalty because restructuring the cut may be necessary
to discover a better solution.

Therefore regression is penalized mildly.

STOP
----

STOP is explicitly rewarded/penalized:

    STOP + invalid solution
        -> strong negative reward

    STOP + valid solution
        -> positive reward based on SCB quality

Lower SCB = better reward.
"""


# ==========================================================
# Hyperparameters
# ==========================================================

# ----------------------------------------------------------
# Invalid actions
# ----------------------------------------------------------

INVALID_ACTION_PENALTY = -0.10


# ----------------------------------------------------------
# Separation
# ----------------------------------------------------------

SEPARATION_WEIGHT = 0.40


# ----------------------------------------------------------
# SCB progress
# ----------------------------------------------------------

SCB_WEIGHT = 1.00


# Extra reward when a genuinely new best SCB is found.
NEW_BEST_BONUS = 0.50


# ----------------------------------------------------------
# STOP
# ----------------------------------------------------------

# STOP with no meaningful solution.
INVALID_STOP_PENALTY = -1.00


# Valid STOP reward multiplier.
STOP_WEIGHT = 2.00


# ----------------------------------------------------------
# Regression
# ----------------------------------------------------------

# Mild penalty for destroying a previously valid solution.
#
# Deliberately small so that:
#
#     good -> bad -> better
#
# remains possible.
REGRESSION_PENALTY = 0.15


# ----------------------------------------------------------
# Time
# ----------------------------------------------------------

TIME_PENALTY_WEIGHT = 0.05


# ----------------------------------------------------------
# Numerical stability
# ----------------------------------------------------------

EPSILON = 1e-8


# ==========================================================
# Helpers
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
    Convert SCB into a bounded quality value.

    Lower SCB = better quality.

    Examples:

        SCB = 1   -> 0.50
        SCB = 2   -> 0.33
        SCB = 5   -> 0.17
        SCB = 10  -> 0.09

    Invalid / infinite SCB -> 0.
    """

    if not _is_finite_scb(scb):

        return 0.0

    return (
        1.0
        / (1.0 + float(scb))
    )


def get_step_budget(state):
    """
    Compute a graph-size-dependent exploration budget.

    Larger graphs receive a larger allowed number of
    useful search steps before the time penalty reaches
    its maximum.
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
# Reward Components
# ==========================================================

def compute_reward_components(
    old_state,
    new_state,
    action,
    invalid=False,
    best_scb=None,
    is_stop=False,
    new_best=False,
):
    """
    Compute the V2-D reward breakdown.

    Parameters
    ----------
    old_state : SCBState
        State before the action.

    new_state : SCBState
        State after the action.

    action : Action
        Action taken.

    invalid : bool
        Whether the action was invalid.

    best_scb : float or None
        Best finite SCB known BEFORE this transition.

    is_stop : bool
        Whether this transition is STOP.

    new_best : bool
        Whether the environment has determined that the
        new state is a genuinely better episode-best SCB.

        This is supplied explicitly by the environment so
        the reward calculation does not have to guess
        about bookkeeping.

    Returns
    -------
    dict
        Reward components.
    """

    # ======================================================
    # INVALID ACTION
    # ======================================================

    if invalid:

        return {
            "separation": 0.0,
            "scb": 0.0,
            "best_bonus": 0.0,
            "stop": 0.0,
            "time": INVALID_ACTION_PENALTY,
            "total": INVALID_ACTION_PENALTY,
        }


    # ======================================================
    # STATE INFORMATION
    # ======================================================

    old_sep = old_state.separated_count

    new_sep = new_state.separated_count

    old_scb = old_state.scb

    new_scb = new_state.scb

    num_sessions = max(
        len(old_state.problem.sessions),
        1
    )


    old_valid = _is_finite_scb(
        old_scb
    )

    new_valid = _is_finite_scb(
        new_scb
    )


    # ======================================================
    # 1. SEPARATION REWARD
    # ======================================================

    separation_reward = 0.0

    separation_gain = (
        new_sep - old_sep
    )


    if separation_gain > 0:

        # --------------------------------------------------
        # Useful separation
        #
        # A finite SCB means we know the separation is
        # producing a valid SCB state.
        # --------------------------------------------------

        if new_valid:

            quality = scb_quality(
                new_scb
            )

            separation_reward = (
                separation_gain
                / num_sessions
            )

            separation_reward *= (
                SEPARATION_WEIGHT
                * quality
                * 2.0
            )

        # --------------------------------------------------
        # Separation while SCB is still infinite
        #
        # Give only a tiny exploratory reward.
        #
        # This prevents the agent from learning:
        #
        #     "maximize separation forever"
        # --------------------------------------------------

        else:

            separation_reward = (
                separation_gain
                / num_sessions
            )

            separation_reward *= (
                SEPARATION_WEIGHT
                * 0.10
            )


    elif separation_gain < 0:

        # --------------------------------------------------
        # Losing separation is mildly negative.
        #
        # IMPORTANT:
        # We intentionally do NOT heavily punish this.
        #
        # The agent must be able to restructure its cut.
        # --------------------------------------------------

        separation_reward = (
            separation_gain
            / num_sessions
        )

        separation_reward *= (
            SEPARATION_WEIGHT
            * 0.25
        )


    # ======================================================
    # 2. SCB PROGRESS
    # ======================================================

    scb_reward = 0.0


    # ------------------------------------------------------
    # Valid -> valid
    # ------------------------------------------------------

    if old_valid and new_valid:

        improvement = (
            old_scb - new_scb
        ) / max(
            abs(old_scb),
            EPSILON
        )

        scb_reward = (
            improvement
            * SCB_WEIGHT
        )


    # ------------------------------------------------------
    # First finite SCB discovered
    # ------------------------------------------------------

    elif (
        not old_valid
        and new_valid
    ):

        scb_reward = (
            scb_quality(new_scb)
            * SCB_WEIGHT
        )


    # ------------------------------------------------------
    # Valid solution destroyed
    # ------------------------------------------------------

    elif (
        old_valid
        and not new_valid
    ):

        scb_reward = (
            -REGRESSION_PENALTY
        )


    # ======================================================
    # 3. NEW BEST BONUS
    # ======================================================

    best_bonus = 0.0


    if new_best and new_valid:

        best_bonus = (
            NEW_BEST_BONUS
            * scb_quality(new_scb)
        )


    # ======================================================
    # 4. STOP REWARD
    # ======================================================

    stop_reward = 0.0


    if is_stop:

        # --------------------------------------------------
        # A valid STOP requires:
        #
        #   separated sessions > 0
        #   finite SCB
        # --------------------------------------------------

        valid_solution = (
            new_sep > 0
            and new_valid
        )


        if not valid_solution:

            stop_reward = (
                INVALID_STOP_PENALTY
            )

        else:

            # ------------------------------------------------
            # Lower SCB = larger STOP reward.
            #
            # This is deliberately independent of GA.
            # ------------------------------------------------

            stop_reward = (
                STOP_WEIGHT
                * scb_quality(new_scb)
            )


    # ======================================================
    # 5. GRAPH-SCALED TIME PENALTY
    # ======================================================

    step = max(
        new_state.step,
        1
    )

    step_budget = get_step_budget(
        new_state
    )

    time_penalty = (
        TIME_PENALTY_WEIGHT
        * step
        / step_budget
    )

    time_penalty = min(
        time_penalty,
        TIME_PENALTY_WEIGHT
    )


    # ======================================================
    # TOTAL
    # ======================================================

    total = (
        separation_reward
        + scb_reward
        + best_bonus
        + stop_reward
        - time_penalty
    )


    return {

        "separation":
            separation_reward,

        "scb":
            scb_reward,

        "best_bonus":
            best_bonus,

        "stop":
            stop_reward,

        "time":
            -time_penalty,

        "total":
            total,
    }


# ==========================================================
# Main Reward Function
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
):
    """
    Compute the scalar V2-D transition reward.

    GA information is deliberately absent.

    The function is kept under the existing name so that
    other project files can continue importing it.
    """

    components = compute_reward_components(

        old_state=old_state,

        new_state=new_state,

        action=action,

        invalid=invalid,

        best_scb=best_scb,

        is_stop=is_stop,

        new_best=new_best,
    )

    return components["total"]