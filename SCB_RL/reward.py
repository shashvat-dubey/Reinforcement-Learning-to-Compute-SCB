"""
reward.py

V2-B reward function for Sparsest Cut Bound RL.

Reward philosophy
-----------------
Normal steps:
    - Encourage separating sessions.
    - Encourage improving SCB.
    - Penalize unnecessarily long trajectories.

Terminal step:
    - Compare the final RL solution against the GA teacher.
    - Penalize inefficient completion.

The GA teacher is NOT used during normal steps.
"""

# ==========================================================
# Hyperparameters
# ==========================================================

INVALID_ACTION_PENALTY = -0.10

# Relative weight of session-separation progress.
SEPARATION_WEIGHT = 0.50

# Relative weight of normalized SCB improvement.
SCB_WEIGHT = 1.00

# Per-step time penalty scale.
TIME_PENALTY_WEIGHT = 0.05

# Terminal GA reward limits.
# TERMINAL_MIN = -1.0
# TERMINAL_MAX = 2.0

# Prevent division by zero.
EPSILON = 1e-8


# ==========================================================
# Graph-dependent step budget
# ==========================================================

def get_step_budget(state):
    """
    Estimate a reasonable number of actions for the graph.

    Larger graphs receive a larger exploration budget.

    Budget depends on:
        - number of edges
        - number of sessions
    """

    problem = state.problem

    num_edges = len(problem.edges)
    num_sessions = len(problem.sessions)

    # Initial conservative scaling.
    #
    # Edges dominate because the agent operates on edges.
    # Sessions contribute additional difficulty.
    budget = (
        0.25 * num_edges
        + 2.0 * num_sessions
    )

    return max(budget, 1.0)


# ==========================================================
# Reward breakdown
# ==========================================================

def compute_reward_components(
    old_state,
    new_state,
    action,
    invalid=False,
):
    """
    Compute normal-step reward components.

    GA teacher information is intentionally NOT used here.
    """

    if invalid:

        return {
            "separation": 0.0,
            "scb": 0.0,
            "time": INVALID_ACTION_PENALTY,
            "total": INVALID_ACTION_PENALTY,
        }

    # ------------------------------------------------------
    # Separation progress
    # ------------------------------------------------------

    old_sep = old_state.separated_count
    new_sep = new_state.separated_count

    num_sessions = max(
        len(old_state.problem.sessions),
        1
    )

    separation_reward = (
        (new_sep - old_sep)
        / num_sessions
    )

    separation_reward *= SEPARATION_WEIGHT

    # ------------------------------------------------------
    # SCB improvement
    # ------------------------------------------------------

    old_scb = old_state.scb
    new_scb = new_state.scb

    scb_reward = 0.0

    # SCB is not meaningful while no sessions are separated.
    if old_sep > 0 and old_scb > EPSILON:

        if new_sep > 0 and new_scb > EPSILON:

            # Lower SCB is better.
            #
            # Example:
            # 10 -> 8 = +0.20
            # 10 -> 5 = +0.50
            # 10 -> 12 = -0.20

            scb_reward = (
                (old_scb - new_scb)
                / max(old_scb, EPSILON)
            )

        else:

            # We destroyed a previously valid separation.
            scb_reward = -1.0

    scb_reward *= SCB_WEIGHT

    # ------------------------------------------------------
    # Dynamic time penalty
    # ------------------------------------------------------

    step = max(new_state.step, 1)

    step_budget = get_step_budget(new_state)

    # Penalty grows gradually with episode progress.
    time_penalty = (
        TIME_PENALTY_WEIGHT
        * (step / step_budget)
    )

    # Prevent the time penalty from becoming ridiculous
    # on unusually long episodes.
    time_penalty = min(
        time_penalty,
        TIME_PENALTY_WEIGHT
    )

    # ------------------------------------------------------
    # Total
    # ------------------------------------------------------

    total = (
        separation_reward
        + scb_reward
        - time_penalty
    )

    return {
        "separation": separation_reward,
        "scb": scb_reward,
        "time": -time_penalty,
        "total": total,
    }


# ==========================================================
# Main reward function
# ==========================================================

def compute_reward(
    old_state,
    new_state,
    action,
    done=False,
    invalid=False,
):
    """
    Compute the normal transition reward.

    IMPORTANT:
        This function does NOT use the GA teacher.

    Terminal GA reward is handled separately by
    SCBEnvironment._compute_terminal_reward().
    """

    components = compute_reward_components(
        old_state,
        new_state,
        action,
        invalid=invalid,
    )

    return components["total"]