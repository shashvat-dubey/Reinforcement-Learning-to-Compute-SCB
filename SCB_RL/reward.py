"""
reward.py

Reward function for the Sparsest Cut Bound Reinforcement Learning environment.

The reward module is intentionally independent of the environment,
PPO and the GNN.

Author: SCB-RL
"""

INVALID_ACTION_PENALTY = -0.1
TERMINAL_BONUS = 0.0


def compute_reward(
    old_state,
    new_state,
    action,
    done=False,
    invalid=False,
):
    """
    Computes the reward for one environment transition.

    Parameters
    ----------
    old_state : SCBState
        State before applying the action.

    new_state : SCBState
        State after applying the action.

    action : Action
        Action selected by the policy.

    done : bool
        Whether the episode has terminated.

    invalid : bool
        True if the action was invalid.

    Returns
    -------
    float
        Reward assigned to the transition.
    """

    # ------------------------------------------------------
    # Invalid Action
    # ------------------------------------------------------

    if invalid:
        return INVALID_ACTION_PENALTY

    # ------------------------------------------------------
    # SCB Improvement
    # ------------------------------------------------------

    reward = new_state.scb - old_state.scb

    # ------------------------------------------------------
    # Future Reward Shaping
    # ------------------------------------------------------

    # reward += ...
    # reward -= ...

    if done:
        reward += TERMINAL_BONUS

    return reward