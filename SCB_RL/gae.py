"""
gae.py

Generalized Advantage Estimation (GAE)
used by PPO.

Author: SCB-RL
"""

import torch


def compute_gae(
    rewards,
    values,
    dones,
    gamma=0.99,
    lam=0.95
):
    """
    Computes Generalized Advantage Estimation (GAE).

    Parameters
    ----------
    rewards : list[float]
        Rewards collected during the episode.

    values : list[Tensor]
        Value estimates predicted by the critic.

    dones : list[bool]
        Episode termination flags.

    gamma : float
        Discount factor.

    lam : float
        GAE lambda.

    Returns
    -------
    advantages : Tensor
        Normalized advantages.

    returns : Tensor
        Target returns for critic training.
    """

    advantages = []

    # Keep everything as tensors
    gae = torch.tensor(0.0)

    # Bootstrap value after final state
    values = values + [torch.tensor(0.0)]

    # --------------------------------------------------
    # Compute GAE backwards
    # --------------------------------------------------

    for t in reversed(range(len(rewards))):

        reward = torch.tensor(
            rewards[t],
            dtype=torch.float32
        )

        mask = 1.0 - float(dones[t])

        delta = (

            reward

            + gamma
            * values[t + 1]
            * mask

            - values[t]

        )

        gae = (

            delta

            + gamma
            * lam
            * mask
            * gae

        )

        advantages.insert(0, gae)

    # --------------------------------------------------
    # Convert to tensors
    # --------------------------------------------------

    advantages = torch.stack(advantages)

    values_tensor = torch.stack(values[:-1])

    # --------------------------------------------------
    # Compute Returns
    # --------------------------------------------------

    returns = advantages + values_tensor

    # --------------------------------------------------
    # Normalize Advantages
    # --------------------------------------------------

    advantages = (

        advantages
        - advantages.mean()

    ) / (

        advantages.std()
        + 1e-8

    )

    return advantages, returns


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    rewards = [

        0.0,

        0.2,

        -0.1,

        1.0

    ]

    values = [

        torch.tensor(0.1),

        torch.tensor(0.3),

        torch.tensor(0.2),

        torch.tensor(0.9)

    ]

    dones = [

        False,

        False,

        False,

        True

    ]

    advantages, returns = compute_gae(

        rewards,

        values,

        dones

    )

    print()

    print("Advantages")

    print(advantages)

    print()

    print("Returns")

    print(returns)