"""
memory.py

Trajectory buffer used by PPO.

Stores one complete episode before the policy
is updated.

Author: SCB-RL
"""

import torch


class PPOMemory:
    """
    Stores one rollout trajectory.
    """

    def __init__(self):

        self.clear()

    # ==========================================================
    # Reset Memory
    # ==========================================================

    def clear(self):

        self.states = []

        self.actions = []

        self.log_probs = []

        self.rewards = []

        self.dones = []

        self.values = []

    # ==========================================================
    # Store Transition
    # ==========================================================

    def add(

        self,

        state,

        action,

        log_prob,

        reward,

        done,

        value

    ):

        self.states.append(state)

        self.actions.append(action)

        self.log_probs.append(log_prob)

        self.rewards.append(reward)

        self.dones.append(done)

        self.values.append(value)

    # ==========================================================
    # Length
    # ==========================================================

    def __len__(self):

        return len(self.states)

    # ==========================================================
    # Representation
    # ==========================================================

    def __repr__(self):

        return (

            "PPOMemory(\n"

            f"    Steps={len(self)}\n"

            ")"

        )


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    memory = PPOMemory()

    print(memory)

    memory.add(

        state="state",

        action="ADD",

        log_prob=torch.tensor(-1.2),

        reward=1.5,

        done=False,

        value=torch.tensor(0.73)

    )

    memory.add(

        state="state2",

        action="REMOVE",

        log_prob=torch.tensor(-0.4),

        reward=-0.2,

        done=True,

        value=torch.tensor(0.51)

    )

    print(memory)

    print()

    print("Rewards")

    print(memory.rewards)

    print()

    print("Values")

    print(memory.values)