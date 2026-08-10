"""
critic.py

Value network for PPO.

Predicts the expected future reward of a graph state.

Author: SCB-RL
"""

import torch
import torch.nn as nn

class SCBCritic(nn.Module):
    """
    PPO Value Network.
    """

    def __init__(self):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, 64),

            nn.ReLU(),

            nn.Linear(64, 1)

        )

    def forward(
        self,
        graph_embedding
    ):
        """
        Parameters
        ----------
        graph_embedding : Tensor
            Shape (64,)

        Returns
        -------
        Tensor
            Scalar value estimate.
        """

        return self.network(
            graph_embedding
        ).squeeze(-1)

if __name__ == "__main__":

    critic = SCBCritic()

    embedding = torch.randn(64)

    value = critic(embedding)

    print(value)

    print(value.shape)