"""
trainer.py

PPO Trainer for the SCB Reinforcement Learning Agent.

Collects trajectories, computes GAE and updates
the encoder, actor and critic.

Author: SCB-RL
"""

from numpy import rint
import numpy as np
import torch
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn.functional as F
import os
from torch.optim import Adam
from SCB_RL.environment import SCBEnvironment
from SCB_RL.memory import PPOMemory
from SCB_RL.gae import compute_gae
import psutil


class PPOTrainer:
    """
    PPO Trainer.
    """

    def __init__(

        self,

        environment,

        encoder,

        policy,

        critic,

        lr=3e-4,

        gamma=0.99,

        gae_lambda=0.95,

        clip_eps=0.2,

        entropy_coef=0.01,

        value_coef=0.5,

        ppo_epochs=1

    ):

        self.env = environment

        self.encoder = encoder

        self.policy = policy

        self.critic = critic

        self.memory = PPOMemory()

        # ---------------------------------------------
        # PPO Hyperparameters
        # ---------------------------------------------

        self.gamma = gamma

        self.gae_lambda = gae_lambda

        self.clip_eps = clip_eps

        self.entropy_coef = entropy_coef

        self.value_coef = value_coef

        self.ppo_epochs = ppo_epochs

        # ---------------------------------------------
        # One Optimizer
        # ---------------------------------------------

        self.optimizer = Adam(

            list(self.encoder.parameters())

            +

            list(self.policy.parameters())

            +

            list(self.critic.parameters()),

            lr=lr

        )

    def collect_episode(self):
        """
        Runs one complete episode and stores the rollout
        inside PPO memory.

        Returns
        -------
        float
            Total episode reward.
        """

        self.memory.clear()

        state = self.env.reset()

        total_reward = 0.0

        done = False

        while not done:

            print(
                f"    Step {state.step} | "
                f"cut={state.cut_size} | "
                f"sep={state.separated_count}",
                flush=True
            )

            # ------------------------------------------
            # Encode State
            # ------------------------------------------

            print("      Encoding...", flush=True)

            encoding = self.encoder(state)

            # ------------------------------------------
            # Check GNN
            # ------------------------------------------

            self._check_tensor(
                "graph_embedding",
                encoding["graph_embedding"]
            )

            print("      Encoding done", flush=True)

            # ------------------------------------------
            # Policy
            # ------------------------------------------

            print("      Sampling action...", flush=True)

            action, log_prob, entropy = self.policy.sample_action(
                encoding,
                state
            )

            self._check_tensor(
                "new_log_prob",
                log_prob
            )

            self._check_tensor(
                "entropy",
                entropy
            )

            print(
                f"      Action: {action}",
                flush=True
            )

            # ------------------------------------------
            # Critic
            # ------------------------------------------

            print("      Critic...", flush=True)

            # value = self.critic(
            #     encoding["graph_embedding"]
            # )

            value = self.critic(
                encoding["graph_embedding"]
            ).squeeze()

            self._check_tensor(
                "value",
                value
            )

            print("      Critic done", flush=True)

            # ------------------------------------------
            # Environment
            # ------------------------------------------

            print("      Environment step...", flush=True)

            next_state, reward, done, info = self.env.step(
                action
            )

            print(
                f"      Step done | reward={reward} | done={done}",
                flush=True
            )

            # ------------------------------------------
            # Store Transition
            # ------------------------------------------

            self.memory.add(
                state=state,
                action=action,
                log_prob=log_prob.detach(),
                reward=reward,
                done=done,
                value=value.detach()
            )

            total_reward += reward

            state = next_state

        return total_reward

    def update(self):
        """
        Performs one PPO update.
        """

        # --------------------------------------------------
        # Compute Advantages and Returns
        # --------------------------------------------------

        advantages, returns = compute_gae(

            rewards=self.memory.rewards,

            values=self.memory.values,

            dones=self.memory.dones,

            gamma=self.gamma,

            lam=self.gae_lambda

        )

        self._check_tensor(
            "advantages",
            advantages
        )

        self._check_tensor(
            "returns",
            returns
        )
                # --------------------------------------------------
        # Normalize Advantages
        # --------------------------------------------------

        advantages = (

            advantages - advantages.mean()

        ) / (

            advantages.std() + 1e-8

        )

        # --------------------------------------------------
        # PPO Epochs
        # --------------------------------------------------

        for _ in range(self.ppo_epochs):

            policy_losses = []

            value_losses = []

            entropies = []

            # ----------------------------------------------
            # Loop over rollout
            # ----------------------------------------------

            for i in range(len(self.memory)):

                state = self.memory.states[i]

                action = self.memory.actions[i]

                old_log_prob = self.memory.log_probs[i]

                advantage = advantages[i]

                target_return = returns[i]

                # ------------------------------------------
                # Forward
                # ------------------------------------------

                encoding = self.encoder(state)

                new_log_prob, entropy = self.policy.evaluate_actions(

                    encoding,

                    state,

                    action

                )

                self._check_tensor(
                    "new_log_prob",
                    new_log_prob
                )

                self._check_tensor(
                    "entropy",
                    entropy
                )

                value = self.critic(

                    encoding["graph_embedding"]

                ).squeeze()

                self._check_tensor(
                    "value",
                    value
                )

                # ------------------------------------------
                # PPO Ratio
                # ------------------------------------------

                ratio = torch.exp(

                    new_log_prob - old_log_prob

                )

                self._check_tensor(
                    "ratio",
                    ratio
                )

                # ------------------------------------------
                # PPO Objective
                # ------------------------------------------

                surr1 = ratio * advantage

                surr2 = torch.clamp(

                    ratio,

                    1.0 - self.clip_eps,

                    1.0 + self.clip_eps

                ) * advantage

                policy_loss = -torch.min(

                    surr1,

                    surr2

                )

                # ------------------------------------------
                # Critic Loss
                # ------------------------------------------

                value_loss = F.mse_loss(

                    value,

                    target_return

                )

                policy_losses.append(policy_loss)

                value_losses.append(value_loss)

                entropies.append(entropy)

            # ----------------------------------------------
            # Mean Losses
            # ----------------------------------------------

            policy_loss = torch.stack(

                policy_losses

            ).mean()

            value_loss = torch.stack(

                value_losses

            ).mean()

            entropy = torch.stack(

                entropies

            ).mean()

            # ----------------------------------------------
            # Total Loss
            # ----------------------------------------------

            loss = (

                policy_loss

                + self.value_coef * value_loss

                - self.entropy_coef * entropy

            )

            self._check_tensor(
                "policy_loss",
                policy_loss
            )

            self._check_tensor(
                "value_loss",
                value_loss
            )

            self._check_tensor(
                "total_loss",
                loss
            )

            # ----------------------------------------------
            # Optimize
            # ----------------------------------------------

            self.optimizer.zero_grad()

            print(
                "    Backward...",
                flush=True
            )

            loss.backward()

            print(
                "    Backward done",
                flush=True
            )

            print(
                "    Gradient clipping...",
                flush=True
            )

            grad_norm = torch.nn.utils.clip_grad_norm_(

                list(self.encoder.parameters())
                + list(self.policy.parameters())
                + list(self.critic.parameters()),

                max_norm=0.5

            )

            print(
                f"    Grad norm: {grad_norm}",
                flush=True
            )

            print(
                "    Optimizer step...",
                flush=True
            )

            self.optimizer.step()

            print(
                "    Optimizer step done",
                flush=True
            )

            # --------------------------------------------------
            # Clear Memory
            # --------------------------------------------------

            print("    About to clear memory...", flush=True)

            self.memory.clear()

            print("    Memory cleared.", flush=True)

            print("    Reading loss values...", flush=True)

            result = {

                "policy_loss": policy_loss.item(),

                "value_loss": value_loss.item(),

                "entropy": entropy.item(),

                "total_loss": loss.item()

            }

            print("    Loss values read.", flush=True)

            return result


    def train(
        self,
        dataset,
        episodes=None,
        log_interval=10,
        checkpoint_interval=100
    ):
        """
        Trains PPO on the supplied graph dataset.

        Parameters
        ----------
        dataset : GraphDataset
            Labelled graph dataset.

        episodes : int or None
            Number of training episodes.
            If None, uses the full dataset.

        log_interval : int
            How often to print training statistics.

        Returns
        -------
        history : list
            Training statistics for every episode.
        """

        history = []

        # --------------------------------------------------
        # Number of Episodes
        # --------------------------------------------------

        if episodes is None:

            episodes = len(dataset)

        episodes = min(
            episodes,
            len(dataset)
        )

        # --------------------------------------------------
        # Training Loop
        # --------------------------------------------------

        for episode in range(episodes):

            print()
            print("=" * 70)
            print(f"STARTING EPISODE {episode + 1}/{episodes}")
            print("=" * 70)

            # ----------------------------------------------
            # Select Graph
            # ----------------------------------------------

            graph = dataset[episode]

            print(
                f"Graph {graph['graph_id']} | "
                f"Nodes={len(graph['nodes'])} | "
                f"Edges={len(graph['edges'])} | "
                f"Sessions={len(graph['sessions'])}"
            )

            # ----------------------------------------------
            # Create Environment
            # ----------------------------------------------

            print("Creating environment...")

            self.env = SCBEnvironment(graph)

            print("Environment created.")

            # ----------------------------------------------
            # Collect Episode
            # ----------------------------------------------

            print("Collecting episode...")

            episode_reward = self.collect_episode()

            episode_length = len(self.memory)

            print(
                f"Episode collected | "
                f"Reward={episode_reward} | "
                f"Length={episode_length}"
            )

            # ----------------------------------------------
            # PPO Update
            # ----------------------------------------------

            print("Starting PPO update...")

            
            process = psutil.Process(os.getpid())

            print(
                f"RAM before update: "
                f"{process.memory_info().rss / (1024 ** 3):.2f} GB"
            )

            stats = self.update()

            print(
                    f"RAM after update:  "
                    f"{process.memory_info().rss / (1024 ** 3):.2f} GB"
                )

            print("PPO update finished.")

            # ----------------------------------------------
            # Record
            # ----------------------------------------------

            result = {

                "episode": episode + 1,

                "reward": episode_reward,

                "episode_length": episode_length,

                "policy_loss": stats["policy_loss"],

                "value_loss": stats["value_loss"],

                "entropy": stats["entropy"],

                "total_loss": stats["total_loss"],

            }

            history.append(result)

            if (episode + 1) % checkpoint_interval == 0:

                os.makedirs(
                    "checkpoints",
                    exist_ok=True
                )

                self.save_checkpoint(

                    path=f"checkpoints/ppo_episode_{episode + 1}.pt",

                    episode=episode + 1,

                    history=history

                )

            print(
                f"Episode {episode + 1:4d}/{episodes} | "
                f"Reward {episode_reward:8.3f} | "
                f"Length {episode_length:3d} | "
                f"Policy {stats['policy_loss']:12.8f} | "
                f"Value {stats['value_loss']:12.8f} | "
                f"Entropy {stats['entropy']:8.4f}"

            )
        os.makedirs(
            "checkpoints",
            exist_ok=True
        )

        self.save_checkpoint(

            path="checkpoints/ppo_final.pt",

            episode=episodes,

            history=history

                )
            
        return history

    def save_checkpoint(
        self,
        path,
        episode,
        history
    ):
        """
        Saves the complete PPO training state.
        """

        checkpoint = {

            "episode": episode,

            "encoder_state_dict":
                self.encoder.state_dict(),

            "policy_state_dict":
                self.policy.state_dict(),

            "critic_state_dict":
                self.critic.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "history": history,

            "hyperparameters": {

                "gamma": self.gamma,

                "gae_lambda": self.gae_lambda,

                "clip_eps": self.clip_eps,

                "entropy_coef": self.entropy_coef,

                "value_coef": self.value_coef,

                "ppo_epochs": self.ppo_epochs,

            }

        }

        torch.save(
            checkpoint,
            path
        )

        print(
            f"Checkpoint saved: {path}"
        )

    def load_checkpoint(self, path):
        """
        Loads a PPO checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location="cpu"
        )

        self.encoder.load_state_dict(
            checkpoint["encoder_state_dict"]
        )

        self.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.critic.load_state_dict(
            checkpoint["critic_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        print(
            f"Checkpoint loaded: {path}"
        )

        print(
            f"Episode: {checkpoint['episode']}"
        )

        return (

            checkpoint["episode"],

            checkpoint["history"]

        )

    def _check_tensor(self, name, tensor):

        if not torch.isfinite(tensor).all():

            print(
                f"\n!!! NON-FINITE TENSOR: {name}",
                flush=True
            )

            print(
                tensor,
                flush=True
            )

            raise RuntimeError(
                f"Non-finite tensor detected: {name}"
            )



if __name__ == "__main__":

    from Data.loader import GraphDataset

    from SCB_RL.gnn import SCBGraphEncoder
    from SCB_RL.policy import HierarchicalSCBPolicy
    from SCB_RL.critic import SCBCritic

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------

    dataset = GraphDataset()

    print("=" * 70)
    print("PPO TRAINING TEST")
    print("=" * 70)

    print("Dataset Size :", len(dataset))

    # --------------------------------------------------
    # Networks
    # --------------------------------------------------

    encoder = SCBGraphEncoder()

    policy = HierarchicalSCBPolicy()

    critic = SCBCritic()

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = PPOTrainer(

        environment=None,

        encoder=encoder,

        policy=policy,

        critic=critic

    )

    # --------------------------------------------------
    # Train
    # --------------------------------------------------

    history = trainer.train(
    dataset,
    episodes=10,
    log_interval=1,
    checkpoint_interval=5
)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    print("Episodes:", len(history))

    print(
        "Final Reward:",
        history[-1]["reward"]
    )