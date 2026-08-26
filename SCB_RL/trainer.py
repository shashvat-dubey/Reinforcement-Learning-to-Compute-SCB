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

# SEED = 42
num_rounds = 2
SHUFFLE_GRAPHS = True
# random.seed(SEED)
# np.random.seed(SEED)
# torch.manual_seed(SEED)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)

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

        # ---------------------------------------------
        # Device
        # ---------------------------------------------

        self.device = torch.device(

            "cuda"
            if torch.cuda.is_available()
            else "cpu"

        )

        print(
            f"Using device: {self.device}"
        )

        # ---------------------------------------------
        # Networks
        # ---------------------------------------------

        self.encoder = encoder.to(
            self.device
        )

        self.policy = policy.to(
            self.device
        )

        self.critic = critic.to(
            self.device
        )

        # ---------------------------------------------
        # Memory
        # ---------------------------------------------

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

            print("CUT:", sorted(state.cut))
            print("EDGE COUNT:", len(state.problem.edges))
            print("NODE COUNT:", state.problem.N)

            encoding = self.encoder(state)

            embedding = encoding["graph_embedding"]

            print(
                f"embedding | "
                f"min={embedding.min().item():.6f} "
                f"max={embedding.max().item():.6f} "
                f"mean={embedding.mean().item():.6f}",
                flush=True
            )

            if not torch.isfinite(embedding).all():
                raise RuntimeError(
                    f"NON-FINITE GRAPH EMBEDDING:\n{embedding}"
                )
           
            # ------------------------------------------
            # Check GNN
            # ------------------------------------------

            # self._check_tensor(
            #     "graph_embedding",
            #     encoding["graph_embedding"]
            # )

            # print("      Encoding done", flush=True)

            # ------------------------------------------
            # Policy
            # ------------------------------------------

            # print("      Sampling action...", flush=True)

            action, log_prob, entropy = self.policy.sample_action(
                encoding,
                state
            )

            if not torch.isfinite(log_prob):
                raise RuntimeError(
                    f"NON-FINITE LOG_PROB: {log_prob}"
                )

            if not torch.isfinite(entropy):
                raise RuntimeError(
                    f"NON-FINITE ENTROPY: {entropy}"
    )


            # self._check_tensor(
            #     "new_log_prob",
            #     log_prob
            # )

            # self._check_tensor(
            #     "entropy",
            #     entropy
            # )

            # print(
            #     f"      Action: {action}",
            #     flush=True
            # )

            # ------------------------------------------
            # Critic
            # ------------------------------------------

            # print("      Critic...", flush=True)

            # value = self.critic(
            #     encoding["graph_embedding"]
            # )

            value = self.critic(
                encoding["graph_embedding"]
            ).squeeze()

            if not torch.isfinite(value):
                raise RuntimeError(
                    f"NON-FINITE VALUE: {value}"
                    )
            # self._check_tensor(
            #     "value",
            #     value
            # )

            # print("      Critic done", flush=True)

            # ------------------------------------------
            # Environment
            # ------------------------------------------

            # print("      Environment step...", flush=True)

            next_state, reward, done, info = self.env.step(
                action
            )
            
            print(
                            f"value={value.item():.6f} "
                            f"log_prob={log_prob.item():.6f} "
                            f"entropy={entropy.item():.6f}"
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

        # self._check_tensor(
        #     "advantages",
        #     advantages
        # )

        # self._check_tensor(
        #     "returns",
        #     returns
        # )
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

                # self._check_tensor(
                #     "new_log_prob",
                #     new_log_prob
                # )

                # self._check_tensor(
                #     "entropy",
                #     entropy
                # )

                value = self.critic(

                    encoding["graph_embedding"]

                ).squeeze()

                # self._check_tensor(
                #     "value",
                #     value
                # )

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

            # self._check_tensor(
            #     "policy_loss",
            #     policy_loss
            # )

            # self._check_tensor(
            #     "value_loss",
            #     value_loss
            # )

            # self._check_tensor(
            #     "total_loss",
            #     loss
            # )

            # ----------------------------------------------
            # Optimize
            # ----------------------------------------------

            self.optimizer.zero_grad()

            # print(
            #     "    Backward...",
            #     flush=True
            # )

            loss.backward()

            # print(
            #     "    Backward done",
            #     flush=True
            # )

            # print(
            #     "    Gradient clipping...",
            #     flush=True
            # )

            grad_norm = torch.nn.utils.clip_grad_norm_(

                list(self.encoder.parameters())
                + list(self.policy.parameters())
                + list(self.critic.parameters()),

                max_norm=0.5

            )

            # print(
            #     f"    Grad norm: {grad_norm}",
            #     flush=True
            # )

            # print(
            #     "    Optimizer step...",
            #     flush=True
            # )

            self.optimizer.step()

            # print(
            #     "    Optimizer step done",
            #     flush=True
            # )

            # --------------------------------------------------
            # Clear Memory
            # --------------------------------------------------

            # print("    About to clear memory...", flush=True)

            self.memory.clear()

            # print("    Memory cleared.", flush=True)

            # print("    Reading loss values...", flush=True)

            result = {

                "policy_loss": policy_loss.item(),

                "value_loss": value_loss.item(),

                "entropy": entropy.item(),

                "total_loss": loss.item()

            }

            # print("    Loss values read.", flush=True)

            return result


    def train(
        self,
        dataset,
        episodes=None,
        log_interval=10,
        checkpoint_interval=10,
        start_episode=0,
        history=None,
        checkpoint_dir="checkpoints",
        max_checkpoints=5
    ):
        """
        Trains PPO on the supplied graph dataset.

        Supports automatic resume from a checkpoint.

        Parameters
        ----------
        dataset : GraphDataset
            Labelled graph dataset.

        episodes : int or None
            Total target number of episodes.

        log_interval : int
            How often to print training statistics.

        checkpoint_interval : int
            How often to save checkpoints.

        start_episode : int
            Episode already completed before resuming.

        history : list or None
            Previous training history.

        checkpoint_dir : str
            Directory for checkpoints.

        max_checkpoints : int
            Maximum number of rolling checkpoints to retain.

        Returns
        -------
        history : list
            Complete training history.
        """

        # --------------------------------------------------
        # History
        # --------------------------------------------------

        if history is None:
            history = []

        # --------------------------------------------------
        # Number of Episodes
        # --------------------------------------------------

        # if episodes is None:
        #     episodes = len(dataset)

        # episodes = min(
        #     episodes,
        #     len(dataset)
        # )

        # --------------------------------------------------
        # Checkpoint Directory
        # --------------------------------------------------

        os.makedirs(
            checkpoint_dir,
            exist_ok=True
        )

        # --------------------------------------------------
        # Training Loop
        # --------------------------------------------------
       
        total_episodes = num_rounds * len(dataset)

        episode = start_episode

        for round_idx in range(num_rounds):

            # Create graph ordering for this round
            graph_indices = list(range(len(dataset)))

            if SHUFFLE_GRAPHS:
                random.shuffle(graph_indices)

            print()
            print("=" * 70)
            print(f"ROUND {round_idx + 1}/{num_rounds}")
            print("=" * 70)

            for graph_idx in graph_indices:

                # ----------------------------------------------
                # Select Graph
                # ----------------------------------------------

                graph = dataset[graph_idx]

                print(
                    f"Graph {graph['graph_id']} | "
                    f"Nodes={len(graph['nodes'])} | "
                    f"Edges={len(graph['edges'])} | "
                    f"Sessions={len(graph['sessions'])}"
                )

                # ----------------------------------------------
                # Create Environment
                # ----------------------------------------------

                self.env = SCBEnvironment(
                    graph
                )

                # ----------------------------------------------
                # Collect Episode
                # ----------------------------------------------

                episode_reward = (
                    self.collect_episode()
                )

                # IMPORTANT:
                # Capture this BEFORE update() clears memory.

                episode_length = len(
                    self.memory
                )

                print(
                    f"Episode collected | "
                    f"Reward={episode_reward} | "
                    f"Length={episode_length}"
                )

                # ----------------------------------------------
                # PPO Update
                # ----------------------------------------------

                stats = self.update()

                # ----------------------------------------------
                # Record Result
                # ----------------------------------------------

                result = {

                    "episode":
                        episode + 1,

                    "reward":
                        episode_reward,

                    "episode_length":
                        episode_length,

                    "policy_loss":
                        stats["policy_loss"],

                    "value_loss":
                        stats["value_loss"],

                    "entropy":
                        stats["entropy"],

                    "total_loss":
                        stats["total_loss"],

                }

                history.append(
                    result
                )

                # ----------------------------------------------
                # Training Output
                # ----------------------------------------------

                print(
                    f"Episode {episode + 1:4d}/"
                    f"{total_episodes} | "
                    f"Reward {episode_reward:8.3f} | "
                    f"Length {episode_length:3d} | "
                    f"Policy {stats['policy_loss']:12.8f} | "
                    f"Value {stats['value_loss']:12.8f} | "
                    f"Entropy {stats['entropy']:8.4f}"
                )

                # ----------------------------------------------
                # Checkpoint
                # ----------------------------------------------

                current_episode = (
                    episode + 1
                )

                if (
                    current_episode
                    % checkpoint_interval
                    == 0
                ):

                    # ------------------------------------------
                    # Rolling checkpoint
                    # ------------------------------------------

                    checkpoint_path = os.path.join(

                        checkpoint_dir,

                        f"ppo_episode_{current_episode}.pt"

                    )

                    self.save_checkpoint(

                        path=checkpoint_path,

                        episode=current_episode,

                        history=history

                    )

                    # ------------------------------------------
                    # Latest checkpoint
                    # ------------------------------------------

                    latest_path = os.path.join(

                        checkpoint_dir,

                        "latest.pt"

                    )

                    self.save_checkpoint(

                        path=latest_path,

                        episode=current_episode,

                        history=history

                    )

                    # ------------------------------------------
                    # Remove old rolling checkpoints
                    # ------------------------------------------

                    rolling = []

                    for filename in os.listdir(
                        checkpoint_dir
                    ):

                        if (
                            filename.startswith(
                                "ppo_episode_"
                            )
                            and filename.endswith(
                                ".pt"
                            )
                        ):

                            full_path = os.path.join(

                                checkpoint_dir,

                                filename

                            )

                            rolling.append(
                                full_path
                            )

                    # Sort by episode number
                    rolling.sort(
                        key=lambda path:
                            int(
                                os.path.basename(
                                    path
                                )
                                .replace(
                                    "ppo_episode_",
                                    ""
                                )
                                .replace(
                                    ".pt",
                                    ""
                                )
                            )
                    )

                    # ------------------------------------------
                    # Keep only newest N
                    # ------------------------------------------

                    while len(rolling) > max_checkpoints:

                        old_checkpoint = rolling.pop(
                            0
                        )

                        try:

                            os.remove(
                                old_checkpoint
                            )

                            print(
                                f"Removed old checkpoint: "
                                f"{old_checkpoint}"
                            )

                        except OSError as e:

                            print(
                                f"Warning: could not remove "
                                f"{old_checkpoint}: {e}"
                            )

                episode += 1

        # --------------------------------------------------
        # Return History
        # --------------------------------------------------

        return history

    def save_checkpoint(
        self,
        path,
        episode,
        history
    ):
        """
        Saves the complete PPO training state so training
        can be resumed from this checkpoint.
        """

        checkpoint = {

            # ----------------------------------------------
            # Training position
            # ----------------------------------------------

            "episode": episode,

            # ----------------------------------------------
            # Model states
            # ----------------------------------------------

            "encoder_state_dict":
                self.encoder.state_dict(),

            "policy_state_dict":
                self.policy.state_dict(),

            "critic_state_dict":
                self.critic.state_dict(),

            # ----------------------------------------------
            # Optimizer
            # ----------------------------------------------

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            # ----------------------------------------------
            # Training history
            # ----------------------------------------------

            "history":
                history,

            # ----------------------------------------------
            # Hyperparameters
            # ----------------------------------------------

            "hyperparameters": {

                "gamma": self.gamma,

                "gae_lambda":
                    self.gae_lambda,

                "clip_eps":
                    self.clip_eps,

                "entropy_coef":
                    self.entropy_coef,

                "value_coef":
                    self.value_coef,

                "ppo_epochs":
                    self.ppo_epochs,

            },

            # ----------------------------------------------
            # Random number generator state
            # ----------------------------------------------

            "torch_rng_state":
                torch.get_rng_state(),

        }

        # --------------------------------------------------
        # CUDA RNG state
        # --------------------------------------------------

        if torch.cuda.is_available():

            checkpoint["cuda_rng_state"] = (
                torch.cuda.get_rng_state_all()
            )

        # --------------------------------------------------
        # Save
        # --------------------------------------------------

        torch.save(
            checkpoint,
            path
        )

        print(
            f"Checkpoint saved: {path}"
        )


    def load_checkpoint(
        self,
        path
    ):
        """
        Loads a PPO checkpoint and restores the complete
        training state.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device
        )

        # --------------------------------------------------
        # Restore model states
        # --------------------------------------------------

        self.encoder.load_state_dict(
            checkpoint["encoder_state_dict"]
        )

        self.policy.load_state_dict(
            checkpoint["policy_state_dict"]
        )

        self.critic.load_state_dict(
            checkpoint["critic_state_dict"]
        )

        # --------------------------------------------------
        # Restore optimizer
        # --------------------------------------------------

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        # --------------------------------------------------
        # Restore RNG state
        # --------------------------------------------------

        if "torch_rng_state" in checkpoint:

            torch.set_rng_state(
                checkpoint["torch_rng_state"]
            )

        if (
            "cuda_rng_state" in checkpoint
            and torch.cuda.is_available()
        ):

            torch.cuda.set_rng_state_all(
                checkpoint["cuda_rng_state"]
            )

        # --------------------------------------------------
        # Information
        # --------------------------------------------------

        episode = checkpoint["episode"]

        history = checkpoint["history"]

        print(
            f"Checkpoint loaded: {path}"
        )

        print(
            f"Episode: {episode}"
        )

        return (
            episode,
            history
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

    import os
    import glob
    import torch

    from Data.loader import GraphDataset

    from SCB_RL.gnn import SCBGraphEncoder
    from SCB_RL.policy import HierarchicalSCBPolicy
    from SCB_RL.critic import SCBCritic


    # ==========================================================
    # CONFIGURATION
    # ==========================================================

    NUM_ROUNDS = 2

    LOG_FILE = "ppo_training.txt"

    CHECKPOINT_DIR = "checkpoints_testing"

    LATEST_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "latest.pt"
    )

    FINAL_CHECKPOINT = os.path.join(
        CHECKPOINT_DIR,
        "ppo_final.pt"
    )

    CHECKPOINT_INTERVAL = 4

    MAX_ROLLING_CHECKPOINTS = 5


    # ==========================================================
    # CREATE DIRECTORIES
    # ==========================================================

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )


    # ==========================================================
    # LOGGER
    # ==========================================================

    log_file = open(
        LOG_FILE,
        "a",
        encoding="utf-8"
    )


    def log(msg=""):

        print(
            msg,
            flush=True
        )

        log_file.write(
            str(msg) + "\n"
        )

        log_file.flush()


    # ==========================================================
    # DATASET
    # ==========================================================

    dataset = GraphDataset()


    log()
    log("=" * 70)
    log("PPO TRAINING")
    log("=" * 70)

    log(
        f"Dataset Size : {len(dataset)}"
    )

    total_episodes = NUM_ROUNDS * len(dataset)

    log(
        f"Training Rounds : {NUM_ROUNDS}"
    )

    log(
        f"Total Episodes : {total_episodes}"
    )


    # ==========================================================
    # NETWORKS
    # ==========================================================

    encoder = SCBGraphEncoder()

    policy = HierarchicalSCBPolicy()

    critic = SCBCritic()


    # ==========================================================
    # TRAINER
    # ==========================================================

    trainer = PPOTrainer(

        environment=None,

        encoder=encoder,

        policy=policy,

        critic=critic

    )


    # ==========================================================
    # RESUME STATE
    # ==========================================================

    start_episode = 0

    history = []


    if os.path.exists(
        LATEST_CHECKPOINT
    ):

        log()
        log("=" * 70)
        log("CHECKPOINT FOUND")
        log("=" * 70)

        try:

            start_episode, history = (
                trainer.load_checkpoint(
                    LATEST_CHECKPOINT
                )
            )

            log(
                f"Resuming from episode "
                f"{start_episode + 1}"
            )

            log(
                f"Previous history entries : "
                f"{len(history)}"
            )

        except Exception as e:

            log(
                "WARNING: Failed to load "
                "latest checkpoint."
            )

            log(
                f"Reason: {e}"
            )

            log(
                "Starting training from episode 1."
            )

            start_episode = 0

            history = []


    else:

        log()
        log(
            "No checkpoint found."
        )

        log(
            "Starting fresh training."
        )


    # ==========================================================
    # ALREADY FINISHED?
    # ==========================================================

    if start_episode >= total_episodes:

        log()
        log("=" * 70)
        log("TRAINING ALREADY COMPLETE")
        log("=" * 70)

        log(
            f"Completed Episodes : "
            f"{start_episode}"
        )

        log(
            f"Final Checkpoint : "
            f"{FINAL_CHECKPOINT}"
        )

        log_file.close()

        raise SystemExit


    # ==========================================================
    # TRAINING
    # ==========================================================

    try:

        log()
        log("=" * 70)
        log("STARTING / RESUMING TRAINING")
        log("=" * 70)

        history = trainer.train(

            dataset,

            log_interval=5,

            checkpoint_interval=CHECKPOINT_INTERVAL,

            start_episode=start_episode,

            history=history,

            checkpoint_dir=CHECKPOINT_DIR,

            max_checkpoints=MAX_ROLLING_CHECKPOINTS

        )


    # ==========================================================
    # MANUAL INTERRUPTION
    # ==========================================================

    except KeyboardInterrupt:

        log()
        log("=" * 70)
        log("TRAINING INTERRUPTED BY USER")
        log("=" * 70)

        if history:

            interrupted_episode = (
                history[-1]["episode"]
            )

        else:

            interrupted_episode = start_episode

        try:

            trainer.save_checkpoint(

                os.path.join(
                    CHECKPOINT_DIR,
                    "latest.pt"
                ),

                interrupted_episode,

                history

            )

            log(
                "Emergency checkpoint saved."
            )

            log(
                f"Resume from episode "
                f"{interrupted_episode + 1}"
            )

        except Exception as e:

            log(
                "WARNING: Failed to save "
                "emergency checkpoint."
            )

            log(
                f"Reason: {e}"
            )

        log_file.close()

        raise SystemExit


    # ==========================================================
    # FINAL CHECKPOINT
    # ==========================================================

    try:

        trainer.save_checkpoint(

            FINAL_CHECKPOINT,

            total_episodes,

            history

        )

        log()
        log("=" * 70)
        log("TRAINING COMPLETE")
        log("=" * 70)

        log(
            f"Episodes : "
            f"{len(history)}"
        )

        if len(history) > 0:

            log(
                f"Final Reward : "
                f"{history[-1]['reward']}"
            )

        log(
            f"Final checkpoint : "
            f"{FINAL_CHECKPOINT}"
        )

    except Exception as e:

        log()
        log(
            "WARNING: Training finished, "
            "but final checkpoint failed."
        )

        log(
            f"Reason: {e}"
        )


    # ==========================================================
    # CLOSE LOG
    # ==========================================================

    log_file.close()