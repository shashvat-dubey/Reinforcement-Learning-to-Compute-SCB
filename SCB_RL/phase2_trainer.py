"""
SCB-RL PHASE 2 TRAINER
======================

Curriculum:
    Phase 1 checkpoint -> Phase 2 termination learning.

Phase 2 goal:
    Learn to construct a useful finite SCB, recognize when the
    CURRENT state is good enough, and STOP.

Important:
    - Only graphs 0..799 are used for training.
    - Graphs 800..999 are permanently held out for validation.
    - Training graphs are shuffled every round.
    - Phase 1 GA SCB is logged for evaluation only.
    - Phase 2 starts from Phase 1 model weights.
    - Phase 2 uses a FRESH optimizer.
    - Phase 2 checkpoints are stored separately in checkpoints/phase2.
    - No artificial max-step limit is imposed by this trainer.
    - Per-step spam is removed.
    - Detailed per-graph/per-episode data is written to CSV.
"""

import csv
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from SCB_RL.environment import SCBEnvironment
from SCB_RL.memory import PPOMemory
from SCB_RL.gae import compute_gae

from Data.loader import GraphDataset

from SCB_RL.gnn import SCBGraphEncoder
from SCB_RL.policy import HierarchicalSCBPolicy
from SCB_RL.critic import SCBCritic


# ==========================================================
# CONFIGURATION
# ==========================================================

PHASE = 2

# Dataset is expected to contain 1000 graphs.
TOTAL_DATASET_GRAPHS = 1000

# FIRST 800 = training
TRAIN_GRAPH_COUNT = 800

# LAST 200 = permanently unseen validation set
VALIDATION_GRAPH_START = 800
VALIDATION_GRAPH_COUNT = 200

# 100 rounds x 800 graphs = 80,000 training episodes.
NUM_ROUNDS = 1

SHUFFLE_TRAINING_GRAPHS = True

# ----------------------------------------------------------
# Checkpoints
# ----------------------------------------------------------

CHECKPOINT_ROOT = "checkpoints"

PHASE1_CHECKPOINT_DIR = os.path.join(
    CHECKPOINT_ROOT,
    "phase 1 checkpoints",
)

PHASE2_CHECKPOINT_DIR = os.path.join(
    CHECKPOINT_ROOT,
    "phase 2 checkpoints",
)

PHASE2_LATEST = os.path.join(
    PHASE2_CHECKPOINT_DIR,
    "latest.pt",
)

PHASE2_FINAL = os.path.join(
    PHASE2_CHECKPOINT_DIR,
    "ppo_phase2_final.pt",
)

# If a Phase-2 checkpoint exists, resume Phase 2.
RESUME_PHASE2 = True

# If no Phase-2 checkpoint exists, automatically find a
# Phase-1 checkpoint in checkpoints/.
AUTO_FIND_PHASE1_CHECKPOINT = True

# Optional explicit override.
# Set to a path string if you ever want a specific checkpoint.
PHASE1_CHECKPOINT = None

CHECKPOINT_INTERVAL = 1000

# Keep all Phase-2 checkpoints.
MAX_ROLLING_CHECKPOINTS = None

# ----------------------------------------------------------
# Logs
# ----------------------------------------------------------

PHASE2_LOG_FILE = "phase2_training.txt"

# One row per episode.
GRAPH_STATS_CSV = os.path.join(
    PHASE2_CHECKPOINT_DIR,
    "phase2_graph_stats.csv",
)

# One cumulative row per graph, updated after every occurrence.
GRAPH_PROGRESS_CSV = os.path.join(
    PHASE2_CHECKPOINT_DIR,
    "phase2_graph_progress.csv",
)

# ----------------------------------------------------------
# PPO
# ----------------------------------------------------------

LEARNING_RATE = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENTROPY_COEF = 0.01
VALUE_COEF = 0.5
PPO_EPOCHS = 1


# ==========================================================
# DETERMINISM / PERFORMANCE
# ==========================================================

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# Keep deterministic behavior compatible with the previous run.
try:
    torch.use_deterministic_algorithms(True)
except Exception:
    pass


# ==========================================================
# LOGGER
# ==========================================================

class Tee:

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


# ==========================================================
# TRAINER
# ==========================================================

class Phase2PPOTrainer:

    def __init__(
        self,
        environment,
        encoder,
        policy,
        critic,
        lr=LEARNING_RATE,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_eps=CLIP_EPS,
        entropy_coef=ENTROPY_COEF,
        value_coef=VALUE_COEF,
        ppo_epochs=PPO_EPOCHS,
    ):

        self.env = environment

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}",
            flush=True,
        )

        self.encoder = encoder.to(self.device)
        self.policy = policy.to(self.device)
        self.critic = critic.to(self.device)

        self.memory = PPOMemory()

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.ppo_epochs = ppo_epochs

        # IMPORTANT:
        # This is intentionally a FRESH Phase-2 optimizer.
        # Phase-1 optimizer state is NOT restored.
        self.optimizer = Adam(
            list(self.encoder.parameters())
            + list(self.policy.parameters())
            + list(self.critic.parameters()),
            lr=lr,
        )

        self.last_episode_metrics = {}

    # ======================================================
    # EPISODE COLLECTION
    # ======================================================

    def collect_episode(self):

        self.memory.clear()

        state = self.env.reset()

        total_reward = 0.0
        done = False

        action_counts = {
            "ADD": 0,
            "REMOVE": 0,
            "SWAP": 0,
            "STOP": 0,
        }

        final_info = None

        while not done:

            encoding = self.encoder(state)

            embedding = encoding["graph_embedding"]

            if not torch.isfinite(embedding).all():
                raise RuntimeError(
                    "NON-FINITE GRAPH EMBEDDING"
                )

            action, log_prob, entropy = (
                self.policy.sample_action(
                    encoding,
                    state,
                )
            )

            if not torch.isfinite(log_prob):
                raise RuntimeError(
                    "NON-FINITE LOG PROB"
                )

            if not torch.isfinite(entropy):
                raise RuntimeError(
                    "NON-FINITE ENTROPY"
                )

            value = self.critic(
                encoding["graph_embedding"]
            ).squeeze()

            if not torch.isfinite(value):
                raise RuntimeError(
                    "NON-FINITE VALUE"
                )

            action_name = action.action_type.name

            action_counts.setdefault(
                action_name,
                0,
            )

            action_counts[action_name] += 1

            next_state, reward, done, info = (
                self.env.step(action)
            )

            self.memory.add(
                state=state,
                action=action,
                log_prob=log_prob.detach(),
                reward=reward,
                done=done,
                value=value.detach(),
            )

            total_reward += reward

            final_info = info
            state = next_state

        total_actions = sum(
            action_counts.values()
        )

        self.last_episode_metrics = {
            "action_counts": action_counts,
            "total_actions": total_actions,
            "final_info": final_info or {},
            "final_state": state,
        }

        return total_reward

    # ======================================================
    # PPO UPDATE
    # ======================================================

    def update(self):

        advantages, returns = compute_gae(
            rewards=self.memory.rewards,
            values=self.memory.values,
            dones=self.memory.dones,
            gamma=self.gamma,
            lam=self.gae_lambda,
        )

        advantages = (
            advantages - advantages.mean()
        ) / (
            advantages.std() + 1e-8
        )

        policy_losses = []
        value_losses = []
        entropies = []

        for i in range(len(self.memory)):

            state = self.memory.states[i]
            action = self.memory.actions[i]
            old_log_prob = self.memory.log_probs[i]

            advantage = advantages[i]
            target_return = returns[i]

            encoding = self.encoder(state)

            new_log_prob, entropy = (
                self.policy.evaluate_actions(
                    encoding,
                    state,
                    action,
                )
            )

            value = self.critic(
                encoding["graph_embedding"]
            ).squeeze()

            ratio = torch.exp(
                new_log_prob - old_log_prob
            )

            if not torch.isfinite(ratio):
                raise RuntimeError(
                    "NON-FINITE PPO RATIO"
                )

            surr1 = ratio * advantage

            surr2 = torch.clamp(
                ratio,
                1.0 - self.clip_eps,
                1.0 + self.clip_eps,
            ) * advantage

            policy_loss = -torch.min(
                surr1,
                surr2,
            )

            value_loss = F.mse_loss(
                value,
                target_return,
            )

            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
            entropies.append(entropy)

        policy_loss = torch.stack(
            policy_losses
        ).mean()

        value_loss = torch.stack(
            value_losses
        ).mean()

        entropy = torch.stack(
            entropies
        ).mean()

        loss = (
            policy_loss
            + self.value_coef * value_loss
            - self.entropy_coef * entropy
        )

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.policy.parameters())
            + list(self.critic.parameters()),
            max_norm=0.5,
        )

        self.optimizer.step()

        result = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "total_loss": loss.item(),
        }

        self.memory.clear()

        return result

    # ======================================================
    # PHASE-1 WEIGHT TRANSFER
    # ======================================================

    def load_phase1_weights(
        self,
        path,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
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

        # DELIBERATELY do not restore optimizer.
        #
        # Phase 2 gets a clean PPO optimizer while retaining
        # the representation/policy learned in Phase 1.

        print(
            f"[Phase 1 -> Phase 2] Loaded model weights: {path}",
            flush=True,
        )

        print(
            "[Phase 1 -> Phase 2] Optimizer reset for Phase 2.",
            flush=True,
        )

        return checkpoint

    # ======================================================
    # FULL PHASE-2 RESUME
    # ======================================================

    def load_phase2_checkpoint(
        self,
        path,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
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

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

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

        episode = int(
            checkpoint.get("episode", 0)
        )

        history_tail = checkpoint.get(
            "history_tail",
            [],
        )

        print(
            f"[Phase 2 Resume] Episode {episode}",
            flush=True,
        )

        return episode, history_tail

    # ======================================================
    # SAVE PHASE-2 CHECKPOINT
    # ======================================================

    def save_checkpoint(
        self,
        path,
        episode,
        history_tail,
        graph_progress,
    ):

        checkpoint = {
            "phase": 2,
            "episode": episode,

            "encoder_state_dict":
                self.encoder.state_dict(),

            "policy_state_dict":
                self.policy.state_dict(),

            "critic_state_dict":
                self.critic.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            # Keep only a small tail inside the checkpoint.
            # Full episode history lives in CSV.
            "history_tail":
                history_tail[-1000:],

            "graph_progress":
                graph_progress,

            "hyperparameters": {
                "learning_rate": LEARNING_RATE,
                "gamma": self.gamma,
                "gae_lambda": self.gae_lambda,
                "clip_eps": self.clip_eps,
                "entropy_coef": self.entropy_coef,
                "value_coef": self.value_coef,
                "ppo_epochs": self.ppo_epochs,
            },

            "train_graph_count":
                TRAIN_GRAPH_COUNT,

            "validation_graph_start":
                VALIDATION_GRAPH_START,

            "validation_graph_count":
                VALIDATION_GRAPH_COUNT,

            "torch_rng_state":
                torch.get_rng_state(),
        }

        if torch.cuda.is_available():
            checkpoint["cuda_rng_state"] = (
                torch.cuda.get_rng_state_all()
            )

        torch.save(
            checkpoint,
            path,
        )

        print(
            f"[Checkpoint] Phase 2 episode {episode} saved -> {path}",
            flush=True,
        )


# ==========================================================
# CSV HELPERS
# ==========================================================

GRAPH_STATS_FIELDS = [
    "episode",
    "round",
    "graph_id",

    "nodes",
    "edges",
    "sessions",

    "ga_scb",

    "best_rl_scb",
    "final_rl_scb",

    "best_step",
    "steps_taken",

    "best_cut_size",
    "final_cut_size",

    "best_separation",
    "final_separation",

    "stop_step",
    "stop_valid",

    "total_reward",

    "add_count",
    "remove_count",
    "swap_count",
    "stop_count",

    "add_percent",
    "remove_percent",
    "swap_percent",
    "stop_percent",

    "best_to_ga_ratio",
    "final_to_ga_ratio",

    "policy_loss",
    "value_loss",
    "entropy",
    "total_loss",
]


def ensure_csv(
    path,
    fields,
):

    Path(path).parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not os.path.exists(path):
        with open(
            path,
            "w",
            newline="",
            encoding="utf-8",
        ) as f:

            writer = csv.DictWriter(
                f,
                fieldnames=fields,
            )

            writer.writeheader()


def append_csv_row(
    path,
    fields,
    row,
):

    with open(
        path,
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writerow(row)


# ==========================================================
# SAFE VALUE HELPERS
# ==========================================================

def fmt_scb(value):

    if value is None:
        return "None"

    try:
        value = float(value)

        if np.isinf(value):
            return "inf"

        if np.isnan(value):
            return "nan"

        return f"{value:.6f}"

    except (
        TypeError,
        ValueError,
    ):
        return str(value)


def finite_scb(
    value,
):

    if value is None:
        return None

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except (
        TypeError,
        ValueError,
    ):
        return None


def ratio_to_ga(
    value,
    ga,
):

    value = finite_scb(value)
    ga = finite_scb(ga)

    if value is None or ga is None:
        return None

    if abs(ga) < 1e-12:
        return None

    return value / ga


def get_final_metric(
    info,
    final_state,
    key,
    state_attr,
):

    value = info.get(
        key,
        None,
    )

    if value is not None:
        return value

    if final_state is not None:
        return getattr(
            final_state,
            state_attr,
            None,
        )

    return None


# ==========================================================
# PHASE-1 CHECKPOINT DISCOVERY
# ==========================================================

def find_phase1_checkpoint():

    if PHASE1_CHECKPOINT:
        if os.path.exists(
            PHASE1_CHECKPOINT
        ):
            return PHASE1_CHECKPOINT

        raise FileNotFoundError(
            f"Explicit Phase-1 checkpoint does not exist: "
            f"{PHASE1_CHECKPOINT}"
        )

    root = Path(
    PHASE1_CHECKPOINT_DIR
    )

    if not root.exists():
        return None

    # Prefer root/latest.pt because this is how the previous
    # trainer saved the latest Phase-1 model.
    latest = root / "latest.pt"

    if latest.exists():
        return str(latest)

    # Otherwise choose the highest-numbered root-level
    # ppo_episode_*.pt checkpoint.
    candidates = []

    for file in root.glob(
        "ppo_episode_*.pt"
    ):

        try:

            number = int(
                file.stem.replace(
                    "ppo_episode_",
                    "",
                )
            )

            candidates.append(
                (number, file)
            )

        except ValueError:
            continue

    if candidates:

        candidates.sort(
            key=lambda x: x[0]
        )

        return str(
            candidates[-1][1]
        )

    return None


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    os.makedirs(
        PHASE2_CHECKPOINT_DIR,
        exist_ok=True,
    )

    # ------------------------------------------------------
    # Redirect stdout to both console and log file.
    # ------------------------------------------------------

    log_handle = open(
        PHASE2_LOG_FILE,
        "a",
        encoding="utf-8",
    )

    sys.stdout = Tee(
        sys.__stdout__,
        log_handle,
    )

    print()
    print("=" * 78)
    print("SCB-RL PHASE 2 TRAINING")
    print("=" * 78)
    print(
        "Goal: learn when a sufficiently good current SCB should STOP."
    )
    print(
        "GA SCB: evaluation only; NEVER part of Phase-2 reward."
    )
    print()

    # ------------------------------------------------------
    # Dataset
    # ------------------------------------------------------

    dataset = GraphDataset()

    if len(dataset) < TOTAL_DATASET_GRAPHS:
        raise RuntimeError(
            f"Expected at least {TOTAL_DATASET_GRAPHS} graphs, "
            f"but dataset contains {len(dataset)}."
        )

    train_graphs = list(
        range(
            0,
            TRAIN_GRAPH_COUNT,
        )
    )

    validation_graphs = list(
        range(
            VALIDATION_GRAPH_START,
            VALIDATION_GRAPH_START
            + VALIDATION_GRAPH_COUNT,
        )
    )

    # Sanity checks.
    if set(train_graphs) & set(validation_graphs):
        raise RuntimeError(
            "Training and validation graph sets overlap."
        )

    if validation_graphs[-1] >= len(dataset):
        raise RuntimeError(
            "Validation split exceeds dataset."
        )

    print(
        f"Dataset size              : {len(dataset)}"
    )

    print(
        f"Training graphs            : "
        f"{train_graphs[0]}..{train_graphs[-1]} "
        f"({len(train_graphs)})"
    )

    print(
        f"Validation graphs          : "
        f"{validation_graphs[0]}..{validation_graphs[-1]} "
        f"({len(validation_graphs)})"
    )

    print(
        "Validation graphs are NEVER passed to the trainer."
    )

    total_episodes = (
        NUM_ROUNDS
        * TRAIN_GRAPH_COUNT
    )

    print(
        f"Rounds                     : {NUM_ROUNDS}"
    )

    print(
        f"Episodes / round           : {TRAIN_GRAPH_COUNT}"
    )

    print(
        f"Total Phase-2 episodes     : {total_episodes}"
    )

    print(
        f"Phase-2 checkpoints        : "
        f"{PHASE2_CHECKPOINT_DIR}"
    )

    print()

    # ------------------------------------------------------
    # Networks
    # ------------------------------------------------------

    encoder = SCBGraphEncoder()
    policy = HierarchicalSCBPolicy()
    critic = SCBCritic()

    trainer = Phase2PPOTrainer(
        environment=None,
        encoder=encoder,
        policy=policy,
        critic=critic,
    )

    # ------------------------------------------------------
    # CSV
    # ------------------------------------------------------

    ensure_csv(
        GRAPH_STATS_CSV,
        GRAPH_STATS_FIELDS,
    )

    # Graph progress has one cumulative record per graph.
    progress_fields = [
        "graph_id",
        "nodes",
        "edges",
        "sessions",
        "ga_scb",

        "episodes_seen",

        "best_rl_scb_ever",
        "best_step_ever",

        "latest_final_scb",
        "latest_steps",

        "latest_final_cut",
        "latest_final_separation",

        "latest_add_percent",
        "latest_remove_percent",
        "latest_swap_percent",
        "latest_stop_percent",

        "latest_reward",

        "latest_stop_step",
        "latest_stop_valid",

        "best_to_ga_ratio_ever",
    ]

    ensure_csv(
        GRAPH_PROGRESS_CSV,
        progress_fields,
    )

    # ------------------------------------------------------
    # Resume / transfer
    # ------------------------------------------------------

    start_episode = 0
    history_tail = []
    graph_progress = {}

    if (
        RESUME_PHASE2
        and os.path.exists(
            PHASE2_LATEST
        )
    ):

        print()
        print("=" * 78)
        print("PHASE 2 CHECKPOINT FOUND")
        print("=" * 78)

        start_episode, history_tail = (
            trainer.load_phase2_checkpoint(
                PHASE2_LATEST
            )
        )

        checkpoint = torch.load(
            PHASE2_LATEST,
            map_location="cpu",
        )

        graph_progress = checkpoint.get(
            "graph_progress",
            {},
        )

        print(
            f"Resuming Phase 2 from episode "
            f"{start_episode + 1}."
        )

    else:

        phase1_path = find_phase1_checkpoint()

        if phase1_path is None:
            raise FileNotFoundError(
                "No Phase-1 checkpoint found in "
                f"'{CHECKPOINT_ROOT}'. "
                "Put the Phase-1 checkpoint there or set "
                "PHASE1_CHECKPOINT explicitly."
            )

        print()
        print("=" * 78)
        print("INITIALIZING PHASE 2 FROM PHASE 1")
        print("=" * 78)

        print(
            f"Phase-1 checkpoint: {phase1_path}"
        )

        trainer.load_phase1_weights(
            phase1_path
        )

        print(
            "Starting Phase 2 at episode 1."
        )

    # ------------------------------------------------------
    # Already complete?
    # ------------------------------------------------------

    if start_episode >= total_episodes:

        print()
        print("=" * 78)
        print("PHASE 2 ALREADY COMPLETE")
        print("=" * 78)

        print(
            f"Completed episodes: {start_episode}"
        )

        print(
            f"Final checkpoint: {PHASE2_FINAL}"
        )

        log_handle.close()
        raise SystemExit

    # ------------------------------------------------------
    # Training
    # ------------------------------------------------------

    try:

        for episode_zero_based in range(
            start_episode,
            total_episodes,
        ):

            current_episode = (
                episode_zero_based + 1
            )

            round_idx = (
                episode_zero_based
                // TRAIN_GRAPH_COUNT
            )

            # Within each round, randomize ONLY the 800
            # training graphs.
            position_in_round = (
                episode_zero_based
                % TRAIN_GRAPH_COUNT
            )

            # Build a deterministic per-round ordering.
            #
            # For resumed training we can reproduce the ordering
            # from the round seed without relying on the global
            # RNG history.
            graph_order = list(
                train_graphs
            )

            if SHUFFLE_TRAINING_GRAPHS:

                round_rng = random.Random(
                    1000003 + round_idx
                )

                round_rng.shuffle(
                    graph_order
                )

            graph_idx = graph_order[
                position_in_round
            ]

            graph = dataset[
                graph_idx
            ]

            # --------------------------------------------------
            # Environment
            # --------------------------------------------------

            trainer.env = SCBEnvironment(
                graph
            )

            # --------------------------------------------------
            # Collect
            # --------------------------------------------------

            episode_reward = (
                trainer.collect_episode()
            )

            episode_length = len(
                trainer.memory
            )

            # --------------------------------------------------
            # PPO update
            # --------------------------------------------------

            stats = trainer.update()

            metrics = trainer.last_episode_metrics

            action_counts = metrics[
                "action_counts"
            ]

            total_actions = metrics[
                "total_actions"
            ]

            final_info = metrics[
                "final_info"
            ]

            final_state = metrics[
                "final_state"
            ]

            # --------------------------------------------------
            # Metrics
            # --------------------------------------------------

            ga_scb = graph.get(
                "ga_scb",
                None,
            )

            best_rl_scb = final_info.get(
                "best_scb",
                getattr(
                    trainer.env,
                    "best_scb",
                    None,
                ),
            )

            best_rl_step = final_info.get(
                "best_step",
                getattr(
                    trainer.env,
                    "best_step",
                    None,
                ),
            )

            final_rl_scb = get_final_metric(
                final_info,
                final_state,
                "current_scb",
                "scb",
            )

            final_cut_size = get_final_metric(
                final_info,
                final_state,
                "current_cut",
                "cut_size",
            )

            final_separation = get_final_metric(
                final_info,
                final_state,
                "current_sep",
                "separated_count",
            )

            best_state = getattr(
                trainer.env,
                "best_state",
                None,
            )

            best_cut_size = (
                getattr(
                    best_state,
                    "cut_size",
                    None,
                )
                if best_state is not None
                else None
            )

            best_separation = (
                getattr(
                    best_state,
                    "separated_count",
                    None,
                )
                if best_state is not None
                else None
            )

            stop_step = final_info.get(
                "stop_step",
                getattr(
                    trainer.env,
                    "stop_step",
                    None,
                ),
            )

            stop_valid = final_info.get(
                "stop_valid",
                getattr(
                    trainer.env,
                    "stop_was_valid",
                    False,
                ),
            )

            def pct(name):
                return (
                    100.0
                    * action_counts.get(
                        name,
                        0,
                    )
                    / max(
                        total_actions,
                        1,
                    )
                )

            best_ga_ratio = ratio_to_ga(
                best_rl_scb,
                ga_scb,
            )

            final_ga_ratio = ratio_to_ga(
                final_rl_scb,
                ga_scb,
            )

            # --------------------------------------------------
            # Per-episode graph record
            # --------------------------------------------------

            row = {
                "episode":
                    current_episode,

                "round":
                    round_idx + 1,

                "graph_id":
                    graph["graph_id"],

                "nodes":
                    len(graph["nodes"]),

                "edges":
                    len(graph["edges"]),

                "sessions":
                    len(graph["sessions"]),

                "ga_scb":
                    ga_scb,

                "best_rl_scb":
                    best_rl_scb,

                "final_rl_scb":
                    final_rl_scb,

                "best_step":
                    best_rl_step,

                "steps_taken":
                    episode_length,

                "best_cut_size":
                    best_cut_size,

                "final_cut_size":
                    final_cut_size,

                "best_separation":
                    best_separation,

                "final_separation":
                    final_separation,

                "stop_step":
                    stop_step,

                "stop_valid":
                    bool(stop_valid),

                "total_reward":
                    episode_reward,

                "add_count":
                    action_counts.get(
                        "ADD",
                        0,
                    ),

                "remove_count":
                    action_counts.get(
                        "REMOVE",
                        0,
                    ),

                "swap_count":
                    action_counts.get(
                        "SWAP",
                        0,
                    ),

                "stop_count":
                    action_counts.get(
                        "STOP",
                        0,
                    ),

                "add_percent":
                    pct("ADD"),

                "remove_percent":
                    pct("REMOVE"),

                "swap_percent":
                    pct("SWAP"),

                "stop_percent":
                    pct("STOP"),

                "best_to_ga_ratio":
                    best_ga_ratio,

                "final_to_ga_ratio":
                    final_ga_ratio,

                "policy_loss":
                    stats["policy_loss"],

                "value_loss":
                    stats["value_loss"],

                "entropy":
                    stats["entropy"],

                "total_loss":
                    stats["total_loss"],
            }

            append_csv_row(
                GRAPH_STATS_CSV,
                GRAPH_STATS_FIELDS,
                row,
            )

            history_tail.append(
                row
            )

            # --------------------------------------------------
            # Cumulative per-graph progress
            # --------------------------------------------------

            graph_id_key = str(
                graph["graph_id"]
            )

            previous = graph_progress.get(
                graph_id_key,
                {},
            )

            previous_best = finite_scb(
                previous.get(
                    "best_rl_scb_ever"
                )
            )

            current_best = finite_scb(
                best_rl_scb
            )

            if (
                current_best is not None
                and (
                    previous_best is None
                    or current_best < previous_best
                )
            ):
                best_ever = current_best
                best_ever_step = best_rl_step
            else:
                best_ever = previous_best
                best_ever_step = previous.get(
                    "best_step_ever"
                )

            if best_ever is not None:
                best_ever_ratio = ratio_to_ga(
                    best_ever,
                    ga_scb,
                )
            else:
                best_ever_ratio = None

            graph_progress[
                graph_id_key
            ] = {

                "graph_id":
                    graph["graph_id"],

                "nodes":
                    len(graph["nodes"]),

                "edges":
                    len(graph["edges"]),

                "sessions":
                    len(graph["sessions"]),

                "ga_scb":
                    ga_scb,

                "episodes_seen":
                    int(
                        previous.get(
                            "episodes_seen",
                            0,
                        )
                    ) + 1,

                "best_rl_scb_ever":
                    best_ever,

                "best_step_ever":
                    best_ever_step,

                "latest_final_scb":
                    final_rl_scb,

                "latest_steps":
                    episode_length,

                "latest_final_cut":
                    final_cut_size,

                "latest_final_separation":
                    final_separation,

                "latest_add_percent":
                    pct("ADD"),

                "latest_remove_percent":
                    pct("REMOVE"),

                "latest_swap_percent":
                    pct("SWAP"),

                "latest_stop_percent":
                    pct("STOP"),

                "latest_reward":
                    episode_reward,

                "latest_stop_step":
                    stop_step,

                "latest_stop_valid":
                    bool(stop_valid),

                "best_to_ga_ratio_ever":
                    best_ever_ratio,
            }

            # --------------------------------------------------
            # Human-readable log
            # --------------------------------------------------

            print(
                f"EP {current_episode:6d}/{total_episodes} | "
                f"G {graph['graph_id']:4d} | "
                f"N={len(graph['nodes']):3d} "
                f"E={len(graph['edges']):4d} "
                f"S={len(graph['sessions']):2d} | "
                f"GA={fmt_scb(ga_scb)} | "
                f"Best={fmt_scb(best_rl_scb)} | "
                f"Final={fmt_scb(final_rl_scb)} | "
                f"Steps={episode_length:4d} | "
                f"Stop@={stop_step if stop_step is not None else '-'}",
                flush=True,
            )

            print(
                f"    Cut(best/final)="
                f"{best_cut_size if best_cut_size is not None else '-'}"
                f"/"
                f"{final_cut_size if final_cut_size is not None else '-'} | "
                f"Sep(best/final)="
                f"{best_separation if best_separation is not None else '-'}"
                f"/"
                f"{final_separation if final_separation is not None else '-'} | "
                f"Reward={episode_reward:.4f}",
                flush=True,
            )

            print(
                f"    Actions: "
                f"ADD={pct('ADD'):5.1f}% "
                f"REM={pct('REMOVE'):5.1f}% "
                f"SWAP={pct('SWAP'):5.1f}% "
                f"STOP={pct('STOP'):5.1f}% | "
                f"STOP_VALID={bool(stop_valid)}",
                flush=True,
            )

            # --------------------------------------------------
            # Periodic graph-progress CSV snapshot
            # --------------------------------------------------

            if (
                current_episode
                % 100
                == 0
            ):

                with open(
                    GRAPH_PROGRESS_CSV,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as f:

                    writer = csv.DictWriter(
                        f,
                        fieldnames=progress_fields,
                    )

                    writer.writeheader()

                    for graph_id in sorted(
                        graph_progress,
                        key=lambda x: int(x),
                    ):
                        writer.writerow(
                            graph_progress[
                                graph_id
                            ]
                        )

            # --------------------------------------------------
            # Checkpoint
            # --------------------------------------------------

            if (
                current_episode
                % CHECKPOINT_INTERVAL
                == 0
            ):

                checkpoint_path = os.path.join(
                    PHASE2_CHECKPOINT_DIR,
                    f"ppo_phase2_episode_{current_episode}.pt",
                )

                trainer.save_checkpoint(
                    path=checkpoint_path,
                    episode=current_episode,
                    history_tail=history_tail,
                    graph_progress=graph_progress,
                )

                trainer.save_checkpoint(
                    path=PHASE2_LATEST,
                    episode=current_episode,
                    history_tail=history_tail,
                    graph_progress=graph_progress,
                )

                # Optional rolling retention.
                if MAX_ROLLING_CHECKPOINTS is not None:

                    candidates = sorted(
                        glob.glob(
                            os.path.join(
                                PHASE2_CHECKPOINT_DIR,
                                "ppo_phase2_episode_*.pt",
                            )
                        ),
                        key=lambda p: int(
                            Path(p).stem.split("_")[-1]
                        ),
                    )

                    while len(
                        candidates
                    ) > MAX_ROLLING_CHECKPOINTS:

                        old = candidates.pop(0)

                        try:
                            os.remove(old)
                        except OSError:
                            pass

        # ------------------------------------------------------
        # Final checkpoint
        # ------------------------------------------------------

        trainer.save_checkpoint(
            path=PHASE2_FINAL,
            episode=total_episodes,
            history_tail=history_tail,
            graph_progress=graph_progress,
        )

        print()
        print("=" * 78)
        print("PHASE 2 TRAINING COMPLETE")
        print("=" * 78)
        print(
            f"Training episodes completed : {total_episodes}"
        )
        print(
            f"Phase-2 final checkpoint     : {PHASE2_FINAL}"
        )
        print(
            f"Episode CSV                  : {GRAPH_STATS_CSV}"
        )
        print(
            f"Graph progress CSV           : {GRAPH_PROGRESS_CSV}"
        )

    except KeyboardInterrupt:

        print()
        print("=" * 78)
        print("PHASE 2 INTERRUPTED")
        print("=" * 78)

        # If interrupted between checkpoints, latest remains
        # the last safe checkpoint.
        print(
            f"Last safe checkpoint: {PHASE2_LATEST}"
        )

    finally:

        log_handle.flush()
        log_handle.close()
