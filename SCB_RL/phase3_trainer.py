"""
SCB-RL PHASE 3 TRAINER — FINAL
==============================

Pipeline
--------
For every training graph:

    Graph
      ↓
    Phase-2 inference
      ↓
    Phase-2 final cut
      ↓
    Phase-3 environment
      ↓
    Phase-3 PPO episode
      ↓
    STOP
      ↓
    PPO update
      ↓
    Phase-3 checkpoint

Phase 3 is trained from scratch.

Phase-2 is inference-only:
    Phase-2 weights are NEVER optimized in Phase 3.

Dataset:
    graphs 0..799   = training
    graphs 800..999 = permanently unseen validation

Logging:
    phase3_graph_stats.csv
        One row per graph visit / episode.

    phase3_progression.csv
        Same information organized around repeated visits to the
        same graph.  This lets us answer:

            "When graph 7 appears again later, did the agent
             perform better than its previous visit?"

    phase3_move_analysis.csv
        One row per action taken inside every Phase-3 episode.

    phase3_training.txt
        Human-readable training progression.

No normal max-step termination is used.  A watchdog exists only
to detect a broken policy/environment and raise an error.
"""

import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from Data.loader import GraphDataset

from SCB_RL.gnn import SCBGraphEncoder
from SCB_RL.critic import SCBCritic
from SCB_RL.memory import PPOMemory
from SCB_RL.gae import compute_gae

from .phase_2.phase2_environment import SCBEnvironment as Phase2Environment
from .phase_2.phase2_policy import HierarchicalSCBPolicy as Phase2Policy

from SCB_RL.phase3_environment import Phase3Environment
from SCB_RL.phase3_policy import HierarchicalSCBPolicy as Phase3Policy


# ==========================================================
# DATASET
# ==========================================================

TOTAL_DATASET_GRAPHS = 1000

TRAIN_GRAPH_START = 0
TRAIN_GRAPH_COUNT = 800

VALIDATION_GRAPH_START = 800
VALIDATION_GRAPH_COUNT = 200

NUM_ROUNDS = 1

SHUFFLE_TRAINING_GRAPHS = True


# ==========================================================
# CHECKPOINTS
# ==========================================================

CHECKPOINT_ROOT = "checkpoints"

PHASE2_CHECKPOINT = os.path.join(
    CHECKPOINT_ROOT,
    "phase 2 checkpoints",
    "latest.pt",
)

PHASE3_CHECKPOINT_DIR = os.path.join(
    CHECKPOINT_ROOT,
    "phase 3 checkpoints",
)

PHASE3_LATEST = os.path.join(
    PHASE3_CHECKPOINT_DIR,
    "latest.pt",
)

PHASE3_FINAL = os.path.join(
    PHASE3_CHECKPOINT_DIR,
    "ppo_phase3_final.pt",
)

RESUME_PHASE3 = True

CHECKPOINT_INTERVAL = 1000


# ==========================================================
# LOGGING
# ==========================================================

PHASE3_LOG_FILE = "phase3_training.txt"

GRAPH_STATS_CSV = os.path.join(
    PHASE3_CHECKPOINT_DIR,
    "phase3_graph_stats.csv",
)

PROGRESSION_CSV = os.path.join(
    PHASE3_CHECKPOINT_DIR,
    "phase3_progression.csv",
)

MOVE_ANALYSIS_CSV = os.path.join(
    PHASE3_CHECKPOINT_DIR,
    "phase3_move_analysis.csv",
)


# ==========================================================
# PPO
# ==========================================================

LEARNING_RATE = 3e-4

GAMMA = 0.99
GAE_LAMBDA = 0.95

CLIP_EPS = 0.20

ENTROPY_COEF = 0.01
VALUE_COEF = 0.50

PPO_EPOCHS = 1


# ==========================================================
# PHASE-3 POLICY
# ==========================================================

STOP_EXPLORATION = 0.01
MIN_STEPS_BEFORE_STOP = 5


# ==========================================================
# WATCHDOG
# ==========================================================

# This is NOT a training termination condition.
# It only catches an implementation/policy that never terminates.
WATCHDOG_STEPS = 5000


# ==========================================================
# PERFORMANCE
# ==========================================================

torch.set_num_threads(1)

try:
    torch.set_num_interop_threads(1)
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
# HELPERS
# ==========================================================

def finite_scb(value):

    if value is None:
        return None

    try:
        value = float(value)

        if not np.isfinite(value):
            return None

        return value

    except Exception:
        return None


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

    except Exception:
        return str(value)


def percentage(count, total):

    if total <= 0:
        return 0.0

    return 100.0 * count / total


def graph_value(graph, *names, default=None):

    for name in names:

        if isinstance(graph, dict):

            if name in graph:
                return graph[name]

        else:

            if hasattr(graph, name):
                return getattr(graph, name)

    return default


def graph_id_of(graph, fallback):

    value = graph_value(
        graph,
        "graph_id",
        "id",
        default=fallback,
    )

    try:
        return int(value)
    except Exception:
        return fallback


def graph_nodes(graph):

    nodes = graph_value(
        graph,
        "nodes",
        default=[],
    )

    try:
        return len(nodes)
    except Exception:
        return int(
            graph_value(
                graph,
                "num_nodes",
                default=0,
            )
        )


def graph_edges(graph):

    edges = graph_value(
        graph,
        "edges",
        default=[],
    )

    try:
        return len(edges)
    except Exception:
        return int(
            graph_value(
                graph,
                "num_edges",
                default=0,
            )
        )


def graph_sessions(graph):

    sessions = graph_value(
        graph,
        "sessions",
        default=[],
    )

    try:
        return len(sessions)
    except Exception:
        return int(
            graph_value(
                graph,
                "num_sessions",
                default=0,
            )
        )


def action_name(action):

    value = getattr(
        action,
        "action_type",
        None,
    )

    if value is None:
        return "UNKNOWN"

    name = getattr(
        value,
        "name",
        None,
    )

    if name is not None:
        return name

    return str(value)


def action_edge_text(action):

    fields = [
        "edge",
        "edge_id",
        "add_edge",
        "remove_edge",
        "swap_edge",
        "remove_edge_id",
        "add_edge_id",
    ]

    values = []

    for field in fields:

        if hasattr(action, field):

            value = getattr(
                action,
                field,
            )

            if value is not None:
                values.append(
                    f"{field}={value}"
                )

    if not values:
        return ""

    return "; ".join(values)


# ==========================================================
# CSV SCHEMAS
# ==========================================================

GRAPH_STATS_FIELDS = [
    "episode",
    "round",
    "graph_id",
    "graph_visit",

    "nodes",
    "edges",
    "sessions",

    "ga_scb",

    "phase2_final_scb",
    "phase2_final_cut",
    "phase2_final_sep",
    "phase2_steps",

    "phase3_initial_scb",
    "phase3_best_scb",
    "phase3_final_scb",

    "best_improvement_abs",
    "best_improvement_pct",

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

    # Cross-visit learning indicators.
    "previous_visit_best_scb",
    "previous_visit_final_scb",
    "best_scb_delta_vs_previous",
    "final_scb_delta_vs_previous",
    "steps_delta_vs_previous",
]


PROGRESSION_FIELDS = [
    "graph_id",
    "episode",
    "round",
    "graph_visit",

    "phase2_final_scb",
    "phase3_initial_scb",
    "phase3_best_scb",
    "phase3_final_scb",

    "best_improvement_abs",
    "best_improvement_pct",

    "steps_taken",
    "best_step",
    "stop_step",

    "add_percent",
    "remove_percent",
    "swap_percent",
    "stop_percent",

    "total_reward",

    "previous_visit_best_scb",
    "best_scb_delta_vs_previous",

    "previous_visit_final_scb",
    "final_scb_delta_vs_previous",

    "steps_delta_vs_previous",
]


MOVE_FIELDS = [
    "episode",
    "round",
    "graph_id",
    "graph_visit",

    "step",

    "action",
    "action_detail",

    "scb_before",
    "scb_after",
    "delta_scb",

    "current_best_scb",
    "best_improved",

    "cut_size_before",
    "cut_size_after",

    "separation_before",
    "separation_after",

    "reward",

    "done",
]


def ensure_csv(path, fields):

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


def append_csv(path, fields, row):

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
# PHASE-2 INITIALIZER
# ==========================================================

class Phase2Initializer:

    """
    Frozen Phase-2 inference model.

    It is deliberately separate from the Phase-3 optimizer.
    """

    def __init__(
        self,
        checkpoint_path,
        device,
    ):

        self.device = device

        self.encoder = (
            SCBGraphEncoder()
            .to(device)
            .eval()
        )

        self.policy = (
            Phase2Policy(
                stop_exploration=0.01,
                min_steps_before_stop=5,
            )
            .to(device)
            .eval()
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

        self.encoder.load_state_dict(
            checkpoint[
                "encoder_state_dict"
            ]
        )

        self.policy.load_state_dict(
            checkpoint[
                "policy_state_dict"
            ]
        )

        print(
            "[Phase 2 -> Phase 3] "
            f"Loaded: {checkpoint_path}",
            flush=True,
        )

    @torch.no_grad()
    def get_final_cut(self, graph):

        env = Phase2Environment(
            graph
        )

        state = env.reset()

        steps = 0

        while not env.done:

            steps += 1

            if (
                WATCHDOG_STEPS is not None
                and steps > WATCHDOG_STEPS
            ):

                raise RuntimeError(
                    "Phase-2 initializer watchdog "
                    f"triggered on graph "
                    f"{graph_id_of(graph, -1)} "
                    f"after {WATCHDOG_STEPS} steps."
                )

            encoding = self.encoder(
                state
            )

            action, _, _ = (
                self.policy.sample_action(
                    encoding,
                    state,
                )
            )

            _, _, done, _ = env.step(
                action
            )

            state = env.current_state

        final_state = (
            env.current_state
        )

        return {
            "cut": set(
                final_state.cut
            ),

            "cut_size":
                final_state.cut_size,

            "scb":
                final_state.scb,

            "separation":
                final_state.separated_count,

            "steps":
                steps,

            "best_scb":
                env.best_scb,
        }


# ==========================================================
# PHASE-3 TRAINER
# ==========================================================

class Phase3Trainer:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Using device: {self.device}",
            flush=True,
        )

        # --------------------------------------------------
        # FRESH Phase-3 networks.
        # --------------------------------------------------

        self.encoder = (
            SCBGraphEncoder()
            .to(self.device)
        )

        self.policy = (
            Phase3Policy(
                stop_exploration=
                    STOP_EXPLORATION,
                min_steps_before_stop=
                    MIN_STEPS_BEFORE_STOP,
            )
            .to(self.device)
        )

        self.critic = (
            SCBCritic()
            .to(self.device)
        )

        # --------------------------------------------------
        # Fresh Phase-3 optimizer.
        # --------------------------------------------------

        self.optimizer = Adam(
            list(
                self.encoder.parameters()
            )
            + list(
                self.policy.parameters()
            )
            + list(
                self.critic.parameters()
            ),
            lr=LEARNING_RATE,
        )

        self.memory = PPOMemory()

        self.last_metrics = {}

    # ======================================================
    # EPISODE COLLECTION
    # ======================================================

    def collect_episode(
        self,
        env,
        initial_cut,
        graph_id,
        graph_visit,
        episode,
        round_idx,
    ):

        self.memory.clear()

        state = env.reset(
            initial_cut=initial_cut
        )

        done = False

        total_reward = 0.0

        action_counts = {
            "ADD": 0,
            "REMOVE": 0,
            "SWAP": 0,
            "STOP": 0,
        }

        move_rows = []

        step = 0

        while not done:

            step += 1

            if (
                WATCHDOG_STEPS is not None
                and step > WATCHDOG_STEPS
            ):

                raise RuntimeError(
                    "Phase-3 watchdog triggered on "
                    f"graph {graph_id}, "
                    f"episode {episode}, "
                    f"after {WATCHDOG_STEPS} steps. "
                    "No normal max-step termination "
                    "is being used."
                )

            scb_before = finite_scb(
                getattr(
                    state,
                    "scb",
                    None,
                )
            )

            cut_before = getattr(
                state,
                "cut_size",
                None,
            )

            sep_before = getattr(
                state,
                "separated_count",
                None,
            )

            encoding = self.encoder(
                state
            )

            graph_embedding = encoding[
                "graph_embedding"
            ]

            if not torch.isfinite(
                graph_embedding
            ).all():

                raise RuntimeError(
                    "Non-finite Phase-3 "
                    "graph embedding."
                )

            action, log_prob, entropy = (
                self.policy.sample_action(
                    encoding,
                    state,
                )
            )

            value = self.critic(
                graph_embedding
            ).squeeze()

            if not torch.isfinite(
                log_prob
            ):

                raise RuntimeError(
                    "Non-finite Phase-3 "
                    "log probability."
                )

            if not torch.isfinite(
                entropy
            ):

                raise RuntimeError(
                    "Non-finite Phase-3 "
                    "entropy."
                )

            if not torch.isfinite(
                value
            ):

                raise RuntimeError(
                    "Non-finite Phase-3 "
                    "critic value."
                )

            name = action_name(
                action
            )

            action_counts[name] = (
                action_counts.get(
                    name,
                    0,
                )
                + 1
            )

            (
                next_state,
                reward,
                done,
                info,
            ) = env.step(action)

            reward = float(reward)

            if not np.isfinite(
                reward
            ):

                raise RuntimeError(
                    "Non-finite Phase-3 reward."
                )

            scb_after = finite_scb(
                getattr(
                    next_state,
                    "scb",
                    None,
                )
            )

            cut_after = getattr(
                next_state,
                "cut_size",
                None,
            )

            sep_after = getattr(
                next_state,
                "separated_count",
                None,
            )

            delta_scb = None

            if (
                scb_before is not None
                and scb_after is not None
            ):

                delta_scb = (
                    scb_after
                    - scb_before
                )

            best_scb = finite_scb(
                getattr(
                    env,
                    "best_scb",
                    None,
                )
            )

            best_improved = False

            if (
                scb_before is not None
                and scb_after is not None
                and scb_after < scb_before
            ):

                best_improved = True

            move_rows.append({
                "episode":
                    episode,

                "round":
                    round_idx + 1,

                "graph_id":
                    graph_id,

                "graph_visit":
                    graph_visit,

                "step":
                    step,

                "action":
                    name,

                "action_detail":
                    action_edge_text(
                        action
                    ),

                "scb_before":
                    scb_before,

                "scb_after":
                    scb_after,

                "delta_scb":
                    delta_scb,

                "current_best_scb":
                    best_scb,

                "best_improved":
                    best_improved,

                "cut_size_before":
                    cut_before,

                "cut_size_after":
                    cut_after,

                "separation_before":
                    sep_before,

                "separation_after":
                    sep_after,

                "reward":
                    reward,

                "done":
                    done,
            })

            self.memory.add(
                state=state,
                action=action,
                log_prob=log_prob.detach(),
                reward=reward,
                done=done,
                value=value.detach(),
            )

            total_reward += reward

            state = next_state

        self.last_metrics = {
            "action_counts":
                action_counts,

            "total_actions":
                sum(
                    action_counts.values()
                ),

            "move_rows":
                move_rows,

            "total_reward":
                total_reward,

            "final_state":
                state,

            "initial_scb":
                getattr(
                    env,
                    "initial_scb",
                    None,
                ),

            "best_scb":
                getattr(
                    env,
                    "best_scb",
                    None,
                ),

            "best_step":
                getattr(
                    env,
                    "best_step",
                    None,
                ),

            "best_state":
                getattr(
                    env,
                    "best_state",
                    None,
                ),

            "final_info":
                getattr(
                    env,
                    "last_info",
                    None,
                ),
        }

        return total_reward

    # ======================================================
    # PPO
    # ======================================================

    def update(self):

        advantages, returns = compute_gae(
            rewards=self.memory.rewards,
            values=self.memory.values,
            dones=self.memory.dones,
            gamma=GAMMA,
            lam=GAE_LAMBDA,
        )

        if len(advantages) > 1:

            advantages = (
                advantages
                - advantages.mean()
            ) / (
                advantages.std()
                + 1e-8
            )

        policy_loss = torch.tensor(
            0.0,
            device=self.device,
        )

        value_loss = torch.tensor(
            0.0,
            device=self.device,
        )

        entropy = torch.tensor(
            0.0,
            device=self.device,
        )

        for _ in range(
            PPO_EPOCHS
        ):

            policy_terms = []
            value_terms = []
            entropy_terms = []

            for i in range(
                len(self.memory)
            ):

                state = (
                    self.memory.states[i]
                )

                action = (
                    self.memory.actions[i]
                )

                old_log_prob = (
                    self.memory.log_probs[i]
                )

                encoding = self.encoder(
                    state
                )

                (
                    new_log_prob,
                    action_entropy,
                ) = (
                    self.policy.evaluate_actions(
                        encoding,
                        state,
                        action,
                    )
                )

                value = self.critic(
                    encoding[
                        "graph_embedding"
                    ]
                ).squeeze()

                ratio = torch.exp(
                    new_log_prob
                    - old_log_prob
                )

                advantage = advantages[i]

                target_return = returns[i]

                unclipped = (
                    ratio
                    * advantage
                )

                clipped = (
                    torch.clamp(
                        ratio,
                        1.0 - CLIP_EPS,
                        1.0 + CLIP_EPS,
                    )
                    * advantage
                )

                policy_terms.append(
                    -torch.min(
                        unclipped,
                        clipped,
                    )
                )

                value_terms.append(
                    F.mse_loss(
                        value,
                        target_return,
                    )
                )

                entropy_terms.append(
                    action_entropy
                )

            policy_loss = torch.stack(
                policy_terms
            ).mean()

            value_loss = torch.stack(
                value_terms
            ).mean()

            entropy = torch.stack(
                entropy_terms
            ).mean()

            loss = (
                policy_loss
                + VALUE_COEF
                * value_loss
                - ENTROPY_COEF
                * entropy
            )

            if not torch.isfinite(
                loss
            ):

                raise RuntimeError(
                    "Non-finite PPO loss."
                )

            self.optimizer.zero_grad()

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(
                    self.encoder.parameters()
                )
                + list(
                    self.policy.parameters()
                )
                + list(
                    self.critic.parameters()
                ),
                max_norm=1.0,
            )

            self.optimizer.step()

        return {
            "policy_loss":
                float(
                    policy_loss.detach()
                    .cpu()
                ),

            "value_loss":
                float(
                    value_loss.detach()
                    .cpu()
                ),

            "entropy":
                float(
                    entropy.detach()
                    .cpu()
                ),

            "total_loss":
                float(
                    loss.detach()
                    .cpu()
                ),
        }

    # ======================================================
    # CHECKPOINT
    # ======================================================

    def save_checkpoint(
        self,
        path,
        episode,
        history,
        graph_visits,
    ):

        checkpoint = {
            "phase": 3,

            "episode":
                episode,

            "encoder_state_dict":
                self.encoder.state_dict(),

            "policy_state_dict":
                self.policy.state_dict(),

            "critic_state_dict":
                self.critic.state_dict(),

            "optimizer_state_dict":
                self.optimizer.state_dict(),

            "history_tail":
                history[-1000:],

            "graph_visits":
                graph_visits,

            "hyperparameters": {
                "learning_rate":
                    LEARNING_RATE,

                "gamma":
                    GAMMA,

                "gae_lambda":
                    GAE_LAMBDA,

                "clip_eps":
                    CLIP_EPS,

                "entropy_coef":
                    ENTROPY_COEF,

                "value_coef":
                    VALUE_COEF,

                "ppo_epochs":
                    PPO_EPOCHS,

                "stop_exploration":
                    STOP_EXPLORATION,

                "min_steps_before_stop":
                    MIN_STEPS_BEFORE_STOP,
            },

            "torch_rng_state":
                torch.get_rng_state(),
        }

        if torch.cuda.is_available():

            checkpoint[
                "cuda_rng_state"
            ] = torch.cuda.get_rng_state_all()

        torch.save(
            checkpoint,
            path,
        )

        print(
            f"[Checkpoint] "
            f"Episode {episode} -> {path}",
            flush=True,
        )

    def load_checkpoint(
        self,
        path,
    ):

        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.encoder.load_state_dict(
            checkpoint[
                "encoder_state_dict"
            ]
        )

        self.policy.load_state_dict(
            checkpoint[
                "policy_state_dict"
            ]
        )

        self.critic.load_state_dict(
            checkpoint[
                "critic_state_dict"
            ]
        )

        self.optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        if "torch_rng_state" in checkpoint:

            torch.set_rng_state(
                checkpoint[
                    "torch_rng_state"
                ]
            )

        if (
            "cuda_rng_state" in checkpoint
            and torch.cuda.is_available()
        ):

            torch.cuda.set_rng_state_all(
                checkpoint[
                    "cuda_rng_state"
                ]
            )

        return (
            int(
                checkpoint.get(
                    "episode",
                    0,
                )
            ),

            checkpoint.get(
                "history_tail",
                [],
            ),

            checkpoint.get(
                "graph_visits",
                {},
            ),
        )


# ==========================================================
# MAIN
# ==========================================================

if __name__ == "__main__":

    os.makedirs(
        PHASE3_CHECKPOINT_DIR,
        exist_ok=True,
    )

    log_handle = open(
        PHASE3_LOG_FILE,
        "a",
        encoding="utf-8",
    )

    sys.stdout = Tee(
        sys.__stdout__,
        log_handle,
    )

    print()
    print("=" * 78)
    print("SCB-RL PHASE 3 TRAINING — FINAL")
    print("=" * 78)
    print(
        "Goal: optimize the Phase-2 final cut "
        "for lower SCB, then STOP."
    )
    print(
        "Phase-3 networks: FRESH."
    )
    print(
        "Phase-2 model: INFERENCE ONLY."
    )
    print(
        "GA SCB: EVALUATION ONLY."
    )
    print()

    dataset = GraphDataset()

    if len(dataset) < TOTAL_DATASET_GRAPHS:

        raise RuntimeError(
            f"Expected at least "
            f"{TOTAL_DATASET_GRAPHS} graphs; "
            f"got {len(dataset)}."
        )

    train_graphs = list(
        range(
            TRAIN_GRAPH_START,
            TRAIN_GRAPH_START
            + TRAIN_GRAPH_COUNT,
        )
    )

    validation_graphs = list(
        range(
            VALIDATION_GRAPH_START,
            VALIDATION_GRAPH_START
            + VALIDATION_GRAPH_COUNT,
        )
    )

    if (
        set(train_graphs)
        & set(validation_graphs)
    ):

        raise RuntimeError(
            "Training and validation "
            "graph overlap detected."
        )

    print(
        f"Dataset size              : "
        f"{len(dataset)}"
    )

    print(
        f"Training graphs            : "
        f"0..{TRAIN_GRAPH_COUNT - 1} "
        f"({TRAIN_GRAPH_COUNT})"
    )

    print(
        f"Validation graphs          : "
        f"{VALIDATION_GRAPH_START}.."
        f"{VALIDATION_GRAPH_START + VALIDATION_GRAPH_COUNT - 1} "
        f"({VALIDATION_GRAPH_COUNT})"
    )

    print(
        "Validation graphs are NEVER "
        "passed to training."
    )

    print(
        f"Rounds                     : "
        f"{NUM_ROUNDS}"
    )

    print(
        f"Episodes                   : "
        f"{NUM_ROUNDS * TRAIN_GRAPH_COUNT}"
    )

    print(
        f"Phase-2 checkpoint         : "
        f"{PHASE2_CHECKPOINT}"
    )

    print(
        f"Phase-3 checkpoint dir     : "
        f"{PHASE3_CHECKPOINT_DIR}"
    )

    print()

    if not os.path.exists(
        PHASE2_CHECKPOINT
    ):

        raise FileNotFoundError(
            "Phase-2 checkpoint not found:\n"
            f"{PHASE2_CHECKPOINT}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # ------------------------------------------------------
    # Frozen Phase-2 initializer.
    # ------------------------------------------------------

    print("=" * 78)
    print("INITIALIZING PHASE-2 INFERENCE MODEL")
    print("=" * 78)

    phase2 = Phase2Initializer(
        PHASE2_CHECKPOINT,
        device,
    )

    # ------------------------------------------------------
    # Fresh Phase-3 trainer.
    # ------------------------------------------------------

    print()
    print("=" * 78)
    print("INITIALIZING FRESH PHASE-3 MODEL")
    print("=" * 78)

    trainer = Phase3Trainer()

    ensure_csv(
        GRAPH_STATS_CSV,
        GRAPH_STATS_FIELDS,
    )

    ensure_csv(
        PROGRESSION_CSV,
        PROGRESSION_FIELDS,
    )

    ensure_csv(
        MOVE_ANALYSIS_CSV,
        MOVE_FIELDS,
    )

    # ------------------------------------------------------
    # Resume state.
    # ------------------------------------------------------

    start_episode = 0

    history = []

    # graph_visits stores the latest performance information
    # for each graph, allowing direct comparison when the same
    # graph appears again.
    graph_visits = {}

    if (
        RESUME_PHASE3
        and os.path.exists(
            PHASE3_LATEST
        )
    ):

        print()
        print("=" * 78)
        print("PHASE-3 CHECKPOINT FOUND")
        print("=" * 78)

        try:

            (
                start_episode,
                history,
                graph_visits,
            ) = trainer.load_checkpoint(
                PHASE3_LATEST
            )

            print(
                f"Resuming at episode "
                f"{start_episode + 1}.",
                flush=True,
            )

        except Exception as exc:

            print(
                "WARNING: checkpoint could "
                f"not be loaded: {exc}"
            )

            print(
                "Starting Phase 3 fresh."
            )

            start_episode = 0
            history = []
            graph_visits = {}

    else:

        print(
            "No Phase-3 checkpoint found."
        )

        print(
            "Starting Phase 3 from scratch."
        )

    total_episodes = (
        NUM_ROUNDS
        * TRAIN_GRAPH_COUNT
    )

    if start_episode >= total_episodes:

        print(
            "Phase-3 training already complete."
        )

        log_handle.close()

        raise SystemExit

    current_episode = start_episode

    try:

        for round_idx in range(
            NUM_ROUNDS
        ):

            graph_order = list(
                train_graphs
            )

            if SHUFFLE_TRAINING_GRAPHS:

                rng = random.Random(
                    3000001
                    + round_idx
                )

                rng.shuffle(
                    graph_order
                )

            for graph_position, graph_idx in enumerate(
                graph_order
            ):

                if (
                    current_episode
                    >= total_episodes
                ):
                    break

                graph = dataset[
                    graph_idx
                ]

                gid = graph_id_of(
                    graph,
                    graph_idx,
                )

                # --------------------------------------------------
                # Repeated-graph tracking.
                # --------------------------------------------------

                previous = (
                    graph_visits.get(
                        str(gid)
                    )
                    or graph_visits.get(
                        gid
                    )
                )

                if previous is None:

                    graph_visit = 1

                else:

                    graph_visit = int(
                        previous.get(
                            "graph_visit",
                            0,
                        )
                    ) + 1

                # --------------------------------------------------
                # Phase 2 produces the starting cut.
                # --------------------------------------------------

                phase2_result = (
                    phase2.get_final_cut(
                        graph
                    )
                )

                phase2_cut = (
                    phase2_result["cut"]
                )

                # --------------------------------------------------
                # Phase 3 starts from Phase-2 cut.
                # --------------------------------------------------

                env = Phase3Environment(
                    graph
                )

                episode_number = (
                    current_episode + 1
                )

                episode_reward = (
                    trainer.collect_episode(
                        env,
                        phase2_cut,
                        graph_id=gid,
                        graph_visit=graph_visit,
                        episode=episode_number,
                        round_idx=round_idx,
                    )
                )

                ppo_stats = (
                    trainer.update()
                )

                metrics = (
                    trainer.last_metrics
                )

                action_counts = (
                    metrics[
                        "action_counts"
                    ]
                )

                total_actions = (
                    metrics[
                        "total_actions"
                    ]
                )

                final_state = (
                    metrics[
                        "final_state"
                    ]
                )

                best_state = (
                    metrics[
                        "best_state"
                    ]
                )

                initial_scb = (
                    metrics[
                        "initial_scb"
                    ]
                )

                best_scb = (
                    metrics[
                        "best_scb"
                    ]
                )

                final_scb = getattr(
                    final_state,
                    "scb",
                    None,
                )

                best_finite = finite_scb(
                    best_scb
                )

                initial_finite = finite_scb(
                    initial_scb
                )

                final_finite = finite_scb(
                    final_scb
                )

                best_improvement_abs = None
                best_improvement_pct = None

                if (
                    initial_finite is not None
                    and best_finite is not None
                ):

                    best_improvement_abs = (
                        initial_finite
                        - best_finite
                    )

                    if abs(
                        initial_finite
                    ) > 1e-12:

                        best_improvement_pct = (
                            100.0
                            * best_improvement_abs
                            / initial_finite
                        )

                previous_best = None
                previous_final = None
                previous_steps = None

                if previous is not None:

                    previous_best = (
                        finite_scb(
                            previous.get(
                                "phase3_best_scb"
                            )
                        )
                    )

                    previous_final = (
                        finite_scb(
                            previous.get(
                                "phase3_final_scb"
                            )
                        )
                    )

                    previous_steps = (
                        previous.get(
                            "steps_taken"
                        )
                    )

                best_delta_previous = None
                final_delta_previous = None
                steps_delta_previous = None

                if (
                    best_finite is not None
                    and previous_best is not None
                ):

                    # Negative = current best SCB is lower
                    # than the previous visit.
                    best_delta_previous = (
                        best_finite
                        - previous_best
                    )

                if (
                    final_finite is not None
                    and previous_final is not None
                ):

                    final_delta_previous = (
                        final_finite
                        - previous_final
                    )

                if (
                    previous_steps is not None
                ):

                    steps_delta_previous = (
                        len(
                            trainer.memory
                        )
                        - int(
                            previous_steps
                        )
                    )

                ga_scb = graph_value(
                    graph,
                    "ga_scb",
                    "GA_scb",
                    "best_fitness",
                    default=None,
                )

                ga_finite = finite_scb(
                    ga_scb
                )

                best_to_ga_ratio = None
                final_to_ga_ratio = None

                if (
                    best_finite is not None
                    and ga_finite is not None
                    and abs(ga_finite) > 1e-12
                ):

                    best_to_ga_ratio = (
                        best_finite
                        / ga_finite
                    )

                if (
                    final_finite is not None
                    and ga_finite is not None
                    and abs(ga_finite) > 1e-12
                ):

                    final_to_ga_ratio = (
                        final_finite
                        / ga_finite
                    )

                final_info = (
                    metrics.get(
                        "final_info"
                    )
                    or {}
                )

                stop_step = final_info.get(
                    "step",
                    len(
                        trainer.memory
                    ),
                )

                stop_valid = bool(
                    final_info.get(
                        "stopped",
                        True,
                    )
                    and not final_info.get(
                        "invalid",
                        False,
                    )
                )

                # --------------------------------------------------
                # Per-step move log.
                # --------------------------------------------------

                for move in metrics[
                    "move_rows"
                ]:

                    append_csv(
                        MOVE_ANALYSIS_CSV,
                        MOVE_FIELDS,
                        move,
                    )

                # --------------------------------------------------
                # Main graph stats row.
                # --------------------------------------------------

                row = {
                    "episode":
                        episode_number,

                    "round":
                        round_idx + 1,

                    "graph_id":
                        gid,

                    "graph_visit":
                        graph_visit,

                    "nodes":
                        graph_nodes(graph),

                    "edges":
                        graph_edges(graph),

                    "sessions":
                        graph_sessions(graph),

                    "ga_scb":
                        ga_scb,

                    "phase2_final_scb":
                        phase2_result[
                            "scb"
                        ],

                    "phase2_final_cut":
                        phase2_result[
                            "cut_size"
                        ],

                    "phase2_final_sep":
                        phase2_result[
                            "separation"
                        ],

                    "phase2_steps":
                        phase2_result[
                            "steps"
                        ],

                    "phase3_initial_scb":
                        initial_scb,

                    "phase3_best_scb":
                        best_scb,

                    "phase3_final_scb":
                        final_scb,

                    "best_improvement_abs":
                        best_improvement_abs,

                    "best_improvement_pct":
                        best_improvement_pct,

                    "best_step":
                        metrics[
                            "best_step"
                        ],

                    "steps_taken":
                        len(
                            trainer.memory
                        ),

                    "best_cut_size":
                        (
                            best_state.cut_size
                            if best_state is not None
                            else None
                        ),

                    "final_cut_size":
                        final_state.cut_size,

                    "best_separation":
                        (
                            best_state.separated_count
                            if best_state is not None
                            else None
                        ),

                    "final_separation":
                        final_state.separated_count,

                    "stop_step":
                        stop_step,

                    "stop_valid":
                        stop_valid,

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
                        percentage(
                            action_counts.get(
                                "ADD",
                                0,
                            ),
                            total_actions,
                        ),

                    "remove_percent":
                        percentage(
                            action_counts.get(
                                "REMOVE",
                                0,
                            ),
                            total_actions,
                        ),

                    "swap_percent":
                        percentage(
                            action_counts.get(
                                "SWAP",
                                0,
                            ),
                            total_actions,
                        ),

                    "stop_percent":
                        percentage(
                            action_counts.get(
                                "STOP",
                                0,
                            ),
                            total_actions,
                        ),

                    "best_to_ga_ratio":
                        best_to_ga_ratio,

                    "final_to_ga_ratio":
                        final_to_ga_ratio,

                    "policy_loss":
                        ppo_stats[
                            "policy_loss"
                        ],

                    "value_loss":
                        ppo_stats[
                            "value_loss"
                        ],

                    "entropy":
                        ppo_stats[
                            "entropy"
                        ],

                    "total_loss":
                        ppo_stats[
                            "total_loss"
                        ],

                    "previous_visit_best_scb":
                        previous_best,

                    "previous_visit_final_scb":
                        previous_final,

                    "best_scb_delta_vs_previous":
                        best_delta_previous,

                    "final_scb_delta_vs_previous":
                        final_delta_previous,

                    "steps_delta_vs_previous":
                        steps_delta_previous,
                }

                append_csv(
                    GRAPH_STATS_CSV,
                    GRAPH_STATS_FIELDS,
                    row,
                )

                # --------------------------------------------------
                # Explicit repeated-graph progression log.
                # --------------------------------------------------

                progression_row = {
                    "graph_id":
                        gid,

                    "episode":
                        episode_number,

                    "round":
                        round_idx + 1,

                    "graph_visit":
                        graph_visit,

                    "phase2_final_scb":
                        phase2_result[
                            "scb"
                        ],

                    "phase3_initial_scb":
                        initial_scb,

                    "phase3_best_scb":
                        best_scb,

                    "phase3_final_scb":
                        final_scb,

                    "best_improvement_abs":
                        best_improvement_abs,

                    "best_improvement_pct":
                        best_improvement_pct,

                    "steps_taken":
                        len(
                            trainer.memory
                        ),

                    "best_step":
                        metrics[
                            "best_step"
                        ],

                    "stop_step":
                        stop_step,

                    "add_percent":
                        percentage(
                            action_counts.get(
                                "ADD",
                                0,
                            ),
                            total_actions,
                        ),

                    "remove_percent":
                        percentage(
                            action_counts.get(
                                "REMOVE",
                                0,
                            ),
                            total_actions,
                        ),

                    "swap_percent":
                        percentage(
                            action_counts.get(
                                "SWAP",
                                0,
                            ),
                            total_actions,
                        ),

                    "stop_percent":
                        percentage(
                            action_counts.get(
                                "STOP",
                                0,
                            ),
                            total_actions,
                        ),

                    "total_reward":
                        episode_reward,

                    "previous_visit_best_scb":
                        previous_best,

                    "best_scb_delta_vs_previous":
                        best_delta_previous,

                    "previous_visit_final_scb":
                        previous_final,

                    "final_scb_delta_vs_previous":
                        final_delta_previous,

                    "steps_delta_vs_previous":
                        steps_delta_previous,
                }

                append_csv(
                    PROGRESSION_CSV,
                    PROGRESSION_FIELDS,
                    progression_row,
                )

                # --------------------------------------------------
                # Update graph's latest visit AFTER calculating
                # deltas, so comparison is always against the
                # previous visit.
                # --------------------------------------------------

                graph_visits[
                    str(gid)
                ] = {
                    "graph_visit":
                        graph_visit,

                    "episode":
                        episode_number,

                    "phase3_best_scb":
                        best_scb,

                    "phase3_final_scb":
                        final_scb,

                    "steps_taken":
                        len(
                            trainer.memory
                        ),
                }

                history.append(
                    row
                )

                # --------------------------------------------------
                # Console diagnostics.
                # --------------------------------------------------

                print()
                print(
                    f"EP {episode_number:5d}/"
                    f"{total_episodes} | "
                    f"G {gid:4d} | "
                    f"VISIT {graph_visit:3d}"
                )

                print(
                    f"  Graph: "
                    f"N={graph_nodes(graph):3d} "
                    f"E={graph_edges(graph):4d} "
                    f"S={graph_sessions(graph):3d} "
                    f"GA={fmt_scb(ga_scb)}"
                )

                print(
                    f"  Phase2: "
                    f"SCB={fmt_scb(phase2_result['scb'])} "
                    f"Cut={phase2_result['cut_size']} "
                    f"Sep={phase2_result['separation']} "
                    f"Steps={phase2_result['steps']}"
                )

                print(
                    f"  Phase3: "
                    f"Start={fmt_scb(initial_scb)} "
                    f"Best={fmt_scb(best_scb)} "
                    f"Final={fmt_scb(final_scb)}"
                )

                print(
                    f"  Improvement: "
                    f"{fmt_scb(best_improvement_abs)} "
                    f"({fmt_scb(best_improvement_pct)}%) "
                    f"| BestStep={metrics['best_step']} "
                    f"| Steps={len(trainer.memory)}"
                )

                if previous is not None:

                    print(
                        f"  Previous visit: "
                        f"Best={fmt_scb(previous_best)} "
                        f"Final={fmt_scb(previous_final)} "
                        f"Steps={previous_steps}"
                    )

                    print(
                        f"  Vs previous: "
                        f"Best Δ={fmt_scb(best_delta_previous)} "
                        f"Final Δ={fmt_scb(final_delta_previous)} "
                        f"Steps Δ={steps_delta_previous}"
                    )

                print(
                    f"  Reward={episode_reward:.6f} "
                    f"| STOP@{stop_step} "
                    f"| valid={stop_valid}"
                )

                print(
                    "  Actions: "
                    f"ADD {action_counts.get('ADD', 0):4d} "
                    f"({percentage(action_counts.get('ADD', 0), total_actions):5.1f}%) | "
                    f"REMOVE {action_counts.get('REMOVE', 0):4d} "
                    f"({percentage(action_counts.get('REMOVE', 0), total_actions):5.1f}%) | "
                    f"SWAP {action_counts.get('SWAP', 0):4d} "
                    f"({percentage(action_counts.get('SWAP', 0), total_actions):5.1f}%) | "
                    f"STOP {action_counts.get('STOP', 0):4d} "
                    f"({percentage(action_counts.get('STOP', 0), total_actions):5.1f}%)"
                )

                print(
                    f"  PPO: "
                    f"policy={ppo_stats['policy_loss']:.8f} "
                    f"value={ppo_stats['value_loss']:.8f} "
                    f"entropy={ppo_stats['entropy']:.6f} "
                    f"total={ppo_stats['total_loss']:.8f}"
                )

                current_episode += 1

                # --------------------------------------------------
                # Periodic checkpoint.
                # --------------------------------------------------

                if (
                    current_episode
                    % CHECKPOINT_INTERVAL
                    == 0
                ):

                    trainer.save_checkpoint(
                        PHASE3_LATEST,
                        current_episode,
                        history,
                        graph_visits,
                    )

    except KeyboardInterrupt:

        print()
        print("=" * 78)
        print("PHASE 3 INTERRUPTED")
        print("=" * 78)

        trainer.save_checkpoint(
            PHASE3_LATEST,
            current_episode,
            history,
            graph_visits,
        )

        print(
            f"Emergency checkpoint saved at "
            f"episode {current_episode}."
        )

    finally:

        if (
            current_episode
            >= total_episodes
        ):

            trainer.save_checkpoint(
                PHASE3_FINAL,
                current_episode,
                history,
                graph_visits,
            )

            trainer.save_checkpoint(
                PHASE3_LATEST,
                current_episode,
                history,
                graph_visits,
            )

            print()
            print("=" * 78)
            print("PHASE 3 TRAINING COMPLETE")
            print("=" * 78)
            print(
                f"Episodes completed : "
                f"{current_episode}"
            )
            print(
                f"Graph stats        : "
                f"{GRAPH_STATS_CSV}"
            )
            print(
                f"Progression        : "
                f"{PROGRESSION_CSV}"
            )
            print(
                f"Move analysis      : "
                f"{MOVE_ANALYSIS_CSV}"
            )
            print(
                f"Final checkpoint   : "
                f"{PHASE3_FINAL}"
            )

        log_handle.close()
