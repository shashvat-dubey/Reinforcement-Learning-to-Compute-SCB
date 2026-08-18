"""
Diagnostic PPO training runner for the SCB RL project.

Purpose
-------
Run the CURRENT RL formulation repeatedly over the labelled graph dataset
without changing the PPO/SCB objective. The runner adds diagnostics so that
we can answer:

    1. Is the RL agent improving with repeated exposure?
    2. How close is RL to the GA reference?
    3. Did RL ever find a good state but fail to stop there?
    4. Is reward improving because of step rewards or terminal rewards?
    5. Are PPO losses/entropy changing during training?

IMPORTANT
---------
This runner intentionally does NOT log actual edge identities/cut contents.
Only aggregate counts are stored.

Training schedule
-----------------
By default:
    800 graphs
    100 rounds
    = 80,000 episodes

Each round shuffles the selected graphs, so the agent does not train on the
same graph 100 times consecutively.

This file is designed around the existing PPOTrainer interface supplied in
trainer(3).py. It subclasses the trainer only to make episode collection
diagnostic-aware; the PPO update itself remains the existing implementation.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from SCB_RL.trainer import PPOTrainer
from SCB_RL.environment import SCBEnvironment


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_SIZE = 800
ROUNDS = 100

SEED = 42

OUTPUT_DIR = Path("diagnostics")
EPISODE_LOG = OUTPUT_DIR / "episode_results.csv"
GRAPH_LOG = OUTPUT_DIR / "graph_summary.csv"
ROUND_LOG = OUTPUT_DIR / "round_summary.csv"
TEXT_LOG = OUTPUT_DIR / "training_summary.txt"

CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_INTERVAL_ROUNDS = 10
MAX_ROLLING_CHECKPOINTS = 5

# Do not print every environment step.
PRINT_PROGRESS_EVERY = 100
PRINT_GRAPH_EVERY = 25

# Resume support.
LATEST_CHECKPOINT = CHECKPOINT_DIR / "latest.pt"

# ============================================================
# SAFE METRIC HELPERS
# ============================================================

def safe_float(x, default=np.nan):
    try:
        value = float(x)
        return value if np.isfinite(value) else default
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default


def get_field(obj, name, default=None):
    """Read either an object attribute or dictionary key."""
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def extract_ga_metrics(graph):
    """
    Extract only aggregate GA reference metrics.

    No GA cut-edge identities are returned or logged.
    """
    ga_cut = graph.get("ga_cut", None)
    ga_sep = graph.get("ga_sep", graph.get("ga_sessions_sep", None))
    ga_scb = graph.get("ga_scb", None)

    # Fall back to common labelled-dataset fields if necessary.
    if ga_cut is None and graph.get("ga_cut_edges") is not None:
        ga_cut = len(graph["ga_cut_edges"])

    if ga_sep is None and graph.get("ga_sessions_sep") is not None:
        ga_sep = graph["ga_sessions_sep"]

    # Keep the dataset's labelled GA SCB if available.
    if ga_scb is None:
        if ga_cut is not None and ga_sep not in (None, 0):
            ga_scb = float(ga_cut) / float(ga_sep)

    return {
        "ga_cut": safe_float(ga_cut),
        "ga_sep": safe_float(ga_sep),
        "ga_scb": safe_float(ga_scb),
    }


def compute_scb(cut, sep):
    """
    Current project's reported SCB convention:
        cut / separated

    This is intentionally kept consistent with the current GA/RL logging
    convention. We are NOT changing the objective in this diagnostic run.
    """
    cut = safe_float(cut)
    sep = safe_float(sep)

    if not np.isfinite(cut) or not np.isfinite(sep):
        return np.nan

    if cut <= 0 or sep <= 0:
        return np.inf if cut > 0 else np.nan

    return cut / sep


def gap_percent(rl_scb, ga_scb):
    if not np.isfinite(rl_scb) or not np.isfinite(ga_scb) or ga_scb == 0:
        return np.nan
    return 100.0 * (rl_scb - ga_scb) / ga_scb


# ============================================================
# DIAGNOSTIC TRAINER
# ============================================================

class DiagnosticPPOTrainer(PPOTrainer):
    """
    Existing PPOTrainer + diagnostic episode collection.

    PPO update logic is inherited unchanged.
    """

    def collect_episode_diagnostic(self):
        """
        Run one episode while recording aggregate diagnostics.

        Actual cut edge identities are never returned.
        """

        self.memory.clear()

        state = self.env.reset()

        initial_cut = safe_int(get_field(state, "cut_size", 0))
        initial_sep = safe_int(get_field(state, "separated_count", 0))

        total_reward = 0.0
        step_reward_sum = 0.0
        terminal_reward = 0.0

        best_scb = np.nan
        best_cut = initial_cut
        best_sep = initial_sep
        best_step = 0

        done = False
        final_info = {}

        while not done:

            encoding = self.encoder(state)
            embedding = encoding["graph_embedding"]

            if not torch.isfinite(embedding).all():
                raise RuntimeError("NON-FINITE GRAPH EMBEDDING")

            action, log_prob, entropy = self.policy.sample_action(
                encoding,
                state
            )

            if not torch.isfinite(log_prob):
                raise RuntimeError("NON-FINITE LOG_PROB")

            if not torch.isfinite(entropy):
                raise RuntimeError("NON-FINITE ENTROPY")

            value = self.critic(
                encoding["graph_embedding"]
            ).squeeze()

            if not torch.isfinite(value):
                raise RuntimeError("NON-FINITE VALUE")

            next_state, reward, done, info = self.env.step(action)

            reward = safe_float(reward, 0.0)
            info = info if isinstance(info, dict) else {}

            next_cut = safe_int(get_field(next_state, "cut_size", 0))
            next_sep = safe_int(get_field(next_state, "separated_count", 0))
            next_scb = compute_scb(next_cut, next_sep)

            # Track best state encountered during the episode.
            if np.isfinite(next_scb):
                if not np.isfinite(best_scb) or next_scb < best_scb:
                    best_scb = next_scb
                    best_cut = next_cut
                    best_sep = next_sep
                    best_step = safe_int(get_field(next_state, "step", 0))

            if done:
                terminal_reward = reward
            else:
                step_reward_sum += reward

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
            final_info = info

        # Final state.
        final_cut = safe_int(get_field(state, "cut_size", 0))
        final_sep = safe_int(get_field(state, "separated_count", 0))
        final_scb = compute_scb(final_cut, final_sep)

        # Some environments maintain best_state explicitly.
        env_best = getattr(self.env, "best_state", None)
        if env_best is not None:
            env_best_cut = safe_int(get_field(env_best, "cut_size", best_cut))
            env_best_sep = safe_int(
                get_field(env_best, "separated_count", best_sep)
            )
            env_best_scb = compute_scb(env_best_cut, env_best_sep)

            if np.isfinite(env_best_scb) and (
                not np.isfinite(best_scb) or env_best_scb < best_scb
            ):
                best_scb = env_best_scb
                best_cut = env_best_cut
                best_sep = env_best_sep
                best_step = safe_int(get_field(env_best, "step", best_step))

        return {
            "total_reward": total_reward,
            "step_reward_sum": step_reward_sum,
            "terminal_reward": terminal_reward,
            "episode_length": len(self.memory),

            "final_cut": final_cut,
            "final_sep": final_sep,
            "final_scb": final_scb,

            "best_cut": best_cut,
            "best_sep": best_sep,
            "best_scb": best_scb,
            "best_step": best_step,

            "final_info": final_info,
        }


# ============================================================
# LOGGING
# ============================================================

EPISODE_FIELDS = [
    "round",
    "episode",
    "graph_id",
    "nodes",
    "edges",
    "sessions",

    "ga_cut",
    "ga_sep",
    "ga_scb",

    "rl_final_cut",
    "rl_final_sep",
    "rl_final_scb",

    "rl_best_cut",
    "rl_best_sep",
    "rl_best_scb",
    "best_step",

    "reward_total",
    "step_reward_sum",
    "terminal_reward",
    "episode_length",

    "final_gap_pct",
    "best_gap_pct",

    "final_matches_ga",
    "best_matches_ga",
    "best_beats_ga",

    "policy_loss",
    "value_loss",
    "entropy",
    "total_loss",

    "runtime_sec",
]


GRAPH_FIELDS = [
    "graph_id",
    "nodes",
    "edges",
    "sessions",
    "ga_cut",
    "ga_sep",
    "ga_scb",

    "episodes_seen",
    "mean_reward",
    "max_reward",

    "best_rl_scb",
    "best_rl_cut",
    "best_rl_sep",
    "best_rl_round",

    "final_match_count",
    "best_match_count",
    "best_better_count",

    "first_best_match_round",
    "first_positive_reward_round",

    "mean_episode_length",
]


ROUND_FIELDS = [
    "round",
    "graphs",
    "mean_reward",
    "max_reward",

    "mean_ga_scb",
    "mean_final_rl_scb",
    "mean_best_rl_scb",

    "mean_final_gap_pct",
    "mean_best_gap_pct",

    "final_matches",
    "best_matches",
    "best_better_than_ga",

    "final_match_rate_pct",
    "best_match_rate_pct",

    "mean_final_cut",
    "mean_best_cut",
    "mean_ga_cut",

    "mean_final_sep",
    "mean_best_sep",
    "mean_ga_sep",

    "mean_episode_length",
    "mean_policy_loss",
    "mean_value_loss",
    "mean_entropy",
    "mean_total_loss",

    "round_runtime_sec",
]


def append_csv(path, fields, row):
    path.parent.mkdir(parents=True, exist_ok=True)

    new_file = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fields})


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_diagnostic(dataset, trainer):
    """
    Execute the full repeated-graph diagnostic experiment.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)

    # Select a fixed subset once. The same graphs are revisited every round.
    n = min(DATASET_SIZE, len(dataset))
    indices = list(range(n))

    graph_records = {}

    start_time = time.time()

    for round_idx in range(1, ROUNDS + 1):

        round_start = time.time()

        order = indices.copy()
        rng.shuffle(order)

        round_rows = []

        for position, dataset_index in enumerate(order, start=1):

            graph = dataset[dataset_index]
            graph_id = graph.get("graph_id", dataset_index)

            trainer.env = SCBEnvironment(graph)

            episode_start = time.time()

            diagnostics = trainer.collect_episode_diagnostic()

            # PPO update is inherited from the supplied trainer.
            stats = trainer.update()

            runtime = time.time() - episode_start

            ga = extract_ga_metrics(graph)

            final_gap = gap_percent(
                diagnostics["final_scb"],
                ga["ga_scb"]
            )

            best_gap = gap_percent(
                diagnostics["best_scb"],
                ga["ga_scb"]
            )

            final_match = (
                np.isfinite(diagnostics["final_scb"])
                and np.isfinite(ga["ga_scb"])
                and abs(diagnostics["final_scb"] - ga["ga_scb"]) < 1e-9
            )

            best_match = (
                np.isfinite(diagnostics["best_scb"])
                and np.isfinite(ga["ga_scb"])
                and abs(diagnostics["best_scb"] - ga["ga_scb"]) < 1e-9
            )

            best_better = (
                np.isfinite(diagnostics["best_scb"])
                and np.isfinite(ga["ga_scb"])
                and diagnostics["best_scb"] < ga["ga_scb"] - 1e-9
            )

            row = {
                "round": round_idx,
                "episode": (round_idx - 1) * n + position,
                "graph_id": graph_id,
                "nodes": len(graph["nodes"]),
                "edges": len(graph["edges"]),
                "sessions": len(graph["sessions"]),

                "ga_cut": ga["ga_cut"],
                "ga_sep": ga["ga_sep"],
                "ga_scb": ga["ga_scb"],

                "rl_final_cut": diagnostics["final_cut"],
                "rl_final_sep": diagnostics["final_sep"],
                "rl_final_scb": diagnostics["final_scb"],

                "rl_best_cut": diagnostics["best_cut"],
                "rl_best_sep": diagnostics["best_sep"],
                "rl_best_scb": diagnostics["best_scb"],
                "best_step": diagnostics["best_step"],

                "reward_total": diagnostics["total_reward"],
                "step_reward_sum": diagnostics["step_reward_sum"],
                "terminal_reward": diagnostics["terminal_reward"],
                "episode_length": diagnostics["episode_length"],

                "final_gap_pct": final_gap,
                "best_gap_pct": best_gap,

                "final_matches_ga": int(final_match),
                "best_matches_ga": int(best_match),
                "best_beats_ga": int(best_better),

                "policy_loss": stats["policy_loss"],
                "value_loss": stats["value_loss"],
                "entropy": stats["entropy"],
                "total_loss": stats["total_loss"],

                "runtime_sec": runtime,
            }

            append_csv(EPISODE_LOG, EPISODE_FIELDS, row)
            round_rows.append(row)

            # Update per-graph aggregate.
            record = graph_records.setdefault(
                graph_id,
                {
                    "graph_id": graph_id,
                    "nodes": len(graph["nodes"]),
                    "edges": len(graph["edges"]),
                    "sessions": len(graph["sessions"]),
                    "ga_cut": ga["ga_cut"],
                    "ga_sep": ga["ga_sep"],
                    "ga_scb": ga["ga_scb"],
                    "rows": [],
                }
            )
            record["rows"].append(row)

            if position % PRINT_GRAPH_EVERY == 0 or position == n:
                print(
                    f"[Round {round_idx:03d}/{ROUNDS}] "
                    f"Graph {position:04d}/{n} | "
                    f"Reward={row['reward_total']:+.3f} | "
                    f"BestSCB={row['rl_best_scb']:.4f} | "
                    f"GASCB={row['ga_scb']:.4f}",
                    flush=True
                )

        # ----------------------------------------------------
        # Round aggregate
        # ----------------------------------------------------

        def avg(field):
            vals = [
                safe_float(r[field])
                for r in round_rows
                if np.isfinite(safe_float(r[field]))
            ]
            return mean(vals) if vals else np.nan

        final_matches = sum(r["final_matches_ga"] for r in round_rows)
        best_matches = sum(r["best_matches_ga"] for r in round_rows)
        better = sum(r["best_beats_ga"] for r in round_rows)

        round_row = {
            "round": round_idx,
            "graphs": len(round_rows),

            "mean_reward": avg("reward_total"),
            "max_reward": max(
                (safe_float(r["reward_total"]) for r in round_rows),
                default=np.nan
            ),

            "mean_ga_scb": avg("ga_scb"),
            "mean_final_rl_scb": avg("rl_final_scb"),
            "mean_best_rl_scb": avg("rl_best_scb"),

            "mean_final_gap_pct": avg("final_gap_pct"),
            "mean_best_gap_pct": avg("best_gap_pct"),

            "final_matches": final_matches,
            "best_matches": best_matches,
            "best_better_than_ga": better,

            "final_match_rate_pct": 100.0 * final_matches / len(round_rows),
            "best_match_rate_pct": 100.0 * best_matches / len(round_rows),

            "mean_final_cut": avg("rl_final_cut"),
            "mean_best_cut": avg("rl_best_cut"),
            "mean_ga_cut": avg("ga_cut"),

            "mean_final_sep": avg("rl_final_sep"),
            "mean_best_sep": avg("rl_best_sep"),
            "mean_ga_sep": avg("ga_sep"),

            "mean_episode_length": avg("episode_length"),
            "mean_policy_loss": avg("policy_loss"),
            "mean_value_loss": avg("value_loss"),
            "mean_entropy": avg("entropy"),
            "mean_total_loss": avg("total_loss"),

            "round_runtime_sec": time.time() - round_start,
        }

        append_csv(ROUND_LOG, ROUND_FIELDS, round_row)

        print()
        print("=" * 78)
        print(f"ROUND {round_idx}/{ROUNDS} COMPLETE")
        print("=" * 78)
        print(
            f"Graphs             : {len(round_rows)}"
        )
        print(
            f"Mean reward        : {round_row['mean_reward']:+.4f}"
        )
        print(
            f"Mean GA SCB        : {round_row['mean_ga_scb']:.4f}"
        )
        print(
            f"Mean final RL SCB  : {round_row['mean_final_rl_scb']:.4f}"
        )
        print(
            f"Mean best RL SCB   : {round_row['mean_best_rl_scb']:.4f}"
        )
        print(
            f"Final GA matches   : "
            f"{final_matches}/{len(round_rows)} "
            f"({round_row['final_match_rate_pct']:.2f}%)"
        )
        print(
            f"Best GA matches    : "
            f"{best_matches}/{len(round_rows)} "
            f"({round_row['best_match_rate_pct']:.2f}%)"
        )
        print(
            f"Best beats GA      : {better}"
        )
        print(
            f"Mean episode len   : "
            f"{round_row['mean_episode_length']:.2f}"
        )
        print(
            f"Round runtime      : "
            f"{round_row['round_runtime_sec']:.1f}s"
        )
        print("=" * 78)

        # ----------------------------------------------------
        # Periodic checkpoint
        # ----------------------------------------------------

        if round_idx % CHECKPOINT_INTERVAL_ROUNDS == 0:

            checkpoint_path = (
                CHECKPOINT_DIR /
                f"round_{round_idx:03d}.pt"
            )

            trainer.save_checkpoint(
                path=str(checkpoint_path),
                episode=round_idx * n,
                history=[]
            )

            trainer.save_checkpoint(
                path=str(LATEST_CHECKPOINT),
                episode=round_idx * n,
                history=[]
            )

    # ========================================================
    # GRAPH SUMMARY
    # ========================================================

    for graph_id, record in graph_records.items():

        rows = record["rows"]

        best_row = min(
            rows,
            key=lambda r: (
                safe_float(r["rl_best_scb"], np.inf)
            )
        )

        match_rows = [
            r for r in rows if r["best_matches_ga"]
        ]

        positive_rows = [
            r for r in rows if r["reward_total"] > 0
        ]

        def avg(field):
            vals = [
                safe_float(r[field])
                for r in rows
                if np.isfinite(safe_float(r[field]))
            ]
            return mean(vals) if vals else np.nan

        graph_row = {
            "graph_id": graph_id,
            "nodes": record["nodes"],
            "edges": record["edges"],
            "sessions": record["sessions"],
            "ga_cut": record["ga_cut"],
            "ga_sep": record["ga_sep"],
            "ga_scb": record["ga_scb"],

            "episodes_seen": len(rows),
            "mean_reward": avg("reward_total"),
            "max_reward": max(
                (safe_float(r["reward_total"]) for r in rows),
                default=np.nan
            ),

            "best_rl_scb": best_row["rl_best_scb"],
            "best_rl_cut": best_row["rl_best_cut"],
            "best_rl_sep": best_row["rl_best_sep"],
            "best_rl_round": best_row["round"],

            "final_match_count": sum(
                r["final_matches_ga"] for r in rows
            ),
            "best_match_count": len(match_rows),
            "best_better_count": sum(
                r["best_beats_ga"] for r in rows
            ),

            "first_best_match_round": (
                min(r["round"] for r in match_rows)
                if match_rows else ""
            ),

            "first_positive_reward_round": (
                min(r["round"] for r in positive_rows)
                if positive_rows else ""
            ),

            "mean_episode_length": avg("episode_length"),
        }

        append_csv(
            GRAPH_LOG,
            GRAPH_FIELDS,
            graph_row
        )

    total_runtime = time.time() - start_time

    with TEXT_LOG.open("a", encoding="utf-8") as f:
        f.write("\n")
        f.write("=" * 78 + "\n")
        f.write("DIAGNOSTIC RUN COMPLETE\n")
        f.write("=" * 78 + "\n")
        f.write(f"Graphs: {n}\n")
        f.write(f"Rounds: {ROUNDS}\n")
        f.write(f"Episodes: {n * ROUNDS}\n")
        f.write(f"Runtime seconds: {total_runtime:.2f}\n")
        f.write("=" * 78 + "\n")

    print()
    print("DIAGNOSTIC RUN COMPLETE")
    print(f"Graphs   : {n}")
    print(f"Rounds   : {ROUNDS}")
    print(f"Episodes : {n * ROUNDS}")
    print(f"Runtime  : {total_runtime / 3600:.2f} hours")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    from Data.loader import GraphDataset
    from SCB_RL.gnn import SCBGraphEncoder
    from SCB_RL.policy import HierarchicalSCBPolicy
    from SCB_RL.critic import SCBCritic

    print("=" * 78)
    print("SCB RL DIAGNOSTIC TRAINING")
    print("=" * 78)
    print(
        f"Graphs: {DATASET_SIZE} | "
        f"Rounds: {ROUNDS} | "
        f"Episodes: {DATASET_SIZE * ROUNDS}"
    )
    print("Actual cut identities will NOT be logged.")
    print("=" * 78)

    # Reproducible dataset ordering/shuffling.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    dataset = GraphDataset()

    encoder = SCBGraphEncoder()
    policy = HierarchicalSCBPolicy()
    critic = SCBCritic()

    trainer = DiagnosticPPOTrainer(
        environment=None,
        encoder=encoder,
        policy=policy,
        critic=critic,
    )

    run_diagnostic(
        dataset=dataset,
        trainer=trainer,
    )
