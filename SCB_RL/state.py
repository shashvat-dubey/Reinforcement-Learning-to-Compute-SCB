"""
state.py

Defines the SCBState object used by the SCB Reinforcement Learning
environment.

A state is simply a snapshot of the current optimization process.

It wraps the evaluation dictionary returned by the GA and exposes
convenient properties for the RL environment.

Author: SCB-RL
"""

from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

from GA_SCB.graph import SCBProblem


@dataclass
class SCBState:
    """
    Represents one state in the SCB optimization process.
    """

    problem: SCBProblem

    evaluation: dict

    step: int = 0

    # ==========================================================
    # Convenience Properties
    # ==========================================================

    @property
    def cut(self):
        """
        Current cut edge set.
        """
        return {
        self.problem.edge_to_idx[e]
        for e in self.evaluation["cut_edges"]
    }

    @property
    def cut_edges(self):
        return set(self.evaluation["cut_edges"])

    @property
    def cut_size(self):
        """
        Number of cut edges.
        """
        return self.evaluation["cut"]

    @property
    def separated_count(self):
        """
        Number of separated sessions.
        """
        return self.evaluation["sep"]

    @property
    def scb(self):
        fitness2 = self.evaluation["fitness2"]

        if fitness2 <= 0:
            return float("inf")

        return 1.0 / fitness2

    @property
    def full_cut_fitness(self):
        """
        Full-cut fitness used by the GA.
        """
        return self.evaluation["fitness1"]

    @property
    def components(self):
        """
        Connected component labels.
        """
        return self.evaluation["component"]

    @property
    def separated_sessions(self):
        """
        Separated source-destination session pairs.
        """
        return self.evaluation["sessions_sep"]

    # ==========================================================
    # Utilities
    # ==========================================================

    def copy(self):
        """
        Returns a deep copy of the current state.
        """

        return SCBState(
            problem=self.problem,
            evaluation=deepcopy(self.evaluation),
            step=self.step
        )

    # ==========================================================
    # Representation
    # ==========================================================

    def __repr__(self):

        return (
            "SCBState(\n"
            f"    step={self.step},\n"
            f"    cut_size={self.cut_size},\n"
            f"    separated={self.separated_count},\n"
            f"    scb={self.scb:.6f}\n"
            ")"
        )