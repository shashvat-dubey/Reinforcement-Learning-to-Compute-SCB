"""
SCB-RL PHASE 3 ENVIRONMENT
===========================

Phase 3:
    Phase-2 final cut -> optimize SCB -> STOP

The environment uses the project's real:
    - SCBProblem
    - evaluate()
    - SCBState
    - Action / ActionType

There is NO GA/teacher reward signal here.

Action semantics:
    ADD(edge)
    REMOVE(edge)
    SWAP(remove_edge, add_edge)
    STOP
"""

from typing import Optional, Sequence

import numpy as np
import torch

from SCB_RL.state import SCBState
from SCB_RL.actions import Action, ActionType
from GA_SCB.graph import SCBProblem, evaluate
from SCB_RL.phase3_reward import compute_reward_components


ACTION_NAMES = {
    ActionType.ADD: "ADD",
    ActionType.REMOVE: "REMOVE",
    ActionType.SWAP: "SWAP",
    ActionType.STOP: "STOP",
}


class Phase3Environment:
    """
    Phase-3 optimization environment.

    A Phase-2 episode supplies its FINAL CUT to reset():

        env.reset(initial_cut=phase2_final_cut)

    Phase 3 then edits that cut until the policy chooses STOP.

    There is intentionally no hard maximum-step termination.
    """

    def __init__(
        self,
        graph_data,
        device="cpu",
        reward_module=None,
    ):
        self.graph_data = graph_data
        self.device = device
        self.reward_module = reward_module

        # Build the actual project problem object.
        self.problem = SCBProblem(
            graph_data["nodes"],
            graph_data["edges"],
            graph_data["sessions"],
        )

        # Evaluation-only information.
        self.ga_scb = graph_data.get("ga_scb")
        self.ga_cut = graph_data.get("ga_cut")
        self.ga_sep = graph_data.get("ga_sep")
        self.ga_cut_edges = graph_data.get("ga_cut_edges")
        self.ga_components = graph_data.get("ga_components")

        self.state: Optional[SCBState] = None
        self._initial_state: Optional[SCBState] = None

        self.best_state: Optional[SCBState] = None
        self.best_scb = float("inf")
        self.best_step = None

        self.step_count = 0
        self.done = False

        self.action_counts = {
            ActionType.ADD: 0,
            ActionType.REMOVE: 0,
            ActionType.SWAP: 0,
            ActionType.STOP: 0,
        }

        self._last_reward_components = {
            "scb": 0.0,
            "best_bonus": 0.0,
            "stop": 0.0,
            "regression": 0.0,
            "time": 0.0,
            "total": 0.0,
        }

    # ==========================================================
    # RESET
    # ==========================================================

    def reset(
        self,
        initial_cut: Optional[Sequence] = None,
        cut: Optional[Sequence] = None,
    ):
        """
        Start Phase 3 from the final cut produced by Phase 2.

        initial_cut may contain:
            - edge indices
            - edge tuples

        The cut is converted to edge indices internally because
        SCBState.cut uses edge indices.
        """

        if initial_cut is None:
            initial_cut = cut

        if initial_cut is None:
            raise ValueError(
                "Phase 3 requires the final Phase-2 cut."
            )

        self.step_count = 0
        self.done = False

        self.action_counts = {
            ActionType.ADD: 0,
            ActionType.REMOVE: 0,
            ActionType.SWAP: 0,
            ActionType.STOP: 0,
        }

        self._last_reward_components = {
            "scb": 0.0,
            "best_bonus": 0.0,
            "stop": 0.0,
            "regression": 0.0,
            "time": 0.0,
            "total": 0.0,
        }

        normalized_cut = self._normalize_cut(initial_cut)

        self.state = self._make_state(normalized_cut)

        # Freeze the Phase-2 starting point.
        self._initial_state = self.state.copy()

        # Phase-2 solution is already a valid candidate.
        if self._is_valid_state(self.state):
            self.best_state = self.state.copy()
            self.best_scb = float(self.state.scb)
            self.best_step = 0
        else:
            self.best_state = None
            self.best_scb = float("inf")
            self.best_step = None

        return self._get_observation()

    # ==========================================================
    # STEP
    # ==========================================================

    def step(self, action: Action):
        """
        Apply one real Action selected by the Phase-3 policy.

        Returns:
            observation, reward, done, info
        """

        if self.state is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        if self.done:
            raise RuntimeError(
                "Episode is already terminated."
            )

        if not isinstance(action, Action):
            raise TypeError(
                "Phase 3 expects an actions.Action object."
            )

        action.validate()

        self.step_count += 1
        self.state.step = self.step_count

        old_state = self.state.copy()

        # ------------------------------------------------------
        # STOP
        # ------------------------------------------------------

        if action.is_stop():

            self.action_counts[ActionType.STOP] += 1

            new_state = self.state.copy()
            new_state.step = self.step_count

            components = compute_reward_components(
                old_state=old_state,
                new_state=new_state,
                action=action,
                invalid=False,
                best_scb=self.best_scb,
                is_stop=True,
                new_best=False,
                initial_scb=self._initial_scb(),
            )

            self._last_reward_components = components
            self.done = True

            return (
                self._get_observation(),
                float(components["total"]),
                True,
                self._make_info(
                    action,
                    invalid=False,
                    components=components,
                    stopped=True,
                    new_best=False,
                ),
            )

        # ------------------------------------------------------
        # Validate graph edit
        # ------------------------------------------------------

        if not self._is_valid_action(action):

            self.action_counts[action.action_type] += 1

            new_state = self.state.copy()
            new_state.step = self.step_count
            self.state = new_state

            components = compute_reward_components(
                old_state=old_state,
                new_state=new_state,
                action=action,
                invalid=True,
                best_scb=self.best_scb,
                is_stop=False,
                new_best=False,
                initial_scb=self._initial_scb(),
            )

            self._last_reward_components = components

            return (
                self._get_observation(),
                float(components["total"]),
                False,
                self._make_info(
                    action,
                    invalid=True,
                    components=components,
                    stopped=False,
                    new_best=False,
                ),
            )

        # ------------------------------------------------------
        # Apply real ADD / REMOVE / SWAP
        # ------------------------------------------------------

        self.action_counts[action.action_type] += 1

        new_cut = set(self.state.cut)

        if action.is_add():
            new_cut.add(action.edge)

        elif action.is_remove():
            new_cut.remove(action.edge)

        elif action.is_swap():
            new_cut.remove(action.remove_edge)
            new_cut.add(action.add_edge)

        new_state = self._make_state(new_cut)
        new_state.step = self.step_count

        candidate_scb = self._finite_or_inf(
            new_state.scb
        )

        new_best = (
            self._is_valid_state(new_state)
            and candidate_scb < self.best_scb
        )

        # Reward is computed against the previous best.
        components = compute_reward_components(
            old_state=old_state,
            new_state=new_state,
            action=action,
            invalid=False,
            best_scb=self.best_scb,
            is_stop=False,
            new_best=new_best,
            initial_scb=self._initial_scb(),
        )

        self._last_reward_components = components
        self.state = new_state

        if new_best:
            self.best_scb = candidate_scb
            self.best_state = new_state.copy()
            self.best_step = self.step_count

        return (
            self._get_observation(),
            float(components["total"]),
            False,
            self._make_info(
                action,
                invalid=False,
                components=components,
                stopped=False,
                new_best=new_best,
            ),
        )

    # ==========================================================
    # REAL STATE / EVALUATION
    # ==========================================================

    def _make_state(self, cut):
        """
        Evaluate the proposed cut using the project's real SCB
        evaluation pipeline.
        """

        cut_indices = self._normalize_cut(cut)

        cut_edges = {
            self.problem.edges[int(i)]
            for i in cut_indices
        }

        evaluation = evaluate(
            self.problem,
            cut_edges,
        )

        return SCBState(
            problem=self.problem,
            evaluation=evaluation,
            step=self.step_count,
        )

    # ==========================================================
    # ACTION VALIDATION
    # ==========================================================

    def _is_valid_action(self, action):
        current_cut = set(self.state.cut)

        if action.is_add():
            return (
                action.edge is not None
                and 0 <= int(action.edge) < len(self.problem.edges)
                and int(action.edge) not in current_cut
            )

        if action.is_remove():
            return (
                action.edge is not None
                and int(action.edge) in current_cut
            )

        if action.is_swap():
            return (
                action.remove_edge is not None
                and action.add_edge is not None
                and int(action.remove_edge) in current_cut
                and int(action.add_edge) not in current_cut
                and 0 <= int(action.add_edge) < len(self.problem.edges)
                and int(action.remove_edge) != int(action.add_edge)
            )

        return False

    # ==========================================================
    # OBSERVATION
    # ==========================================================

    def _get_observation(self):
        """
        Return the current SCBState.

        The Phase-3 trainer/policy can feed this through the same
        encoder pipeline used elsewhere in the project.
        """

        return self.state

    # ==========================================================
    # INFO / LOGGING
    # ==========================================================

    def _make_info(
        self,
        action,
        invalid,
        components,
        stopped,
        new_best,
    ):
        return {
            "action": ACTION_NAMES.get(
                action.action_type,
                str(action.action_type),
            ),
            "action_type": action.action_type,

            "invalid": bool(invalid),
            "stopped": bool(stopped),
            "new_best": bool(new_best),

            "step": self.step_count,

            "cut_size": int(self.state.cut_size),
            "separated_sessions": int(
                self.state.separated_count
            ),
            "scb": float(
                self._finite_or_inf(self.state.scb)
            ),

            "initial_scb": float(
                self._finite_or_inf(self.initial_scb)
            ),
            "best_scb": float(
                self._finite_or_inf(self.best_scb)
            ),
            "best_step": self.best_step,

            "ga_scb": self.ga_scb,

            "reward_components": dict(
                components
            ),

            "action_counts": {
                ACTION_NAMES[k]: v
                for k, v in self.action_counts.items()
            },
        }

    # ==========================================================
    # CUT HELPERS
    # ==========================================================

    def _normalize_cut(self, cut):
        """
        Convert a cut into edge indices.

        Supports:
            [0, 4, 7]
            [(u, v), (x, y)]
            numpy arrays
            torch tensors
            sets / tuples
        """

        if isinstance(cut, torch.Tensor):
            cut = cut.detach().cpu().tolist()

        elif isinstance(cut, np.ndarray):
            cut = cut.tolist()

        elif isinstance(cut, set):
            cut = list(cut)

        elif isinstance(cut, tuple):
            cut = list(cut)

        elif not isinstance(cut, list):
            cut = [cut]

        result = []

        for item in cut:

            # Edge index.
            if isinstance(item, (int, np.integer)):
                idx = int(item)

                if not 0 <= idx < len(self.problem.edges):
                    raise ValueError(
                        f"Invalid edge index {idx}."
                    )

                result.append(idx)
                continue

            # Edge tuple.
            if isinstance(item, (tuple, list)):
                if len(item) != 2:
                    raise ValueError(
                        f"Invalid edge tuple: {item}"
                    )

                edge = tuple(item)

                if edge in self.problem.edge_to_idx:
                    result.append(
                        self.problem.edge_to_idx[edge]
                    )
                    continue

                reverse = (edge[1], edge[0])

                if reverse in self.problem.edge_to_idx:
                    result.append(
                        self.problem.edge_to_idx[reverse]
                    )
                    continue

                raise ValueError(
                    f"Edge {edge} does not exist."
                )

            raise TypeError(
                f"Unsupported cut element: {item!r}"
            )

        return list(dict.fromkeys(result))

    # ==========================================================
    # VALIDITY / NUMERICAL HELPERS
    # ==========================================================

    def _is_valid_state(self, state):
        return (
            state.cut_size > 0
            and state.separated_count > 0
            and self._finite_or_none(state.scb) is not None
        )

    def _initial_scb(self):
        if self._initial_state is None:
            return None

        return self._finite_or_none(
            self._initial_state.scb
        )

    @staticmethod
    def _finite_or_inf(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return float("inf")

        if not np.isfinite(value):
            return float("inf")

        return value

    @staticmethod
    def _finite_or_none(value):
        value = Phase3Environment._finite_or_inf(value)

        if np.isfinite(value):
            return value

        return None

    # ==========================================================
    # CONVENIENCE PROPERTIES
    # ==========================================================

    @property
    def current_scb(self):
        if self.state is None:
            return float("inf")
        return self.state.scb

    @property
    def current_cut(self):
        if self.state is None:
            return set()
        return set(self.state.cut)

    @property
    def current_cut_edges(self):
        if self.state is None:
            return set()
        return set(self.state.cut_edges)

    @property
    def initial_scb(self):
        return self._initial_scb()

    @property
    def best_cut(self):
        if self.best_state is None:
            return set()
        return set(self.best_state.cut)

    @property
    def best_cut_edges(self):
        if self.best_state is None:
            return set()
        return set(self.best_state.cut_edges)
