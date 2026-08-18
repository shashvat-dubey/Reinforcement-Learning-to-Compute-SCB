"""
environment.py

Gym-style environment for the Sparsest Cut Bound Reinforcement
Learning project.

The environment reuses the GA evaluation function while exposing
a standard RL interface.

Author: SCB-RL
"""
from GA_SCB.graph import SCBProblem
from GA_SCB.graph import evaluate
from .reward import compute_reward
from .actions import ActionType
from .state import SCBState


class SCBEnvironment:

    def __init__(self, graph, max_steps=100):   

        # -------------------------------------------------
        # Store original dataset entry
        # -------------------------------------------------

        self.graph = graph

        # -------------------------------------------------
        # Build GA problem
        # -------------------------------------------------

        self.problem = SCBProblem(

            graph["nodes"],

            graph["edges"],

            graph["sessions"]

        )

        # -------------------------------------------------
        # Teacher Solution
        # -------------------------------------------------

        self.teacher_scb = graph["ga_scb"]

        self.teacher_cut = graph["ga_cut"]

        self.teacher_sep = graph["ga_sep"]

        self.teacher_cut_edges = graph["ga_cut_edges"]

        self.teacher_components = graph["ga_components"]

        # -------------------------------------------------
        # Environment State
        # -------------------------------------------------

        self.max_steps = max_steps

        self.current_state = None

        self.best_state = None

        self.episode_step = 0

        self.done = False

        # -------------------------------------------------
        # Action Dispatch
        # -------------------------------------------------

        self._ACTION_MAP = {

            ActionType.ADD: self._apply_add,

            ActionType.REMOVE: self._apply_remove,

            ActionType.SWAP: self._apply_swap,

            ActionType.STOP: self._apply_stop,

        }
    # ==========================================================
    # Private Helpers
    # ==========================================================

    def _build_state(self, cut):

        # --------------------------------------------
        # Convert RL edge indices -> GA edge tuples
        # --------------------------------------------

        ga_cut = {
            self.problem.edges[i]
            for i in cut
        }

        evaluation = evaluate(
            self.problem,
            ga_cut
        )

        return SCBState(
            problem=self.problem,
            evaluation=evaluation,
            step=self.episode_step
        )

    def _update_best_state(self):

        if self.current_state.scb < self.best_state.scb:

            self.best_state = self.current_state.copy()

    def _get_info(self):

        return {

            "teacher_scb": self.teacher_scb,

            "current_scb": self.current_state.scb,

            "current_cut": self.current_state.cut_size,

            "current_sep": self.current_state.separated_count,

            "best_scb": self.best_state.scb,

            "cut_size": self.current_state.cut_size,

            "sessions_separated": self.current_state.separated_count,

            "step": self.episode_step
        }

    # ==========================================================
    # Reset
    # ==========================================================

    def reset(self):

        self.done = False

        self.episode_step = 0

        empty_cut = set()

        self.current_state = self._build_state(empty_cut)

        self.best_state = self.current_state.copy()

        return self.current_state.copy()

    # ==========================================================
    # Action Validation
    # ==========================================================

    def _is_valid_action(self, action):

        action.validate()

        cut = self.current_state.cut

        if action.is_add():

            return action.edge not in cut

        if action.is_remove():

            return action.edge in cut

        if action.is_swap():

            if action.remove_edge not in cut:
                return False

            if action.add_edge in cut:
                return False

            return True

        return True

    # ==========================================================
    # Action Application
    # ==========================================================

    def _apply_add(self, action):

        cut = self.current_state.cut.copy()

        cut.add(action.edge)

        return cut

    def _apply_remove(self, action):

        cut = self.current_state.cut.copy()

        cut.remove(action.edge)

        return cut

    def _apply_swap(self, action):

        cut = self.current_state.cut.copy()

        cut.remove(action.remove_edge)

        cut.add(action.add_edge)

        return cut

    def _apply_stop(self, action):

        self.done = True

        return self.current_state.cut.copy()

    # ==========================================================
    # Step
    # ==========================================================

    def step(self, action):

        if self.done:

            raise RuntimeError(
                "Episode already finished. Call reset()."
            )

        # ------------------------------------------------------
        # Invalid Action
        # ------------------------------------------------------

        if not self._is_valid_action(action):

            self.episode_step += 1

            if self.episode_step >= self.max_steps:

                self.done = True

            reward = compute_reward(
                self.current_state,
                self.current_state,
                action,
                invalid=True
            )

            # --------------------------------------------------
            # If invalid action also ended the episode,
            # apply terminal teacher reward.
            # --------------------------------------------------

            if self.done:

                reward += self._compute_terminal_reward()

            state = self.current_state.copy()

            state.step = self.episode_step

            return (

                state,

                reward,

                self.done,

                self._get_info()

            )

        # ------------------------------------------------------
        # STOP
        # ------------------------------------------------------

        if action.is_stop():

            self.episode_step += 1

            self.done = True

            state = self.current_state.copy()

            state.step = self.episode_step

            # --------------------------------------------------
            # Compare final RL solution against GA
            # --------------------------------------------------

            terminal_reward = self._compute_terminal_reward()

            return (

                state,

                terminal_reward,

                True,

                self._get_info()

            )

        # ------------------------------------------------------
        # Apply Action
        # ------------------------------------------------------

        new_cut = self._ACTION_MAP[
            action.action_type
        ](action)

        old_state = self.current_state.copy()

        self.episode_step += 1

        self.current_state = self._build_state(new_cut)

        # ------------------------------------------------------
        # Check whether max_steps has been reached
        # ------------------------------------------------------

        if self.episode_step >= self.max_steps:

            self.done = True

        # ------------------------------------------------------
        # Normal reward
        # ------------------------------------------------------

        reward = compute_reward(

            old_state,

            self.current_state,

            action,

            done=self.done

        )

        self._update_best_state()

        # ------------------------------------------------------
        # Terminal teacher reward
        # ------------------------------------------------------

        if self.done:

            reward += self._compute_terminal_reward()

        # ------------------------------------------------------
        # Return state
        # ------------------------------------------------------

        state = self.current_state.copy()

        state.step = self.episode_step

        return (

            state,

            reward,

            self.done,

            self._get_info()

        )

    def _compute_terminal_reward(self):
        """
        Compares the final RL SCB against the GA teacher SCB.

        Higher reward = better RL solution.

        Returns
        -------
        float
            Terminal teacher reward.
        """

        cut = self.current_state.cut_size
        sep = self.current_state.separated_count

        # --------------------------------------------------
        # No separated sessions = invalid SCB solution
        # --------------------------------------------------

        if sep <= 0 or cut <= 0:

            return -1.0

        # --------------------------------------------------
        # Actual RL SCB = cut / separated sessions
        # --------------------------------------------------

        rl_scb = cut / sep

        teacher_scb = self.teacher_scb

        # --------------------------------------------------
        # Compare against GA
        #
        # teacher / RL:
        #
        #   = 1  -> matched GA
        #   > 1  -> RL is better
        #   < 1  -> RL is worse
        # --------------------------------------------------

        reward = teacher_scb / rl_scb

        # --------------------------------------------------
        # Prevent extreme values
        # --------------------------------------------------

        reward = max(
            -1.0,
            min(reward, 2.0)
        )

        return reward

    # ==========================================================
    # Render
    # ==========================================================


    def render(self):

        print(self.current_state)