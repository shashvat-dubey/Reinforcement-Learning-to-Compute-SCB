"""
actions.py

Defines the action space for the SCB Reinforcement Learning environment.

The RL agent does not directly modify the graph. Instead, it outputs an
Action object describing a graph-edit operation to be applied to the
current candidate cut.

Author: SCB-RL
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class ActionType(Enum):
    """
    Primitive graph-edit operations available to the RL agent.
    """

    ADD = 0
    REMOVE = 1
    SWAP = 2
    STOP = 3


@dataclass(frozen=True)
class Action:
    """
    Represents a single action selected by the RL policy.

    Depending on the action type:

    ADD:
        edge must be specified.

    REMOVE:
        edge must be specified.

    SWAP:
        remove_edge and add_edge must be specified.

    STOP:
        No additional fields are required.
    """

    action_type: ActionType

    edge: Optional[int] = None

    remove_edge: Optional[int] = None
    add_edge: Optional[int] = None

    # --------------------------------------------------------
    # Helper Functions
    # --------------------------------------------------------

    def is_add(self) -> bool:
        return self.action_type == ActionType.ADD

    def is_remove(self) -> bool:
        return self.action_type == ActionType.REMOVE

    def is_swap(self) -> bool:
        return self.action_type == ActionType.SWAP

    def is_stop(self) -> bool:
        return self.action_type == ActionType.STOP

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    def validate(self) -> bool:
        """
        Validates that this Action contains the required fields.

        Returns
        -------
        bool
            True if the action is valid.

        Raises
        ------
        ValueError
            If required fields are missing.
        """

        if self.is_add():

            if self.edge is None:
                raise ValueError(
                    "ADD action requires 'edge'."
                )

            return True

        if self.is_remove():

            if self.edge is None:
                raise ValueError(
                    "REMOVE action requires 'edge'."
                )

            return True

        if self.is_swap():

            if self.remove_edge is None:
                raise ValueError(
                    "SWAP action requires 'remove_edge'."
                )

            if self.add_edge is None:
                raise ValueError(
                    "SWAP action requires 'add_edge'."
                )

            if self.remove_edge == self.add_edge:
                raise ValueError(
                    "Cannot swap an edge with itself."
                )

            return True

        if self.is_stop():

            if (
                self.edge is not None
                or self.remove_edge is not None
                or self.add_edge is not None
            ):
                raise ValueError(
                    "STOP action should not contain edge information."
                )

            return True

        raise ValueError("Unknown action type.")

    # --------------------------------------------------------
    # Pretty Representation
    # --------------------------------------------------------

    def __repr__(self):

        if self.is_add():
            return f"Action(ADD edge={self.edge})"

        if self.is_remove():
            return f"Action(REMOVE edge={self.edge})"

        if self.is_swap():
            return (
                f"Action(SWAP remove={self.remove_edge}, "
                f"add={self.add_edge})"
            )

        return "Action(STOP)"