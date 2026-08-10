"""
loader.py

Loads labelled graphs for PPO training.

Author: SCB-RL
"""

import pickle


class GraphDataset:
    """
    Loads the labelled graph dataset.
    """

    def __init__(
        self,
        dataset_path="labelled_dataset.pkl"
    ):

        with open(
            dataset_path,
            "rb"
        ) as f:

            self.dataset = pickle.load(f)

    # --------------------------------------------------
    # Length
    # --------------------------------------------------

    def __len__(self):

        return len(self.dataset)

    # --------------------------------------------------
    # Index
    # --------------------------------------------------

    def __getitem__(
        self,
        index
    ):

        return self.dataset[index]

    # --------------------------------------------------
    # Convenience
    # --------------------------------------------------

    def get_graph(
        self,
        index
    ):

        return self.dataset[index]


if __name__ == "__main__":

    dataset = GraphDataset()

    print("Graphs :", len(dataset))

    graph = dataset[0]

    print()

    print("Graph ID :", graph["graph_id"])

    print("Nodes    :", len(graph["nodes"]))

    print("Edges    :", len(graph["edges"]))

    print("Sessions :", len(graph["sessions"]))

    print()

    print("Teacher SCB :", graph["ga_scb"])