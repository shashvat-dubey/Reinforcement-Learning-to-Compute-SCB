from .graph import SCBProblem
from .genetic import run_ga
from .solution import print_solution


class SCBGeneticSolver:
    """
    Genetic Algorithm solver for the Sparsest Cut Bound.
    """

    def __init__(
        self,
        generations=50,
        pop_size=100
    ):
        self.generations = generations
        self.pop_size = pop_size

    def solve(
        self,
        nodes,
        edges,
        sessions,
        verbose=True
    ):

        # -------------------------------------
        # Create Problem
        # -------------------------------------

        problem = SCBProblem(
            nodes,
            edges,
            sessions
        )

        # -------------------------------------
        # Run GA with both objectives
        # -------------------------------------

        sol_full = run_ga(
            problem=problem,
            fitness_key="fitness1",
            generations=self.generations,
            pop_size=self.pop_size,
            verbose=False
        )

        sol_sparse = run_ga(
            problem=problem,
            fitness_key="fitness2",
            generations=self.generations,
            pop_size=self.pop_size,
            verbose=False
        )

        # -------------------------------------
        # Compare final SCB
        # Higher sep/cut is better
        # -------------------------------------

        scb_full = sol_full["fitness1"]
        scb_sparse = sol_sparse["fitness2"]

        if scb_full <= scb_sparse:
            best = sol_full
            objective = "fitness1"
        else:
            best = sol_sparse
            objective = "fitness2"

        # -------------------------------------
        # Display final solution
        # -------------------------------------

        if verbose:
            print(f"\nBest objective : {objective}")
            print_solution(problem, best)

        return {
            "problem": problem,
            "solution": best,
            "objective": objective,
            "fitness1_solution": scb_full,
            "fitness2_solution": scb_sparse
        }