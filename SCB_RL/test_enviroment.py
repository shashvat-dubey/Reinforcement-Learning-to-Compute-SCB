"""
SCB RL Environment Validation Suite

This script verifies that the RL environment behaves exactly like the
GA evaluation function.

Every result is written both to the console and to

environment_test_results.txt
"""

import random
import traceback
from pathlib import Path

from GA_SCB.graph import SCBProblem, evaluate
from SCB_RL.environment import SCBEnvironment
from SCB_RL.actions import Action, ActionType


# ==========================================================
# Logger
# ==========================================================

LOG_FILE = Path("environment_test_results.txt")

# Clear previous log
LOG_FILE.write_text("")


def log(*args):
    msg = " ".join(str(a) for a in args)

    print(msg)

    with LOG_FILE.open("a") as f:
        f.write(msg + "\n")


def divider():

    log("=" * 70)


# ==========================================================
# Statistics
# ==========================================================

passed = 0
failed = 0


def check(name, condition):

    global passed, failed

    if condition:

        passed += 1

        log(f"[PASS] {name}")

    else:

        failed += 1

        log(f"[FAIL] {name}")

        raise AssertionError(name)


# ==========================================================
# Test Graph
# ==========================================================

nodes = [

    "v1", "v2", "v3",

    "v4", "v5", "v6",

    "v7"

]

sessions = [

    ("v1", "v4"),

    ("v2", "v5"),

    ("v3", "v6")

]

edges = [

    ("v4","v3"),

    ("v5","v7"),

    ("v3","v7"),

    ("v6","v7"),

    ("v6","v1"),

    ("v7","v1"),

    ("v7","v4"),

    ("v7","v2"),

    ("v4","v2"),

    ("v1","v2")

]

problem = SCBProblem(

    nodes,

    edges,

    sessions

)

env = SCBEnvironment(problem)


# ==========================================================
# Helper Functions
# ==========================================================

def verify_against_ga(state):
    """
    Verify that the environment state exactly matches
    the GA evaluation.
    """

    ga = evaluate(

        problem,

        state.cut

    )

    check(

        "SCB",

        abs(

            ga["fitness2"]

            - state.scb

        ) < 1e-9

    )

    check(

        "Cut Size",

        ga["cut"]

        == state.cut_size

    )

    check(

        "Separated Sessions",

        ga["sep"]

        == state.separated_count

    )

    check(

        "Cut Edges",

        set(ga["cut_edges"])

        == state.cut

    )


def header(title):

    divider()

    log(title)

    divider()

# ==========================================================
# RESET TEST
# ==========================================================

header("TEST 1 : RESET")

state = env.reset()

log(state)

check(
    "Initial SCB",
    state.scb == 0
)

check(
    "Initial Cut Size",
    state.cut_size == 0
)

check(
    "Initial Step",
    state.step == 0
)

verify_against_ga(state)


# ==========================================================
# ADD TEST
# ==========================================================

header("TEST 2 : ADD")

action = Action(
    ActionType.ADD,
    edge=0
)

state, reward, done, info = env.step(action)

log("Action :", action)
log("Reward :", reward)
log(state)

verify_against_ga(state)

check(
    "Episode Not Done",
    done is False
)

check(
    "Cut Size After Add",
    state.cut_size == 1
)


# ==========================================================
# INVALID ADD TEST
# ==========================================================

header("TEST 3 : INVALID ADD")

previous_cut = state.cut.copy()
previous_step = state.step

state, reward, done, info = env.step(action)

log("Action :", action)
log("Reward :", reward)
log(state)

check(
    "Penalty Applied",
    reward == -0.1
)

check(
    "Cut Unchanged",
    state.cut == previous_cut
)

check(
    "Step Increased",
    state.step == previous_step + 1
)

verify_against_ga(state)


# ==========================================================
# REMOVE TEST
# ==========================================================

header("TEST 4 : REMOVE")

action = Action(
    ActionType.REMOVE,
    edge=0
)

state, reward, done, info = env.step(action)

log("Action :", action)
log("Reward :", reward)
log(state)

verify_against_ga(state)

check(
    "Cut Empty",
    state.cut_size == 0
)


# ==========================================================
# INVALID REMOVE TEST
# ==========================================================

header("TEST 5 : INVALID REMOVE")

previous_cut = state.cut.copy()
previous_step = state.step

state, reward, done, info = env.step(action)

log("Action :", action)
log("Reward :", reward)
log(state)

check(
    "Penalty Applied",
    reward == -0.1
)

check(
    "Cut Unchanged",
    state.cut == previous_cut
)

check(
    "Step Increased",
    state.step == previous_step + 1
)

verify_against_ga(state)

# ==========================================================
# SWAP TEST
# ==========================================================

header("TEST 6 : SWAP")

# Reset environment
state = env.reset()

# Add edge 0
state, reward, done, info = env.step(
    Action(ActionType.ADD, edge=0)
)

old_cut = state.cut.copy()

# Swap edge 0 -> edge 1
action = Action(
    ActionType.SWAP,
    remove_edge=0,
    add_edge=1
)

state, reward, done, info = env.step(action)

log("Action :", action)
log("Reward :", reward)
log(state)

verify_against_ga(state)

check(
    "Old Edge Removed",
    problem.idx_to_edge[0] not in state.cut
)

check(
    "New Edge Added",
    problem.idx_to_edge[1] in state.cut
)

check(
    "Swap Keeps Cut Size",
    state.cut_size == len(old_cut)
)


# ==========================================================
# STOP TEST
# ==========================================================

header("TEST 7 : STOP")

action = Action(ActionType.STOP)

state, reward, done, info = env.step(action)

log("Action :", action)
log("Reward :", reward)
log(state)

check(
    "Episode Finished",
    done
)

# Calling step() again should fail

try:

    env.step(action)

    check(
        "Step After Done",
        False
    )

except RuntimeError:

    check(
        "Step After Done",
        True
    )


# ==========================================================
# BEST STATE TEST
# ==========================================================

header("TEST 8 : BEST STATE")

env.reset()

best_seen = 0.0

actions = [

    Action(ActionType.ADD, edge=0),

    Action(ActionType.ADD, edge=1),

    Action(ActionType.REMOVE, edge=0),

    Action(ActionType.ADD, edge=2),

]

for action in actions:

    state, reward, done, info = env.step(action)

    best_seen = max(best_seen, state.scb)

check(

    "Best State",

    abs(env.best_state.scb - best_seen) < 1e-9

)

log("Best SCB :", env.best_state.scb)


# ==========================================================
# REPLAY TEST
# ==========================================================

header("TEST 9 : REPLAY CONSISTENCY")

sequence = [

    Action(ActionType.ADD, edge=0),

    Action(ActionType.ADD, edge=3),

    Action(ActionType.REMOVE, edge=0),

    Action(ActionType.ADD, edge=5),

]

# ---------- Run 1 ----------

env.reset()

results1 = []

for action in sequence:

    state, reward, done, info = env.step(action)

    results1.append(

        (

            state.scb,

            state.cut.copy(),

            reward

        )

    )

# ---------- Run 2 ----------

env.reset()

results2 = []

for action in sequence:

    state, reward, done, info = env.step(action)

    results2.append(

        (

            state.scb,

            state.cut.copy(),

            reward

        )

    )

check(

    "Replay Deterministic",

    results1 == results2

)

log("Replay matched successfully.")

# ==========================================================
# RANDOM EPISODE TEST
# ==========================================================

header("TEST 10 : RANDOM EPISODES")

ACTIONS = []

for edge in range(problem.E):

    ACTIONS.append(
        Action(ActionType.ADD, edge=edge)
    )

    ACTIONS.append(
        Action(ActionType.REMOVE, edge=edge)
    )

for remove_edge in range(problem.E):

    for add_edge in range(problem.E):

        if remove_edge != add_edge:

            ACTIONS.append(

                Action(

                    ActionType.SWAP,

                    remove_edge=remove_edge,

                    add_edge=add_edge

                )

            )

ACTIONS.append(

    Action(ActionType.STOP)

)

episodes = 100

random.seed(42)

for episode in range(episodes):

    state = env.reset()

    while True:

        action = random.choice(ACTIONS)

        state, reward, done, info = env.step(action)

        # --------------------------------------------
        # Verify Environment matches GA
        # --------------------------------------------

        verify_against_ga(state)

        if done:

            break

check(

    "100 Random Episodes",

    True

)

log("Completed", episodes, "episodes successfully.")


# ==========================================================
# STRESS TEST
# ==========================================================

header("TEST 11 : STRESS TEST")

episodes = 500

steps = 0

random.seed(123)

for episode in range(episodes):

    state = env.reset()

    while True:

        action = random.choice(ACTIONS)

        state, reward, done, info = env.step(action)

        steps += 1

        if done:

            break

check(

    "Stress Test",

    True

)

log("Episodes :", episodes)

log("Total Steps :", steps)


# ==========================================================
# RANDOM CONSISTENCY TEST
# ==========================================================

header("TEST 12 : CONTINUOUS GA CONSISTENCY")

env.reset()

for i in range(1000):

    action = random.choice(ACTIONS)

    state, reward, done, info = env.step(action)

    ga = evaluate(

        problem,

        state.cut

    )

    check(

        f"Consistency {i}",

        abs(

            ga["fitness2"]

            - state.scb

        ) < 1e-9

    )

    if done:

        env.reset()

log("Verified 1000 consecutive transitions.")


# ==========================================================
# FINAL SUMMARY
# ==========================================================

divider()

log("FINAL TEST SUMMARY")

divider()

log("Passed :", passed)

log("Failed :", failed)

divider()

if failed == 0:

    log("🎉 ENVIRONMENT VALIDATION SUCCESSFUL")

else:

    log("❌ SOME TESTS FAILED")

divider()