from GA_SCB.graph import SCBProblem
from SCB_RL.environment import SCBEnvironment
from SCB_RL.gnn import SCBGraphEncoder
from SCB_RL.policy import HierarchicalSCBPolicy

import torch

LOG_FILE = "pipeline_log.txt"

log_file = open(
    LOG_FILE,
    "w",
    encoding="utf-8"
)


def log(msg=""):

    print(msg)

    log_file.write(str(msg) + "\n")

    log_file.flush()

nodes = [
    "v1","v2","v3",
    "v4","v5","v6","v7"
]

sessions = [
    ("v1","v4"),
    ("v2","v5"),
    ("v3","v6")
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

encoder = SCBGraphEncoder()

policy = HierarchicalSCBPolicy()

log("=" * 70)
log("PIPELINE STRESS TEST")
log("=" * 70)

NUM_EPISODES = 100

successful = 0
failed = 0

total_reward_all = 0.0
best_scb = 0.0
longest_episode = 0

for episode in range(NUM_EPISODES):

    log()
    log("=" * 70)
    log(f"EPISODE {episode + 1}")
    log("=" * 70)

    try:

        state = env.reset()

        total_reward = 0.0

        step = 0

        while True:

            log()
            log("-" * 70)

            log(f"STEP : {step}")
            log()
            log(state)

            # ----------------------------------------
            # Encode
            # ----------------------------------------

            encoding = encoder(state)

            # ----------------------------------------
            # Policy
            # ----------------------------------------

            action, log_prob, entropy = policy.sample_action(
                encoding,
                state
            )

            log(f"Action   : {action}")
            log(f"Log Prob : {log_prob.item():.6f}")
            log(f"Entropy  : {entropy.item():.6f}")

            # ----------------------------------------
            # Environment
            # ----------------------------------------

            next_state, reward, done, info = env.step(action)

            total_reward += reward

            log(f"Reward   : {reward:.6f}")
            log()
            log(next_state)

            state = next_state

            step += 1

            if state.scb > best_scb:
                best_scb = state.scb

            if done:
                break

        successful += 1

        total_reward_all += total_reward

        longest_episode = max(
            longest_episode,
            step
        )

        log()
        log(f"Episode Reward : {total_reward:.3f}")
        log(f"Final SCB      : {state.scb:.6f}")

    except Exception as e:

        failed += 1

        log()
        log("EPISODE CRASHED")
        log(str(e))

log()
log("=" * 70)
log("FINAL SUMMARY")
log("=" * 70)

log(f"Episodes          : {NUM_EPISODES}")
log(f"Successful        : {successful}")
log(f"Failed            : {failed}")

if successful > 0:

    log(f"Average Reward    : {total_reward_all / successful:.4f}")

log(f"Best SCB          : {best_scb:.6f}")
log(f"Longest Episode   : {longest_episode}")

if failed == 0:

    log()
    log("🎉 PIPELINE PASSED ALL TESTS")

else:

    log()
    log("❌ PIPELINE HAS FAILURES")

log_file.close()