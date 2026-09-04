import os

from breakout_env import BreakoutEnv
from dqn import DQNAgent


CHECKPOINT_DIR = "checkpoints"


def create_agent():
    """Create a DQN agent with the same state/action sizes as Breakout."""

    env = BreakoutEnv()

    state = env.reset(seed=42)

    agent = DQNAgent(
        state_size=len(state),
        action_size=3
    )

    return env, agent


def evaluate_agent(
    checkpoint_path,
    seed=42,
    max_steps=2000
):
    """Evaluate a trained checkpoint on a fresh seeded game."""

    env, agent = create_agent()

    checkpoint_episode = agent.load(
        checkpoint_path
    )

    state = env.reset(
        seed=seed
    )

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:

        # epsilon=0 means no random actions.
        action = agent.act(
            state,
            epsilon=0.0
        )

        (
            next_state,
            reward,
            done,
            info
        ) = env.step(action)

        state = next_state

        total_reward += reward

        steps += 1

    return {
        "checkpoint": checkpoint_path,
        "checkpoint_episode": checkpoint_episode,
        "reward": total_reward,
        "steps": steps,
        "score": info["score"],
        "lives": info["lives"],
        "bricks_remaining": info["bricks_remaining"]
    }


def compare_agents(
    checkpoint_a,
    checkpoint_b,
    seed=42
):
    """Evaluate two checkpoints using the same seed."""

    result_a = evaluate_agent(
        checkpoint_a,
        seed=seed
    )

    result_b = evaluate_agent(
        checkpoint_b,
        seed=seed
    )

    if result_a["reward"] > result_b["reward"]:
        winner = "Agent A"

    elif result_b["reward"] > result_a["reward"]:
        winner = "Agent B"

    else:
        winner = "Tie"

    return {
        "agent_a": result_a,
        "agent_b": result_b,
        "winner": winner
    }


def list_checkpoints():

    if not os.path.exists(
        CHECKPOINT_DIR
    ):
        return []

    files = []

    for filename in os.listdir(
        CHECKPOINT_DIR
    ):

        if filename.endswith(".pt"):
            files.append(filename)

    return sorted(files)