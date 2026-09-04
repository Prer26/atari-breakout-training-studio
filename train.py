import os
import random

import numpy as np
import torch

from breakout_env import BreakoutEnv
from dqn import DQNAgent


# ============================================================
# CONFIGURATION
# ============================================================

NUM_EPISODES = 100

LEARNING_RATE = 0.0005
GAMMA = 0.99

BATCH_SIZE = 64

EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.98

TARGET_UPDATE = 20

MAX_STEPS = 2000

CHECKPOINT_EVERY = 20

CHECKPOINT_DIR = "checkpoints"


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# SETUP
# ============================================================

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

env = BreakoutEnv()

initial_state = env.reset(seed=SEED)

state_size = len(initial_state)

action_size = 3


agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    learning_rate=LEARNING_RATE,
    gamma=GAMMA,
)


# ============================================================
# TRAINING METRICS
# ============================================================

reward_history = []
length_history = []
loss_history = []

best_reward = float("-inf")


# ============================================================
# TRAINING LOOP
# ============================================================

print("=" * 60)
print("ATARI BREAKOUT — DQN TRAINING")
print("=" * 60)

print()
print("Configuration")
print("-" * 60)
print(f"Episodes          : {NUM_EPISODES}")
print(f"Learning rate     : {LEARNING_RATE}")
print(f"Gamma             : {GAMMA}")
print(f"Batch size        : {BATCH_SIZE}")
print(f"Epsilon start     : {EPSILON_START}")
print(f"Epsilon end       : {EPSILON_END}")
print(f"Epsilon decay     : {EPSILON_DECAY}")
print(f"Target update     : {TARGET_UPDATE}")
print(f"Max steps         : {MAX_STEPS}")
print(f"Checkpoint every  : {CHECKPOINT_EVERY}")
print(f"Seed              : {SEED}")
print("=" * 60)
print()


epsilon = EPSILON_START


for episode in range(1, NUM_EPISODES + 1):

    # --------------------------------------------------------
    # Reset environment
    # --------------------------------------------------------

    state = env.reset(
        seed=SEED + episode
    )

    episode_reward = 0.0
    episode_loss = []

    done = False

    step = 0


    # --------------------------------------------------------
    # Episode
    # --------------------------------------------------------

    while not done and step < MAX_STEPS:

        step += 1


        # ----------------------------------------------------
        # Select action
        # ----------------------------------------------------

        action = agent.act(
            state,
            epsilon=epsilon
        )


        # ----------------------------------------------------
        # Environment step
        # ----------------------------------------------------

        next_state, reward, done, info = env.step(
            action
        )


        # ----------------------------------------------------
        # Store experience
        # ----------------------------------------------------

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )


        # ----------------------------------------------------
        # Learn
        # ----------------------------------------------------

        if len(agent.memory) >= BATCH_SIZE:

            loss = agent.learn(
                BATCH_SIZE
            )

            if loss is not None:
                episode_loss.append(
                    float(loss)
                )


        # ----------------------------------------------------
        # Update state
        # ----------------------------------------------------

        state = next_state

        episode_reward += reward


    # ========================================================
    # END OF EPISODE
    # ========================================================

    reward_history.append(
        episode_reward
    )

    length_history.append(
        step
    )


    if episode_loss:

        mean_loss = float(
            np.mean(episode_loss)
        )

        loss_history.append(
            mean_loss
        )

    else:

        mean_loss = 0.0

        loss_history.append(
            mean_loss
        )


    # --------------------------------------------------------
    # Epsilon decay
    # --------------------------------------------------------

    epsilon = max(
        EPSILON_END,
        epsilon * EPSILON_DECAY,
    )


    # --------------------------------------------------------
    # Target network update
    # --------------------------------------------------------

    if episode % TARGET_UPDATE == 0:

        agent.update_target()


    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    if episode_reward > best_reward:

        best_reward = episode_reward

        best_path = os.path.join(
            CHECKPOINT_DIR,
            "best.pt",
        )

        agent.save(
            best_path,
            episode=episode,
        )


    # --------------------------------------------------------
    # Periodic checkpoint
    # --------------------------------------------------------

    if episode % CHECKPOINT_EVERY == 0:

        checkpoint_path = os.path.join(
            CHECKPOINT_DIR,
            f"checkpoint_{episode}.pt",
        )

        agent.save(
            checkpoint_path,
            episode=episode,
        )


    # --------------------------------------------------------
    # Running average
    # --------------------------------------------------------

    recent_rewards = reward_history[
        -10:
    ]

    average_reward = float(
        np.mean(recent_rewards)
    )


    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    print(
        f"Episode {episode:3d}/{NUM_EPISODES} | "
        f"Reward: {episode_reward:7.2f} | "
        f"Avg(10): {average_reward:7.2f} | "
        f"Score: {env.score:4d} | "
        f"Lives: {env.lives} | "
        f"Steps: {step:4d} | "
        f"Epsilon: {epsilon:.3f} | "
        f"Loss: {mean_loss:.4f}"
    )


# ============================================================
# SAVE FINAL CHECKPOINT
# ============================================================

final_path = os.path.join(
    CHECKPOINT_DIR,
    "final.pt",
)

agent.save(
    final_path,
    episode=NUM_EPISODES,
)


# ============================================================
# SAVE TRAINING DATA
# ============================================================

np.save(
    "episode_rewards.npy",
    np.array(reward_history),
)

np.save(
    "episode_lengths.npy",
    np.array(length_history),
)

np.save(
    "loss_history.npy",
    np.array(loss_history),
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Best reward       : {best_reward:.2f}"
)

print(
    f"Final reward      : {reward_history[-1]:.2f}"
)

print(
    f"Average reward    : "
    f"{np.mean(reward_history):.2f}"
)

print(
    f"Last 10 avg       : "
    f"{np.mean(reward_history[-10:]):.2f}"
)

print(
    f"Best episode      : "
    f"{np.argmax(reward_history) + 1}"
)

print(
    f"Final episode len : "
    f"{length_history[-1]}"
)

print(
    f"Final epsilon     : "
    f"{epsilon:.3f}"
)

print()
print("Saved:")
print(f"  {best_path}")
print(f"  {final_path}")
print("  episode_rewards.npy")
print("  episode_lengths.npy")
print("  loss_history.npy")

print("=" * 60)