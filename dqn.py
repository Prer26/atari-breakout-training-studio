import random

from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Q NETWORK
# ============================================================

class DQN(nn.Module):

    def __init__(
        self,
        state_size,
        action_size,
    ):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                state_size,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                action_size
            ),
        )

    def forward(self, state):

        return self.network(state)


# ============================================================
# REPLAY BUFFER
# ============================================================

class ReplayBuffer:

    def __init__(
        self,
        capacity=50_000,
    ):

        self.buffer = deque(
            maxlen=capacity
        )

    def push(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.buffer.append(
            (
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(
        self,
        batch_size,
    ):

        batch = random.sample(
            self.buffer,
            batch_size
        )

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = zip(*batch)

        return (
            np.array(
                states,
                dtype=np.float32
            ),

            np.array(
                actions,
                dtype=np.int64
            ),

            np.array(
                rewards,
                dtype=np.float32
            ),

            np.array(
                next_states,
                dtype=np.float32
            ),

            np.array(
                dones,
                dtype=np.float32
            ),
        )

    def __len__(self):

        return len(self.buffer)


# ============================================================
# DQN AGENT
# ============================================================

class DQNAgent:

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=5e-4,
        gamma=0.99,
        buffer_size=50_000,
    ):

        self.state_size = state_size

        self.action_size = action_size

        self.gamma = gamma

        # ----------------------------------------------------
        # Policy network
        # ----------------------------------------------------

        self.policy_net = DQN(
            state_size,
            action_size,
        ).to(DEVICE)

        # ----------------------------------------------------
        # Target network
        # ----------------------------------------------------

        self.target_net = DQN(
            state_size,
            action_size,
        ).to(DEVICE)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        # ----------------------------------------------------
        # Optimizer
        # ----------------------------------------------------

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate,
        )

        # ----------------------------------------------------
        # Replay memory
        # ----------------------------------------------------

        self.memory = ReplayBuffer(
            buffer_size
        )

    # ========================================================
    # ACTION
    # ========================================================

    def act(
        self,
        state,
        epsilon=0.1,
    ):

        # Exploration
        if random.random() < epsilon:

            return random.randrange(
                self.action_size
            )

        # Exploitation
        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=DEVICE,
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

        return q_values.argmax(
            dim=1
        ).item()

    # ========================================================
    # REMEMBER
    # ========================================================

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):

        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    # ========================================================
    # LEARN
    # ========================================================

    def learn(
        self,
        batch_size=64,
    ):

        if len(self.memory) < batch_size:

            return None

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(
            batch_size
        )

        states = torch.tensor(
            states,
            dtype=torch.float32,
            device=DEVICE,
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=DEVICE,
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=DEVICE,
        )

        next_states = torch.tensor(
            next_states,
            dtype=torch.float32,
            device=DEVICE,
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=DEVICE,
        )

        # ----------------------------------------------------
        # Current Q values
        # ----------------------------------------------------

        current_q = (
            self.policy_net(states)
            .gather(
                1,
                actions
            )
            .squeeze(1)
        )

        # ----------------------------------------------------
        # Double DQN target
        # ----------------------------------------------------

        with torch.no_grad():

            # Policy network chooses the action.
            next_actions = (
                self.policy_net(next_states)
                .argmax(
                    dim=1,
                    keepdim=True
                )
            )

            # Target network evaluates it.
            next_q = (
                self.target_net(next_states)
                .gather(
                    1,
                    next_actions
                )
                .squeeze(1)
            )

            target_q = (
                rewards
                + self.gamma
                * next_q
                * (1 - dones)
            )

        # ----------------------------------------------------
        # Huber loss
        # ----------------------------------------------------

        loss = nn.functional.smooth_l1_loss(
            current_q,
            target_q
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        self.optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            5.0
        )

        self.optimizer.step()

        return loss.item()

    # ========================================================
    # TARGET UPDATE
    # ========================================================

    def update_target(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    # ========================================================
    # SAVE
    # ========================================================

    def save(
        self,
        path,
        episode,
    ):

        torch.save(
            {
                "episode": episode,

                "policy_net":
                    self.policy_net.state_dict(),

                "target_net":
                    self.target_net.state_dict(),

                "optimizer":
                    self.optimizer.state_dict(),
            },
            path,
        )

    # ========================================================
    # LOAD
    # ========================================================

    def load(self, path):

        checkpoint = torch.load(
            path,
            map_location=DEVICE
        )

        self.policy_net.load_state_dict(
            checkpoint["policy_net"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_net"]
        )

        # Optimizer may not exist in an older checkpoint.
        if "optimizer" in checkpoint:

            self.optimizer.load_state_dict(
                checkpoint["optimizer"]
            )

        return checkpoint.get(
            "episode",
            0
        )