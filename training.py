import os
import threading
import time

import numpy as np

from breakout_env import BreakoutEnv
from dqn import DQNAgent


class TrainingController:

    def __init__(self):

        # ====================================================
        # Thread control
        # ====================================================

        self.lock = threading.Lock()

        self.running = False
        self.paused = False

        self.thread = None

        # ====================================================
        # Environment / Agent
        # ====================================================

        self.env = None
        self.agent = None

        # ====================================================
        # Training state
        # ====================================================

        self.episode = 0
        self.step = 0

        self.current_reward = 0.0
        self.episode_reward = 0.0

        self.average_reward = 0.0
        self.best_reward = float("-inf")

        self.episode_length = 0

        self.epsilon = 1.0

        # ====================================================
        # Metrics
        # ====================================================

        self.reward_history = []
        self.length_history = []
        self.loss_history = []

        # ====================================================
        # Hyperparameters
        # ====================================================

        self.config = {

            "learning_rate": 5e-4,

            "gamma": 0.99,

            "batch_size": 64,

            "epsilon_start": 1.0,

            "epsilon_end": 0.05,

            "epsilon_decay": 0.98,

            "target_update": 20,

            "max_steps": 2000,

            "checkpoint_every": 50,

            "num_episodes": 100,
        }

        # ====================================================
        # Improved experiment checkpoint names
        # ====================================================

        self.checkpoint_dir = "checkpoints"

        os.makedirs(
            self.checkpoint_dir,
            exist_ok=True
        )

        # IMPORTANT:
        # This experiment will NOT overwrite the
        # previous best.pt.

        self.best_checkpoint_name = (
            "improved_best.pt"
        )

        self.final_checkpoint_name = (
            "improved_final.pt"
        )

        # ====================================================
        # Latest environment information
        # ====================================================

        self.last_info = {}

    # ========================================================
    # START TRAINING
    # ========================================================

    def start(self, config=None):

        with self.lock:

            # Don't allow another run while
            # training is already active.

            if self.running:

                return False

            if config:

                for key, value in config.items():

                    if value is not None:

                        self.config[key] = value

            self.running = True
            self.paused = False

            self._reset_training_state()

            self.thread = threading.Thread(
                target=self._training_loop,
                daemon=True,
            )

            self.thread.start()

            return True

    # ========================================================
    # RESET TRAINING STATE
    # ========================================================

    def _reset_training_state(self):

        self.env = BreakoutEnv()

        state = self.env.reset(
            seed=42
        )

        self.agent = DQNAgent(
            state_size=len(state),

            action_size=3,

            learning_rate=self.config[
                "learning_rate"
            ],

            gamma=self.config[
                "gamma"
            ],
        )

        self.episode = 0
        self.step = 0

        self.current_reward = 0.0
        self.episode_reward = 0.0

        self.average_reward = 0.0
        self.best_reward = float("-inf")

        self.episode_length = 0

        self.epsilon = self.config[
            "epsilon_start"
        ]

        self.reward_history = []
        self.length_history = []
        self.loss_history = []

        self.last_info = (
            self._get_env_info()
        )

    # ========================================================
    # PAUSE
    # ========================================================

    def pause(self):

        with self.lock:

            if not self.running:

                return False

            self.paused = True

            return True

    # ========================================================
    # RESUME
    # ========================================================

    def resume(self):

        with self.lock:

            if not self.running:

                return False

            self.paused = False

            return True

    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        with self.lock:

            self.running = False
            self.paused = False

            return True

    # ========================================================
    # ENVIRONMENT INFO
    # ========================================================

    def _get_env_info(self):

        if self.env is None:

            return {}

        try:

            info = self.env.get_info()

            if isinstance(info, dict):

                return info

        except Exception:

            pass

        return {

            "score":
                getattr(
                    self.env,
                    "score",
                    0
                ),

            "lives":
                getattr(
                    self.env,
                    "lives",
                    3
                ),

            "bricks_remaining":
                sum(
                    1
                    for brick in getattr(
                        self.env,
                        "bricks",
                        []
                    )
                    if brick.get(
                        "alive",
                        True
                    )
                ),
        }

    # ========================================================
    # SAFE ATTRIBUTE
    # ========================================================

    def _get_attr(
        self,
        name,
        default=0,
    ):

        if self.env is None:

            return default

        try:

            return getattr(
                self.env,
                name
            )

        except Exception:

            return default

    # ========================================================
    # LIVE GAME STATE
    # ========================================================

    def _get_game_state(self):

        if self.env is None:

            return {

                "paddle_x": 0,
                "paddle_y": 0,

                "paddle_width": 80,
                "paddle_height": 10,

                "ball_x": 0,
                "ball_y": 0,

                "ball_vx": 0,
                "ball_vy": 0,

                "ball_radius": 5,

                "bricks": [],
            }

        game = {

            "paddle_x":
                float(
                    self._get_attr(
                        "paddle_x",
                        0
                    )
                ),

            "paddle_y":
                float(
                    self.HEIGHT
                    if hasattr(
                        self,
                        "HEIGHT"
                    )
                    else 470
                ),

            "paddle_width":
                float(
                    self._get_attr(
                        "PADDLE_WIDTH",
                        70
                    )
                ),

            "paddle_height":
                float(
                    self._get_attr(
                        "PADDLE_HEIGHT",
                        10
                    )
                ),

            "ball_x":
                float(
                    self._get_attr(
                        "ball_x",
                        0
                    )
                ),

            "ball_y":
                float(
                    self._get_attr(
                        "ball_y",
                        0
                    )
                ),

            "ball_vx":
                float(
                    self._get_attr(
                        "ball_vx",
                        0
                    )
                ),

            "ball_vy":
                float(
                    self._get_attr(
                        "ball_vy",
                        0
                    )
                ),

            "ball_radius":
                float(
                    self._get_attr(
                        "BALL_SIZE",
                        5
                    )
                ),

            "bricks": [],
        }

        # Actual paddle Y in the environment.
        game["paddle_y"] = (
            self.env.HEIGHT - 30
        )

        # ----------------------------------------------------
        # Bricks
        # ----------------------------------------------------

        bricks = getattr(
            self.env,
            "bricks",
            []
        )

        for brick in bricks:

            if isinstance(
                brick,
                dict
            ):

                game["bricks"].append({

                    "x":
                        float(
                            brick.get(
                                "x",
                                0
                            )
                        ),

                    "y":
                        float(
                            brick.get(
                                "y",
                                0
                            )
                        ),

                    "width":
                        float(
                            self.env.BRICK_WIDTH
                        ),

                    "height":
                        float(
                            self.env.BRICK_HEIGHT
                        ),

                    "alive":
                        bool(
                            brick.get(
                                "alive",
                                True
                            )
                        ),
                })

        return game

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    def _training_loop(self):

        total_episodes = int(
            self.config[
                "num_episodes"
            ]
        )

        batch_size = int(
            self.config[
                "batch_size"
            ]
        )

        max_steps = int(
            self.config[
                "max_steps"
            ]
        )

        epsilon_end = float(
            self.config[
                "epsilon_end"
            ]
        )

        epsilon_decay = float(
            self.config[
                "epsilon_decay"
            ]
        )

        target_update = int(
            self.config[
                "target_update"
            ]
        )

        checkpoint_every = int(
            self.config[
                "checkpoint_every"
            ]
        )

        # ====================================================
        # Episode loop
        # ====================================================

        while (
            self.running
            and self.episode
            < total_episodes
        ):

            # ------------------------------------------------
            # Pause
            # ------------------------------------------------

            while (
                self.paused
                and self.running
            ):

                time.sleep(0.1)

            if not self.running:

                break

            # ------------------------------------------------
            # New episode
            # ------------------------------------------------

            self.episode += 1

            state = self.env.reset(
                seed=self.episode
            )

            episode_reward = 0.0
            episode_steps = 0

            done = False

            # ------------------------------------------------
            # Episode
            # ------------------------------------------------

            while (
                not done
                and episode_steps
                < max_steps
                and self.running
            ):

                # --------------------------------------------
                # Pause
                # --------------------------------------------

                while (
                    self.paused
                    and self.running
                ):

                    time.sleep(0.1)

                if not self.running:

                    break

                # --------------------------------------------
                # Choose action
                # --------------------------------------------

                action = self.agent.act(
                    state,
                    self.epsilon
                )

                # --------------------------------------------
                # Environment
                # --------------------------------------------

                (
                    next_state,
                    reward,
                    done,
                    info,
                ) = self.env.step(
                    action
                )

                # --------------------------------------------
                # Store
                # --------------------------------------------

                self.agent.remember(
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                )

                # --------------------------------------------
                # Learn
                # --------------------------------------------

                loss = self.agent.learn(
                    batch_size
                )

                if loss is not None:

                    self.loss_history.append(
                        float(loss)
                    )

                # --------------------------------------------
                # State
                # --------------------------------------------

                state = next_state

                episode_reward += float(
                    reward
                )

                episode_steps += 1

                # --------------------------------------------
                # Live metrics
                # --------------------------------------------

                self.step = episode_steps

                self.current_reward = float(
                    reward
                )

                self.episode_reward = float(
                    episode_reward
                )

                if isinstance(
                    info,
                    dict
                ):

                    self.last_info = info

                else:

                    self.last_info = (
                        self._get_env_info()
                    )

                # Keep UI responsive without making
                # training painfully slow.
                time.sleep(0.001)

            # =================================================
            # EPSILON DECAY
            # =================================================

            self.epsilon = max(
                epsilon_end,
                self.epsilon
                * epsilon_decay
            )

            # =================================================
            # TARGET NETWORK
            # =================================================

            if (
                target_update > 0
                and self.episode
                % target_update == 0
            ):

                try:

                    self.agent.update_target()

                except Exception as error:

                    print(
                        "Target update error:",
                        error
                    )

            # =================================================
            # METRICS
            # =================================================

            self.reward_history.append(
                float(episode_reward)
            )

            self.length_history.append(
                int(episode_steps)
            )

            recent_rewards = (
                self.reward_history[-20:]
            )

            if recent_rewards:

                self.average_reward = float(
                    np.mean(
                        recent_rewards
                    )
                )

            self.episode_length = (
                episode_steps
            )

            # =================================================
            # IMPROVED BEST CHECKPOINT
            # =================================================

            if (
                episode_reward
                > self.best_reward
            ):

                self.best_reward = float(
                    episode_reward
                )

                try:

                    self.agent.save(
                        os.path.join(
                            self.checkpoint_dir,
                            self.best_checkpoint_name,
                        ),
                        self.episode,
                    )

                except Exception as error:

                    print(
                        "Best checkpoint error:",
                        error
                    )

            # =================================================
            # PERIODIC CHECKPOINT
            # =================================================

            if (
                checkpoint_every > 0
                and self.episode
                % checkpoint_every == 0
            ):

                try:

                    self.agent.save(
                        os.path.join(
                            self.checkpoint_dir,
                            f"improved_checkpoint_{self.episode}.pt",
                        ),
                        self.episode,
                    )

                except Exception as error:

                    print(
                        "Checkpoint error:",
                        error
                    )

        # ====================================================
        # FINAL CHECKPOINT
        # ====================================================

        if self.agent is not None:

            try:

                self.agent.save(
                    os.path.join(
                        self.checkpoint_dir,
                        self.final_checkpoint_name,
                    ),
                    self.episode,
                )

            except Exception as error:

                print(
                    "Final checkpoint error:",
                    error
                )

        self.running = False
        self.paused = False

    # ========================================================
    # STATUS
    # ========================================================

    def get_status(self):

        with self.lock:

            # ------------------------------------------------
            # Initialize environment before training.
            # ------------------------------------------------

            if self.env is None:

                self.env = BreakoutEnv()

                self.env.reset(
                    seed=42
                )

                self.last_info = (
                    self._get_env_info()
                )

            info = (
                self.last_info
                or {}
            )

            bricks = getattr(
                self.env,
                "bricks",
                []
            )

            bricks_remaining = sum(
                1
                for brick in bricks
                if isinstance(
                    brick,
                    dict
                )
                and brick.get(
                    "alive",
                    True
                )
            )

            return {

                # --------------------------------------------
                # Training
                # --------------------------------------------

                "running":
                    self.running,

                "paused":
                    self.paused,

                "episode":
                    self.episode,

                "step":
                    self.step,

                "current_reward":
                    self.current_reward,

                "episode_reward":
                    self.episode_reward,

                "average_reward":
                    self.average_reward,

                "best_reward":
                    (
                        self.best_reward
                        if self.best_reward
                        != float("-inf")
                        else 0.0
                    ),

                "episode_length":
                    self.episode_length,

                "epsilon":
                    self.epsilon,

                # --------------------------------------------
                # Game
                # --------------------------------------------

                "score":
                    info.get(
                        "score",
                        getattr(
                            self.env,
                            "score",
                            0
                        )
                    ),

                "lives":
                    info.get(
                        "lives",
                        getattr(
                            self.env,
                            "lives",
                            3
                        )
                    ),

                "bricks_remaining":
                    info.get(
                        "bricks_remaining",
                        bricks_remaining
                    ),

                # --------------------------------------------
                # Live game
                # --------------------------------------------

                "game":
                    self._get_game_state(),

                # --------------------------------------------
                # Charts
                # --------------------------------------------

                "reward_history":
                    self.reward_history,

                "length_history":
                    self.length_history,

                "loss_history":
                    self.loss_history[-100:],

                # --------------------------------------------
                # Hyperparameters
                # --------------------------------------------

                "config":
                    self.config,
            }


# ============================================================
# GLOBAL TRAINING CONTROLLER
# ============================================================

training_controller = TrainingController()