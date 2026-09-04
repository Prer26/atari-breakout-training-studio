import random

import numpy as np


class BreakoutEnv:
    """
    Simple deterministic Breakout environment.

    Actions:
        0 = move left
        1 = stay
        2 = move right

    State:
        Normalized:
        paddle_x
        ball_x
        ball_y
        ball_vx
        ball_vy
        ball_to_paddle_dx
        ball_vertical_distance
        brick_alive[40]

    Rewards:
        +0.20 -> successful paddle hit
        +1.00 -> brick destroyed
        +5.00 -> level cleared
        -1.00 -> life lost
        -5.00 -> game over
    """

    WIDTH = 400
    HEIGHT = 500

    PADDLE_WIDTH = 70
    PADDLE_HEIGHT = 10
    PADDLE_SPEED = 7

    BALL_SIZE = 8

    BRICK_ROWS = 5
    BRICK_COLS = 8
    BRICK_WIDTH = 42
    BRICK_HEIGHT = 18
    BRICK_GAP = 5

    STARTING_LIVES = 3

    def __init__(self):
        self.rng = random.Random()
        self.np_rng = np.random.default_rng()

        self.paddle_x = 0.0

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0

        self.bricks = []

        self.lives = self.STARTING_LIVES
        self.score = 0
        self.done = False

    # ========================================================
    # RESET
    # ========================================================

    def reset(self, seed=None):
        """
        Reset the environment.

        Using the same seed produces the same initial state.
        """

        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.paddle_x = (
            self.WIDTH - self.PADDLE_WIDTH
        ) / 2

        self.ball_x = self.WIDTH / 2
        self.ball_y = self.HEIGHT / 2

        direction = self.rng.choice([-1, 1])

        self.ball_vx = direction * 3
        self.ball_vy = -3

        # ----------------------------------------------------
        # Create bricks
        # ----------------------------------------------------

        self.bricks = []

        for row in range(self.BRICK_ROWS):

            for col in range(self.BRICK_COLS):

                self.bricks.append(
                    {
                        "x": 20
                        + col
                        * (
                            self.BRICK_WIDTH
                            + self.BRICK_GAP
                        ),

                        "y": 40
                        + row
                        * (
                            self.BRICK_HEIGHT
                            + self.BRICK_GAP
                        ),

                        "alive": True,
                    }
                )

        self.lives = self.STARTING_LIVES
        self.score = 0
        self.done = False

        return self.get_state()

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):
        """
        Return normalized numerical state for the agent.

        The additional relationship features help the agent
        understand where the paddle is relative to the ball.
        """

        paddle_center = (
            self.paddle_x
            + self.PADDLE_WIDTH / 2
        )

        ball_center = (
            self.ball_x
            + self.BALL_SIZE / 2
        )

        # Horizontal distance from paddle center to ball center.
        ball_to_paddle_dx = (
            ball_center - paddle_center
        ) / self.WIDTH

        # Vertical distance between ball and paddle.
        paddle_y = self.HEIGHT - 30

        ball_vertical_distance = (
            paddle_y - self.ball_y
        ) / self.HEIGHT

        state = [
            # Paddle
            self.paddle_x / self.WIDTH,

            # Ball
            self.ball_x / self.WIDTH,
            self.ball_y / self.HEIGHT,

            # Ball velocity
            self.ball_vx / 10.0,
            self.ball_vy / 10.0,

            # Useful relationship features
            ball_to_paddle_dx,
            ball_vertical_distance,
        ]

        # Brick state
        for brick in self.bricks:

            state.append(
                1.0
                if brick["alive"]
                else 0.0
            )

        return np.array(
            state,
            dtype=np.float32
        )

    # ========================================================
    # STEP
    # ========================================================

    def step(self, action):
        """
        Advance environment by one frame.

        Returns:
            state
            reward
            done
            info
        """

        if self.done:

            return (
                self.get_state(),
                0.0,
                True,
                self.get_info(),
            )

        reward = 0.0

        # ====================================================
        # 1. Move paddle
        # ====================================================

        if action == 0:

            self.paddle_x -= (
                self.PADDLE_SPEED
            )

        elif action == 2:

            self.paddle_x += (
                self.PADDLE_SPEED
            )

        self.paddle_x = max(
            0,
            min(
                self.WIDTH
                - self.PADDLE_WIDTH,
                self.paddle_x,
            ),
        )

        # ====================================================
        # 2. Move ball
        # ====================================================

        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # ====================================================
        # 3. Wall collision
        # ====================================================

        if self.ball_x <= 0:

            self.ball_x = 0
            self.ball_vx *= -1

        elif (
            self.ball_x
            + self.BALL_SIZE
            >= self.WIDTH
        ):

            self.ball_x = (
                self.WIDTH
                - self.BALL_SIZE
            )

            self.ball_vx *= -1

        if self.ball_y <= 0:

            self.ball_y = 0
            self.ball_vy *= -1

        # ====================================================
        # 4. Paddle collision
        # ====================================================

        paddle_y = self.HEIGHT - 30

        ball_bottom = (
            self.ball_y
            + self.BALL_SIZE
        )

        paddle_hit = (
            ball_bottom >= paddle_y
            and self.ball_y
            <= paddle_y
            + self.PADDLE_HEIGHT
            and self.ball_x
            + self.BALL_SIZE
            >= self.paddle_x
            and self.ball_x
            <= self.paddle_x
            + self.PADDLE_WIDTH
            and self.ball_vy > 0
        )

        if paddle_hit:

            self.ball_y = (
                paddle_y
                - self.BALL_SIZE
            )

            self.ball_vy *= -1

            paddle_center = (
                self.paddle_x
                + self.PADDLE_WIDTH / 2
            )

            ball_center = (
                self.ball_x
                + self.BALL_SIZE / 2
            )

            offset = (
                ball_center
                - paddle_center
            ) / (
                self.PADDLE_WIDTH / 2
            )

            self.ball_vx = (
                offset * 5
            )

            if abs(self.ball_vx) < 1:

                self.ball_vx = (
                    1
                    if self.ball_vx >= 0
                    else -1
                )

            # Small positive reward for surviving
            # and successfully returning the ball.
            reward += 0.20

        # ====================================================
        # 5. Brick collision
        # ====================================================

        for brick in self.bricks:

            if not brick["alive"]:
                continue

            collision = (
                self.ball_x
                + self.BALL_SIZE
                >= brick["x"]
                and self.ball_x
                <= brick["x"]
                + self.BRICK_WIDTH
                and self.ball_y
                + self.BALL_SIZE
                >= brick["y"]
                and self.ball_y
                <= brick["y"]
                + self.BRICK_HEIGHT
            )

            if collision:

                brick["alive"] = False

                self.ball_vy *= -1

                self.score += 10

                reward += 1.0

                break

        # ====================================================
        # 6. Ball falls below paddle
        # ====================================================

        if self.ball_y > self.HEIGHT:

            self.lives -= 1

            reward -= 1.0

            if self.lives <= 0:

                self.done = True

                reward -= 5.0

            else:

                # Reset ball while preserving
                # bricks and score.

                self.ball_x = (
                    self.WIDTH / 2
                )

                self.ball_y = (
                    self.HEIGHT / 2
                )

                self.ball_vx = (
                    self.rng.choice([-1, 1])
                    * 3
                )

                self.ball_vy = -3

        # ====================================================
        # 7. Win condition
        # ====================================================

        if all(
            not brick["alive"]
            for brick in self.bricks
        ):

            self.done = True

            reward += 5.0

        return (
            self.get_state(),
            reward,
            self.done,
            self.get_info(),
        )

    # ========================================================
    # INFO
    # ========================================================

    def get_info(self):

        return {
            "score": self.score,

            "lives": self.lives,

            "paddle_x": self.paddle_x,

            "ball_x": self.ball_x,

            "ball_y": self.ball_y,

            "bricks_remaining": sum(
                brick["alive"]
                for brick in self.bricks
            ),
        }