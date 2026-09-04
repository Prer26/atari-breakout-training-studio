import pygame
from breakout_env import BreakoutEnv


pygame.init()

env = BreakoutEnv()

screen = pygame.display.set_mode(
    (env.WIDTH, env.HEIGHT)
)

pygame.display.set_caption("Atari Breakout")

clock = pygame.time.Clock()

font = pygame.font.SysFont(None, 28)

state = env.reset(seed=42)

running = True

while running:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # -------------------------
    # Human controls
    # -------------------------

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        action = 0

    elif keys[pygame.K_RIGHT]:
        action = 2

    else:
        action = 1

    state, reward, done, info = env.step(action)

    # -------------------------
    # Draw background
    # -------------------------

    screen.fill((15, 15, 25))

    # -------------------------
    # Draw bricks
    # -------------------------

    for brick in env.bricks:

        if not brick["alive"]:
            continue

        rect = pygame.Rect(
            brick["x"],
            brick["y"],
            env.BRICK_WIDTH,
            env.BRICK_HEIGHT
        )

        pygame.draw.rect(
            screen,
            (220, 80, 80),
            rect
        )

    # -------------------------
    # Draw paddle
    # -------------------------

    paddle_y = env.HEIGHT - 30

    paddle = pygame.Rect(
        env.paddle_x,
        paddle_y,
        env.PADDLE_WIDTH,
        env.PADDLE_HEIGHT
    )

    pygame.draw.rect(
        screen,
        (240, 240, 240),
        paddle
    )

    # -------------------------
    # Draw ball
    # -------------------------

    ball = pygame.Rect(
        env.ball_x,
        env.ball_y,
        env.BALL_SIZE,
        env.BALL_SIZE
    )

    pygame.draw.ellipse(
        screen,
        (240, 240, 240),
        ball
    )

    # -------------------------
    # HUD
    # -------------------------

    score_text = font.render(
        f"Score: {info['score']}",
        True,
        (240, 240, 240)
    )

    lives_text = font.render(
        f"Lives: {info['lives']}",
        True,
        (240, 240, 240)
    )

    screen.blit(score_text, (10, 5))
    screen.blit(lives_text, (300, 5))

    # -------------------------
    # Game over
    # -------------------------

    if done:

        message = font.render(
            "GAME OVER",
            True,
            (255, 255, 255)
        )

        screen.blit(
            message,
            (
                env.WIDTH // 2 - 60,
                env.HEIGHT // 2
            )
        )

    pygame.display.flip()

    clock.tick(60)


pygame.quit()