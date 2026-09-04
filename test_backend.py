from breakout_env import BreakoutEnv
from training import training_controller
from evaluate import list_checkpoints


print("=" * 60)
print("ATARI BREAKOUT BACKEND TEST")
print("=" * 60)


# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

print("\n[1] Environment")

env = BreakoutEnv()

state = env.reset(seed=42)

print("  Reset:       OK")
print("  State size: ", len(state))

next_state, reward, done, info = env.step(1)

print("  Step:        OK")
print("  Reward:     ", reward)
print("  Score:      ", info.get("score"))
print("  Lives:      ", info.get("lives"))


# ------------------------------------------------------------
# Training controller
# ------------------------------------------------------------

print("\n[2] Training Controller")

status = training_controller.get_status()

print("  Controller:  OK")
print("  Running:    ", status["running"])
print("  Episodes:   ", status["episode"])


# ------------------------------------------------------------
# Checkpoints
# ------------------------------------------------------------

print("\n[3] Checkpoints")

checkpoints = list_checkpoints()

print("  Checkpoints: OK")

if checkpoints:
    for checkpoint in checkpoints:
        print("   -", checkpoint)
else:
    print("   - No checkpoints yet")


# ------------------------------------------------------------
# Game state
# ------------------------------------------------------------

print("\n[4] Game State")

game = training_controller._get_game_state()

required = [
    "paddle_x",
    "ball_x",
    "ball_y",
    "ball_vx",
    "ball_vy",
    "bricks"
]

for key in required:

    if key in game:
        print(
            f"  {key:<15} OK"
        )
    else:
        print(
            f"  {key:<15} MISSING"
        )


# ------------------------------------------------------------
# Result
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("BACKEND TEST COMPLETE")
print("=" * 60)