from breakout_env import BreakoutEnv


env = BreakoutEnv()

state = env.reset(seed=42)

print("Initial state shape:", state.shape)
print("Initial info:", env.get_info())

for i in range(20):
    action = i % 3

    state, reward, done, info = env.step(action)

    print(
        f"Step {i:02d} | "
        f"Action: {action} | "
        f"Reward: {reward} | "
        f"Score: {info['score']} | "
        f"Lives: {info['lives']}"
    )

    if done:
        break