import gymnasium as gym
from stable_baselines3 import PPO

env = gym.make(
    "CarRacing-v3",
    render_mode="human"
)

model = PPO.load("models/car_model")

obs, info = env.reset()

while True:
    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, info = env.reset()