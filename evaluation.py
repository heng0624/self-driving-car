from stable_baselines3 import PPO

from env.carla_env import CarlaEnv

env = CarlaEnv()

model = PPO.load(
    "models/self_driving_agent"
)

obs, info = env.reset()

while True:

    action, _ = model.predict(
        obs,
        deterministic=True
    )

    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:

        obs, info = env.reset()