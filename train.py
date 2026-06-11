from stable_baselines3 import PPO

from env.carla_env import CarlaEnv

env = CarlaEnv()

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(
    total_timesteps=1000000
)

model.save(
    "models/self_driving_agent"
)