import gymnasium as gym
import numpy as np

class CarlaEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.action_space = gym.spaces.Box(
            low=np.array([-1,0,0]),
            high=np.array([1,1,1]),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(120,160,3),
            dtype=np.uint8
        )

    def reset(
        self,
        seed=None,
        options=None
    ):

        observation = np.zeros(
            (120,160,3),
            dtype=np.uint8
        )

        return observation, {}

    def step(self, action):

        observation = np.zeros(
            (120,160,3),
            dtype=np.uint8
        )

        reward = 0

        terminated = False

        truncated = False

        return (
            observation,
            reward,
            terminated,
            truncated,
            {}
        )