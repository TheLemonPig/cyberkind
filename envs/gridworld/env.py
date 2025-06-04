import gym
from gym.spaces import Discrete, Box
import numpy as np

class GridWorldEnv:
    def __init__(self, size=5):
        self.size = size
        self.action_space = Discrete(4)
        self.observation_space = Box(low=0, high=1, shape=(size, size), dtype=np.float32)
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        self.done = False
        return self._get_obs()

    def step(self, action):
        if self.done:
            raise RuntimeError("Cannot call step() on a finished environment. Call reset() first.")
        self.agent_pos[0] = min(self.size - 1, max(0, self.agent_pos[0] + (action == 1) - (action == 3)))
        self.agent_pos[1] = min(self.size - 1, max(0, self.agent_pos[1] + (action == 2) - (action == 0)))
        obs = self._get_obs()
        reward = 1.0 if self.agent_pos == [self.size - self.size//2, self.size - self.size//2] else 0.0
        self.done = reward == 1.0
        return obs, reward, self.done, {}

    def _get_obs(self):
        obs = np.zeros((self.size, self.size), dtype=np.float32)
        obs[self.agent_pos[0], self.agent_pos[1]] = 1.0
        return obs