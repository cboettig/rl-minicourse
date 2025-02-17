import numpy as np
import gymnasium as gym


# Convert an environment to take actions in original units and
# report observations in original (natural) units.

class rescale_env(gym.Env):
    def __init__(self, env):
      self.rl_env = env
      self.Tmax = env.Tmax
    def reset(self, *, seed=None, options=None):
      state, info = self.rl_env.reset()
      return self.rl_env.population(), info
    def step(self, effort):
      action = 2 * effort - 1
      observation, reward, terminated, done, info = self.rl_env.step(action)
      obs = np.array((observation + 1) * self.rl_env.bound / 2, dtype=np.float32)
      obs = np.clip(obs, 0, np.Inf)
      return(obs, reward, terminated, done, info)
      

