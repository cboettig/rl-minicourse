import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
import pandas as pd
import numpy as np
from plotnine import ggplot, aes, geom_line
from envs import one_fish, rescale_env

# RL envs work in transformed units
# rescale_wrapper lets us humans play in natural units
rl_env = one_fish.one_fish()
env = rescale_env.rescale_env(rl_env)

df = []
episode_reward = 0
observation, _ = env.reset()
for t in range(100):
  status = ("t: " + str(t) + 
            ", Stock: " + format(observation[0],  '.3f') + 
            ", profits: "+ format(episode_reward, '.2f'))
  txt = input(status + ". Set harvest effort [0,1]:  ")
  action = np.float32(txt)
  df.append([t, action, episode_reward, observation[0]])
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break


cols = ["t","action", "reward", "state"]
df = pd.DataFrame(df, columns = cols)

ggplot(df, aes("t", "action")) + geom_line()
ggplot(df, aes("t", "state")) + geom_line()
ggplot(df, aes("t", "reward")) + geom_line()

episode_reward

