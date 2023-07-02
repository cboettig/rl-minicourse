import gymnasium as gym
from stable_baselines3.common.env_checker import check_env
from envs import one_fish
import numpy as np
import pandas as pd
from plotnine import ggplot, geom_point, aes, geom_line

env = one_fish.one_fish()
check_env(env)

def const_esc(obs, esc=0.5, bound=2):
  state = (obs+1) * bound / 2
  harvest = np.max([state[0] - esc, 0])
  effort = harvest / state[0]
  action = np.array([effort * 2 -1])
  return(action)

df = []
episode_reward = 0
observation, _ = env.reset()

for t in range(env.Tmax):
  #effort = env.parameters["r"]/2
  #action = np.array([effort * 2 -1])
  action = const_esc(observation)
  df.append([t, action[0], episode_reward, observation[0]])
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break


cols = ["t","action", "reward", "X"]
df = pd.DataFrame(df, columns = cols)

df["state"] = (df.X + 1) * env.bound / 2
df["effort"] = (df.action + 1) / 2
df["escapement"] = (df.state - df.effort * df.state)
ggplot(df, aes("t", "escapement")) + geom_line()
ggplot(df, aes("t", "state")) + geom_line()
episode_reward

