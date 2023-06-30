# Initialize saved copy of eval environment:
from envs import one_fish
#from ray.rllib.algorithms import ppo
from ray.rllib.algorithms import td3
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch


config = ppo.PPOConfig()
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
config.env="one_fish"
agent = config.build()
config = agent.evaluation_config.env_config

config.update({'seed': 42})
env = agent.env_creator(config)

df = []
episode_reward = 0
observation, _ = env.reset()
for t in range(env.Tmax):
  action = agent.compute_single_action(observation)
  df.append([t, action[0], episode_reward, observation[0]])
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break


cols = ["t","action", "reward", "X"]
df = pd.DataFrame(df, columns = cols)
df.to_csv(f"data/PPO{iterations}.csv.xz", index = False)

## Plots ## 
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
## Timeseries

df["escapement"] = (df.X - df.action*df.X)
ggplot(df, aes("t", "escapement")) + geom_line()
