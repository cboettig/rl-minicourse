# Initialize saved copy of eval environment:
from envs import one_fish
from ray.rllib.algorithms import ppo, td3
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch
from plotnine import ggplot, aes, geom_line


register_env("one_fish",one_fish.one_fish)

run_id = "ppo"
iterations=100
checkpoint = (f"run_{run_id}"+"/checkpoint_000{}".format(iterations))
config = ppo.PPOConfig()
config.env="one_fish"
agent = config.build()
agent.restore(checkpoint)

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
#df.to_csv(f"data/PPO{iterations}.csv.xz", index = False)


df["state"] = (df.X + 1) * env.bound / 2
df["effort"] = (df.action + 1) / 2
df["escapement"] = (df.state - df.effort * df.state)
ggplot(df, aes("t", "escapement")) + geom_line()
ggplot(df, aes("t", "state")) + geom_line()
episode_reward

