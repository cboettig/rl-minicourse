import polars as pl
import numpy as np
import ray

from envs.one_fish import one_fish
from envs.rescale_env import rescale_env
# rescale_wrapper lets us humans play in natural units
rl_env = one_fish()
env = rescale_env(rl_env)

@ray.remote
def simulate(env, action):
  df = []
  for rep in range(30):
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      # NOTE we use the same action across all t...
      df.append(np.append([t, rep, action, episode_reward], observation))
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated:
        break
  return(df)

actions = np.linspace(0,1,101)

# define parllel loop and execute
parallel = [simulate.remote(env, i) for i in actions]
df = ray.get(parallel)

# convert to polars
cols = ["t", "rep", "action", "reward", "X"]
df2 = pl.DataFrame(np.vstack(df), schema=cols)
max_reward = df2.max().select("reward")
df2.filter(pl.col("reward") == max_reward)


# Surplus production MSY is r*K/4, achieved by F*B_MSY,  (r/2) * (K/2)
rl_env.parameters["r"] / 2
