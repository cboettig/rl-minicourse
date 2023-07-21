import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import TQC, ARS
from envs.one_fish import one_fish

agent = ARS.load("ars_fish")
env = one_fish()

df = []
episode_reward = 0
observation, _ = env.reset()

for t in range(env.Tmax):
  action, _ = agent.predict(observation, deterministic=True)
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward  
  obs = (observation + 1 ) / 2 # natural units
  effort = (action + 1)/2      # natural units
  df.append([t, episode_reward, *effort, *obs])
  if terminated or done:
    break

episode_reward


# optional plotting code
import polars as pl
from plotnine import ggplot, aes, geom_line
cols = ["t", "reward", "action", "state"]

dfl = (pl.DataFrame(df, schema=cols).
        select(["t", "action", "state"]).
        melt("t")
      )
ggplot(dfl, aes("t", "value", color="variable")) + geom_line()
