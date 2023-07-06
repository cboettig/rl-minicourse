import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import TQC, ARS
from envs.one_fish import one_fish
from envs.three_fish import three_fish
from envs.s3a2 import s3a2

agent = ARS.load("ars_fish")
env = one_fish()

df = []
episode_reward = 0
observation, _ = env.reset()
for t in range(env.Tmax):
  action, _ = agent.predict(observation, deterministic=True)
  
  obs = (observation + 1 ) / 2 # natural units
  effort = (action[0] + 1)/2
  df.append(np.append([t, episode_reward, effort], obs))
  
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break

episode_reward


# optional plotting code
import polars as pl
from plotnine import ggplot, aes, geom_line
cols = ["t", "action", "reward", "state"]

dfl = (pl.DataFrame(df, schema=cols).
        select(["t", "action", "state"]).
        melt("t")
      )
ggplot(dfl, aes("t", "value", color="variable")) + geom_line()
