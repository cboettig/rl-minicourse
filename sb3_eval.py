import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO, A2C
from sb3_contrib import TQC
from stable_baselines3.common.env_checker import check_env
import pandas as pd
from plotnine import ggplot, aes, geom_line
from envs.one_fish import one_fish
from envs.three_fish import three_fish
from envs.s3a2 import s3a2

agent = PPO.load("ppo_fish")
#agent = TQC.load("tqc_s3a2")
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

#cols = ["t","reward", "action",  "X", "Y", "Z"]
cols = ["t","reward", "action",  "X"]
df = pd.DataFrame(df, columns = cols)

df["escapement"] = (df.X - df.action * df.X)
ggplot(df, aes("t", "escapement")) + geom_line()
ggplot(df, aes("t", "action")) + geom_line()
ggplot(df, aes("t", "reward")) + geom_line()


episode_reward

