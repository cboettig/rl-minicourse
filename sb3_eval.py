import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs import one_fish

env = one_fish.one_fish()
check_env(env)


agent = PPO.load("ppo_fish")

df = []
episode_reward = 0
observation, _ = env.reset()
for t in range(100):
  action, _ = agent.predict(observation)
  df.append([t, action[0], episode_reward, observation[0]])
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break



import pandas as pd
cols = ["t","action", "reward", "X"]
df = pd.DataFrame(df, columns = cols)
df.to_csv(f"data/PPO{iterations}.csv.xz", index = False)

## Plots ## 
import plotnine
from plotnine import ggplot, geom_point, aes, geom_line, facet_wrap, geom_path
## Timeseries

df["escapement"] = (df.X - df.action*df.X)
ggplot(df, aes("t", "X")) + geom_line()
