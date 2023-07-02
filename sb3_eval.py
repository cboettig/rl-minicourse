import gymnasium as gym
from stable_baselines3 import PPO, A2C, TQC
from stable_baselines3.common.env_checker import check_env
import pandas as pd
from plotnine import ggplot, aes, geom_line
from envs import one_fish

#agent = PPO.load("ppo_fish")
agent = TQC.load("tqc_fish")
env = one_fish.one_fish()

df = []
episode_reward = 0
observation, _ = env.reset()
for t in range(env.Tmax):
  action, _ = agent.predict(observation, deterministic=True)
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

