from envs.one_fish import one_fish
from envs.rescale_env import rescale_env
import numpy as np

# RL envs work in transformed units
rl_env = one_fish()
env = rescale_env(rl_env)

def const_esc(obs, esc=0.5):
  harvest = np.max([obs[0] - esc, 0])
  effort = harvest / obs
  return(effort)

# consider an alternative initial state
env.rl_env.initial_pop = np.array([0.1], dtype=np.float32)
df = []
episode_reward = 0
observation, _ = env.reset()

for t in range(env.Tmax):
  action = const_esc(observation)
  df.append([t, episode_reward, action[0], observation[0]])
  observation, reward, terminated, done, info = env.step(action)
  episode_reward += reward
  if terminated:
    break

# optional plotting code
import polars as pl
from plotnine import ggplot, aes, geom_line
cols = ["t", "reward", "action", "state"]
df2 = pl.DataFrame(df, schema=cols)
df2.with_columns(escapement = pl.col("state") - pl.col("action") * pl.col("state"))
dfl = df2.melt("t")
ggplot(dfl, aes("t", "value", color="variable")) + geom_line()

