import numpy as np

# A simple agent
class fixed_action:
    def __init__(self, effort):
        self.effort = np.array(effort, dtype=np.float32)

    def predict(self, observation, **kwargs):
        action = self.effort * 2 - 1
        return action, {}


def simulate(agent, env, timeseries = True):
    df = []
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      obs = env.population_units(observation) # natural units
      action, _ = agent.predict(obs, deterministic=True)
      if timeseries:
          df.append([t, episode_reward, *action, *obs])
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated or done:
        break
    
    return df, episode_reward



import polars as pl
from plotnine import ggplot, aes, geom_line
cols = ["t", "reward",  "effort", "X"]

def plot_sim(df):
    dfl = (pl.DataFrame(df, schema=cols).
            select(["t", "effort", "X"]).
            melt("t")
          )
    return ggplot(dfl, aes("t", "value", color="variable")) + geom_line()

plot_sim(df)



def policy_fn(agent, env, timeseries = True):
    df = []
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      obs = env.population_units(observation) # natural units
      action, _ = agent.predict(obs, deterministic=True)
      if timeseries:
          df.append([t, episode_reward, *action, *obs])
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward
      if terminated or done:
        break
    
    return df, episode_reward

