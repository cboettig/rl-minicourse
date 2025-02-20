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
from plotnine import ggplot, aes, geom_line, geom_point


def plot_sim(df, 
             scnema = ["t", "reward",  "effort", "X"],
             variables = ["t", "effort", "X"],
             geom = "line"
            ):
    dfl = (pl.DataFrame(df, schema=scnema, orient="row").
            select(variables).
            unpivot(index = "t")
          )
    if geom == "line":
        return ggplot(dfl, aes("t", "value", color="variable")) + geom_line()
    else:
        return ggplot(dfl, aes("t", "value", color="variable")) + geom_point()



def policy_fn(agent, env, N = 10, geom = "line"):
    df = []
    state_space = np.linspace(env.observation_space.low, env.observation_space.high, N)
    for obs in state_space:
      action, _ = agent.predict(obs, deterministic=True)
      df.append([*obs, *action])
        
    dfl = pl.DataFrame(df, schema=['obs', 'action'], orient = 'row')
    if geom == 'line':
        return ggplot(dfl, aes("obs", "action")) + geom_line()
    return ggplot(dfl, aes("obs", "action")) + geom_point()
    

