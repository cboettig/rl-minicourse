import numpy as np

def simulate(agent, env, timeseries = True):
    # initial conditions
    df = []
    episode_reward = 0
    obs, _ = env.reset()
    
    for t in range(env.Tmax):
      action = agent.predict(obs, deterministic=True)
      obs, reward, done = env.time_step(action)
      episode_reward += reward
      if timeseries:
          df.append([t, episode_reward, action, obs])
      if done:
        break
    return df, episode_reward


# When agent.predict uses RL's -1, 1
def simulate_rl(agent, env, timeseries = True):
    df = []
    episode_reward = 0
    observation, _ = env.reset()
    for t in range(env.Tmax):
      action, _ = agent.predict(observation, deterministic=True)
      observation, reward, terminated, done, info = env.step(action)
      episode_reward += reward

      if timeseries:
          effort = env.effort_units(action)
          obs = env.population_units(observation) # natural units
          df.append([t, episode_reward, *effort, *obs])
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
    

