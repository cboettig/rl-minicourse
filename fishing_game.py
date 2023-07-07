import numpy as np
from envs.one_fish import one_fish
from envs.rescale_env import rescale_env
from stable_baselines3.common.env_checker import check_env

# RL envs work in transformed units
# rescale_wrapper lets us humans play in natural units
rl_env = one_fish()
env = rescale_env(rl_env)

# verify custom environment is correctly implemented
check_env(rl_env)

# Consider differnt starting population sizes
env.rl_env.initial_pop[0] = 0.9

df = []
episode_reward = 0
observation, _ = env.reset()

# Try longer management intervals by increasing range here:
for t in range(10):
    status = ("t: " + str(t) +
              ", Stock: " + format(observation[0],  '.3f') +
              ", profits: " + format(episode_reward, '.2f'))
    txt = input(status + ". Set harvest effort [0,1]:  ")
    action = np.float32(txt)
    df.append([t, episode_reward, action, observation[0]])
    observation, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    if terminated or truncated:
        break

print("Final score: " + format(episode_reward, '.4f'))


# optional plotting code
import polars as pl
from plotnine import ggplot, aes, geom_line
cols = ["t", "reward", "action", "state"]

dfl = (pl.DataFrame(df, schema=cols).
        select(["t", "action", "state"]).
        melt("t")
      )
ggplot(dfl, aes("t", "value", color="variable")) + geom_line()
