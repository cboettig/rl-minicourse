from envs import one_fish
from sb3_contrib import TQC
from stable_baselines3 import PPO
import os

env = one_fish.one_fish()

#model = PPO("MlpPolicy", env, verbose=1,  tensorboard_log= os.path.expanduser("~/ray_results/sb3/"))
#model.learn(total_timesteps=200000, log_interval=4)
#model.save("ppo_fish")

model = TQC("MlpPolicy", env, verbose=1,  tensorboard_log="~/ray_results/sb3/")
model.learn(total_timesteps=200000, log_interval=4)
model.save("tqc_fish")
