import os
from sb3_contrib import TQC
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from envs import three_fish
env = three_fish.three_fish
vec_env = make_vec_env(env, n_envs=4)
log = os.path.expanduser("~/ray_results/sb3/")

model = TQC("MlpPolicy", vec_env, verbose=0, tensorboard_log=log)
model.learn(total_timesteps=300_000, progress_bar=True)
model.save("tqc_3fish")
