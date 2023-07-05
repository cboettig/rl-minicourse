from envs import one_fish
from ray.rllib.algorithms import ppo, td3, sac
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch

register_env("s3a2",one_fish.one_fish)

#config = sac.SACConfig()

config = ppo.PPOConfig()
config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
config.env="s3a2"
agent = config.build()

run_id = "PPO"
iterations = 40
checkpoint = (f"run_{run_id}"+"/checkpoint_" + str(iterations).zfill(6))

if not os.path.exists(checkpoint): # train only if no trained agent saved
  for _ in range(iterations):
    print(f"iteration {_}", end = "\r")
    agent.train()
  checkpoint = agent.save(f"run_{run_id}")

agent.restore(checkpoint)

stats = agent.evaluate() # built-in method to evaluate agent on eval env
print({"mean": stats['evaluation']['episode_reward_mean'],
       "max": stats['evaluation']['episode_reward_max'],
       "min": stats['evaluation']['episode_reward_min'],
       "mean_len": stats['evaluation']['episode_len_mean']})
