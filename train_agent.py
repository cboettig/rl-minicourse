from envs import one_fish
from ray.rllib.algorithms import ppo
from ray.tune import register_env
import os
import pandas as pd
import numpy as np
import torch


register_env("one_fish",one_fish.one_fish)

## We could call env directly without this if only  our envs took a env_config dict argument

config = ppo.PPOConfig()
#config = config.resources(num_gpus=torch.cuda.device_count())
config.framework_str="torch"
config.create_env_on_local_worker = True
config.env="one_fish"
agent = config.build()

run_id = "A"
iterations = 500
checkpoint = (f"run{run_id}"+"/checkpoint_000{}".format(iterations))

if not os.path.exists(checkpoint): # train only if no trained agent saved
  for _ in range(iterations):
    print(f"iteration {_}", end = "\r")
    agent.train()
  checkpoint = agent.save("run{run_id}")

agent.restore(checkpoint)

stats = agent.evaluate() # built-in method to evaluate agent on eval env
stats['evaluation']['episode_reward_mean']
stats['evaluation']['episode_reward_max']
stats['evaluation']['episode_reward_min']
stats['evaluation']['episode_len_mean']
