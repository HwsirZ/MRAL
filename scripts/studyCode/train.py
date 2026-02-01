import torch
import numpy as np
from src.env.make_env import make_env
from src.mapping.occupancy_mapper import OccupancyMapper
from src.rl.policy import Policy
from src.rl.ppo import PPO
from src.viz.recorder import Recorder

env = make_env("/data1/user2025263063/project/MARLObjectNav/configs/studyCode/pointnav_slam.yaml")
mapper = OccupancyMapper()
rec = Recorder()

policy = Policy(obs_dim=2, act_dim=4)
ppo = PPO(policy)

obs = env.reset()
for step in range(300):
    depth = obs["depth"].squeeze()
    pose = obs["agent_pose"]
    mapper.update(depth, pose)

    obs_tensor = torch.tensor(obs["pointgoal_with_gps_compass"], dtype=torch.float32)
    action, _ = ppo.act(obs_tensor)

    obs, _, done, info = env.step(action)

    rec.add(obs["rgb"], depth, mapper.explored)

    if done:
        break

rec.close()
env.close()
