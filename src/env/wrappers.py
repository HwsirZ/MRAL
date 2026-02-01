import numpy as np

class SLAMWrapper:
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return self._process_obs(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._process_obs(obs), reward, done, info

    def _process_obs(self, obs):
        state = self.env.sim.get_agent_state()
        pose = np.array([
            state.position[0],
            state.position[2],
            state.rotation.to_euler_angles()[1]
        ])
        obs["agent_pose"] = pose
        return obs

    def close(self):
        self.env.close()
