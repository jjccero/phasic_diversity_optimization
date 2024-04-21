import numpy as np
from gym.envs.mujoco import HumanoidEnv


class QDHumanoidEnv(HumanoidEnv):
    def __init__(self):
        self.T = 0
        super(QDHumanoidEnv, self).__init__()

    def reset(self):
        self.T = 0
        return super(QDHumanoidEnv, self).reset()

    def step(self, a):
        observation, reward, done, info = super(QDHumanoidEnv, self).step(a)
        self.T += 1
        if done or self.T >= 1000:
            done = True
            mass = np.expand_dims(self.model.body_mass, 1)
            xpos = self.sim.data.xipos
            pos = (np.sum(mass * xpos, 0) / np.sum(mass))[0].copy()
            info['bc'] = np.array((pos / 16, self.T / 1000)).clip(0.0, 1.0)
        return observation, reward, done, info
