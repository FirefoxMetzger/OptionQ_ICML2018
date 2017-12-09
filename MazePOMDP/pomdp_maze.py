from gym.envs.toy_text.discrete import DiscreteEnv
import numpy as np
from gym.envs.registration import register
from gym.core import Env
from gym.spaces.discrete import Discrete
import random
import numpy as np

class POMDPMaze(Env):
    def __init__(self):

        self.T = np.array([
                [3, 1,   5,   6,   8,   10,  6,   7,   8,   9,   10],
                [0, 1,   2,   3,   4,   5,   7,   8,   9,   10,  10],
                [0, 1,   2,   0,   1,   2,   3,   7,   4,   9,   5],
                [0, 1,   2,   3,   4,   5,   6,   6,   7,   8,   9]
                ])
        self.observations = [2,6,2,4,4,4,0,5,1,5,3]

        self.s = 0
        self.rng = random.Random(42)

    def _step(self, action):
        if self.s == 4 and action == 2:
            reward = 8.0
            done = True
        else:
            reward = -1.0
            done = False
            
        self.s = self.T[action, self.s]
            
        return self.observations[self.s], reward, done, ""

    def _reset(self):
        #valid_states = [0,2,3,4,5,6,7,8,9,10]
        valid_states = [0,2]
        self.s = self.rng.sample(valid_states,1)[0]
        return self.observations[self.s]

    def _close(self):
        return

    def _seed(self, seed):
        self.rng.seed(seed)
