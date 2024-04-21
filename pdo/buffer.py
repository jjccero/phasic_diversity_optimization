import numpy as np

from pbrl.algorithms.ppo.buffer import PGBuffer


class PGBufferRemote(PGBuffer):
    def __init__(self, remote):
        super(PGBufferRemote, self).__init__()
        self.remote = remote

    def clear(self):
        # this method will be called after update
        self.remote.send(('append', np.concatenate(self.observations)))
        self.remote.recv()
        super(PGBufferRemote, self).clear()
