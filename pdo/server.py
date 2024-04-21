import copy

import gym
import numpy as np
import torch

from map_elites.archive import Solution, Archive
from pbrl.algorithms.ppo import Policy, Runner
from pbrl.common import update_dict
from pbrl.common.map import auto_map
from pbrl.env import DummyVecEnv
from pbrl.pbt import PBT
from pdo.dse import DSE


class PDO(PBT):
    def __init__(
            self,
            archive: Archive,
            pdo_batch_size: int,
            pdo_epoch: int,
            lr_pdo: float,
            start_exploit_timestep: int,
            dse: DSE,
            **kwargs
    ):
        super(PDO, self).__init__(**kwargs)
        self.archive = archive
        self.pdo_epoch = pdo_epoch
        self.pdo_batch_size = pdo_batch_size
        self.pdo_buffer = None
        self.start_exploit_timestep = start_exploit_timestep

        self.dse = dse
        self.policies = None
        self.runner_test = None
        self.eval_episode_num = kwargs['worker_params']['eval_episode_num']
        self.lr_pdo = lr_pdo
        if self.pdo_epoch > 0:
            env_test = DummyVecEnv([lambda: gym.make(
                kwargs['worker_params']['env']
            ) for _ in range(self.eval_episode_num)])
            self.runner_test = Runner(env=env_test)
            policy_config = kwargs['worker_params']['policy_config']
            self.policies = [
                Policy(
                    observation_space=env_test.observation_space,
                    action_space=env_test.action_space,
                    critic_type=None,
                    **policy_config
                ) for _ in range(self.worker_num)
            ]

    def seed(self, seed):
        super(PDO, self).seed(seed)
        self.archive.random_state = seed
        if self.runner_test is not None:
            self.runner_test.env.seed(seed)

    def run(self):
        info = {}
        while True:
            cmd = self.recv()
            if cmd == 'select':
                returns = []
                timestep = -1
                increment = 0
                for worker_id in range(self.worker_num):
                    (timestep, x, fitness, bd) = self.objs[worker_id]
                    data = self.datas[worker_id]
                    data.exploit = False
                    individual = Solution(x, bd, fitness)
                    returns.append(fitness)
                    if self.archive.add_to_archive(individual):
                        increment += 1

                if self.archive.coverage >= self.worker_num and timestep >= self.start_exploit_timestep:
                    min_fitness_worker = np.argmin(returns)
                    solution = self.archive.random_solutions(1)[0]
                    data = self.datas[min_fitness_worker]
                    data.exploit = self.exploit
                    data.y = solution.x.copy()
                    self.send()
                    if self.pdo_epoch > 0:
                        diversify_info = self.diversify()
                        update_dict(info, diversify_info, 'diversify/')
                else:
                    self.send()

                current_solutions_info = dict(
                    max_fitness=np.max(returns),
                    mean_fitness=np.mean(returns),
                    min_fitness=np.min(returns),
                    med_fitness=np.median(returns),
                    increment=increment
                )
                archive_info = self.archive.info()

                update_dict(info, current_solutions_info, 'current/')
                update_dict(info, archive_info, 'archive/')
                self.logger.log(timestep, info)

            elif cmd == 'append':
                self.pdo_buffer = np.concatenate(self.objs)
                for worker_id in range(self.worker_num):
                    self.remotes[worker_id].send(None)
            elif cmd == 'close':
                self.archive.save()
                self.close()
                break

    def diversify(self):
        diversify_info = dict(
            det=[]
        )
        solutions = self.archive.top_solutions(self.worker_num)
        for policy, solution in zip(self.policies, solutions):
            policy.actor.load_state_dict(solution.x['actor'])
            if policy.obs_norm:
                policy.rms_obs.load(solution.x['rms_obs'])
            policy.actor.train()
        optimizer = torch.optim.Adam(
            tuple({'params': policy.actor.parameters()} for policy in self.policies),
            lr=self.lr_pdo
        )
        for _ in range(self.pdo_epoch):
            sample = self.pdo_buffer[
                self.random_state.randint(self.pdo_buffer.shape[0], size=self.pdo_batch_size)
            ]
            embeddings = []
            for policy in self.policies:
                dists, _ = policy.actor.forward(
                    auto_map(
                        policy.n2t,
                        policy.normalize_observations(sample)
                    )
                )
                embeddings.append(dists)
            det = self.dse.forward(embeddings).exp()
            optimizer.zero_grad()
            (-det).backward()
            optimizer.step()
            diversify_info['det'].append(det.item())

        increment = 0
        for policy, solution in zip(self.policies, solutions):
            policy.actor.eval()
            self.runner_test.reset()
            eval_info = self.runner_test.run(policy=policy, episode_num=self.eval_episode_num)
            fitness = np.mean(eval_info['reward'])
            bd = None
            if 'bc' in eval_info['info'][0]:
                bd = np.mean(np.array([a['bc'] for a in eval_info['info']]), axis=0)
            x = copy.deepcopy(solution.x)
            x['actor'] = policy.actor.state_dict()
            new_solution = Solution(x, bd, fitness)
            if self.archive.add_to_archive(new_solution):
                increment += 1
        diversify_info['increment'] = increment
        return diversify_info
