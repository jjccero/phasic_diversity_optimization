import numpy as np

from dse.bandit import UCB, TS
from map_elites.archive import Solution, Archive
from pbrl.algorithms.ppo import Policy
from pbrl.common import update_dict
from pbrl.common.map import auto_map
from pbrl.pbt import PBT
from pdo.dse import DSE


class DSEBandits(PBT):
    def __init__(
            self,
            archive: Archive,
            arms,
            bandit,
            sample_size,
            start_exploit_timestep: int,
            dse: DSE,
            **kwargs
    ):
        super(DSEBandits, self).__init__(**kwargs)
        self.archive = archive
        self.start_exploit_timestep = start_exploit_timestep
        self.bandit = None
        self.sample_size = sample_size
        self.dse = dse
        self.div_coef = 0.0
        if arms is not None:
            if bandit == 'ts':
                self.bandit = TS(arms=arms)
            elif bandit == 'ucb':
                self.bandit = UCB(arms=arms)

    def seed(self, seed):
        super(DSEBandits, self).seed(seed)
        self.archive.random_state = seed

    def run(self):
        policies: List[Policy] = []
        last_max_return = None
        cmd = self.recv()
        if cmd == 'init':
            for policy in self.objs:
                policy.actor.train()
                policies.append(policy)
            for remote in self.remotes:
                remote.send(None)
        info = {
            'diversify/det': [],
            'diversify/lambda': [],
        }
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

                current_solutions_info = dict(
                    max_fitness=np.max(returns),
                    mean_fitness=np.mean(returns),
                    min_fitness=np.min(returns),
                    med_fitness=np.median(returns),
                    increment=increment
                )
                archive_info = dict(
                    max_fitness=self.archive.max_fitness,
                    coverage=self.archive.coverage
                )
                max_return = np.max(returns)
                if last_max_return is not None:
                    increment = max_return - last_max_return
                    self.bandit.update(increment > 0)
                    self.div_coef = self.bandit.sample()
                last_max_return = max_return

                info['diversify/lambda'].append(self.div_coef)
                update_dict(info, current_solutions_info, 'current/')
                update_dict(info, archive_info, 'archive/')
                self.logger.log(timestep, info)
            elif cmd == 'dse':
                observations_all = []
                for observations in self.objs:
                    observations_all.append(observations)
                observations_all = np.concatenate(observations_all)
                sampled_indices = self.random_state.choice(observations_all.shape[0], self.sample_size)
                sampled_observations = observations_all[sampled_indices]

                def get_dist(policy):
                    dists, _ = policy.actor.forward(
                        auto_map(
                            policy.n2t,
                            policy.normalize_observations(sampled_observations)
                        )
                    )
                    return dists

                dists_all = auto_map(get_dist, tuple(policies))
                det = self.dse.forward(dists_all).exp()
                for policy in policies:
                    policy.actor.zero_grad()
                (-self.div_coef * det).backward()
                det = det.item()
                info['diversify/det'].append(det)
                for policy, remote in zip(policies, self.remotes):
                    grad = {}
                    for k, v in policy.actor.named_parameters():
                        if v.grad is not None:
                            grad[k] = v.grad.cpu()
                    remote.send(grad)
            elif cmd == 'update':
                for worker_id in range(self.worker_num):
                    rms_obs = self.objs[worker_id]
                    policies[worker_id].rms_obs = rms_obs
                    self.remotes[worker_id].send(self.div_coef)
            elif cmd == 'close':
                self.logger = None
                self.archive.save()
                self.close()
                break
