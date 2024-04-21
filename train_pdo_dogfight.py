import argparse
import time

import numpy as np
import torch

from map_elites.archive import MAPElitesArchive
from pbrl.algorithms.ppo import PPO
from pdo.config_dogfight import policy_config, trainer_config
from pdo.dse import DSE
from pdo.server import PDO
from pdo.worker import worker


def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='QDDogfight-v0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--worker_num', type=int, default=10)
    parser.add_argument('--timestep', type=int, default=10000000)
    parser.add_argument('--start_exploit_timestep', type=int, default=600000)
    parser.add_argument('--pdo_epoch', type=int, default=10)

    parser.add_argument('--env_num', type=int, default=16)
    parser.add_argument('--buffer_size', type=int, default=4096)
    parser.add_argument('--eval_episode_num', type=int, default=10)
    parser.add_argument('--update_epoch', type=int, default=25)

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)

    policy_config['device'] = torch.device('cuda:{}'.format(args.cuda))
    # policy_config['device'] = torch.device('cpu')

    log_dir = 'result/pdo-d-{}/{}/{}-{}'.format(args.pdo_epoch, args.env, args.seed, int(time.time()))
    pbt = PDO(
        archive=MAPElitesArchive((10, 10), log_dir + '/archive.pkl'),
        pdo_epoch=args.pdo_epoch,
        pdo_batch_size=256,
        lr_pdo=3e-4,
        dse=DSE(beta=0.99),
        start_exploit_timestep=args.start_exploit_timestep,
        worker_num=args.worker_num,
        worker_fn=worker_dogfight,
        log_dir=log_dir,
        worker_params=dict(
            policy_config=policy_config,
            trainer_config=trainer_config,
            env=args.env,
            env_num=args.env_num,
            eval_episode_num=args.eval_episode_num,
            update_epoch=args.update_epoch,
            seed=args.seed,
            timestep=args.timestep,
            buffer_size=args.buffer_size,
            trainer_type=PPO
        ),
        exploit=True,
    )
    pbt.seed(seed)
    pbt.run()


def worker_dogfight(*args, **kwargs):
    __import__('air_combat')
    worker(*args, **kwargs)


if __name__ == '__main__':
    main()
