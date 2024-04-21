import gym
import numpy as np
import torch

from pbrl.algorithms.ppo import Policy, Runner
from pbrl.common import Logger, update_dict
from pbrl.env import DummyVecEnv
from pdo.worker import evaluate


def worker(
        worker_num: int, worker_id: int, log_dir: str,
        remote,
        policy_config: dict,
        trainer_config: dict,
        env: str,
        env_num: int,
        eval_episode_num: int,
        update_epoch: int,
        seed: int,
        timestep: int,
        buffer_size: int,
        trainer_type
):
    torch.multiprocessing.set_sharing_strategy('file_system')
    seed_worker = seed + worker_id
    torch.manual_seed(seed_worker)
    np.random.seed(seed_worker)
    env_train = DummyVecEnv([lambda: gym.make(env) for _ in range(env_num)])
    env_test = DummyVecEnv([lambda: gym.make(env) for _ in range(eval_episode_num)])
    env_train.seed(seed_worker)
    env_test.seed(seed_worker)

    filename_log = '{}/{}'.format(log_dir, worker_id)
    logger = Logger(filename_log)
    policy = Policy(
        observation_space=env_train.observation_space,
        action_space=env_train.action_space,
        **policy_config,
    )
    trainer = trainer_type(
        remote=remote,
        policy=policy,
        **trainer_config
    )
    runner_train = Runner(env=env_train)
    runner_test = Runner(env=env_test)

    info = dict()
    evaluate(trainer, policy, runner_test, eval_episode_num, info, logger, remote)

    while trainer.timestep < timestep:
        train_info = trainer.learn(
            timestep=buffer_size * update_epoch,
            runner_train=runner_train,
            timestep_update=buffer_size
        )
        update_dict(info, train_info)
        evaluate(trainer, policy, runner_test, eval_episode_num, info, logger, remote)

    remote.send(('close', None))
