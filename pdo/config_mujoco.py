import torch

from pbrl.algorithms.ppg import AuxActor

policy_config = dict(
    hidden_sizes=[64, 64],
    activation=torch.nn.Tanh,
    obs_norm=True,
    reward_norm=True,
    gamma=0.99,
    deterministic=True,
    actor_type=AuxActor
)

trainer_config = dict(
    batch_size=64,
    eps=0.2,
    gamma=0.99,
    gae_lambda=0.95,
    lr=3e-4,
    grad_norm=0.5,
    entropy_coef=0.0,
    aux_batch_size=256,
    lr_aux=3e-4,
    beta_clone=1.0,
    n_pi=10,
    epoch_pi=4,
    epoch_vf=4,
    epoch_aux=6
)
