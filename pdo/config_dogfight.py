import torch

policy_config = dict(
    hidden_sizes=[128, 128],
    activation=torch.nn.Tanh,
    gamma=0.999,
    deterministic=True
)

trainer_config = dict(
    batch_size=256,
    eps=0.2,
    gamma=0.999,
    gae_lambda=0.95,
    repeat=10,
    lr=3e-4,
    entropy_coef=0.05,
    recompute_adv=True
)
