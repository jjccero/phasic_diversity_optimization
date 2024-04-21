import numpy as np
import torch

from pbrl.algorithms.ppo import PPO
from pbrl.common.map import auto_map


class DsePPO(PPO):
    def __init__(self, remote, **kwargs):
        super(DsePPO, self).__init__(**kwargs)
        self.remote = remote
        self.remote.send(('init', self.policy))
        self.remote.recv()
        self.div_coef = 0

    def before_dse(self, observations: np.ndarray):
        self.remote.send(('dse', observations))

    def after_dse(self, module: torch.nn.Module):
        grad = self.remote.recv()
        for k, v in module.named_parameters():
            if k in grad:
                v.grad += grad[k].to(self.policy.device)

    def train_pi_vf(self, loss_info):
        for mini_batch in self.buffer.generator(self.batch_size, self.chunk_len, self.ks):
            self.before_dse(mini_batch['observations'])

            mini_batch['observations'] = self.policy.normalize_observations(mini_batch['observations'])
            mini_batch = auto_map(self.policy.n2t, mini_batch)
            observations = mini_batch['observations']
            actions = mini_batch['actions']
            advantages = mini_batch['advantages']
            log_probs_old = mini_batch['log_probs_old']
            returns = mini_batch['returns']
            dones = None
            if self.policy.rnn:
                dones = mini_batch['dones']
            policy_loss, entropy_loss = self.actor_loss(observations, actions, advantages, log_probs_old, dones)
            value_loss = self.critic_loss(observations, returns, dones)
            loss = (1 - self.div_coef) * (self.vf_coef * value_loss - policy_loss - self.entropy_coef * entropy_loss)
            self.optimizer.zero_grad()
            loss.backward()

            self.after_dse(self.policy.actor)

            if self.grad_norm:
                torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_norm)
                torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_norm)
            self.optimizer.step()
            loss_info['value'].append(value_loss.item())
            loss_info['policy'].append(policy_loss.item())
            loss_info['entropy'].append(entropy_loss.item())

    def update(self):
        self.remote.send(('update', self.policy.rms_obs))
        self.div_coef = self.remote.recv()
        return super(DsePPO, self).update()
