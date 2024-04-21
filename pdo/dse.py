import math

import torch
import torch.nn as nn
from torch.distributions import kl_divergence, Categorical


class DSE(nn.Module):
    def __init__(self, kernel='wd', beta=1.0, alpha=None):
        super(DSE, self).__init__()
        self.beta = beta
        self.kernel = kernel
        self.alpha = alpha

    def l2rbf(self, m_actions):
        actions = torch.stack(m_actions)
        x1 = actions.unsqueeze(0).repeat_interleave(actions.shape[0], 0)
        x2 = actions.unsqueeze(1).repeat_interleave(actions.shape[0], 1)
        d2 = torch.square(x1 - x2)
        # l2 = torch.var(actions, dim=0).detach() + 1e-8
        l2 = 1.0
        return (d2 / (2 * l2)).mean(-1)

    def wd2(self, m_dists):
        means = torch.stack([combined_dist.mean for combined_dist in m_dists])
        a = means.unsqueeze(0).repeat_interleave(means.shape[0], 0)
        b = means.unsqueeze(1).repeat_interleave(means.shape[0], 1)
        d2 = torch.square(a - b)
        l2 = torch.var(means, dim=0).detach() + 1e-8
        if self.alpha is not None:
            sigmas = torch.stack([combined_dist.scale for combined_dist in m_dists])
            a1 = sigmas.unsqueeze(0).repeat_interleave(means.shape[0], 0)
            b1 = sigmas.unsqueeze(1).repeat_interleave(means.shape[0], 1)
            d2 += self.alpha * torch.square(a1 - b1)
            l2 += self.alpha ** 2 * torch.var(sigmas, dim=0).detach()
        return (d2 / (2 * l2)).mean(-1)

    def js(self, m_dists):
        probs = torch.stack([combined_dist.probs for combined_dist in m_dists])
        p = probs.unsqueeze(0).repeat_interleave(probs.shape[0], 0)
        q = probs.unsqueeze(1).repeat_interleave(probs.shape[0], 1)
        m = Categorical(probs=(p + q) / 2.0)
        return 0.5 * (kl_divergence(Categorical(probs=p), m) + kl_divergence(Categorical(probs=q), m))

    def tv(self, m_dists):
        probs = torch.stack([combined_dist.probs for combined_dist in m_dists])
        p = probs.unsqueeze(0).repeat_interleave(probs.shape[0], 0)
        q = probs.unsqueeze(1).repeat_interleave(probs.shape[0], 1)
        return 0.5 * torch.abs(p - q).sum(-1)

    def he(self, m_dists):
        logits = torch.stack([combined_dist.logits for combined_dist in m_dists])
        logp = logits.unsqueeze(0).repeat_interleave(logits.shape[0], 0)
        logq = logits.unsqueeze(1).repeat_interleave(logits.shape[0], 1)
        return torch.exp(0.5 * (logp + logq)).sum(-1)

    def forward(self, m_dists):
        m = len(m_dists)
        if self.kernel == 'wd':
            d = self.wd2(m_dists)
            if len(d.shape) == 4:
                d = d.mean(-1)
            K = (-d).exp()
        elif self.kernel == 'js':
            K = 1 - self.js(m_dists) / math.log(2)
            if len(K.shape) == 4:
                K = K.prod(-1) ** (1 / K.shape[-1])
        elif self.kernel == 'tv':
            K = 1 - self.tv(m_dists)
            if len(K.shape) == 4:
                K = K.prod(-1) ** (1 / K.shape[-1])
        elif self.kernel == 'he':
            K = self.he(m_dists)
            if len(K.shape) == 4:
                K = K.prod(-1) ** (1 / K.shape[-1])
        elif self.kernel == 'l2':
            d = self.l2rbf(m_dists)
            if len(d.shape) == 4:
                d = d.mean(-1)
            K = (-d).exp()
        else:
            raise NotImplementedError
        K = K.mean(-1)
        K_ = self.beta * K + (1 - self.beta) * torch.eye(m, device=K.device)
        L = torch.linalg.cholesky(K_)
        log_det = 2 * torch.log(torch.diag(L)).sum()
        return log_det
