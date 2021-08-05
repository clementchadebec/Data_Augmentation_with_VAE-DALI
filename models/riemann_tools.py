import functools
import warnings
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad


class Exponential_map(object):
    def __init__(self, latent_dim=2, device='cpu'):
        self.starting_pos = None
        self.starting_velo = None
        self.metric_inv = None
        self.latent_dim = latent_dim
        self.device = device

    def _velocity_to_momentum(self, v, p=None):
        if p is None:
            return torch.inverse(self.metric_inv(self.starting_pos)) @ v.unsqueeze(-1)

        else:
            return torch.inverse(self.metric_inv(p)) @ v.unsqueeze(-1)

    def _momentum_to_velocity(self, q, p=None):
        if p is None:
            return self.metric_inv(self.starting_pos) @ q.unsqueeze(-1)

        else:
            return self.metric_inv(p) @ q.unsqueeze(-1)

    def _rk_step(self, p, q, dp, dt):
        mean_p = p.unsqueeze(-1) + 0.5 * dt * self.metric_inv(p) @ q.unsqueeze(-1)
        mean_q = q.unsqueeze(-1) - 0.5 * dt * dp(p, q)

        return (
            p.unsqueeze(-1) + dt * self.metric_inv(mean_p.squeeze(-1)) @ mean_q,
            q.unsqueeze(-1) - dt * dp(mean_p.squeeze(-1), mean_q.squeeze(-1)),
        )

    def hamiltonian(self, p, q):
        return (
            0.5
            * torch.transpose(q.unsqueeze(-1), 1, 2)
            @ self.metric_inv(p)
            @ q.unsqueeze(-1)
        )

    @staticmethod
    def dH_dp(p, q, model):
        a = (
            torch.transpose(q.unsqueeze(-1).unsqueeze(1), 2, 3)
            @ model.M_tens.unsqueeze(0)
            @ q.unsqueeze(-1).unsqueeze(1)
        )
        b = model.centroids_tens.unsqueeze(0) - p.unsqueeze(1)
        return (
            -1
            / (model.T ** 2)
            * b.unsqueeze(-1)
            @ a
            * (
                torch.exp(
                    -torch.norm(
                        model.centroids_tens.unsqueeze(0) - p.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (model.T ** 2)
                )
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        ).sum(dim=1)

    def shoot(self, p=None, v=None, q=None, model=None, n_steps=10):
        """
        Geodesic shooting using Hamiltonian dynamics

        Inputs:
        -------

        p (tensor): Starting position
        v (tensor): Starting velocity
        q (tensor): Starting momentum
        inverse_metric (function): The inverse Riemannian metric should output the matrix form
        """
        assert (
            p is not None
        ), "Provide a starting position (i.e. where the exponential is computed"
        assert (
            v is not None or p is not None
        ), "Provide at least a starting velocity or momentum"

        self.metric_inv = model.G_inv

        if self.device == 'cuda':
            p, v = p.cuda(), v.cuda()
            self.metric_inv(p)

        if q is None:
            q = self._velocity_to_momentum(v, p=p).reshape(-1, self.latent_dim)

        dp = functools.partial(Exponential_map.dH_dp, model=model)

        dt = 1 / float(n_steps)

        pos_path = torch.zeros(p.shape[0], n_steps + 1, self.latent_dim).requires_grad_(
            False
        )
        mom_path = torch.zeros(p.shape[0], n_steps + 1, self.latent_dim).requires_grad_(
            False
        )

        pos_path[:, 0, :] = p.reshape(-1, self.latent_dim)
        mom_path[:, 0, :] = q.reshape(-1, self.latent_dim)

        for i in range(n_steps):
            p_t, q_t = self._rk_step(p, q, dp, dt)

            p, q = p_t.reshape(-1, self.latent_dim), q_t.reshape(-1, self.latent_dim)

            pos_path[:, i + 1, :] = p.detach()
            mom_path[:, i + 1, :] = q.detach()
        return pos_path, mom_path
