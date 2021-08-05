import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

from .base import BaseVAE


class VAE(BaseVAE, nn.Module):
    def __init__(self, args):

        BaseVAE.__init__(self, args)
        nn.Module.__init__(self)

        self.name = args.model_name
        # self.data_type = data_type
        self.input_dim = args.input_dim
        self.latent_dim = args.latent_dim

        # encoder network
        self.fc1 = nn.Linear(args.input_dim, 400)
        self.fc21 = nn.Linear(400, args.latent_dim)
        self.fc22 = nn.Linear(400, args.latent_dim)

        # decoder network
        self.fc3 = nn.Linear(args.latent_dim, 400)
        self.fc4 = nn.Linear(400, args.input_dim)

        self._encoder = self._encode_mlp
        self._decoder = self._decode_mlp

        # define a N(0, I) distribution
        self.normal = torch.distributions.MultivariateNormal(
            loc=torch.zeros(args.latent_dim).to(self.device),
            covariance_matrix=torch.eye(args.latent_dim).to(self.device),
        )

    def forward(self, x):
        """
        The VAE model
        """
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        std = torch.exp(0.5 * log_var)
        z, eps = self._sample_gauss(mu, std)
        recon_x = self.decode(z)
        return recon_x, z, eps, mu, log_var

    def loss_function(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(
            recon_x, x.view(-1, self.input_dim), reduction="sum"
        )
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        return BCE + KLD

    def encode(self, x):
        return self._encoder(x)

    def decode(self, z):
        x_prob = self._decoder(z)
        return x_prob

    def sample_img(
        self,
        z=None,
        x=None,
        step_nbr=1,
        record_path=False,
        n_samples=1,
        verbose=False,
    ):
        """
        Simulate p(x|z) to generate an image
        """

        if z is None:
            z = self.normal.sample(sample_shape=(n_samples,)).to(self.device)

        else:
            n_samples = z.shape[0]

        if x is not None:
            recon_x, z, _, _, _ = self.forward(x)

        z.requires_grad_(True)
        recon_x = self.decode(z)
        return recon_x

    def _encode_mlp(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def _decode_mlp(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def _sample_gauss(self, mu, std):
        # Reparametrization trick
        # Sample N(0, I)
        eps = torch.randn_like(std)
        return mu + eps * std, eps

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k

    ########## Estimate densities ##########

    def get_metrics(self, recon_x, x, z, mu, log_var, sample_size=10):
        """
        Estimates all metrics '(log-densities, kl-dvg)

        Output:
        -------

        log_densities (dict): Dict with keys [
            'log_p_x_given_z',
            'log_p_z_given_x',
            'log_p_x'
            'log_p_z'
            'lop_p_xz'
            ]

        KL-dvg (dict): Dict with keys [
            'kl_prior',
            'kl_cond'
            ]
        """
        metrics = {}

        # metrics["log_p_x_given_z"] = self.log_p_x_given_z(recon_x, x)
        # metrics["log_p_z_given_x"] = self.log_p_z_given_x(
        #    z, recon_x, x, sample_size=sample_size
        # )
        # metrics["log_p_x"] = self.log_p_x(x, sample_size=sample_size)
        # metrics["log_p_z"] = self.log_z(z)
        # metrics["lop_p_xz"] = self.log_p_xz(recon_x, x, z)
        # metrics["kl_prior"] = self.kl_prior(mu, log_var)
        # metrics["kl_cond"] = self.kl_cond(
        #    recon_x, x, z, mu, log_var, sample_size=sample_size
        # )
        return metrics

    def log_p_x_given_z(self, recon_x, x, reduction="none"):
        """
        Estimate the decoder's log-density modelled as follows:
            p(x|z)     = \prod_i Bernouilli(x_i|pi_{theta}(z_i))
            p(x = s|z) = \prod_i (pi(z_i))^x_i * (1 - pi(z_i)^(1 - x_i))"""
        return -F.binary_cross_entropy(
            recon_x, x.view(-1, self.input_dim), reduction=reduction
        ).sum(dim=1)

    def log_z(self, z):
        """
        Return Normal density function as prior on z
        """
        return self.normal.log_prob(z)

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling with q(z|x)
        """

        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)
        recon_X = self.decode(Z)
        bce = F.binary_cross_entropy(
            recon_X, x.view(-1, self.input_dim).repeat(sample_size, 1), reduction="none"
        )

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(x|z))
        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(z))
        logqzx = self.normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)

        logpx = (logpxz + logpz - logqzx).logsumexp(dim=0).mean(dim=0) - torch.log(
            torch.Tensor([sample_size]).to(self.device)
        )

        return logpx

    def log_p_z_given_x(self, z, recon_x, x, sample_size=10):
        """
        Estimate log(p(z|x)) using Bayes rule and Importance Sampling for log(p(x))
        """
        logpx = self.log_p_x(x, sample_size)
        lopgxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return lopgxz + logpz - logpx

    def log_p_xz(self, recon_x, x, z):
        """
        Estimate log(p(x, z)) using Bayes rule
        """
        logpxz = self.log_p_x_given_z(recon_x, x)
        logpz = self.log_z(z)
        return logpxz + logpz

    ########## Kullback-Leiber divergences estimates ##########

    def kl_prior(self, mu, log_var):
        """KL[q(z|y) || p(z)] : exact formula"""
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    def kl_cond(self, recon_x, x, z, mu, log_var, sample_size=10):
        """
        KL[p(z|x) || q(z|x)]

        Note:
        -----
        p(z|x) is approximated using IS on log(p(x))
        """
        logpzx = self.log_p_z_given_x(z, recon_x, x, sample_size=sample_size)
        logqzx = torch.distributions.MultivariateNormal(
            loc=mu, covariance_matrix=torch.diag_embed(torch.exp(log_var))
        ).log_prob(z)

        return (logqzx - logpzx).sum()


class HVAE(VAE):
    def __init__(self, args):
        """
        Inputs:
        -------

        n_lf (int): Number of leapfrog steps to perform
        eps_lf (float): Leapfrog step size
        beta_zero (float): Initial tempering
        tempering (str): Tempering type (free, fixed)
        model_type (str): Model type for VAR (mlp, convnet)
        latent_dim (int): Latentn dimension
        """
        VAE.__init__(self, args)

        self.vae_forward = super().forward
        self.n_lf = args.n_lf

        self.eps_lf = nn.Parameter(torch.Tensor([args.eps_lf]), requires_grad=False)

        assert 0 < args.beta_zero <= 1, "Tempering factor should belong to [0, 1]"

        self.beta_zero_sqrt = nn.Parameter(
            torch.Tensor([args.beta_zero]), requires_grad=False
        )

    def forward(self, x):
        """
        The HVAE model
        """

        recon_x, z0, eps0, mu, log_var = self.vae_forward(x)
        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        z = z0
        beta_sqrt_old = self.beta_zero_sqrt

        recon_x = self.decode(z)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # computes potential energy
            U = -self.log_p_xz(recon_x, x, z).sum()

            # Compute its gradient
            g = grad(U, z, create_graph=True)[0]

            # 1st leapfrog step
            rho_ = rho - (self.eps_lf / 2) * g

            # 2nd leapfrog step
            z = z + self.eps_lf * rho_

            recon_x = self.decode(z)

            U = -self.log_p_xz(recon_x, x, z).sum()
            g = grad(U, z, create_graph=True)[0]

            # 3rd leapfrog step
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering steps
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var

    def loss_function(self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var):

        logpxz = self.log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = self.normal.log_prob(rhoK)  # log p(\rho_K)
        logp = logpxz + logrhoK

        logq = self.normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # q(z_0|x)

        return -(logp - logq).sum()

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling on q(z|x)
        """
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)

        recon_X = self.decode(Z)

        gamma = torch.randn_like(Z, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        rho0 = rho
        beta_sqrt_old = self.beta_zero_sqrt
        X_rep = x.repeat(sample_size, 1, 1, 1).reshape(-1, self.input_dim)

        for k in range(self.n_lf):

            U = self.hamiltonian(recon_X, X_rep, Z, rho)
            g = grad(U, Z, create_graph=True)[0]

            # step 1
            rho_ = rho - (self.eps_lf / 2) * g

            # step 2
            Z = Z + self.eps_lf * rho_

            recon_X = self.decode(Z)

            U = self.hamiltonian(recon_X, X_rep, Z, rho_)
            g = grad(U, Z, create_graph=True)[0]

            # step 3
            rho__ = rho_ - (self.eps_lf / 2) * g

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(recon_X, X_rep, reduction="none")

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(X|Z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(Z))

        logrho0 = self.normal.log_prob(rho0 * self.beta_zero_sqrt).reshape(
            sample_size, -1
        )  # log(p(rho0))
        logrho = self.normal.log_prob(rho).reshape(sample_size, -1)  # log(p(rho_K))
        logqzx = self.normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)  # q(Z_0|X)

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(torch.Tensor([sample_size]).to(self.device))
        return logpx

    def hamiltonian(self, recon_x, x, z, rho, G_inv=None, G_log_det=None):
        """
        Computes the Hamiltonian function.
        used for HVAE and RHVAE
        """
        if self.name == "HVAE":
            return -self.log_p_xz(recon_x, x, z).sum()
        norm = (
            torch.transpose(rho.unsqueeze(-1), 1, 2) @ G_inv @ rho.unsqueeze(-1)
        ).sum()

        return -self.log_p_xz(recon_x, x, z).sum() + 0.5 * norm + 0.5 * G_log_det.sum()

    def _tempering(self, k, K):
        """Perform tempering step"""

        beta_k = (
            (1 - 1 / self.beta_zero_sqrt) * (k / K) ** 2
        ) + 1 / self.beta_zero_sqrt

        return 1 / beta_k


class RHVAE(HVAE):
    def __init__(self, args):

        HVAE.__init__(self, args)
        # defines the Neural net to compute the metric

        # first layer
        self.metric_fc1 = nn.Linear(self.input_dim, args.metric_fc)

        # diagonal
        self.metric_fc21 = nn.Linear(args.metric_fc, self.latent_dim)
        # remaining coefficients
        k = int(self.latent_dim * (self.latent_dim - 1) / 2)
        self.metric_fc22 = nn.Linear(args.metric_fc, k)

        self.T = nn.Parameter(torch.Tensor([args.temperature]), requires_grad=False)
        self.lbd = nn.Parameter(
            torch.Tensor([args.regularization]), requires_grad=False
        )

        # this is used to store the matrices and centroids throughout training for
        # further use in metric update (L is the cholesky decomposition of M)
        self.M = []
        self.centroids = []

        # define a starting metric (c_i = 0 & L = I_d)
        def G(z):
            return (
                torch.eye(self.latent_dim, device=self.device).unsqueeze(0)
                * torch.exp(-torch.norm(z.unsqueeze(1), dim=-1) ** 2)
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G

    def metric_forward(self, x):
        """
        This function returns the outputs of the metric neural network

        Outputs:
        --------

        L (Tensor): The L matrix as used in the metric definition
        M (Tensor): L L^T
        """

        h1 = torch.relu(self.metric_fc1(x.view(-1, self.input_dim)))
        h21, h22 = self.metric_fc21(h1), self.metric_fc22(h1)

        L = torch.zeros((x.shape[0], self.latent_dim, self.latent_dim)).to(self.device)
        indices = torch.tril_indices(
            row=self.latent_dim, col=self.latent_dim, offset=-1
        )

        # get non-diagonal coefficients
        L[:, indices[0], indices[1]] = h22

        # add diagonal coefficients
        L = L + torch.diag_embed(h21.exp())

        return L, L @ torch.transpose(L, 1, 2)

    def update_metric(self):
        """
        As soon as the model has seen all the data points (i.e. at the end of 1 loop)
        we update the final metric function using \mu(x_i) as centroids
        """
        # convert to 1 big tensor
        self.M_tens = torch.cat(self.M)
        self.centroids_tens = torch.cat(self.centroids)

        # define new metric
        def G(z):
            return torch.inverse(
                (
                    self.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1)
                + self.lbd * torch.eye(self.latent_dim).to(self.device)
            )

        def G_inv(z):
            return (
                self.M_tens.unsqueeze(0)
                * torch.exp(
                    -torch.norm(
                        self.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                    )
                    ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        self.G = G
        self.G_inv = G_inv
        self.M = []
        self.centroids = []

    def forward(self, x):
        """
        The RHVAE model
        """

        recon_x, z0, eps0, mu, log_var = self.vae_forward(x)

        z = z0

        if self.training:

            # update the metric using batch data points
            L, M = self.metric_forward(x)

            # store LL^T and mu(x_i) to update final metric
            self.M.append(M.clone().detach())
            self.centroids.append(mu.clone().detach())

            G_inv = (
                M.unsqueeze(0)
                * torch.exp(
                    -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                    / (self.T ** 2)
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

        else:
            G = self.G(z)
            G_inv = self.G_inv(z)
            L = torch.cholesky(G)

        G_log_det = -torch.logdet(G_inv)

        gamma = torch.randn_like(z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        # sample \rho from N(0, G)
        rho = (L @ rho.unsqueeze(-1)).squeeze(-1)

        recon_x = self.decode(z)

        for k in range(self.n_lf):

            # perform leapfrog steps

            # step 1
            rho_ = self.leap_step_1(recon_x, x, z, rho, G_inv, G_log_det)

            # step 2
            z = self.leap_step_2(recon_x, x, z, rho_, G_inv, G_log_det)

            recon_x = self.decode(z)

            if self.training:

                G_inv = (
                    M.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(mu.unsqueeze(0) - z.unsqueeze(1), dim=-1) ** 2
                        / (self.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                ).sum(dim=1) + self.lbd * torch.eye(self.latent_dim).to(self.device)

            else:
                # compute metric value on new z using final metric
                G = self.G(z)
                G_inv = self.G_inv(z)

            G_log_det = -torch.logdet(G_inv)

            # step 3
            rho__ = self.leap_step_3(recon_x, x, z, rho_, G_inv, G_log_det)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        return recon_x, z, z0, rho, eps0, gamma, mu, log_var, G_inv, G_log_det

    def leap_step_1(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves first equation of generalized leapfrog integrator
        using fixed point iterations
        """

        def f_(rho_):
            H = self.hamiltonian(recon_x, x, z, rho_, G_inv, G_log_det)
            gz = grad(H, z, retain_graph=True)[0]
            return rho - 0.5 * self.eps_lf * gz

        rho_ = rho.clone()
        for _ in range(steps):
            rho_ = f_(rho_)
        return rho_

    def leap_step_2(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves second equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H0 = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        grho_0 = grad(H0, rho)[0]

        def f_(z_):
            H = self.hamiltonian(recon_x, x, z_, rho, G_inv, G_log_det)
            grho = grad(H, rho, retain_graph=True)[0]
            return z + 0.5 * self.eps_lf * (grho_0 + grho)

        z_ = z.clone()
        for _ in range(steps):
            z_ = f_(z_)
        return z_

    def leap_step_3(self, recon_x, x, z, rho, G_inv, G_log_det, steps=3):
        """
        Resolves third equation of generalized leapfrog integrator
        using fixed point iterations
        """
        H = self.hamiltonian(recon_x, x, z, rho, G_inv, G_log_det)
        gz = grad(H, z, create_graph=True)[0]
        return rho - 0.5 * self.eps_lf * gz

    def loss_function(
        self, recon_x, x, z0, zK, rhoK, eps0, gamma, mu, log_var, G_inv, G_log_det
    ):

        logpxz = self.log_p_xz(recon_x, x, zK)  # log p(x, z_K)
        logrhoK = (
            -0.5
            * (torch.transpose(rhoK.unsqueeze(-1), 1, 2) @ G_inv @ rhoK.unsqueeze(-1))
            .squeeze()
            .squeeze()
            - 0.5 * G_log_det
        ) - torch.log(
            torch.tensor([2 * np.pi]).to(self.device)
        ) * self.latent_dim / 2  # log p(\rho_K)

        logp = logpxz + logrhoK

        logq = self.normal.log_prob(eps0) - 0.5 * log_var.sum(dim=1)  # log(q(z_0|x))

        return -(logp - logq).sum()

    def log_p_x(self, x, sample_size=10):
        """
        Estimate log(p(x)) using importance sampling on q(z|x)
        """
        # print(sample_size)
        mu, log_var = self.encode(x.view(-1, self.input_dim))
        Eps = torch.randn(sample_size, x.size()[0], self.latent_dim, device=self.device)
        Z = (mu + Eps * torch.exp(0.5 * log_var)).reshape(-1, self.latent_dim)

        Z0 = Z

        recon_X = self.decode(Z)

        # get metric value
        G_rep = self.G(Z)
        G_inv_rep = self.G_inv(Z)

        G_log_det_rep = torch.logdet(G_rep)

        L_rep = torch.cholesky(G_rep)

        G_inv_rep_0 = G_inv_rep
        G_log_det_rep_0 = G_log_det_rep

        # initialization
        gamma = torch.randn_like(Z0, device=self.device)
        rho = gamma / self.beta_zero_sqrt
        beta_sqrt_old = self.beta_zero_sqrt

        rho = (L_rep @ rho.unsqueeze(-1)).squeeze(
            -1
        )  # sample from the multivariate N(0, G)

        rho0 = rho

        X_rep = x.repeat(sample_size, 1, 1, 1).reshape(-1, self.input_dim)

        for k in range(self.n_lf):

            # step 1
            rho_ = self.leap_step_1(recon_X, X_rep, Z, rho, G_inv_rep, G_log_det_rep)

            # step 2
            Z = self.leap_step_2(recon_X, X_rep, Z, rho_, G_inv_rep, G_log_det_rep)

            recon_X = self.decode(Z)

            G_rep_inv = self.G_inv(Z)
            G_log_det_rep = -torch.logdet(G_rep_inv)

            # step 3
            rho__ = self.leap_step_3(recon_X, X_rep, Z, rho_, G_inv_rep, G_log_det_rep)

            # tempering
            beta_sqrt = self._tempering(k + 1, self.n_lf)
            rho = (beta_sqrt_old / beta_sqrt) * rho__
            beta_sqrt_old = beta_sqrt

        bce = F.binary_cross_entropy(recon_X, X_rep, reduction="none")

        # compute densities to recover p(x)
        logpxz = -bce.reshape(sample_size, -1, self.input_dim).sum(dim=2)  # log(p(X|Z))

        logpz = self.log_z(Z).reshape(sample_size, -1)  # log(p(Z))

        logrho0 = (
            (
                -0.5
                * (
                    torch.transpose(rho0.unsqueeze(-1), 1, 2)
                    * self.beta_zero_sqrt
                    @ G_inv_rep_0
                    @ rho0.unsqueeze(-1)
                    * self.beta_zero_sqrt
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det_rep_0
            )
            - torch.log(torch.tensor([2 * np.pi]).to(self.device)) * self.latent_dim / 2
        ).reshape(sample_size, -1)

        # log(p(\rho_0))
        logrho = (
            (
                -0.5
                * (
                    torch.transpose(rho.unsqueeze(-1), 1, 2)
                    @ G_inv_rep
                    @ rho.unsqueeze(-1)
                )
                .squeeze()
                .squeeze()
                - 0.5 * G_log_det_rep
            )
            - torch.log(torch.tensor([2 * np.pi]).to(self.device)) * self.latent_dim / 2
        ).reshape(sample_size, -1)
        # log(p(\rho_K))

        logqzx = self.normal.log_prob(Eps) - 0.5 * log_var.sum(dim=1)  # log(q(Z_0|X))

        logpx = (logpxz + logpz + logrho - logrho0 - logqzx).logsumexp(dim=0).mean(
            dim=0
        ) - torch.log(
            torch.Tensor([sample_size]).to(self.device)
        )  # + self.latent_dim /2 * torch.log(self.beta_zero_sqrt ** 2)

        return logpx
