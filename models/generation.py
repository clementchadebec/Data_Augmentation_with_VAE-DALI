import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from models.riemann_tools import Exponential_map
from tqdm import tqdm


def generate_data(args, args_model, model, dir_path, logger=None):

    full_batch_nbr = int(args.num_samples / args.batch_size)
    last_batch_samples_nbr = args.num_samples % args.batch_size

    if args_model.model_name == "VAE" or args.generation_method=='prior':
        if args.verbose:
            logger.info("Using Prior based generation !\n")

        generated_samples = []

        for i in tqdm(range(full_batch_nbr)):
            z = torch.randn(args.batch_size, args_model.latent_dim).to(args.device)
            x_gen = model.sample_img(z).detach()
            generated_samples.append(x_gen)

        if last_batch_samples_nbr > 0:
            z = torch.randn(args.batch_size, args_model.latent_dim).to(args.device)
            x_gen = model.sample_img(z).detach()
            generated_samples.append(x_gen)

    elif args.generation_method=='metric_sampling':
        if args.verbose:
            logger.info("Metric sampling generation !\n")

        generated_samples = []
        for i in tqdm(range(full_batch_nbr)):
            samples = hmc_manifold_sampling(
                model,
                latent_dim=args_model.latent_dim,
                n_samples=args.batch_size,
                step_nbr=args.mcmc_steps_nbr,
                n_lf=args.n_lf,
                eps_lf=args.eps_lf,
                verbose=True,
            )

            x_gen = model.sample_img(z=samples).detach()
            generated_samples.append(x_gen)

        if last_batch_samples_nbr > 0:
            samples = hmc_manifold_sampling(
                model,
                latent_dim=args_model.latent_dim,
                n_samples=last_batch_samples_nbr,
                step_nbr=args.mcmc_steps_nbr,
                n_lf=args.n_lf,
                eps_lf=args.eps_lf,
                verbose=True,
            )

            x_gen = model.sample_img(z=samples).detach()
            generated_samples.append(x_gen)

    elif args.generation_method=='riemannian_rw':
        if args.verbose:
            logger.info("Riemannian RW generation !\n")

        generated_samples = []
        for i in tqdm(range(full_batch_nbr)):
            samples = random_walk_batch(
                model=model,
                latent_dim=args_model.latent_dim,
                n_steps=args.mcmc_steps_nbr,
                n_samples=args.batch_size,
                delta=1.,
                dt=args.eigenvalues,
                verbose=True
            )

            x_gen = model.sample_img(z=samples).detach()
            generated_samples.append(x_gen)

        if last_batch_samples_nbr > 0:
            samples = random_walk_batch(
                model=model,
                latent_dim=args_model.latent_dim,
                n_steps=last_batch_samples_nbr,
                n_samples=args.batch_size,
                delta=1.,
                dt=args.eigenvalues,
                verbose=True
            )

            x_gen = model.sample_img(z=samples).detach()
            generated_samples.append(x_gen)

    generated_samples = torch.cat(generated_samples)

    torch.save(
        {"args": args, "args_model": args_model, "data": generated_samples},
        os.path.join(dir_path, f"{args.generation_type}.data"),
    )

    if args.verbose:
        logger.info(f"\nGenerated {args.num_samples} samples !\n")
        logger.info("Samples saved !\n")


def random_walk(model=None, z0=None, latent_dim=2, n_steps=10, delta=1.0, dt=1.0, device='cpu'):
    exponential_map = Exponential_map(latent_dim=latent_dim, device=device)

    cov = torch.eye(latent_dim) * delta ** 2 * dt
    norm = torch.distributions.MultivariateNormal(
        loc=torch.zeros(latent_dim), covariance_matrix=cov
    )

    samples = []

    if z0 is None:
        z0 = torch.randn(latent_dim)
    z = z0.reshape(-1, latent_dim)
    for i in range(n_steps):

        v = norm.sample().T.reshape(-1, latent_dim)
        z_traj, q_traj = exponential_map.shoot(p=z, v=v, model=model, n_steps=10)
        z_prop = z_traj[0, -1, :].reshape(-1, latent_dim)

        alpha = (
            torch.det((model.G_inv(z_prop))).sqrt() / torch.det((model.G_inv(z))).sqrt()
        )
        with torch.no_grad():
            u = torch.rand(1)
            if u < min(1, alpha):
                samples.append(z_traj[-1][-1].detach().unsqueeze(0))
                z = z_prop

    return torch.cat(samples)


def random_walk_batch(
    model=None,
    z0=None,
    latent_dim=2,
    n_steps=10,
    n_samples=1,
    delta=1.0,
    dt=1,
    verbose=False,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with torch.no_grad():
        exponential_map = Exponential_map(latent_dim=latent_dim, device=device)
        acc_nbr = torch.zeros((n_samples, 1)).to(device)
        cov = torch.eye(latent_dim) * delta ** 2 * dt
        norm = torch.distributions.MultivariateNormal(
            loc=torch.zeros(latent_dim), covariance_matrix=cov
        )

        if z0 is None:
            idx = torch.randint(len(model.centroids_tens), (n_samples,))
            z0 = model.centroids_tens[idx]
            
        z = z0
        for i in range(n_steps):

            # Sample Velocities
            v = norm.sample((n_samples,))

            # Shoot
            z_traj, q_traj = exponential_map.shoot(p=z, v=v, model=model, n_steps=10)
            z = z_traj[:, -1, :].to(device)

            # Compute acceptance ratio
            alpha = (
                torch.det((model.G_inv(z))).sqrt() / torch.det((model.G_inv(z0))).sqrt()
            )
            acc = torch.rand(n_samples).to(device)
            moves = torch.tensor(acc < alpha).type(torch.int).reshape(n_samples, 1)
            z = z * moves + (1 - moves) * z0
            acc_nbr += moves

            z0 = z

            if i % 100 == 0 and verbose:
                if i == 0:
                    print(f"Iteration {i} / {n_steps}")
                else:
                    print(
                        f"Iteration {i} / {n_steps}\t Mean acc. rate {torch.mean(100*(acc_nbr / (i+1)))}"
                    )
    return z


def hmc_manifold_sampling(
    model,
    log_pi=None,
    grad_func=None,
    latent_dim=2,
    step_nbr=10,
    z0=None,
    n_lf=15,
    eps_lf=0.05,
    n_samples=1,
    beta_zero_sqrt=1.0,
    record_path=False,
    return_acc=False,
    verbose=False,
):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    beta_zero_sqrt = torch.tensor([beta_zero_sqrt]).to(device)
    eps_lf = torch.tensor([eps_lf]).to(device)
    n_lf = torch.tensor([n_lf]).to(device)

    if log_pi is None:
        log_pi = log_sqrt_det_G_inv

    if grad_func is None:
        grad_func = grad_log_prop

    acc_nbr = torch.zeros((n_samples, 1)).to(device)
    with torch.no_grad():
        idx = torch.randint(len(model.centroids_tens), (n_samples,))
        z0 = model.centroids_tens[idx]

       

        n_samples = z0.shape[0]

        Z_i = torch.zeros((step_nbr + 1, z0.shape[0], z0.shape[1]))
        Z_i[0] = z0

        beta_sqrt_old = beta_zero_sqrt
        z = z0
        for i in range(step_nbr):
            if verbose and i % 50 == 0:
                print(f"Iteration {i} / {step_nbr}")
            gamma = torch.randn_like(z, device=device)
            rho = gamma / beta_zero_sqrt
            H0 = -log_pi(z, model) + 0.5 * torch.norm(rho, dim=1) ** 2

            for k in range(n_lf):

                g = -grad_func(z, model).reshape(n_samples, latent_dim)
                # step 1
                rho_ = rho - (eps_lf / 2) * g

                # step 2
                z = z + eps_lf * rho_
                g = -grad_func(z, model).reshape(n_samples, latent_dim)

                # step 3
                rho__ = rho_ - (eps_lf / 2) * g

                # tempering
                beta_sqrt = tempering(k + 1, n_lf, beta_zero_sqrt)
                rho = (beta_sqrt_old / beta_sqrt) * rho__
                beta_sqrt_old = beta_sqrt

            H = -log_pi(z, model) + 0.5 * torch.norm(rho, dim=1) ** 2
            alpha = torch.exp(-H) / (torch.exp(-H0))
            acc = torch.rand(n_samples).to(device)
            moves = torch.tensor(acc < alpha).type(torch.int).reshape(n_samples, 1)
            z = z * moves + (1 - moves) * z0
            acc_nbr += moves
            if record_path:
                Z_i[i] = z

            z0 = z
    if return_acc:
        return z, acc_nbr
    else:
        return z


def tempering(k, K, beta_zero_sqrt):
    beta_k = ((1 - 1 / beta_zero_sqrt) * (k / K) ** 2) + 1 / beta_zero_sqrt

    return 1 / beta_k


def log_sqrt_det_G_inv(z, model):
    return torch.log(torch.sqrt(torch.det(model.G_inv(z))) + 1e-10)


def grad_log_sqrt_det_G_inv(z, model):
    return (
        -0.5
        * torch.transpose(model.G(z), 1, 2)
        @ torch.transpose(
            (
                -2
                / (model.T ** 2)
                * (model.centroids_tens.unsqueeze(0) - z.unsqueeze(1)).unsqueeze(2)
                @ (
                    model.M_tens.unsqueeze(0)
                    * torch.exp(
                        -torch.norm(
                            model.centroids_tens.unsqueeze(0) - z.unsqueeze(1), dim=-1
                        )
                        ** 2
                        / (model.T ** 2)
                    )
                    .unsqueeze(-1)
                    .unsqueeze(-1)
                )
            ).sum(dim=1),
            1,
            2,
        )
    )


def grad_prop_manifold(z, grad_log_density, model):
    return grad_log_sqrt_det_G_inv(z, model) + grad_log_density(z).reshape(-1, 2, 1)


def grad_log_prop(z, model):
    def grad_func(z, model):
        return grad_log_sqrt_det_G_inv(z, model)

    return grad_func(z, model)
