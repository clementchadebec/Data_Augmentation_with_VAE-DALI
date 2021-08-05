import argparse
import datetime
import logging
import os

import torch
from models.generation import generate_data
from utils import create_inverse_metric, create_metric

# Get the top-level logger object
log = logging.getLogger()

# make it print to the console.
console = logging.StreamHandler()
log.addHandler(console)
log.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))

ap = ap = argparse.ArgumentParser()

# Model
ap.add_argument("--path_to_model", required=True, help="path to the model")

# Generation settings
ap.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="batch size for generation (default: 100)",
)
ap.add_argument(
    "--num_samples", type=int, default=1, help="number of samples to generate (default: 1)"
)
ap.add_argument(
    "--generation_method",
    default="metric_sampling",
    choices=['prior', 'metric_sampling', 'riemannian_rw'],
    help="generation method, choices=['prior', 'metric_sampling', 'riemannian_rw'], (default: 'metric_sampling').",
)
ap.add_argument(
    "--mcmc_steps_nbr",
    type=int,
    default=400,
    help="number of MCMC steps for metric sampling and Riemannian RW (default: 400)",
)
ap.add_argument(
    "--n_lf", type=int, default=15, help="n_lf, the number of leapfrog steps in HMC sampler (default: 15)"
)
ap.add_argument(
    "--eps_lf", type=float, default=0.05, help="eps_lf, the leapfrog step size in HMC sampler (default: 0.05)"
)
ap.add_argument(
    "--beta_zero_sqrt",
    type=float,
    default=1.0,
    help="the tempering coefficient beta (default: 1.0 ie no tempering)",
)
ap.add_argument(
    "--eigenvalues",
    type=float,
    default=.5,
    help="the eigenvalues of Sigma for the Riemannian random walk.",
)
# Seed
ap.add_argument("--seed", type=int, default=None, help="random seed")

ap.add_argument(
    "--verbose", action="store_true", default=False, help="verbosity (default: False)"
)

args = ap.parse_args()
args.cuda = torch.cuda.is_available()
args.device = "cuda" if torch.cuda.is_available() else "cpu"

if args.seed is None:
    seed = torch.randint(10000, (1,))
    args.seed = seed

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

print(args.seed)

def main(args):
    args.dataset_signature = (
        str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
    )
    args.model_name = args.path_to_model.split("/")[-1].split(".")[0]

    try:
        checkpoint = torch.load(args.path_to_model, map_location=args.device)
        args_model = checkpoint["args"]
        if args_model.model_name == "VAE":
            from models.vae import VAE

            if args.verbose:
                log.info("\nVAE model loaded !\n")
            args_model.device = args.device
            model = VAE(args_model).to(args.device)
            args.generation_type = "classic_gen"

            folder_name = (
                f"{args.model_name}_classic_generation_{args.num_samples}_samples"
            )

        elif args_model.model_name == "RHVAE":
            from models.vae import RHVAE

            if args.verbose:
                log.info("\nRHVAE model loaded !\n")
            args_model.device = args.device
            model = RHVAE(args_model)
            model.M_tens = checkpoint["M"]
            model.centroids_tens = checkpoint["centroids"]
            model.G = create_metric(model, device=args.device)
            model.G_inv = create_inverse_metric(model, device=args.device)

            if args.generation_method=='prior':
                folder_name = (
                    f"{args.model_name}_classic_generation_{args.num_samples}_samples"
                )
                args.generation_type = "classic_gen"
            elif args.generation_method=='metric_sampling':
                folder_name = f"{args.model_name}_ms_{args.num_samples}_samples_{args.mcmc_steps_nbr}_steps"
                args.generation_type = "metric_sampling"
            elif args.generation_method=='riemannian_rw':
                folder_name = f"{args.model_name}_Riem_RW_{args.num_samples}_samples_{args.mcmc_steps_nbr}_steps"
                args.generation_type = "riemannian_rw"

        else:
            raise NameError(
                f"Wrong model name provided ({args.model_name}). Must be VAE or RHVAE."
            )

    except FileNotFoundError:
        log.error(
            f"Cannot load the model. Provide a valid path (current path: '{args.path_to_model}'')"
        )

    model.load_state_dict(checkpoint["model_state_dict"])

    if args.cuda:
        log.info("Using Cuda !\n")
        model.cuda()

    else:
        log.info("Using cpu !\n")

    # Directory for saving
    generated_path = "generated_data"
    dir_path = os.path.join(
        PATH, generated_path, f"{args.dataset_signature}_{folder_name}"
    )

    if not os.path.exists(dir_path):
        if args.verbose:
            log.info(f"Data will be stored in:\n" f" --> {dir_path}\n")
        os.makedirs(dir_path)

    generate_data(args, args_model, model, dir_path, logger=log)


if __name__ == "__main__":
    log.info(args)
    main(args)
