import argparse
import datetime
import logging
import os

import torch
import torch.optim as optim
from make_experiment import experiment_vae

# Get the top-level logger object
log = logging.getLogger()

# make it print to the console.
console = logging.StreamHandler()
log.addHandler(console)
log.setLevel(logging.INFO)

PATH = os.path.dirname(os.path.abspath(__file__))


ap = ap = argparse.ArgumentParser()


# Training setting
ap.add_argument("--path_to_train_loader", help="path to loader. Data must be loadable using"
" 'checkpoint = torch.load()' and 'data = checkpoint['data']' (targets = checkpoint['targets']' if targets"
" are available)")
ap.add_argument(
    "--batch_size",
    type=int,
    default=10,
    help="batch size for training (default: 10)",
)
ap.add_argument(
    "--max_epochs",
    type=int,
    default=10000,
    help="max number of epochs (default: 10000)",
)
ap.add_argument(
    "--lr",
    type=float,
    default=0.0001,
    help="learning rate (default: 0.0001)",
)
ap.add_argument(
    "--early_stopping_epochs",
    type=int,
    default=50,
    help="number of epochs for early stopping",
)
# Cuda
ap.add_argument(
    "--no_cuda", action="store_true", default=False, help="disables CUDA training"
)
# Seed
ap.add_argument(
    "--seed", type=int, default=None, help="random seed"
)
# Model parameters
ap.add_argument(
    "--model_name",
    type=str,
    default="RHVAE",
    help="choice of model [RHVAE or VAE] (default: RHVAE)",
)
ap.add_argument(
    "--input_dim", type=int, default=50 * 50 * 50, help="input dimension"
)
ap.add_argument(
    "--latent_dim", type=int, default=2, help="latent space dimension"
)

ap.add_argument("--n_lf", type=int, default=3, help="n_lf, the number of leapfrog steps in RHVAE training (default: 3)")
ap.add_argument(
    "--eps_lf",
    type=float,
    default=0.001,
    help="eps_lf, the size of leapfrog step size in RHVAE training (default: 0.001)",
)
ap.add_argument(
    "--beta_zero",
    type=float,
    default=0.3,
    help="beta zero sqrt, the temperature in the leapfrog integrator for RHVAE training  (default: 0.3)",
)
ap.add_argument(
    "--temperature", type=float, default=0.8, help="T, the metric temperature for RHVAE (default: 0.8)"
)
ap.add_argument(
    "--regularization",
    type=float,
    default=0.01,
    help="lambda, the metric regularization factor for RHVAE (default: 0.01)",
)
ap.add_argument(
    "--metric_fc", type=int, default=400, help="metric hidden units, metric's neural net architecture for RHVAE (default: 400)"
)

ap.add_argument(
    "--dynamic_binarization",
    action="store_true",
    default=False,
    help="allow dynamic binarization",
)

ap.add_argument(
    "--verbose", action="store_true", default=False, help="verbosity (default: False)"
)

args = ap.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

args.device = "cuda" if args.cuda else "cpu"

if args.seed is None:
    seed = torch.randint(10000, (1,))
    args.seed = seed

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

def main(args):

    args.dataset = args.path_to_train_loader.split("/")[-1]
    args.model_signature = (
        str(datetime.datetime.now())[0:19].replace(" ", "_").replace(":", "-")
    )

    if args.model_name == "VAE":
        folder_name = f"{args.model_name}_{args.dataset}_ldim_{args.latent_dim}"

    elif args.model_name == "RHVAE":
        folder_name = f"{args.model_name}_{args.dataset}_ldim_{args.latent_dim}_nlf_{args.n_lf}_epslf_{args.eps_lf}_T_{args.temperature}_lbd_{args.regularization}"

    else:
        raise NameError(
            f"Wrong model name provided ({args.model_name}). Must be VAE or RHVAE."
        )

    # DIRECTORY FOR SAVING
    recording_path = "recordings"
    dir_path = os.path.join(PATH, recording_path, f"{folder_name}")

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    log.info("Load data\n")
    try:
        checkpoint = torch.load(
            args.path_to_train_loader, map_location=torch.device(args.device)
        )
        train_data = checkpoint["data"]

        if 'targets' in checkpoint.keys():
            train_targets = checkpoint["targets"]
        else:
            train_targets = torch.ones(train_data.shape[0])

        from utils import Digits

        train = Digits(train_data, train_targets)
        train_loader = torch.utils.data.DataLoader(
            dataset=train, batch_size=args.batch_size, shuffle=True
        )

    except Exception as e:
        log.error(e)

    if args.model_name == "VAE":
        from models.vae import VAE

        model = VAE(args)
    else:
        from models.vae import RHVAE

        model = RHVAE(args)

    if args.cuda:
        log.info("Using Cuda !\n")
        model.cuda()

    else:
        log.info("Using cpu !\n")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.verbose:
        log.info(f"Model Architecture:\n --> {model}\n")

        if args.model_name == "VAE":
            log.info(
                f"VAE hyper-params:\n"
                f" - Latent dim: {args.latent_dim}\n"
                f" - input dim: {args.input_dim}\n"
            )

        else:
            log.info(
                f"RHVAE hyper-params:\n"
                f" - Latent dim: {args.latent_dim}\n"
                f" - input dim: {args.input_dim}\n"
                f" - n_lf: {args.n_lf}\n"
                f" - eps_lf: {args.eps_lf}\n"
                f" - T: {args.temperature}\n"
                f" - lbd: {args.regularization}\n"
            )

        log.info(f"Optimizer \n --> {optimizer}\n")
        log.info(
            f"Training params:\n - max_epochs: {args.max_epochs}\n"
            f" - es: {args.early_stopping_epochs}\n"
            f" - batch_size: {args.batch_size}\n"
        )

    experiment_vae(args, train_loader, model, optimizer, dir_path, logger=log)


if __name__ == "__main__":
    log.info(args)
    main(args)
