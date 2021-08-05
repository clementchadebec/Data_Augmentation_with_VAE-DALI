# Data Augmentation with Variational Autoencoders and Manifold Sampling

This repository is the official implementation of Data Augmentation with Variational Autoencoders and Manifold Sampling

## Requirements

Python 3.8 environment is used for the tests.

To install requirements run:

```setup
pip install -r requirements
```

## Training a model

A commandline to train a model is provided in `experiments_launcher.sh`.

```bash
python experiment_runner.py --path_to_train_loader 'training_data/train_Shapes' --batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'RHVAE' --input_dim 2500 --n_lf 3 --early_stopping_epochs 50 --regularization 0.001 --eps_lf 0.01 --verbose --seed 8
```

Parser arguments
```bash
--path_to_train_loader "path to loader. Data must be loadable using 'checkpoint = torch.load()' and 'data = checkpoint['data']' (targets = checkpoint['targets']' if targets are available)"
--batch_size "batch size for training (default: 10)"
--max_epochs MAX_EPOCHS "Max number of epochs (default: 10000)"
--lr "Learning rate (default: 0.0001)"
--early_stopping_epochs "number of epochs for early stopping"
--no_cuda "disables CUDA training"
--seed "random seed (default: 8)"
--model_name "Choice of model [RHVAE or VAE] (default: RHVAE)"
--input_dim "Input dimension"
--latent_dim "Latent space dimension"
--n_lf "n_lf, the number of leapfrog steps in RHVAE training (default: 3)"
--eps_lf "eps_lf, the size of leapfrog step size in RHVAE training (default: 0.001)"
--beta_zero "beta zero sqrt, the temperature in the leapfrog integrator for RHVAE training"
--temperature "T, the metric temperature for RHVAE (default: 0.8)"
--regularization "lambda, the metric regularization factor for RHVAE (default: 0.01)"
--metric_fc "metric hidden units, metric's neural net architecture for RHVAE (default: 400)"
--dynamic_binarization "allow dynamic binarization"
--verbose "Verbosity (default: False)"
```

At the end of training the model is stored in `recording/model_name__dataset_name__params/model_name__dataset.model`.

## Generation
A commandline to generate data from a trained model is provided in `generation_launcher.sh`

```bash
python generation_parser.py --path_to_model 'recordings/RHVAE_train_Shapes_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_Shapes.model' \
--num_samples 100 --verbose --mcmc_steps 400 --generation_method 'riemannian_rw' --seed 8
``` 

Parser arguments

```bash
--path_to_model "path to the model"
--batch_size "batch size for generation (default: 100)"
--num_samples "number of samples to generate"
--generation_method "generation method, choices=['prior', 'metric_sampling', 'riemannian_rw'], (default: 'metric_sampling')"
--mcmc_steps_nbr "number of MCMC steps for metric sampling and Riemannian RW (default: 400)"
--n_lf "n_lf, the number of leapfrog steps in HMC sampler"
--eps_lf "eps_lf, the leapfrog step size in HMC sampler"
--beta_zero_sqrt "the tempering coefficient beta (default: 1.0 ie no tempering)"
--eigenvalues "the eigenvalues of Sigma for the Riemannian random walk"
--seed "random seed"
--no_cuda "disables CUDA training"
--verbose "verbosity (default: False)"
``` 
Generated data are stored in the folder `generated_data/YYYY-MM-DD_hh_mm_ss__model_name__dataset_name__sampling_method`  with an extension `.data`.

## Short files description
- `experiment_runner.py`: Parser for training.
- `generation_parser.py`: Parser for generation.
- `experiments_launcher.sh`: Example of commandline for training.
- `generation_launcher.sh`: Example of commandline for generating data from trained models.


## Short folders description

- `models`: Contains the main pieces of code (*i.e.*  models, generation procedures ...).
- `trainers`: Contains the `training.py` script to train the models.
- `trained_models`: Contain the pre-trained models used in the paper.
- `training_data`: Contains the data used in the paper
- `vae_vampprior`: The repository from https://github.com/jmtomczak/vae_vampprior (the implemention of (Tomczak, J. and Welling, M.  VAE with a vampprior, 2017)).
- `notebooks`: Contains the demo notebook.