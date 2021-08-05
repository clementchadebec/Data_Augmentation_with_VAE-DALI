#!/bin/bash

# Trainings
## VAEs
python experiment_runner.py --path_to_train_loader 'training_data/train_Shapes' \
--batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'VAE' \
--input_dim 2500 --seed 8 --verbose

python experiment_runner.py --path_to_train_loader 'training_data/train_MNIST' \
--batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'VAE' \
--input_dim 784 --seed 8 --verbose

python experiment_runner.py --path_to_train_loader 'training_data/train_Fashion' \
--batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'VAE' \
--input_dim 784 --seed 8 --verbose

## RHVAEs
python experiment_runner.py --path_to_train_loader 'training_data/train_Shapes' \
--batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'RHVAE' \
--input_dim 2500 --n_lf 3 --early_stopping_epochs 50 --regularization 0.001 --eps_lf 0.01 --seed 8 --verbose

python experiment_runner.py --path_to_train_loader 'training_data/train_MNIST' \
--batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'RHVAE' \
--input_dim 784 --n_lf 3 --early_stopping_epochs 50 --regularization 0.001 --eps_lf 0.01 --seed 8 --verbose

python experiment_runner.py --path_to_train_loader 'training_data/train_Fashion' \
--batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'RHVAE' \
--input_dim 784 --n_lf 3 --early_stopping_epochs 50 --regularization 0.001 --eps_lf 0.01 --seed 8 --verbose

# Generation
## Synthetic data
python generation_parser.py --path_to_model 'recordings/VAE_train_Shapes_ldim_2/VAE_train_Shapes.model' \
--num_samples 100 --verbose --generation_method 'prior' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_Shapes_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_Shapes.model' \
--num_samples 100 --verbose --generation_method 'prior' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_Shapes_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_Shapes.model' \
--num_samples 100 --verbose --mcmc_steps 400 --generation_method 'metric_sampling' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_Shapes_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_Shapes.model' \
--num_samples 100 --verbose --mcmc_steps 200 --batch_size 50 --eigenvalues 0.5 --generation_method 'riemannian_rw' --seed 8

## reduced MNIST
python generation_parser.py --path_to_model 'recordings/VAE_train_MNIST_ldim_2/VAE_train_MNIST.model' \
--num_samples 100 --verbose --generation_method 'prior' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_MNIST_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_MNIST.model' \
--num_samples 100 --verbose --generation_method 'prior' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_MNIST_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_MNIST.model' \
--num_samples 100 --verbose --mcmc_steps 400 --generation_method 'metric_sampling' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_MNIST_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_MNIST.model' \
--num_samples 100 --verbose --mcmc_steps 200 --batch_size 50 --eigenvalues 0.5 --generation_method 'riemannian_rw' --seed 8

# reduced Fashion
python generation_parser.py --path_to_model 'recordings/VAE_train_Fashion_ldim_2/VAE_train_Fashion.model' \
--num_samples 100 --verbose --generation_method 'prior' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_Fashion_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_Fashion.model' \
--num_samples 100 --verbose --generation_method 'prior' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_Fashion_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_Fashion.model' \
--num_samples 100 --verbose --mcmc_steps 400 --generation_method 'metric_sampling' --seed 8

python generation_parser.py --path_to_model 'recordings/RHVAE_train_Fashion_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_train_Fashion.model' \
--num_samples 100 --verbose --mcmc_steps 200 --batch_size 50 --eigenvalues 0.5 --generation_method 'riemannian_rw' --seed 8