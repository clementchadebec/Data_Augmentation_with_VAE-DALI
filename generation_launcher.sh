#!/bin/bash

python generation_parser.py --path_to_model 'trained_models/RHVAE_train_MNIST_ldim_10_nlf_3_epslf_0.01_T_1.5_lbd_0.001/RHVAE_train_MNIST.model' \
--num_samples 100 --verbose --mcmc_steps 100 --generation_method 'metric_sampling' --seed 8