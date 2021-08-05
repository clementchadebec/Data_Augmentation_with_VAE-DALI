#!/bin/bash

python generation_parser.py --path_to_model 'recordings/RHVAE_mnist_50_0.pt_ldim_2_nlf_3_epslf_0.01_T_0.8_lbd_0.001/RHVAE_mnist_50_0.pt.model' \
--num_samples 100 --verbose --mcmc_steps 400 --generation_method 'metric_sampling' --seed 8