#!/bin/bash

python experiment_runner.py --path_to_train_loader 'training_data/train_MNIST' --batch_size 200 --max_epochs 10000 --lr 0.001 --model_name 'RHVAE' --input_dim 784 --latent_dim  2 --n_lf 3 --early_stopping_epochs 50 --regularization 0.001 --eps_lf 0.01 --verbose --seed 8 --temperature 0.8