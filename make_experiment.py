import math
import os
from copy import deepcopy

import torch
from trainers.training import train_vae


def experiment_vae(
    args, train_loader, model, optimizer, dir_path, logger=None
):
    best_loss = 1e10
    e = 0
    train_loss_recording = []

    for epoch in range(1, args.max_epochs):
        model, train_loss = train_vae(epoch, args, model, train_loader, optimizer)

        train_loss_recording.append(train_loss)

        if args.verbose and epoch % 200 == 0:
            logger.info(
                f"Epoch {epoch} / {args.max_epochs}\n"
                f"- Train loss: {train_loss:.2f}\n"
                f"- Early Stopping: {e}/{args.early_stopping_epochs} (Best: {best_loss:.2f})\n"
            )

        # Early stopping
        if train_loss < best_loss:
            e = 0
            best_loss = train_loss

            if args.model_name == "VAE":
                best_model_dict = {
                    "args": args,
                    "model_state_dict": deepcopy(model.state_dict()),
                }

            else:
                best_model_dict = {
                    "args": args,
                    "M": deepcopy(model.M_tens),
                    "centroids": deepcopy(model.centroids_tens),
                    "model_state_dict": deepcopy(model.state_dict()),
                }

        #
        else:
            e += 1
            if e >= args.early_stopping_epochs:
                logger.info(
                    f"Training ended at epoch {epoch}! (Loss did not improved in {e} epochs)\n"
                )

                path_to_save = os.path.join(
                    dir_path, f"{args.model_name}_{args.dataset}.model"
                )
                torch.save(best_model_dict, path_to_save)
                break

        if math.isnan(train_loss):
            logger.error("NaN detected !")
            break

    logger.info(f"Model saved in {dir_path} !\n")
