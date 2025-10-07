# code based on https://github.com/srigas/MTAD-GAT-mlflow/blob/main/train.py
# adjusted to log model suitable for mlflow container / mlserver serving


import argparse
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
import mlflow

from datetime import datetime

from mlflow.types import Schema, TensorSpec, ParamSchema, ParamSpec
from mlflow.models import ModelSignature

from mtad_gat.architecture import MTAD_GAT
from mtad_gat.handler import Handler
from mtad_gat.utils import get_data, SlidingWindowDataset, create_data_loader, find_epsilon, update_json


if __name__ == "__main__":

    # Get arguments from command line
    parser = argparse.ArgumentParser()

    # --- data params ---
    parser.add_argument("--dataset", type=str, default="blade")
    parser.add_argument("--window_size", type=int, default=120)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_end", type=int, default=None)

    # --- model params ---
    # 1D conv layer
    parser.add_argument("--kernel_size", type=int, default=7)
    # GAT layers
    parser.add_argument("--feat_gat_embed_dim", type=int, default=None)
    parser.add_argument("--time_gat_embed_dim", type=int, default=None)
    # GRU layer
    parser.add_argument("--gru_n_layers", type=int, default=1)
    parser.add_argument("--gru_hid_dim", type=int, default=150)
    # Forecasting model
    parser.add_argument("--fc_n_layers", type=int, default=3)
    parser.add_argument("--fc_hid_dim", type=int, default=150)
    # Reconstruction model
    parser.add_argument("--recon_n_layers", type=int, default=1)
    parser.add_argument("--recon_hid_dim", type=int, default=150)
    # other
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=1.0)

    # --- train params ---
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--init_lr", type=float, default=1e-3)
    parser.add_argument("--step_lr", type=int, default=10)
    parser.add_argument("--gamma_lr", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--use_cpu", action='store_true')
    
    # epsilon
    parser.add_argument("--use_mov_av", action='store_true')

    args = parser.parse_args()


    # Get custom id for every run
    id = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    dataset = args.dataset

    experiment = mlflow.set_experiment(experiment_name=f"{dataset}_training")
    exp_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=exp_id, run_name=id):

        window_size = args.window_size

        # --------------------------- START TRAINING -----------------------------
        # Get data from the dataset
        (x_train, _) = get_data(dataset, mode="train", start=args.train_start, end=args.train_end)

        # Cast data into tensor objects
        x_train = torch.from_numpy(x_train).float()
        n_features = x_train.shape[1]

        # We want to perform forecasting/reconstruction on all features
        out_dim = n_features
        print(f"Proceeding with forecasting and reconstruction of all {n_features} input features.")

        # Construct dataset from tensor object
        train_dataset = SlidingWindowDataset(x_train, window_size, args.stride)

        print("Training:")
        # Create the data loader(s)
        train_loader, val_loader = create_data_loader(train_dataset, args.batch_size, args.val_split, True)

        # Initialize the model
        model = MTAD_GAT(
            n_features,
            window_size,
            out_dim,
            kernel_size=args.kernel_size,
            feat_gat_embed_dim=args.feat_gat_embed_dim,
            time_gat_embed_dim=args.time_gat_embed_dim,
            gru_n_layers=args.gru_n_layers,
            gru_hid_dim=args.gru_hid_dim,
            forecast_n_layers=args.fc_n_layers,
            forecast_hid_dim=args.fc_hid_dim,
            recon_n_layers=args.recon_n_layers,
            recon_hid_dim=args.recon_hid_dim,
            dropout=args.dropout,
            alpha=args.alpha
        )

        # Initialize the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)

        # Add a scheduler for variable learning rate
        e_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_lr, gamma=args.gamma_lr)

        # Set the criterion for each process: forecasting & reconstruction
        forecast_criterion = nn.MSELoss()
        recon_criterion = nn.MSELoss()

        # Initialize the Handler module
        handler = Handler(
            model=model,
            optimizer=optimizer,
            scheduler=e_scheduler,
            window_size=window_size,
            n_features=n_features,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            patience=args.patience,
            forecast_criterion=forecast_criterion,
            recon_criterion=recon_criterion,
            use_cuda=not args.use_cpu,
            gamma=args.gamma
        )

        # Start training
        handler.fit(train_loader, val_loader)

        # ---------------------------- END TRAINING ------------------------------

        art_uri = mlflow.get_artifact_uri()

        # Get scores for training data to be used for thresholds later on
        print("Calculating scores on training data to be used for thresholding...")
        anom_scores, _ = handler.score(loader=train_loader, details=False)
        # Also get the ones from the validation data
        if val_loader is not None:
            val_scores, _ = handler.score(loader=val_loader, details=False)
            anom_scores = np.concatenate((anom_scores, val_scores), axis=0)

        # get threshold using epsilon method
        if args.use_mov_av:
            smoothing_window = int(args.batch_size * window_size * 0.05)
            anom_scores = pd.DataFrame(anom_scores).ewm(span=smoothing_window).mean().values.flatten()

        e_thresh = find_epsilon(errors=anom_scores)
        update_json(art_uri, "thresholds.json", {"epsilon":e_thresh})

        # Workaround to write dimensions of dataset in config
        args.__dict__['n_features'] = out_dim

        mlflow.log_dict(args.__dict__, "config.txt")

        mlflow.log_dict({'anom_scores':anom_scores.tolist()}, "anom_scores.json")

        # Don't log all parameters, only some are relevant for tuning
        to_be_logged = ['window_size', 'kernel_size', 'gru_n_layers', 'gru_hid_dim', 'fc_n_layers',
                        'fc_hid_dim', 'recon_n_layers', 'recon_hid_dim', 'alpha', 'gamma', 'dropout']
        for key in to_be_logged:
            mlflow.log_param(key, args.__dict__[key])
            

        # create model signature (required for mlserver)
        signature = ModelSignature(inputs=Schema([TensorSpec(type=np.dtype("float32"), shape=[-1,-1,n_features])])
                                  ,outputs=Schema([TensorSpec(type=np.dtype("float32"), shape=[-1])])
                                  ,params=ParamSchema([ParamSpec(name="mode", dtype="string", default="anomaly")
                                                      ,ParamSpec(name="threshold", dtype="double", default=e_thresh)]))

        # log mlflow model
        mlflow.pytorch.log_model(
            pytorch_model=handler.model,
            signature=signature,
            code_paths=["mtad_gat"],
            name=f"{dataset}_model",
            registered_model_name=f"{dataset}_model",
            pip_requirements=[""], # requirements are already satisfied by custom mlflow base image
            extra_files=[art_uri+"/anom_scores.json", art_uri+"/thresholds.json"]
        )

    print("Finished.")