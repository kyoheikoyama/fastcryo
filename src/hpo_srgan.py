import os, sys, yaml
from argparse import ArgumentParser
from warnings import warn

import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from pl_bolts.models.gans import SRGAN

from models.srgan import SRGAN, MRCImageDataModule
from datasets.mrc import MRCImageDataModule
from pytorch_lightning.callbacks import ModelCheckpoint
import optuna
import logging

# from optuna.integration import PyTorchLightningPruningCallback # this occurs version error with pytorch-lightning==1.8


DATATYPE = "float32"
pl.seed_everything(1234)


def objective(trial):  # Function to be optimized.
    # Define PyTorch Lightning model here.
    # Hyperparameters can be suggested by Optuna.
    """feature_maps_disc: 64
    feature_maps_gen: 64
    generator_checkpoint: null
    image_channels: 1
    learning_rate_disc: 0.0001
    learning_rate_gen: 2.0e-05
    num_res_blocks: 16
    scale_factor: 1
    scheduler_step: 100"""

    model = SRGAN(
        image_channels=1,
        feature_maps_gen=hpdict["feature_maps_gen"],
        feature_maps_disc=trial.suggest_int("feature_maps_disc", 7, 9, log=True),
        num_res_blocks=hpdict["num_res_blocks"],
        # scale_factor=hpdict["scale_factor"],
        learning_rate_gen=trial.suggest_float(  # 0.000025827
            "learning_rate_gen", 2e-05, 3e-05, log=True
        ),
        learning_rate_disc=trial.suggest_float(  # 0.000010296
            "learning_rate_disc", 1e-06, 2e-05, log=True
        ),
        scheduler_step=trial.suggest_int("scheduler_step", 3, 5, log=True),
        generator_checkpoint=None,
    )

    data_module = MRCImageDataModule(
        args.datalocation,
        datatype=DATATYPE,
        batch_size=1,
        datasize=args.datasize,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        filename="best-checkpoint",
        save_top_k=3,
        verbose=True,
        save_weights_only=False,
        # period=1,
    )
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
        max_epochs=100 if (args.datasize == "all" or args.datasize == "hpo") else 2,
        default_root_dir="hpolog",
        limit_val_batches=1.0,
        limit_train_batches=0.5,  # You might want to limit batches to speed up the tuning process.
        callbacks=[
            # PyTorchLightningPruningCallback(trial, monitor="val_loss"),
            checkpoint_callback,
            EarlyStopping(
                patience=5,
                monitor="val_loss",
                mode="min",
                check_finite=True,
            ),
        ],
        logger=pl.loggers.TensorBoardLogger(
            save_dir="/media/kyohei/forAI/lightning_logs",
            name="hpo_srgan",
            # version=f"ver0",
            # default_hp_metric=False,
        ),
        # gradient_clip_val=1.5,
    )

    trainer.fit(model, datamodule=data_module)
    met = trainer.validate(model, datamodule=data_module, ckpt_path="best")[0]
    # [{'loss/perceptual_loss_epoch': 0.6257400512695312, 'loss/adv_loss_epoch': 0.7146725654602051, 'loss/content_loss_epoch': 402.3280029296875, 'val_loss': 402.3324890136719}].
    print(met)
    return met["val_loss"]


if __name__ == "__main__":
    """
    Example::

            python hpo_srgan.py --datasize small
    """
    parser = ArgumentParser()
    parser.add_argument(
        "--datasize",
        default="all",
        choices=["all", "small", "hpo", "dev"],
    )
    # Get the absolute path of the script
    script_path = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument(
        "--hparams",
        type=str,
        default=os.path.join(script_path, "../hparams/hparams.yaml"),
    )
    parser.add_argument(
        "--datalocation",
        default="/media/kyohei/forAI/split_images/",
    )
    args = parser.parse_args()

    with open(args.hparams, "r") as file:
        data = yaml.safe_load(file)
    hpdict = dict(data)

    global DATASIZE
    DATASIZE = args.datasize

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
    study_name = "cryoem-study"  # Unique identifier of the study.
    storage_name = "sqlite:///{}.db".format(study_name)

    study = optuna.create_study(
        # pruner=optuna.pruners.MedianPruner(n_warmup_steps=4),
        direction="minimize",
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
    )
    study.optimize(
        objective,
        n_trials=4,
        #    timeout=600
    )

    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
