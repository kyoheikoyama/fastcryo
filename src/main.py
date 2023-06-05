from argparse import ArgumentParser
import yaml, os
from datasets.mrc import MRCImageDataModule, MRCImageDataset
from models.srgan import SRGAN, DATATYPE
import pytorch_lightning as pl
import pandas as pd
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint


def main(args) -> float:
    # Load the YAML file
    with open(args.hparams, "r") as file:
        data = yaml.safe_load(file)
    hpdict = dict(data)

    data_module = MRCImageDataModule(
        args.datalocation,
        datatype=DATATYPE,
        batch_size=args.batch_size,
        datasize=args.datasize,
        split_way=args.split_way,
    )

    model = SRGAN(
        image_channels=1,
        feature_maps_gen=hpdict["feature_maps_gen"],
        feature_maps_disc=hpdict["feature_maps_disc"],
        num_res_blocks=hpdict["num_res_blocks"],
        # scale_factor=hpdict["scale_factor"],
        learning_rate_gen=hpdict["learning_rate_gen"],
        learning_rate_disc=hpdict["learning_rate_disc"],
        scheduler_step=hpdict["scheduler_step"],
        generator_checkpoint=hpdict["generator_checkpoint"],
    )
    model = model.to(torch.float16) if DATATYPE == "float16" else model

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
        # default_root_dir="srgan_logs",
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
            name="srgan",
            # version=f"ver0",
            # default_hp_metric=False,
        ),
    )
    trainer.fit(model, datamodule=data_module)
    # trainer.save_checkpoint(f"{}srgan.ckpt")
    valloss = trainer.callback_metrics["val_loss"].item()
    print("valloss", valloss)

    return valloss


if __name__ == "__main__":
    """
    Example::

            python srgan.py --datasize dev
            python main.py --datasize all --hparams ../hparams/hparams_srgan.yaml \
            --batch_size 3 --split_way images
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
    parser.add_argument(
        "--batch_size",
        default=3,
        type=int,
    )
    parser.add_argument(
        "--split_way",
        type=str,
        default="images",
    )
    args = parser.parse_args()

    main(args)
