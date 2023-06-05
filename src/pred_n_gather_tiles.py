from argparse import ArgumentParser
import yaml, os
from datasets.mrc import (
    MRCImageDataModule,
    MRCImageDataset,
    remove_padding,
    get_minmax_xy_of_non_padding,
)
from models.srgan import SRGAN, DATATYPE
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm


def main(args) -> float:
    # Load the YAML file
    with open(args.hparams, "r") as file:
        data = yaml.safe_load(file)
    hpdict = dict(data)

    dm = MRCImageDataModule(
        args.datalocation,
        datatype=DATATYPE,
        batch_size=args.batch_size,
        datasize=args.datasize,
        split_way="images",
    )
    dm.setup()
    ds = dm.mrc_dataset

    dl = DataLoader(
        ds,
        shuffle=False,
        num_workers=8,
        batch_size=1,
    )
    model = SRGAN.load_from_checkpoint(args.checkpoint_path)

    ## ## eval mode to stop backprop
    model = model.eval()

    i = 0

    for xx, xsr in tqdm(dl):
        xlow = xx.squeeze().detach().numpy()
        min_x, max_x, min_y, max_y = get_minmax_xy_of_non_padding(xlow)

        yp = model(xx)
        yp = yp.squeeze().detach().numpy()
        yp = yp[min_y : max_y + 1, min_x : max_x + 1]

        # ind = dm.mrc_dataset.indices[i]
        shortfile = dm.mrc_dataset.short_list[i]
        newf = shortfile.replace("/split_images/shortTIFF/", "/ypred/").replace(
            ".mrc", ".npy"
        )
        directory = os.path.dirname(newf)
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(newf, yp)  # Save the array to a binary file in NumPy `.npy` format

        i += 1


if __name__ == "__main__":
    """
    Example::
            python pred_n_gather_tiles.py --datasize all --hparams ../hparams/hparams_srgan.yaml --batch_size 3 \
            --checkpoint_path /media/kyohei/forAI/lightning_logs/srgan/version_5/checkpoints/best-checkpoint-v2.ckpt
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
        default=os.path.join(script_path, "../hparams/hparams_srgan.yaml"),
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
        "--checkpoint_path",
        type=str,
    )
    args = parser.parse_args()

    main(args)
