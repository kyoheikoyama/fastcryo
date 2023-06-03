"""Adapted from: https://github.com/https-deeplearning-ai/GANs-Public."""
from argparse import ArgumentParser
from typing import Any, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_bolts.callbacks import SRImageLoggerCallback

# from pl_bolts.datasets.utils import prepare_sr_datasets
from pl_bolts.utils.stability import under_review

from models.srgan import SRGANGenerator, MRCImageDataModule


@under_review()
class SRResNet(pl.LightningModule):
    """SRResNet implementation from the paper `Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network <https://arxiv.org/abs/1609.04802>`__. A pretrained SRResNet model is used as the generator
    for SRGAN.
    Example CLI::
        python pretrain_srresnets_for_srgan.py
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_maps: int = 64,
        num_res_blocks: int = 8,
        learning_rate: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            image_channels: Number of channels of the images from the dataset
            feature_maps: Number of feature maps to use
            num_res_blocks: Number of res blocks to use in the generator
            learning_rate: Learning rate
        """
        super().__init__()
        self.save_hyperparameters()

        self.srresnet = SRGANGenerator(image_channels, feature_maps, num_res_blocks, 1)

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(
            self.srresnet.parameters(), lr=self.hparams.learning_rate
        )

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        """Creates a high resolution image given a low resolution image.

        Example::

            srresnet = SRResNet.load_from_checkpoint(PATH)
            hr_image = srresnet(lr_image)
        """
        return self.srresnet(lr_image)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("loss/train", loss, on_epoch=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("loss/val", loss, sync_dist=True)
        return loss

    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss = self._loss(batch)
        self.log("loss/test", loss, sync_dist=True)
        return loss

    def _loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        hr_image, lr_image = batch
        fake = self(lr_image)
        loss = F.mse_loss(hr_image, fake)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--feature_maps", default=32, type=int)
        parser.add_argument("--learning_rate", default=2e-4, type=float)
        parser.add_argument("--num_res_blocks", default=8, type=int)
        return parser


@under_review()
def cli_main(args=None):
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--log_interval", default=1, type=int)
    parser.add_argument("--scale_factor", default=4, type=int)
    parser.add_argument(
        "--save_model_checkpoint", dest="save_model_checkpoint", action="store_true"
    )
    parser.add_argument(
        "--datalocation",
        action="store_true",
        default="/media/kyohei/forAI/split_images/",
    )

    parser = SRResNet.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    dm = MRCImageDataModule(
        args.datalocation,
        datatype="float32",
        batch_size=4,  # 4 x 512 x 512 < 8GB
        datasize="hpo",
        imagesize=512,
    )

    from utils.utils import equal_var_init, weights_init

    model = SRResNet(**vars(args), image_channels=1)
    equal_var_init(model=model)
    model.apply(weights_init)

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            SRImageLoggerCallback(log_interval=args.log_interval, scale_factor=1)
        ],
        max_epochs=30,
        logger=pl.loggers.TensorBoardLogger(
            save_dir="/media/kyohei/forAI/lightning_logs",
            name="srresnet",
            # version=f"ver0",
            # default_hp_metric=False,
        ),
        accelerator="gpu",
        devices=1,
    )
    trainer.fit(model, dm)

    if args.save_model_checkpoint:
        torch.save(
            model.srresnet,
            f"/media/kyohei/forAI/model_checkpoints/srresnet-{args.dataset}.pt",
        )


if __name__ == "__main__":
    """
    how to view the tensorboard:
    $ tensorboard --logdir lightning_logs/
    """
    cli_main()
