import torch, glob
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import mrcfile
from PIL import Image
import os
from torchvision import transforms
from pl_bolts.models.gans import SRGAN
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR as CosineAnnealing
import torch.nn as nn

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Tuple
from warnings import warn

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from pl_bolts.callbacks import SRImageLoggerCallback
from pl_bolts.datamodules import TVTDataModule
from pl_bolts.datasets.utils import prepare_sr_datasets
from pl_bolts.models.gans.srgan.components import (
    SRGANDiscriminator,
    SRGANGenerator,
    VGG19FeatureExtractor,
)
from pl_bolts.utils.stability import under_review


DATATYPE = "float32"


class MRCImageDataset(Dataset):
    def __init__(self, dir_path, datatype=DATATYPE, transform=None):
        self.dir_path = dir_path
        self.transform = transform
        if not dir_path.endswith("/"):
            dir_path += "/"
        self.file_list = glob.glob(dir_path + "**/*.mrc", recursive=True)
        self.short_list = sorted(
            [
                f
                for f in self.file_list
                if "shortTIFF" in f
                if os.path.exists(f.replace("/shortTIFF/", "/cryoEM-data/"))
            ]
        )
        original_list = [
            f.replace("/shortTIFF/", "/cryoEM-data/") for f in self.short_list
        ]
        self.original_list = sorted([f for f in original_list if os.path.exists(f)])
        assert len(self.short_list) == len(original_list)
        self.datatype = datatype

    def __len__(self):
        return len(self.original_list)

    def __getitem__(self, idx):
        short_path = self.short_list[idx]
        original_path = self.original_list[idx]
        # print(short_path, original_path)
        assert short_path.split("/")[-1] == original_path.split("/")[-1]

        # image_size = (511, 720) #(1023, 1440)
        image_size = (151, 172)  # (1023, 1440)

        with mrcfile.open(short_path, "r") as mrc:
            short_img = mrc.data.astype(self.datatype)
            short_img = short_img[: image_size[0], : image_size[1]]

        with mrcfile.open(original_path, "r") as mrc:
            original_img = mrc.data.astype(self.datatype)
            original_img = original_img[: image_size[0], : image_size[1]]

        # The original image (higher resolution)
        y = self.transform(original_img) if self.transform else original_img

        # Creating a lower resolution version
        X = self.transform(short_img) if self.transform else short_img

        return X, y


class MRCImageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, datatype="float32"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        self.datatype = datatype

    def setup(self, stage=None):
        self.mrc_dataset = MRCImageDataset(self.data_dir, self.datatype, self.transform)

    def train_dataloader(self):
        return DataLoader(
            self.mrc_dataset, shuffle=True, num_workers=4, batch_size=self.batch_size
        )

    def val_dataloader(self):
        return DataLoader(
            self.mrc_dataset, shuffle=False, num_workers=4, batch_size=self.batch_size
        )

    def test_dataloader(self):
        return DataLoader(
            self.mrc_dataset, shuffle=False, num_workers=4, batch_size=self.batch_size
        )


# From lightning bolts
class SRGAN(pl.LightningModule):
    """SRGAN implementation from the paper `Photo-Realistic Single Image Super-Resolution Using a Generative
    Adversarial Network <https://arxiv.org/abs/1609.04802>`__. It uses a pretrained SRResNet model as the generator
    if available.

    Code adapted from `https-deeplearning-ai/GANs-Public <https://github.com/https-deeplearning-ai/GANs-Public>`_ to
    Lightning by:

        - `Christoph Clement <https://github.com/chris-clem>`_

    You can pretrain a SRResNet model with :code:`srresnet_module.py`.

    Example::

        from pl_bolts.models.gan import SRGAN

        m = SRGAN()
        Trainer(gpus=1).fit(m)

    Example CLI::

        # CelebA dataset, scale_factor 4
        python srgan_module.py --dataset=celeba --scale_factor=4 --gpus=1

        # MNIST dataset, scale_factor 4
        python srgan_module.py --dataset=mnist --scale_factor=4 --gpus=1

        # STL10 dataset, scale_factor 4
        python srgan_module.py --dataset=stl10 --scale_factor=4 --gpus=1
    """

    def __init__(
        self,
        image_channels: int = 3,
        feature_maps_gen: int = 64,
        feature_maps_disc: int = 64,
        num_res_blocks: int = 16,
        # scale_factor: int = 4,
        generator_checkpoint: Optional[str] = None,
        learning_rate: float = 1e-4,
        scheduler_step: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            image_channels: Number of channels of the images from the dataset
            feature_maps_gen: Number of feature maps to use for the generator
            feature_maps_disc: Number of feature maps to use for the discriminator
            num_res_blocks: Number of res blocks to use in the generator
            scale_factor: Not used. Originally, scale factor for the images (either 2 or 4)
            generator_checkpoint: Generator checkpoint created with SRResNet module
            learning_rate: Learning rate
            scheduler_step: Number of epochs after which the learning rate gets decayed
        """
        super().__init__()
        self.save_hyperparameters()

        if generator_checkpoint:
            self.generator = torch.load(generator_checkpoint)
        else:
            self.generator = SRGANGenerator(
                image_channels, feature_maps_gen, num_res_blocks, 1
            )

        self.discriminator = SRGANDiscriminator(image_channels, feature_maps_disc)
        self.vgg_feature_extractor = VGG19FeatureExtractor(image_channels)

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Adam], List[torch.optim.lr_scheduler.MultiStepLR]]:
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.learning_rate
        )
        opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.learning_rate
        )

        sched_disc = torch.optim.lr_scheduler.MultiStepLR(
            opt_disc, milestones=[self.hparams.scheduler_step], gamma=0.1
        )
        sched_gen = torch.optim.lr_scheduler.MultiStepLR(
            opt_gen, milestones=[self.hparams.scheduler_step], gamma=0.1
        )
        return [opt_disc, opt_gen], [sched_disc, sched_gen]

    def forward(self, lr_image: torch.Tensor) -> torch.Tensor:
        """Generates a high resolution image given a low resolution image.

        Example::

            srgan = SRGAN.load_from_checkpoint(PATH)
            hr_image = srgan(lr_image)
        """
        return self.generator(lr_image)

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        hr_image, lr_image = batch

        # Train discriminator
        result = None
        if optimizer_idx == 0:
            result = self._disc_step(hr_image, lr_image)

        # Train generator
        if optimizer_idx == 1:
            result = self._gen_step(hr_image, lr_image)

        return result

    def _disc_step(
        self, hr_image: torch.Tensor, lr_image: torch.Tensor
    ) -> torch.Tensor:
        disc_loss = self._disc_loss(hr_image, lr_image)
        self.log("loss/disc", disc_loss, on_step=True, on_epoch=True)
        return disc_loss

    def _gen_step(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        gen_loss = self._gen_loss(hr_image, lr_image)
        self.log("loss/gen", gen_loss, on_step=True, on_epoch=True)
        return gen_loss

    def _disc_loss(
        self, hr_image: torch.Tensor, lr_image: torch.Tensor
    ) -> torch.Tensor:
        real_pred = self.discriminator(hr_image)
        real_loss = self._adv_loss(real_pred, ones=True)

        _, fake_pred = self._fake_pred(lr_image)
        fake_loss = self._adv_loss(fake_pred, ones=False)

        disc_loss = 0.5 * (real_loss + fake_loss)

        return disc_loss

    def _gen_loss(self, hr_image: torch.Tensor, lr_image: torch.Tensor) -> torch.Tensor:
        fake, fake_pred = self._fake_pred(lr_image)

        perceptual_loss = self._perceptual_loss(hr_image, fake)
        adv_loss = self._adv_loss(fake_pred, ones=True)
        content_loss = self._content_loss(hr_image, fake)

        gen_loss = 0.006 * perceptual_loss + 0.001 * adv_loss + content_loss

        return gen_loss

    def _fake_pred(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fake = self(lr_image)
        fake_pred = self.discriminator(fake)
        return fake, fake_pred

    @staticmethod
    def _adv_loss(pred: torch.Tensor, ones: bool) -> torch.Tensor:
        target = torch.ones_like(pred) if ones else torch.zeros_like(pred)
        adv_loss = F.binary_cross_entropy_with_logits(pred, target)
        return adv_loss

    def _perceptual_loss(
        self, hr_image: torch.Tensor, fake: torch.Tensor
    ) -> torch.Tensor:
        # print("hr_image.shape, fake.shape", hr_image.shape, fake.shape)
        real_features = self.vgg_feature_extractor(hr_image)
        fake_features = self.vgg_feature_extractor(fake)
        # print("real_features.shape, fake_features.shape", real_features.shape, fake_features.shape)
        perceptual_loss = self._content_loss(real_features, fake_features)
        return perceptual_loss

    @staticmethod
    def _content_loss(hr_image: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
        # print("hr_image.shape, fake.shape", hr_image.shape, fake.shape)
        # hr_image.shape, fake.shape torch.Size([2, 512, 3, 4]) torch.Size([2, 512, 12, 18])
        return F.mse_loss(hr_image, fake)

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc", default=64, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--scheduler_step", default=100, type=float)
        return parser


if __name__ == "__main__":
    data_module = MRCImageDataModule(
        "/media/kyohei/forAI/split_images/", datatype=DATATYPE, batch_size=2
    )

    model = SRGAN(
        image_channels=1,
        feature_maps_gen=64,
        feature_maps_disc=64,
        num_res_blocks=16,
        scale_factor=1,
        learning_rate=0.0001,
        scheduler_step=100,
        generator_checkpoint=None,
    )
    model = model.to(torch.float16) if DATATYPE == "float16" else model
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=4)
    trainer.fit(model, datamodule=data_module)
