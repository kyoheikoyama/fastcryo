import os
from argparse import ArgumentParser
from typing import Any, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.nn as nn
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from pl_bolts.models.gans.srgan.components import (
    SRGANDiscriminator,
    # SRGANGenerator,
    VGG19FeatureExtractor,
    ResidualBlock,
)
from pl_bolts.models.gans import SRGAN

from datasets.mrc import MRCImageDataModule, MRCImageDataset


DATATYPE = "float32"
pl.seed_everything(1234)


class SRGANGenerator(nn.Module):
    def __init__(
        self,
        image_channels: int,
        feature_maps: int = 32,
        num_res_blocks: int = 8,
        num_ps_blocks: int = 2,
    ) -> None:
        super().__init__()
        # Input block (k9n64s1)
        self.input_block = nn.Sequential(
            nn.Conv2d(image_channels, feature_maps, kernel_size=9, padding=4),
            nn.PReLU(),
        )

        # B residual blocks (k3n64s1)
        res_blocks = []
        for _ in range(num_res_blocks):
            res_blocks += [ResidualBlock(feature_maps)]

        # k3n64s1
        res_blocks += [
            nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_maps),
        ]
        self.res_blocks = nn.Sequential(*res_blocks)

        # PixelShuffle blocks (k3n256s1)
        ps_blocks = []
        for _ in range(num_ps_blocks):
            ps_blocks += [
                nn.Conv2d(feature_maps, feature_maps, kernel_size=3, padding=1),
                nn.PReLU(),
            ]
        self.ps_blocks = nn.Sequential(*ps_blocks)

        # Output block (k9n3s1)
        self.output_block = nn.Sequential(
            nn.Conv2d(feature_maps, image_channels, kernel_size=9, padding=4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_res = self.input_block(x)
        x = x_res + self.res_blocks(x_res)
        x = self.ps_blocks(x)
        x = self.output_block(x)
        return x


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
        feature_maps_gen: int = 32,
        feature_maps_disc: int = 32,
        num_res_blocks: int = 16,
        # scale_factor: int = 4,
        generator_checkpoint: Optional[str] = None,
        learning_rate_gen: float = 1e-4,
        learning_rate_disc: float = 2e-6,
        scheduler_step: int = 100,
        adv_loss_coeff: float = 10,
        content_loss_coeff: float = 1e-2,
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
        self.generator = SRGANGenerator(
            image_channels, feature_maps_gen, num_res_blocks, 1
        )
        if generator_checkpoint:
            self.generator.load_state_dict(
                torch.load(generator_checkpoint)
            )  # Load the weights
        else:
            assert False

        self.discriminator = SRGANDiscriminator(image_channels, feature_maps_disc)
        self.vgg_feature_extractor = VGG19FeatureExtractor(image_channels)
        self.adv_loss_coeff = adv_loss_coeff
        self.content_loss_coeff = content_loss_coeff

    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Adam], List[torch.optim.lr_scheduler.MultiStepLR]]:
        opt_disc = torch.optim.Adam(
            self.discriminator.parameters(), lr=self.hparams.learning_rate_disc
        )
        opt_gen = torch.optim.Adam(
            self.generator.parameters(), lr=self.hparams.learning_rate_gen
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

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        hr_image, lr_image = batch
        gen_loss = self._gen_loss(hr_image, lr_image)
        self.log("val_loss", gen_loss)

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
        # discriminator prediction on fake image, ultimately we want this to be 0.99 as a perfect discriminator
        # we want the discriminator to think the fake image is real, so we want the discriminator to output 1 for the fake image
        fake, fake_pred = self._fake_pred(lr_image)

        # adversarial loss := log(D(G(z)))
        adv_loss = F.binary_cross_entropy_with_logits(
            fake_pred, torch.ones_like(fake_pred)
        )  # smaller is better
        perceptual_loss = self._perceptual_loss(hr_image, fake)  # smaller is better
        content_loss = self._content_loss(hr_image, fake)  # smaller is better

        self.log("loss/perceptual_loss", perceptual_loss, on_step=True, on_epoch=True)
        self.log("loss/adv_loss", adv_loss, on_step=True, on_epoch=True)
        self.log("loss/content_loss", content_loss, on_step=True, on_epoch=True)

        # gen_loss = 0.006 * perceptual_loss + 0.001 * adv_loss + content_loss
        gen_loss = (
            perceptual_loss
            + self.adv_loss_coeff * adv_loss
            + self.content_loss_coeff * content_loss
        )

        return gen_loss

    def _fake_pred(self, lr_image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fake = self.forward(lr_image)
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
        parser.add_argument("--feature_maps_gen", default=32, type=int)
        parser.add_argument("--feature_maps_disc", default=32, type=int)
        parser.add_argument("--learning_rate", default=1e-4, type=float)
        parser.add_argument("--scheduler_step", default=100, type=float)
        return parser
