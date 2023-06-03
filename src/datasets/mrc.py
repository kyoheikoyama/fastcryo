import mrcfile
import glob
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split
import numpy as np


class MRCImageDataset(Dataset):
    def __init__(self, dir_path, datatype="float32", transform=None, imagesize=512):
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
        self.original_list = [f for f in original_list if os.path.exists(f)]
        assert len(self.short_list) == len(original_list)
        self.datatype = datatype
        self.imagesize = imagesize

    def __len__(self):
        return len(self.original_list)

    def __resize__(self, img):
        return cv2.resize(
            img, (self.imagesize, self.imagesize), interpolation=cv2.INTER_AREA
        )

    def __clip__(self, image):
        upper_limit = np.percentile(image, 99)
        lower_limit = np.percentile(image, 0)
        return np.clip(image, lower_limit, upper_limit)

    def clip_resize_transform(self, img):
        # img = self.__clip__(img)
        img = self.__resize__(img)
        img = self.transform(img) if self.transform else img
        return img

    def __getitem__(self, idx):
        short_path = self.short_list[idx]
        original_path = self.original_list[idx]
        assert short_path.split("/")[-1] == original_path.split("/")[-1]

        with mrcfile.open(short_path, "r") as mrc:
            short_img = mrc.data.astype(self.datatype)

        with mrcfile.open(original_path, "r") as mrc:
            original_img = mrc.data.astype(self.datatype)

        X = self.clip_resize_transform(short_img)
        y = self.clip_resize_transform(original_img)

        return X, y


class MRCImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        datatype="float32",
        datasize=False,
        imagesize=512,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize((0.5,), (0.5,))
            ]
        )
        self.datatype = datatype
        self.datasize = datasize
        self.imagesize = imagesize

    def setup(self, stage=None):
        self.mrc_dataset = MRCImageDataset(
            self.data_dir, self.datatype, self.transform, self.imagesize
        )
        print("Dataset size:", len(self.mrc_dataset))
        print("Dataset size mode:", len(self.datasize))

        if self.datasize == "small":
            train_size = int(0.01 * len(self.mrc_dataset))
            val_size = int(0.01 * len(self.mrc_dataset))
        if self.datasize == "dev":
            train_size = int(0.0005 * len(self.mrc_dataset))
            val_size = int(0.0005 * len(self.mrc_dataset))
        elif self.datasize == "hpo":
            train_size = int(0.1 * len(self.mrc_dataset))
            val_size = int(0.1 * len(self.mrc_dataset))
        elif self.datasize == "all":
            train_size = int(0.8 * len(self.mrc_dataset))
            val_size = int(0.1 * len(self.mrc_dataset))
        else:
            train_size = int(0.8 * len(self.mrc_dataset))
            val_size = int(0.1 * len(self.mrc_dataset))
        test_size = len(self.mrc_dataset) - train_size - val_size

        (
            self.mrc_train_dataset,
            self.mrc_val_dataset,
            self.mrc_test_dataset,
        ) = random_split(self.mrc_dataset, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(
            self.mrc_train_dataset,
            shuffle=True,
            num_workers=8,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return DataLoader(
            self.mrc_val_dataset,
            shuffle=False,
            num_workers=8,
            batch_size=self.batch_size,
        )

    def test_dataloader(self):
        return DataLoader(
            self.mrc_test_dataset,
            shuffle=False,
            num_workers=8,
            batch_size=self.batch_size,
        )
