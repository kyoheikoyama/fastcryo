import mrcfile
import glob
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from torch.utils.data import random_split, Dataset, Subset
import numpy as np
import pandas as pd


def remove_padding(image, padding_value=-1):
    non_padding_indices = np.where(image != padding_value)
    min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
    min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
    non_padding_part = image[min_y : max_y + 1, min_x : max_x + 1]
    return non_padding_part


def get_minmax_xy_of_non_padding(image, padding_value=-1):
    non_padding_indices = np.where(image != padding_value)
    min_y, max_y = np.min(non_padding_indices[0]), np.max(non_padding_indices[0])
    min_x, max_x = np.min(non_padding_indices[1]), np.max(non_padding_indices[1])
    return min_x, max_x, min_y, max_y


def pad_image_with_val(image, target_size=(409, 576), val=-1):
    height, width = image.shape[:2]

    # Calculate the required padding size
    padding_vertical = max(0, target_size[0] - height)
    padding_horizontal = max(0, target_size[1] - width)

    # Calculate the padding on each side
    top_padding = padding_vertical // 2
    bottom_padding = padding_vertical - top_padding
    left_padding = padding_horizontal // 2
    right_padding = padding_horizontal - left_padding

    # Pad the image with val
    padded_image = cv2.copyMakeBorder(
        image,
        top_padding,
        bottom_padding,
        left_padding,
        right_padding,
        cv2.BORDER_CONSTANT,
        value=val,
    )

    return padded_image


def get_split_indices(mrc_dataset, testratio=0.1, shortlist=True):
    EMPIAR_IDS = [10828, 10877, 10997, 11029, 11084, 11217, 11244, 11349, 11351, 11387]
    test_list = []
    train_list = []
    if shortlist:
        filelist = mrc_dataset.short_list

    for em in EMPIAR_IDS:
        images = (
            pd.Series([f.split("__split")[0] for f in filelist if str(em) in f])
            .drop_duplicates()
            .reset_index()
        )

        test = images.sample(frac=testratio, random_state=0)
        train = images.drop(index=test.index)

        testpath = [f for f in filelist if any([t in f for t in test[0]])]
        trainpath = [f for f in filelist if any([t in f for t in train[0]])]

        test_list.extend(testpath)
        train_list.extend(trainpath)

    test_indices = [i for i, x in enumerate(filelist) if x in test_list]
    train_indices = [i for i, x in enumerate(filelist) if x in train_list]
    return train_indices, test_indices


class MRCImageDataset(Dataset):
    def __init__(
        self, dir_path, datatype="float32", transform=None, imagesize=512, paddingval=-1
    ):
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
        self.paddingval = paddingval

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
        # img = self.__resize__(img)
        img = pad_image_with_val(img, val=self.paddingval)
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
        split_way="random",
        paddingval=-1,
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
        self.split_way = split_way
        self.paddingval = paddingval

    def setup(self, stage=None):
        self.mrc_dataset = MRCImageDataset(
            self.data_dir,
            self.datatype,
            self.transform,
            self.imagesize,
            self.paddingval,
        )
        print("Dataset size:", len(self.mrc_dataset))
        print("Dataset size mode:", self.datasize)

        if self.split_way == "random":
            if self.datasize == "dev":
                train_size = int(0.01 * len(self.mrc_dataset))
                val_size = int(0.01 * len(self.mrc_dataset))
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
        elif self.split_way == "images":
            assert self.datasize == "all", "datasize must be all for split_way=images"
            train_val_indices, test_indices = get_split_indices(
                self.mrc_dataset, testratio=0.1, shortlist=True
            )
            val_indices = (
                pd.Series(train_val_indices).sample(frac=float(1 / 9)).values
            )  # train, val, test = 8:1:1
            train_indices = [i for i in train_val_indices if i not in val_indices]
            self.mrc_train_dataset = Subset(self.mrc_dataset, train_indices)
            self.mrc_val_dataset = Subset(self.mrc_dataset, val_indices)
            self.mrc_test_dataset = Subset(self.mrc_dataset, test_indices)

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
