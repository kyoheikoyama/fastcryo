#!/usr/bin/env python
# coding: utf-8

import os, sys
import glob
import argparse
import mrcfile
import numpy as np

sys.path.append("../src/utils/")
from utils import clip_image, split_image


def read_mrc(m):
    with mrcfile.open(m) as mrc:
        data = mrc.data
    return data


def save_image(data, outpath):
    with mrcfile.new(outpath, overwrite=True) as mrc:
        mrc.set_data(data)
    print(outpath, " saved")


def read_clip_split(m, n=10):
    data = read_mrc(m)
    data = clip_image(data)
    tiles = split_image(data, n=n)
    return tiles


def save_n_tiles(tiles, op, n=10):
    for i in range(n):
        for j in range(n):
            save_image(tiles[i][j], op.replace(".mrc", f"__split{i}_{j}.mrc"))


def check():
    dir_path = "/media/kyohei/forAI/split_images/"
    file_list = glob.glob(dir_path + "**/*.mrc", recursive=True)
    short_list = sorted([f for f in file_list if "/shortTIFF/" in f])

    original_list = [f.replace("/shortTIFF/", "/EMPIAR/") for f in short_list]
    original_list = sorted([f for f in original_list if os.path.exists(f)])

    for f, o in zip(file_list, original_list):
        assert read_mrc(f).shape == read_mrc(o).shape

    assert len(short_list) == len(original_list)

    print("OK")


def main(args):
    NSPLIT = args.nsplit
    DIR_NAME = args.dir_name

    file_list = glob.glob(DIR_NAME + "**/*.mrc", recursive=True)
    short_list = sorted(
        [
            f
            for f in file_list
            if "/shortTIFF/" in f
            and os.path.exists(f.replace("/shortTIFF/", "/EMPIAR/"))
        ]
    )
    original_list = sorted(
        [f.replace("/shortTIFF/", "/EMPIAR/") for f in short_list if os.path.exists(f)]
    )

    assert len(short_list) == len(original_list)

    for m in short_list:
        tiles = read_clip_split(m, n=NSPLIT)
        op = m.replace("/forAI/mrc_by_MotionCor/", "/forAI/split_images/")
        opdir = "/".join(op.split("/")[:-1])
        os.system(f"mkdir -p {opdir}")
        save_n_tiles(tiles, op, n=NSPLIT)

    for m in original_list:
        tiles = read_clip_split(m, n=NSPLIT)
        op = m.replace("/forAI/mrc_by_MotionCor/", "/forAI/split_images/")
        opdir = "/".join(op.split("/")[:-1])
        os.system(f"mkdir -p {opdir}")
        save_n_tiles(tiles, op, n=NSPLIT)

    check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nsplit", default=10, type=int, help="The number of split")
    parser.add_argument(
        "--dir_name",
        default="/media/kyohei/forAI/mrc_by_MotionCor/",
        type=str,
        help="The directory path",
    )
    args = parser.parse_args()
    main()
