#!/usr/bin/env python
# coding: utf-8

import os, gc
import argparse
from tqdm import tqdm

from PIL import Image
import pandas as pd
import multiprocessing as mp


def save_first_n_images_as_tiff(input_tiff, n=20):
    output_tiff = input_tiff.replace(f"/{args.datadir}/", "/shortTIFF/")
    assert output_tiff != input_tiff
    outdir = "/".join(output_tiff.split("/")[:-1])
    if not os.path.exists(outdir):
        os.system(f"mkdir {outdir}")
    if os.path.exists(output_tiff):
        return None

    images = []
    with Image.open(input_tiff) as tiff_img:
        for i in range(n):
            try:
                tiff_img.seek(i)
                extracted_image = tiff_img.copy()
                images.append(extracted_image)
                print(f"Extracted image {i+1}")
            except EOFError:
                # Reached the end of the TIFF file
                break

    if images:
        images[0].save(
            output_tiff,
            save_all=True,
            append_images=images[1:],
            compression="tiff_deflate",
        )
        print(f"Saved first {n} images as: {output_tiff}")
    else:
        print("No images found in the input TIFF file.")


def count_frames_in_tiff(tiffpath):
    try:
        with Image.open(tiffpath) as tiff_img:
            nframes = tiff_img.n_frames
        return nframes
    except:
        print(f"Error in {tiffpath}, returning 0    ")
        return 0


flatten = lambda xxlis: [x for xx in xxlis for x in xx]


def main(args):
    datadir = args.hdname + "/" + args.datadir + "/"

    cryoEM_datanumber_list = [
        os.path.join(datadir, f) for f in os.listdir(datadir) if f.isnumeric()
    ]
    print(cryoEM_datanumber_list)
    tiffpath_list = [
        [os.path.join(c, p) for p in os.listdir(c) if ".tif" in p]
        for c in cryoEM_datanumber_list
    ]

    tiffpath_list = flatten(tiffpath_list)

    # with mp.Pool(14) as p:
    #     count_list = p.map(count_frames_in_tiff, tiffpath_list)

    if not os.path.exists(f"data_in_HD_{args.datetime}.csv"):
        count_list = [count_frames_in_tiff(p) for p in tqdm(tiffpath_list)]

        df = pd.DataFrame([count_list, tiffpath_list]).T
        df.columns = ["nframes", "tiffpath"]

        print(df.nframes.value_counts())

        df["data_number"] = df["tiffpath"].str.split("/").apply(lambda x: x[5])
        df["out_path"] = df["tiffpath"].apply(
            lambda t: t.replace(f"/{args.datadir}/", "/shortTIFF/")
        )
        df["filename"] = df["tiffpath"].str.split("/").apply(lambda x: x[-1])
        df.to_csv(f"data_in_HD_{args.datetime}.csv", index=None)
    else:
        df = pd.read_csv(f"data_in_HD_{args.datetime}.csv")

    with mp.Pool(4) as p:
        p.map(save_first_n_images_as_tiff, df[df.nframes > 20]["tiffpath"])
    df.to_csv(f"data_in_HD_{args.datetime}.csv", index=None)

    print(df["out_path"].str.replace("/shortTIFF/", "/summingup/"))


if __name__ == "__main__":
    from datetime import datetime

    yyyymmdd = datetime.now().strftime("%Y-%m%d")

    parser = argparse.ArgumentParser(description="Process some TIFF files.")
    # parser.add_argument('--datadir', type=str, help='The data directory', default="cryoEM-data")
    parser.add_argument(
        "--hdname", type=str, help="The hard disk name", default="/media/kyohei/forAI"
    )
    parser.add_argument(
        "--datadir", type=str, help="The data directory", default="EMPIAR"
    )
    parser.add_argument(
        "--datetime", type=str, help="The data directory", default="2023-0717"
    )

    args = parser.parse_args()

    main(args)
