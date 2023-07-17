import os

MOTION_COR2_PATH = "/home/kyohei/Applications/MotionCor2_1.6.4_Cuda117_Mar312023"


def motioncor2_command(intif, outmrc, gain=None):
    command = f"""{MOTION_COR2_PATH} -InTiff {intif} \
        -OutMrc {outmrc} \
        -Patch 5 5 -Iter 10 -Tol 0.5 -Throw 2 \
        -Kv 300 -PixlSize 0.5 -FmDose 1.2 \
        """
    if gain is None:
        return command
    else:
        print("gain is used")
        return command + f" -Gain {gain}"


def main():
    experiments = sorted(
        [
            os.path.join(CRYOEM_DATADIR, f)
            for f in os.listdir(CRYOEM_DATADIR)
            if f.isnumeric()
        ]
    )

    for ex in experiments:
        tiff_list = sorted([os.path.join(ex, f) for f in os.listdir(ex) if ".tif" in f])
        gain_file = [f for f in tiff_list if "gain" in f]
        for intif in tiff_list:
            exnum, filename = intif.split("/")[-2], intif.split("/")[-1]
            outdir = os.path.join(MRC_OUTDIR, SHORT_OR_ORIGINAL, exnum)
            os.system("mkdir -p " + outdir)

            outmrc = os.path.join(
                outdir, filename.replace(".tiff", ".mrc").replace(".tif", ".mrc")
            )

            if os.path.exists(outmrc):
                continue

            com = motioncor2_command(intif, outmrc)
            os.system(com)


if __name__ == "__main__":
    import argparse

    # SHORT_OR_ORIGINAL = "cryoEM-data" #short_or
    # SHORT_OR_ORIGINAL = "shortTIFF"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--SHORT_OR_ORIGINAL",
        type=str,  # choices=["cryoEM-data", "shortTIFF"]
    )
    args = parser.parse_args()
    print(args)
    SHORT_OR_ORIGINAL = args.SHORT_OR_ORIGINAL
    CRYOEM_DATADIR = f"/media/kyohei/forAI/{SHORT_OR_ORIGINAL}"
    MRC_OUTDIR = f"/media/kyohei/forAI/mrc_by_MotionCor"
    # MRC_OUTDIR = f"/home/kyohei/workspace/fastcryo/mrc_by_MotionCor"

    main()
