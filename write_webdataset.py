"""
Build the WebDataset shards read by `train.py` from TRIDENT or CLAM patch features.

Inputs:
    - the whole-slide images (any format readable by OpenSlide),
    - one TRIDENT / CLAM feature folder per teacher, containing `<slide>.h5` files with
          "features" (N, D) : one embedding per patch,
          "coords"   (N, 2) : level-0 (x, y) coordinates of the patches.

For every patch, the tile is read from the slide at its coordinates and resized to 224x224,
and the features of all teachers at these coordinates are stored with it. Each sample of a
shard contains the tile (`jpg`) and one `<teacher>.npy` embedding per teacher.

Usage:
    python write_webdataset.py --wsi-dir /data/tcga/wsis --out-dir /data/tcga/shards \\
        --features uni2h=/data/trident/20x_256px_0px_overlap/features_uni_v2 \\
                   virchow2=/data/trident/20x_256px_0px_overlap/features_virchow2 ... \\
        --tiles-per-slide 1000 --num-jobs 100 --job-id $SLURM_ARRAY_TASK_ID
"""

import argparse
import os
from multiprocessing import Pool
from pathlib import Path

import cv2
import h5py
import numpy as np
import openslide
import webdataset as wds
from PIL import Image
from tqdm import tqdm

from configs import TEACHER_CLS_DIM

TILE_SIZE = 224


def parse_features(pairs: list) -> dict:
    """
    Parses the `teacher=folder` arguments.

    Args:
        pairs (list[str]): Items of the form "teacher=/path/to/features".

    Returns:
        dict[str, str]: Feature folder of each teacher (in the given order).
    """
    folders = {}
    for p in pairs:
        teacher, sep, folder = p.partition("=")
        assert sep and teacher and folder, f"Expected teacher=folder, got '{p}'"
        assert os.path.isdir(folder), f"Not a folder: {folder}"
        folders[teacher] = folder
    return folders


def list_slides(wsi_dir: str, folders: dict) -> list:
    """
    Lists the slides that have features for every teacher.

    Args:
        wsi_dir (str): Folder of WSIs.
        folders (dict[str, str]): Feature folder of each teacher.

    Returns:
        list[tuple[str, str]]: Sorted (slide name, slide path) pairs.
    """
    slides = {p.stem: str(p) for p in Path(wsi_dir).iterdir() if p.is_file()}
    names = set(slides)
    for folder in folders.values():
        names &= {p.stem for p in Path(folder).glob("*.h5")}
    skipped = len(slides) - len(names)
    if skipped:
        print(f"[skip] {skipped} slides without features for every teacher")
    return [(n, slides[n]) for n in sorted(names)]


def read_features(path: str):
    """
    Reads a TRIDENT / CLAM feature file.

    Args:
        path (str): Path of `<slide>.h5`.

    Returns:
        tuple[np.ndarray, np.ndarray, dict]: Features (N, D), level-0 coordinates (N, 2) and the
            attributes of "coords" (patch geometry, saved by TRIDENT only).
    """
    with h5py.File(path, "r") as f:
        return (
            f["features"][:],
            f["coords"][:].astype(np.int64),
            dict(f["coords"].attrs),
        )


def read_tile(
    slide: openslide.OpenSlide, x: int, y: int, size_level0: int
) -> np.ndarray:
    """
    Reads one patch at the pyramid level closest to the final resolution and resizes it to 224x224.

    Args:
        slide (openslide.OpenSlide): Opened slide.
        x, y (int): Level-0 coordinates of the top-left corner.
        size_level0 (int): Patch side in level-0 pixels.

    Returns:
        np.ndarray: RGB tile of shape (224, 224, 3), uint8.
    """
    level = slide.get_best_level_for_downsample(size_level0 / TILE_SIZE)
    size = max(1, round(size_level0 / slide.level_downsamples[level]))
    region = slide.read_region((int(x), int(y)), level, (size, size)).convert("RGB")
    if region.size != (TILE_SIZE, TILE_SIZE):
        region = region.resize((TILE_SIZE, TILE_SIZE), Image.BILINEAR)
    return np.asarray(region)


def process_slide(job: tuple):
    """
    Builds the samples of one slide. Runs in a worker process.

    The patches kept are those present in the feature files of every teacher (matched by
    coordinates), optionally subsampled to `tiles_per_slide`.

    Args:
        job (tuple): (slide index, name, path, teacher folders, tiles per slide, level-0 patch size, seed).

    Returns:
        tuple[str, list[dict]]: Slide name and its samples (without keys).
    """
    index, name, path, folders, tiles_per_slide, size_level0, seed = job
    feats, coord_rows = {}, {}
    for t, folder in folders.items():
        f, xy, attrs = read_features(os.path.join(folder, f"{name}.h5"))
        assert f.shape[1] >= TEACHER_CLS_DIM.get(
            t, 0
        ), f"{t}: feature dim {f.shape[1]} < {TEACHER_CLS_DIM[t]} ({name})"
        size_level0 = size_level0 or attrs.get("patch_size_level0")
        feats[t] = f
        coord_rows[t] = {(int(a), int(b)): r for r, (a, b) in enumerate(xy)}
    assert (
        size_level0
    ), "The patch size is unknown: pass --patch-size-level0 (needed for CLAM features)."

    # Patches shared by all teachers, in the order of the first teacher.
    first = next(iter(coord_rows.values()))
    coords = [c for c in first if all(c in rows for rows in coord_rows.values())]
    if tiles_per_slide and len(coords) > tiles_per_slide:
        rng = np.random.default_rng([seed, index])  # deterministic per slide
        coords = [
            coords[i]
            for i in np.sort(rng.choice(len(coords), tiles_per_slide, replace=False))
        ]

    slide = openslide.OpenSlide(path)
    samples = []
    for x, y in coords:
        tile = read_tile(slide, x, y, int(size_level0))
        # cv2 keeps the channel order of the array: the RGB tile is decoded as RGB by train.py.
        sample = {
            "jpg": cv2.imencode(".jpg", tile, [cv2.IMWRITE_JPEG_QUALITY, 95])[
                1
            ].tobytes()
        }
        for t in feats:
            sample[f"{t}.npy"] = feats[t][coord_rows[t][(x, y)]].astype(np.float16)
        samples.append(sample)
    slide.close()
    return name, samples


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--wsi-dir", required=True, help="Folder of whole-slide images."
    )
    parser.add_argument(
        "--features",
        nargs="+",
        required=True,
        help="teacher=folder pairs (TRIDENT / CLAM feature folders).",
    )
    parser.add_argument(
        "--out-dir", required=True, help="Output folder of .tar shards."
    )
    parser.add_argument(
        "--tiles-per-slide",
        type=int,
        default=None,
        help="Random patches kept per slide (default: all).",
    )
    parser.add_argument(
        "--patch-size-level0",
        type=int,
        default=None,
        help="Patch side in level-0 pixels (read from TRIDENT files; required for CLAM).",
    )
    parser.add_argument(
        "--shard-size", type=int, default=10_000, help="Samples per shard."
    )
    parser.add_argument(
        "--num-jobs", type=int, default=1, help="Slides are split over this many jobs."
    )
    parser.add_argument("--job-id", type=int, default=0)
    parser.add_argument(
        "--num-workers", type=int, default=8, help="Slides read in parallel."
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    folders = parse_features(args.features)
    slides = list_slides(args.wsi_dir, folders)[
        args.job_id :: args.num_jobs
    ]  # slides of this job
    print(f"Job {args.job_id}: {len(slides)} slides, teachers: {list(folders)}")
    jobs = [
        (
            i * args.num_jobs + args.job_id,
            n,
            p,
            folders,
            args.tiles_per_slide,
            args.patch_size_level0,
            args.seed,
        )
        for i, (n, p) in enumerate(slides)
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    pattern = os.path.join(args.out_dir, f"{args.job_id:03d}_%06d.tar")
    n = 0
    with Pool(args.num_workers) as pool, wds.ShardWriter(
        pattern, maxcount=args.shard_size, verbose=0
    ) as sink:
        for name, samples in tqdm(pool.imap(process_slide, jobs), total=len(jobs)):
            for j, sample in enumerate(samples):
                sink.write(
                    {"__key__": f"{name}_{j:06d}".replace(".", "-"), **sample}
                )  # keys must not contain dots
            n += len(samples)
    print(f"Job {args.job_id}: wrote {n} samples to {args.out_dir}")


if __name__ == "__main__":
    main()
