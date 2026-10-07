"""
Training data: TCGA tiles stored as WebDataset shards, with pre-computed teacher embeddings.

Each sample of a shard contains, for every teacher `t`:
    - {key}.jpg                     : the 224x224 tile,
    - {key}.{t}.npy                 : the teacher CLS embedding, shape (D_cls,),
    - {key}.{t}_dense_embs.npy      : K sampled teacher patch embeddings, shape (K, D_patch),
    - {key}.{t}_sampled_indices.npy : positions of these K patches in the teacher grid, shape (K,).
"""

import glob
import io
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import webdataset as wds
from PIL import Image


def build_transform(img_size: int = 224) -> T.Compose:
    """
    Training preprocessing (no augmentation): resize + ImageNet normalization.

    Args:
        img_size (int): Output resolution.

    Returns:
        T.Compose: Transform mapping a PIL image to a (3, img_size, img_size) tensor.
    """
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def decode_image(data: bytes) -> Image.Image:
    """
    Decodes the JPEG bytes of a tile with OpenCV (as done to train the released models).

    Note: OpenCV returns channels in the order stored by the encoder that wrote the shards.

    Args:
        data (bytes): Encoded image.

    Returns:
        Image.Image: Decoded image.
    """
    return Image.fromarray(
        cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    )


def to_tensor(data) -> torch.Tensor:
    """
    Args:
        data (bytes | np.ndarray): Raw `.npy` bytes or an already decoded array.

    Returns:
        torch.Tensor: The array as a tensor.
    """
    array = data if isinstance(data, np.ndarray) else np.load(io.BytesIO(data))
    return torch.from_numpy(array)


def make_loader(
    root: str, teachers: list, batch_size: int, num_workers: int
) -> wds.WebLoader:
    """
    Builds an infinite, shuffled loader over all `*.tar` shards of `root`.

    Args:
        root (str): Folder containing the WebDataset shards.
        teachers (list[str]): Teachers whose embeddings are loaded, in this order.
        batch_size (int): Batch size.
        num_workers (int): Number of loading workers.

    Returns:
        wds.WebLoader: Yields batches `(images, cls, dense, indices)` where `cls`, `dense`
            and `indices` are lists with one batched tensor per teacher.
    """
    shards = sorted(glob.glob(os.path.join(root, "*.tar")))
    assert shards, f"No .tar shards found in {root}"
    transform = build_transform()
    n = len(teachers)
    keys = (
        ["jpg"]
        + [f"{t}.npy" for t in teachers]
        + [f"{t}_dense_embs.npy" for t in teachers]
        + [f"{t}_sampled_indices.npy" for t in teachers]
    )

    def preprocess(sample):
        image = transform(decode_image(sample[0]))
        tensors = [to_tensor(x) for x in sample[1:]]
        tensors[2 * n :] = [x.long() for x in tensors[2 * n :]]  # patch indices
        return (image, *tensors)

    dataset = (
        wds.WebDataset(shards, resampled=True, shardshuffle=False)
        .shuffle(1000)
        .to_tuple(*keys)
        .map(preprocess)
    )

    def split(batch):
        # (image, cls x n, dense x n, idx x n) -> (image, [cls], [dense], [idx])
        return batch[0], batch[1 : 1 + n], batch[1 + n : 1 + 2 * n], batch[1 + 2 * n :]

    workers = (
        dict(prefetch_factor=2, persistent_workers=True) if num_workers > 0 else {}
    )
    loader = wds.WebLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        **workers,
    )
    return loader.map(split)
