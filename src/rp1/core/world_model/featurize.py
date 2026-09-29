"""Image preprocessing and dataset featurization for pixel world models."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from rp1.data.base import RowBatch


class _EncodableWorldModel(Protocol):
    obs_key: str
    wants_proprio: bool

    def encode(self, info: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]: ...


def image_transform(
    image_size: int, train_resolution: int | None, dtype: torch.dtype
) -> Callable[[object], torch.Tensor]:
    """ImageNet-normalised ``image_size`` frames for a pixel world model.

    ``train_resolution`` first bottlenecks images through the checkpoint's native
    training resolution (e.g. 64 for world models trained on upscaled 64px frames).
    """
    from torchvision.transforms import v2 as T

    steps = [
        T.ToImage(),
        T.ToDtype(dtype, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if train_resolution and int(train_resolution) != int(image_size):
        steps.append(T.Resize(size=int(train_resolution)))
    steps.append(T.Resize(size=image_size))
    return cast(Callable[[object], torch.Tensor], T.Compose(steps))


def build_featurizer(
    wm: nn.Module,
    device: str,
    img_size: int,
    train_res: int | None,
) -> Callable[[RowBatch], torch.Tensor]:
    """``featurizer(rows) -> (B, D)``: the world model's latents of dataset rows.

    Images are decoded and prepared by :func:`image_transform`; ``train_res``
    must match the resolution used at plan time so cache and deployment share
    an image domain.
    """
    wm = wm.to(device).eval()
    if not callable(getattr(wm, "encode", None)):
        raise TypeError(f"{type(wm).__name__} does not expose encode()")
    encoder = cast(_EncodableWorldModel, wm)

    from io import BytesIO

    from PIL import Image

    tf = image_transform(img_size, train_res, torch.float32)

    def _decode(p: object) -> NDArray[Any]:
        if isinstance(p, (bytes, bytearray, np.bytes_)):
            return np.array(Image.open(BytesIO(bytes(p))))
        return np.asarray(p)

    _mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    wants_proprio = bool(getattr(wm, "wants_proprio", False))

    @torch.no_grad()
    def pixel_featurize(rows: RowBatch) -> torch.Tensor:
        px = rows["pixels"]
        if isinstance(px, np.ndarray) and px.dtype == object:
            # lance returns a ragged object array; decode elements (raw arrays
            # pass through) and re-stack so the vectorized path below applies
            px = np.stack([_decode(p) for p in px])
        if (
            isinstance(px, np.ndarray)
            and px.dtype == np.uint8
            and px.ndim == 4
            and px.shape[1] == img_size
            and px.shape[2] == img_size
            and px.shape[3] == 3
        ):
            # raw already-sized frames: the same math as the per-image transform,
            # vectorized on the device
            x = torch.from_numpy(px).to(device).permute(0, 3, 1, 2).float().div_(255)
            if train_res and int(train_res) != int(img_size):
                x = torch.nn.functional.interpolate(x, size=int(train_res), mode="bilinear", antialias=True)
                x = torch.nn.functional.interpolate(x, size=int(img_size), mode="bilinear")
            imgs = (x - _mean) / _std
        else:
            imgs = torch.stack([tf(_decode(p)) for p in px]).to(device)  # (B,C,H,W)
        enc_in = {"pixels": imgs.unsqueeze(1)}  # (B,1,C,H,W)
        if wants_proprio:
            pro = np.asarray(rows["proprio"], dtype=np.float32).reshape(imgs.shape[0], -1)
            enc_in["proprio"] = torch.from_numpy(pro).unsqueeze(1).to(device)  # (B,1,P)
        out = encoder.encode(enc_in)
        return out["emb"][:, 0]

    return pixel_featurize


__all__ = ["build_featurizer", "image_transform"]
