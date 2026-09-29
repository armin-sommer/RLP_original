from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import stable_pretraining as spt
from torchvision.transforms import v2


def nested_transform(transform: Callable[[Any], Any], source: str, target: str) -> Any:
    factory = cast(Callable[..., Any], spt.data.transforms.WrapTorchTransform)
    return factory(transform, source=source, target=target)


def nested_resize(size: int, source: str, target: str) -> Any:
    return nested_transform(v2.Resize(size), source, target)


def image_preprocessor(source: str, target: str, image_size: int) -> Any:
    stats = spt.data.dataset_stats.ImageNet
    compose = cast(Callable[..., Any], spt.data.transforms.Compose)
    to_image = cast(Callable[..., Any], spt.data.transforms.ToImage)
    return compose(
        to_image(**stats, source=source, target=target),
        nested_resize(image_size, source=source, target=target),
    )


__all__ = ["image_preprocessor", "nested_resize", "nested_transform"]
