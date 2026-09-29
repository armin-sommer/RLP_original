"""Rename a PLDM checkpoint's keys to the LeWM module layout.

The authors' PLDM export stores its encoder under Hugging Face ViTModel names
(``encoder.encoder.layer.N.attention.attention.query`` ...), while LeWM expects
``encoder.layers.N.attention.q_proj``. The architectures are the same (ViT-tiny,
patch 14, 224px), every key maps one to one, and the other components already
share names.

Pair the output with a LeWM ``config.json``, such as the one in
``assets/core/world_model/cube_pldm``.

Example::

    pixi run prepare job=convert_pldm preparation.src=<pldm.pt> preparation.dst=<weights.pt>
"""

import re

import torch
from omegaconf import DictConfig

from rp1.utils.config import phase_config
from rp1.utils.logging import logger


def remap_key(key: str) -> str:
    match = re.match(r"encoder\.encoder\.layer\.(\d+)\.(.*)", key)
    if not match:
        return key
    layer, rest = match.groups()
    rest = (
        rest.replace("attention.attention.query", "attention.q_proj")
        .replace("attention.attention.key", "attention.k_proj")
        .replace("attention.attention.value", "attention.v_proj")
        .replace("attention.output.dense", "attention.o_proj")
        .replace("intermediate.dense", "mlp.fc1")
        .replace("output.dense", "mlp.fc2")
    )
    return f"encoder.layers.{layer}.{rest}"


def run(cfg: DictConfig) -> None:
    args = phase_config(cfg, "preparation")
    checkpoint = torch.load(args.src, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    remapped = {remap_key(key): value for key, value in state_dict.items()}
    if len(remapped) != len(state_dict):
        raise ValueError("key collision during remap")
    renamed = sum(1 for key in state_dict if remap_key(key) != key)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        checkpoint["state_dict"] = remapped
        torch.save(checkpoint, args.dst)
    else:
        torch.save(remapped, args.dst)
    logger.success(f"Converted PLDM checkpoint: keys={len(remapped)} renamed={renamed} -> {args.dst}")
