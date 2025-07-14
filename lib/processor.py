import os
import logging
import boto3
from urllib.parse import urlparse
from io import BytesIO
from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
import torch.nn.functional as F
from genwarp import GenWarp
from extern.ZoeDepth.zoedepth.models.zoedepth import ZoeDepth
from extern.ZoeDepth.zoedepth.utils.misc import colorize
from lib.config import *


def initialize_models():
    logger.info("Loading ZoeDepth and GenWarp models...")
    zoe = torch.hub.load(ZOEDEPTH_REPO_PATH, 'ZoeD_N', source='local', pretrained=True, trust_repo=True).to(DEVICE)
    genwarp_cfg = dict(
        pretrained_model_path=GENWARP_MODEL_DIR,
        checkpoint_name=GENWARP_CHECKPOINT_NAME,
        half_precision_weights=True
    )
    genwarp = GenWarp(cfg=genwarp_cfg)
    return zoe, genwarp

def run_genwarp_pipeline(image_tensor: torch.Tensor, zoe, genwarp) -> Image.Image:
    src_image = image_to_device(image_tensor, DEVICE)
    src_depth = zoe.infer(src_image)
    src_depth = image_to_device(src_depth, DEVICE)

    try:
        fovy = focal_length_to_fov(FOCAL_LENGTH_MM, 24.)
    except Exception:
        fovy = np.deg2rad(55.0)
    fovy = torch.tensor([fovy], device=DEVICE)

    proj_mtx = get_projection_matrix(fovy, 1.0, 0.01, 100.0).to(DEVICE)

    z_up = torch.tensor([[0., 0., 1.]], device=DEVICE)
    src_view_mtx = camera_lookat(
        torch.tensor([[0., 0., 0.]], device=DEVICE),
        torch.tensor([[-1., 0., 0.]], device=DEVICE),
        z_up
    )

    mean_depth = src_depth.mean(dim=(2, 3)).squeeze(1)
    azi = torch.tensor(np.deg2rad(AZIMUTH_DEG), device=DEVICE)
    ele = torch.tensor(np.deg2rad(ELEVATION_DEG), device=DEVICE)
    eye = sph2cart(azi, ele, mean_depth + CAMERA_RADIUS).float()
    at = F.pad(-mean_depth[:, None], (0, 2), mode='constant', value=0)
    tar_view_mtx = camera_lookat(eye + at, at, z_up)
    rel_view_mtx = (tar_view_mtx @ torch.linalg.inv(src_view_mtx.float())).to(DEVICE)

    renders = genwarp(
        src_image=src_image,
        src_depth=src_depth,
        rel_view_mtx=rel_view_mtx,
        src_proj_mtx=proj_mtx,
        tar_proj_mtx=proj_mtx
    )

    return tensor_to_pil(renders['synthesized'])
