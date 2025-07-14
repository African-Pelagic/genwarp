from PIL import Image
import torch
from torchvision.transforms.functional import to_tensor, to_pil_image
import numpy as np
import torch.nn.functional as F

def focal_length_to_fov(focal_mm: float, sensor_height_mm: float) -> float:
    return 2 * np.arctan(sensor_height_mm / (2 * focal_mm))

def get_projection_matrix(fovy: torch.Tensor, aspect_wh: float, near: float, far: float) -> torch.Tensor:
    tan_half_fovy = torch.tan(fovy / 2)
    m = torch.zeros((1, 4, 4), dtype=fovy.dtype)
    m[:, 0, 0] = 1 / (aspect_wh * tan_half_fovy)
    m[:, 1, 1] = 1 / tan_half_fovy
    m[:, 2, 2] = -(far + near) / (far - near)
    m[:, 2, 3] = -(2 * far * near) / (far - near)
    m[:, 3, 2] = -1
    return m

def camera_lookat(eye: torch.Tensor, at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    z = F.normalize(eye - at, dim=-1)
    x = F.normalize(torch.cross(up.expand_as(z), z, dim=-1), dim=-1)
    y = torch.cross(z, x, dim=-1)

    view = torch.eye(4).repeat(eye.shape[0], 1, 1)
    view[:, 0, :3] = x
    view[:, 1, :3] = y
    view[:, 2, :3] = z
    view[:, :3, 3] = -torch.bmm(view[:, :3, :3], eye[..., None]).squeeze(-1)
    return view

def sph2cart(azimuth: torch.Tensor, elevation: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    x = radius * torch.cos(elevation) * torch.cos(azimuth)
    y = radius * torch.cos(elevation) * torch.sin(azimuth)
    z = radius * torch.sin(elevation)
    return torch.stack([x, y, z], dim=-1)
