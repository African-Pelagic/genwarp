import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor


def crop_square(img: Image.Image) -> Image.Image:
    W, H = img.size
    if W < H:
        top = (H - W) // 2
        return img.crop((0, top, W, top + W))
    else:
        left = (W - H) // 2
        return img.crop((left, 0, left + H, H))

def prepare_image(image: Image.Image, res: int) -> torch.Tensor:
    return to_tensor(crop_square(image).resize((res, res)))[None]  # BCHW

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    return to_pil_image(tensor.squeeze(0).clamp(0, 1))

def image_to_device(image: torch.Tensor, device: str, half: bool = False) -> torch.Tensor:
    image = image.to(device)
    return image.half() if half else image
