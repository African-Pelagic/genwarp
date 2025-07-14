from PIL import Image
from lib.image_utils import crop_square, prepare_image, tensor_to_pil
import torch

def test_crop_square():
    """Ensures image is cropped to square correctly."""
    img = Image.new("RGB", (400, 300))
    cropped = crop_square(img)
    assert cropped.size == (300, 300)

def test_prepare_image():
    """Tests image preprocessing to tensor."""
    img = Image.new("RGB", (400, 400))
    tensor = prepare_image(img, 256)
    assert tensor.shape == (1, 3, 256, 256)
    assert tensor.dtype in [torch.float32, torch.float16]

def test_tensor_to_pil():
    """Tests tensor-to-image conversion."""
    img = Image.new("RGB", (128, 128))
    tensor = prepare_image(img, 128)
    pil = tensor_to_pil(tensor[0])
    assert isinstance(pil, Image.Image)
