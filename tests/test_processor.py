import torch
from unittest.mock import MagicMock
from libprocessor import run_genwarp_pipeline
from lib.image_utils import prepare_image
from PIL import Image

def test_run_genwarp_pipeline():
    """Tests GenWarp pipeline with mocked models."""
    dummy_img = Image.new("RGB", (512, 512))
    dummy_tensor = prepare_image(dummy_img, 512)
    mock_zoe = MagicMock()
    mock_zoe.infer.return_value = torch.ones((1, 1, 512, 512))
    mock_genwarp = MagicMock()
    mock_genwarp.return_value = {"synthesized": dummy_tensor}
    output = run_genwarp_pipeline(dummy_tensor, mock_zoe, mock_genwarp)
    assert isinstance(output, Image.Image)
