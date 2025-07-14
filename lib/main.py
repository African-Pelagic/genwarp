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


if __name__ == "__main__":
    import sys
    from image_utils import prepare_image
    from s3_utils import download_image_from_s3, upload_image_to_s3

    s3_uris = sys.argv[1:]
    if not s3_uris:
        logger.error("No input S3 URIs provided.")
        sys.exit(1)

    logger.info(f"Processing {len(s3_uris)} images...")
    zoe, genwarp = initialize_models()

    for s3_uri in s3_uris:
        try:
            image = download_image_from_s3(s3_uri)
            image_tensor = prepare_image(image, IMAGE_RES)
            synthesized = run_genwarp_pipeline(image_tensor, zoe, genwarp)
            upload_image_to_s3(synthesized, s3_uri)
            logger.info(f"Successfully processed {s3_uri}")
        except Exception as e:
            logger.error(f"Error processing {s3_uri}: {e}")
