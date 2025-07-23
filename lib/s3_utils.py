from io import BytesIO
from urllib.parse import urlparse
from logging import getLogger
import os

import boto3
from PIL import Image

from lib import config

logger = getLogger(__file__)

s3_client = boto3.client("s3", region_name=config.S3_REGION)

def parse_s3_path(s3_uri):
    """Parse s3://bucket/key URI into (bucket, key)"""
    parsed = urlparse(s3_uri)
    return parsed.netloc, parsed.path.lstrip("/")

def download_image_from_s3(s3_uri):
    """Download image from S3 and return as PIL.Image"""
    bucket, key = parse_s3_path(s3_uri)
    logger.info(f"Downloading image from s3://{bucket}/{key}")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return Image.open(BytesIO(response['Body'].read())).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to download image from {s3_uri}: {e}")
        raise

def upload_image_to_s3(image: Image.Image, source_s3_uri: str):
    """Upload image to s3://{bucket}/{prefix}/synthetic/{filename}"""
    bucket, key = parse_s3_path(source_s3_uri)
    prefix, filename = os.path.split(key)
    output_key = f"{prefix}/synthetic/{filename}"
    logger.info(f"Uploading synthesized image to s3://{bucket}/{output_key}")
    
    buf = BytesIO()
    image_format = filename.split(".")[-1].upper()
    if image_format == 'JPG':
        image_format = 'JPEG'
    image.save(buf, format=image_format)
    buf.seek(0)

    try:
        s3_client.upload_fileobj(buf, bucket, output_key)
    except Exception as e:
        logger.error(f"Failed to upload image to s3://{bucket}/{output_key}: {e}")
        raise
