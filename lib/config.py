import logging
import os

S3_REGION = os.environ.get("AWS_REGION", "us-east-1")
S3_INPUT_BUCKET = os.environ.get("S3_INPUT_BUCKET")
S3_OUTPUT_BUCKET = os.environ.get("S3_OUTPUT_BUCKET")
TMP_DIR = os.environ.get("TMP_DIR", "/tmp/genwarp")
GENWARP_MODEL_DIR = os.environ.get("GENWARP_MODEL_DIR", "./checkpoints")
GENWARP_CHECKPOINT_NAME = os.environ.get("GENWARP_CHECKPOINT_NAME", "multi1")
ZOEDEPTH_REPO_PATH = os.environ.get("ZOEDEPTH_REPO_PATH", "./extern/ZoeDepth")
IMAGE_RES = int(os.environ.get("IMAGE_RES", 512))
FOCAL_LENGTH_MM = float(os.environ.get("FOCAL_LENGTH_MM", 19))
AZIMUTH_DEG = float(os.environ.get("AZIMUTH_DEG", 20))
ELEVATION_DEG = float(os.environ.get("ELEVATION_DEG", 10))
CAMERA_RADIUS = float(os.environ.get("CAMERA_RADIUS", 0.5))
DEVICE = "cuda" if os.environ.get("USE_CUDA", "1") == "1" else "cpu"
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
