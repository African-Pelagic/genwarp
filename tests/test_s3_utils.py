import io
from unittest import mock

import pytest
from PIL import Image

from lib.s3_utils import (download_image_from_s3, parse_s3_path,
                          upload_image_to_s3)


@pytest.fixture
def mock_s3():
    """Mocks boto3.client to avoid actual AWS calls."""
    return mock.patch("boto3.client")

def test_parse_s3_path():
    """Verifies that s3:// URIs are correctly split into bucket and key."""
    bucket, key = parse_s3_path("s3://my-bucket/some/path/image.jpg")
    assert bucket == "my-bucket"
    assert key == "some/path/image.jpg"

def test_download_image_from_s3(mock_s3):
    """Mimics an S3 download and checks for PIL.Image return."""
    with mock.patch("lib.s3_utils.s3_client") as s3_mock:
        client = s3_mock
        image_bytes = io.BytesIO()
        Image.new("RGB", (64, 64)).save(image_bytes, format="JPEG")
        client.get_object.return_value = {"Body": io.BytesIO(image_bytes.getvalue())}
        image = download_image_from_s3("s3://test-bucket/sample.jpg")
        assert isinstance(image, Image.Image)

def test_upload_image_to_s3(mock_s3):
    """Tests image upload by checking S3 client 'put_object' is called."""
    with mock.patch("lib.s3_utils.s3_client") as s3_mock:
        client = s3_mock
        image_bytes = io.BytesIO()
        img = Image.new("RGB", (64, 64))
        img.save(image_bytes, format='JPEG')
        upload_image_to_s3(img, "s3://test-bucket/abc123/input.jpg")
        assert client.upload_fileobj.called
