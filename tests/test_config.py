def test_env_defaults():
    """Tests that the DEVICE and IMAGE_RES env-configured values are valid."""
    from lib.config import DEVICE, IMAGE_RES
    assert DEVICE in ("cuda", "cpu")
    assert isinstance(IMAGE_RES, int) and IMAGE_RES > 0
