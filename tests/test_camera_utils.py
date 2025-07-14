import torch
from lib.camera_utils import focal_length_to_fov, get_projection_matrix, camera_lookat, sph2cart

def test_focal_length_to_fov():
    """Tests FOV conversion from focal length."""
    fov = focal_length_to_fov(35, 24)
    assert fov > 0 and fov < 3.2

def test_projection_matrix():
    """Tests shape of projection matrix."""
    mtx = get_projection_matrix(torch.tensor([0.5]), 1.0, 0.01, 100.0)
    assert mtx.shape == (1, 4, 4)

def test_camera_lookat():
    """Tests look-at matrix structure."""
    eye = torch.tensor([[0., 0., 0.]])
    target = torch.tensor([[1., 0., 0.]])
    up = torch.tensor([[0., 0., 1.]])
    mtx = camera_lookat(eye, target, up)
    assert mtx.shape == (1, 4, 4)

def test_sph2cart():
    """Tests spherical to cartesian conversion output shape."""
    out = sph2cart(torch.tensor(0.5), torch.tensor(0.3), torch.tensor([1.0]))
    assert out.shape == (1, 3)
