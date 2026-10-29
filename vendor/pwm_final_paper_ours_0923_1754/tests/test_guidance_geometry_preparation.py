"""Numerical mesh tolerance must not admit a physically meaningful concavity."""
import numpy as np
import pytest

pytest.importorskip('scipy')
from scripts.dev.prepare_pick_guidance_geometry import supporting_plane_violation


def test_support_planes_distinguish_numerical_noise_from_a_dented_surface():
    corners=np.array([[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=float)*.01
    faces=np.array([[0,1,4],[1,2,4],[2,3,4],[3,0,4]])
    def surface(center_z):
        return np.concatenate([corners,np.array([[0.,0.,center_z]])])
    assert supporting_plane_violation(surface(.01),faces)<1e-12
    assert 0<supporting_plane_violation(surface(.01-2e-8),faces)<2e-6
    assert supporting_plane_violation(surface(.008),faces)>.003
