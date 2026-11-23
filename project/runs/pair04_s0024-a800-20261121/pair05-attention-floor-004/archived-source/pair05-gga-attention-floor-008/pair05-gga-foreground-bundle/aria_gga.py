"""Inverse of the Pair01 prior renderer's Project Aria fisheye model."""

import numpy as np


def project_aria(renderer_xy, distortion):
    """Map normalized renderer rays to distorted coordinates before intrinsics."""
    radial = distortion[:6]
    tangential = distortion[6:8]
    prism = distortion[8:].reshape(2, 2)

    radius = np.linalg.norm(renderer_xy, axis=1)
    theta = np.arctan(radius)
    radial_powers = theta[:, None] ** (2 * np.arange(1, 7))
    theta_over_radius = np.divide(theta, radius, out=np.ones_like(radius), where=radius > 0)
    radial_xy = renderer_xy * (theta_over_radius * (1 + radial_powers @ radial))[:, None]
    radius_squared = np.sum(radial_xy**2, axis=1)
    tangential_xy = (
        2 * (radial_xy @ tangential)[:, None] * radial_xy
        + radius_squared[:, None] * tangential
    )
    prism_xy = np.column_stack((radius_squared, radius_squared**2)) @ prism.T
    return radial_xy + tangential_xy + prism_xy


def undistort_aria_for_gga(pixel_coords_cv, scaled_k, distortion_coeffs):
    """Return float32 (N, 1, 2) rays in EgoX's pre-rotation camera axes.

    Invert the same front-facing lens extension at every latent grid cell,
    including the corners outside the nominal image circle. Keep each cell's
    original index and use a forward-facing ray with z=1.
    """
    pixels = np.asarray(pixel_coords_cv, dtype=np.float64)
    k = np.asarray(scaled_k, dtype=np.float64)
    distortion = np.asarray(distortion_coeffs, dtype=np.float64).reshape(-1)
    if pixels.ndim != 3 or pixels.shape[1:] != (1, 2) or len(pixels) == 0:
        raise ValueError("Expected EgoX pixel coordinates with shape (N, 1, 2).")
    if k.shape != (3, 3) or distortion.shape != (12,):
        raise ValueError("Expected camera intrinsics (3, 3) and 12 Aria coefficients.")
    if not all(np.isfinite(array).all() for array in (pixels, k, distortion)):
        raise ValueError("Expected finite Aria projection inputs.")
    # The frozen Pair01 renderer uses a centered square image and one focal length.
    focal = k[0, 0]
    if focal <= 0 or not np.array_equal(k, [[focal, 0, 28], [0, focal, 28], [0, 0, 1]]):
        raise ValueError("Expected Pair01's centered 56×56 intrinsics with one focal length.")

    gga_distorted = (pixels[:, 0] - k[:2, 2]) / focal
    # Renderer maps a camera-plane row vector (x, y) to (y, -x).
    # Keep the same pixel ordering and downstream torch.rot90 as official EgoX.
    target = np.column_stack((gga_distorted[:, 1], -gga_distorted[:, 0]))
    angle = np.linalg.norm(target, axis=1)
    if (angle >= np.pi / 2).any():
        raise ValueError("Aria pixel grid leaves the supported front-facing ray domain.")
    initial_scale = np.divide(np.tan(angle), angle, out=np.ones_like(angle), where=angle > 0)
    renderer_xy = target * initial_scale[:, None]

    step = 1e-5
    basis = np.eye(2) * step
    for _ in range(8):
        residual = project_aria(renderer_xy, distortion) - target
        jacobian = np.stack([
            (project_aria(renderer_xy + axis, distortion) - project_aria(renderer_xy - axis, distortion))
            / (2 * step)
            for axis in basis
        ], axis=2)
        jacobian_det = np.linalg.det(jacobian)
        if not np.isfinite(jacobian_det).all() or (jacobian_det <= 0).any():
            raise ValueError("Aria inverse left the orientation-preserving front-facing branch.")
        renderer_xy -= np.linalg.solve(jacobian, residual[..., None])[..., 0]

    residual = project_aria(renderer_xy, distortion) - target
    if not np.isfinite(residual).all() or np.max(np.abs(residual)) > 1e-9:
        raise ValueError("Aria inverse failed to reproject the full Pair01 ray grid.")
    gga_xy = np.column_stack((-renderer_xy[:, 1], renderer_xy[:, 0]))
    return gga_xy[:, None, :].astype(np.float32)
