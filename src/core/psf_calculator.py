import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import center_of_mass
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BeadPSF:
    bead_id: int
    x_pix: int
    y_pix: int
    # FWHM values in µm
    x_fwhm: Optional[float] = None    # lateral, X direction
    y_fwhm: Optional[float] = None    # lateral, Y direction
    xz_fwhm: Optional[float] = None   # axial, XZ plane
    yz_fwhm: Optional[float] = None   # axial, YZ plane
    # Profile data: (dist_um_centered_at_peak, intensities)
    x_data: tuple = field(default_factory=tuple)
    y_data: tuple = field(default_factory=tuple)
    xz_data: tuple = field(default_factory=tuple)
    yz_data: tuple = field(default_factory=tuple)
    # Fitted curves: (x_fine_centered_at_peak, y_fit)
    x_fit: tuple = field(default_factory=tuple)
    y_fit: tuple = field(default_factory=tuple)
    xz_fit: tuple = field(default_factory=tuple)
    yz_fit: tuple = field(default_factory=tuple)
    # Image patches for display
    xy_patch: Optional[np.ndarray] = None
    xz_patch: Optional[np.ndarray] = None
    yz_patch: Optional[np.ndarray] = None
    error: Optional[str] = None


def _gaussian_1d(x, A, mu, sigma, B):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2)) + B


def _fit_gaussian(distances, intensities):
    """Fit 1D Gaussian. Returns (fwhm, x_centered, y_fit, mu).

    x_centered is the fit-curve x-axis shifted so Gaussian peak is at x=0.
    mu is the fit peak position in the original distance units.
    """
    d = np.asarray(distances, dtype=float)
    v = np.asarray(intensities, dtype=float)
    A0 = float(v.max() - v.min())
    mu0 = float(d[np.argmax(v)])
    sigma0 = float((d[-1] - d[0]) / 4)
    B0 = float(v.min())
    p0 = [max(A0, 1e-9), mu0, max(sigma0, 1e-9), B0]
    bounds = ([0, float(d.min()), 0, -np.inf], [np.inf, float(d.max()), np.inf, np.inf])
    popt, _ = curve_fit(_gaussian_1d, d, v, p0=p0, bounds=bounds, maxfev=5000)
    mu = float(popt[1])
    fwhm = 2.355 * abs(float(popt[2]))
    x_fine = np.linspace(float(d[0]), float(d[-1]), 300)
    y_fit = _gaussian_1d(x_fine, *popt)
    return fwhm, x_fine - mu, y_fit, mu


def _get_best_z(stack, y, x, half):
    y0, y1 = max(0, y - half), min(stack.shape[1], y + half)
    x0, x1 = max(0, x - half), min(stack.shape[2], x + half)
    return int(np.argmax(stack[:, y0:y1, x0:x1].max(axis=(1, 2))))


def compute_psf(
    stack: np.ndarray,
    bead_id: int,
    x_pix: int,
    y_pix: int,
    x_um: float,
    y_um: float,
    z_um: float,
    use_max_projection: bool = True,
    current_z: Optional[int] = None,
    half_window: int = 15,
) -> BeadPSF:
    """Compute lateral (X, Y directions) and axial (XZ, YZ) PSF for a bead.

    Lateral: X-profile (mean of center rows) and Y-profile (mean of center columns)
    from the XY patch (max-projection or specific Z-slice).
    Axial: Z-intensity profile through bead center from XZ and YZ cross-sections.
    All distance axes are centered at the fitted Gaussian peak (x=0 at peak).

    use_max_projection=False + current_z=None → auto-find best-focus Z-slice.
    use_max_projection=False + current_z=k → use stack[k] for XY patch.
    """
    result = BeadPSF(bead_id=bead_id, x_pix=x_pix, y_pix=y_pix)
    hw = half_window
    nz, ny, nx = stack.shape
    y, x = int(y_pix), int(x_pix)
    y0, y1 = max(0, y - hw), min(ny, y + hw)
    x0, x1 = max(0, x - hw), min(nx, x + hw)

    # ── XY patch & lateral profiles ──────────────────────────────────────────
    try:
        if use_max_projection:
            xy_patch = stack[:, y0:y1, x0:x1].max(axis=0).astype(float)
        else:
            z_idx = current_z if current_z is not None else _get_best_z(stack, y, x, hw)
            z_idx = int(np.clip(z_idx, 0, nz - 1))
            xy_patch = stack[z_idx, y0:y1, x0:x1].astype(float)
        result.xy_patch = xy_patch
        h_p, w_p = xy_patch.shape

        # Refine bead center via centroid
        thresh = xy_patch.max() * 0.3
        masked = xy_patch * (xy_patch > thresh)
        if masked.any():
            cy_f, cx_f = center_of_mass(masked)
        else:
            cy_f, cx_f = h_p / 2.0, w_p / 2.0
        if np.isnan(cy_f) or np.isnan(cx_f):
            cy_f, cx_f = h_p / 2.0, w_p / 2.0
        cy_int, cx_int = int(round(cy_f)), int(round(cx_f))

        # Averaging width: a few rows/cols for better SNR
        avg = max(1, hw // 8)

        # X-profile: average rows around center → fit in X direction
        r0, r1 = max(0, cy_int - avg), min(h_p, cy_int + avg + 1)
        x_profile = xy_patch[r0:r1, :].mean(axis=0)
        x_dist = (np.arange(w_p) - cx_f) * x_um
        if len(x_dist) > 4:
            fx, x_xf, x_yf, mu_x = _fit_gaussian(x_dist, x_profile)
            result.x_fwhm = fx
            result.x_data = (x_dist - mu_x, x_profile)
            result.x_fit = (x_xf, x_yf)

        # Y-profile: average columns around center → fit in Y direction
        c0, c1 = max(0, cx_int - avg), min(w_p, cx_int + avg + 1)
        y_profile = xy_patch[:, c0:c1].mean(axis=1)
        y_dist = (np.arange(h_p) - cy_f) * y_um
        if len(y_dist) > 4:
            fy, y_xf, y_yf, mu_y = _fit_gaussian(y_dist, y_profile)
            result.y_fwhm = fy
            result.y_data = (y_dist - mu_y, y_profile)
            result.y_fit = (y_xf, y_yf)

    except Exception as e:
        result.error = f"XY fit failed: {e}"

    # ── XZ cross-section ─────────────────────────────────────────────────────
    try:
        xz_patch = stack[:, y, x0:x1].astype(float)   # (Z, X-window)
        result.xz_patch = xz_patch
        z_idx_ax = np.arange(nz) * z_um
        xz_line = xz_patch[:, xz_patch.shape[1] // 2]
        fxz, xz_xf, xz_yf, mu_xz = _fit_gaussian(z_idx_ax, xz_line)
        result.xz_fwhm = fxz
        result.xz_data = (z_idx_ax - mu_xz, xz_line)
        result.xz_fit = (xz_xf, xz_yf)
    except Exception as e:
        err = f"XZ fit failed: {e}"
        result.error = (result.error + "; " + err) if result.error else err

    # ── YZ cross-section ─────────────────────────────────────────────────────
    try:
        yz_patch = stack[:, y0:y1, x].astype(float)   # (Z, Y-window)
        result.yz_patch = yz_patch
        z_idx_ax = np.arange(nz) * z_um
        yz_line = yz_patch[:, yz_patch.shape[1] // 2]
        fyz, yz_xf, yz_yf, mu_yz = _fit_gaussian(z_idx_ax, yz_line)
        result.yz_fwhm = fyz
        result.yz_data = (z_idx_ax - mu_yz, yz_line)
        result.yz_fit = (yz_xf, yz_yf)
    except Exception as e:
        err = f"YZ fit failed: {e}"
        result.error = (result.error + "; " + err) if result.error else err

    return result
