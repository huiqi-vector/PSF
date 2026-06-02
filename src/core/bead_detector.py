import numpy as np
from scipy.ndimage import gaussian_filter, label, center_of_mass
from typing import Optional


def _max_project(stack: np.ndarray) -> np.ndarray:
    return stack.max(axis=0).astype(float)


def auto_detect_beads(
    stack: np.ndarray,
    n_beads: int,
    rois: Optional[list] = None,
    min_distance_px: int = 20,
    bead_sigma_px: float = 2.0,
) -> list[tuple[int, int]]:
    """Detect up to n_beads clean bead candidates in stack.

    rois: list of (x1, y1, x2, y2) rectangles to restrict search.
    Returns list of (x_pix, y_pix) sorted by quality (high SNR, isolated).
    """
    proj = _max_project(stack)

    # Optional ROI mask
    mask = np.ones(proj.shape, dtype=bool)
    if rois:
        mask[:] = False
        for x1, y1, x2, y2 in rois:
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            mask[y1:y2, x1:x2] = True

    proj_masked = proj * mask

    # Smooth to suppress noise, then find local maxima
    smoothed = gaussian_filter(proj_masked, sigma=bead_sigma_px)
    threshold = smoothed.mean() + 3.0 * smoothed.std()
    binary = smoothed > threshold

    labeled, n_features = label(binary)
    if n_features == 0:
        return []

    candidates = []
    for i in range(1, n_features + 1):
        component = labeled == i
        cy, cx = center_of_mass(smoothed * component)
        cy, cx = int(round(cy)), int(round(cx))
        peak = float(proj[cy, cx])
        # background = mean of annular region
        r_inner, r_outer = min_distance_px, min_distance_px + 5
        yy, xx = np.ogrid[:proj.shape[0], :proj.shape[1]]
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        annulus = proj[(r >= r_inner) & (r < r_outer)]
        bg = annulus.mean() if annulus.size > 0 else 1.0
        snr = peak / max(bg, 1.0)
        candidates.append((cx, cy, snr, peak))

    # Filter by minimum isolation distance between candidates
    candidates.sort(key=lambda c: -c[2])  # sort by SNR desc
    selected = []
    for cx, cy, snr, peak in candidates:
        too_close = any(
            np.sqrt((cx - sx) ** 2 + (cy - sy) ** 2) < min_distance_px
            for sx, sy in selected
        )
        if not too_close:
            selected.append((cx, cy))
        if len(selected) >= n_beads:
            break

    return selected
