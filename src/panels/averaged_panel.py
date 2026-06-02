import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QCheckBox, QDoubleSpinBox
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def _average_fits(fit_list):
    """Average a list of (x_centered, y_fit) curves that are already peak-aligned.

    All x arrays have their Gaussian peak at x=0, so we just need a common
    x range, interpolate each curve, then average.
    Returns (x_common, y_mean, y_std) or None if fit_list is empty.
    """
    valid = [f for f in fit_list if f and len(f) == 2 and len(f[0]) > 1]
    if not valid:
        return None
    half = max(abs(f[0]).max() for f in valid)
    x_common = np.linspace(-half, half, 300)
    y_stack = np.array([np.interp(x_common, f[0], f[1]) for f in valid])
    return x_common, y_stack.mean(axis=0), y_stack.std(axis=0)


class AveragedPanel(QWidget):
    """Right lower panel: 1×3 averaged PSF plots across all selected beads."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._psf_results: dict = {}
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ── Header row: info label + save ─────────────────────────────────────
        header = QHBoxLayout()
        self._info_label = QLabel("No beads selected")
        header.addWidget(self._info_label)
        header.addStretch()
        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self._save_plot)
        header.addWidget(save_btn)
        layout.addLayout(header)

        # ── Axis limit row ────────────────────────────────────────────────────
        lim_row = QHBoxLayout()
        self._lock_cb = QCheckBox("Lock Axes")
        self._lock_cb.setToolTip("When checked, applies the xlim/ylim values to all averaged fit plots.")
        self._lock_cb.toggled.connect(self._reapply_limits)
        lim_row.addWidget(self._lock_cb)
        lim_row.addSpacing(10)

        def _sb(lo, hi, val, step, dec):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.setFixedWidth(72)
            sb.valueChanged.connect(self._reapply_limits)
            return sb

        lim_row.addWidget(QLabel("Lat X:"))
        self._lat_xmin = _sb(-999, 999, -3.0, 0.5, 2)
        self._lat_xmax = _sb(-999, 999,  3.0, 0.5, 2)
        lim_row.addWidget(self._lat_xmin)
        lim_row.addWidget(QLabel("to"))
        lim_row.addWidget(self._lat_xmax)

        lim_row.addSpacing(10)
        lim_row.addWidget(QLabel("Axial X:"))
        self._ax_xmin = _sb(-9999, 9999, -20.0, 1.0, 1)
        self._ax_xmax = _sb(-9999, 9999,  20.0, 1.0, 1)
        lim_row.addWidget(self._ax_xmin)
        lim_row.addWidget(QLabel("to"))
        lim_row.addWidget(self._ax_xmax)

        lim_row.addSpacing(10)
        lim_row.addWidget(QLabel("Y:"))
        self._ymin = _sb(-1e7, 1e7,     0, 500, 0)
        self._ymax = _sb(-1e7, 1e7, 65535, 500, 0)
        lim_row.addWidget(self._ymin)
        lim_row.addWidget(QLabel("to"))
        lim_row.addWidget(self._ymax)

        lim_row.addStretch()
        layout.addLayout(lim_row)

        # ── Canvas ────────────────────────────────────────────────────────────
        self._fig = Figure(figsize=(8, 2.5), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_beads(self, psf_results: dict):
        self._psf_results = psf_results
        self._redraw()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _reapply_limits(self):
        """Called when lock is toggled or a limit spinbox changes."""
        if self._psf_results:
            self._redraw()

    def _redraw(self):
        self._fig.clear()
        valid = list(self._psf_results.values())
        n = len(valid)

        if n == 0:
            self._info_label.setText("No beads selected")
            self._canvas.draw()
            return

        summary = [f"n = {n}"]

        # ── XY subplot: X-direction (blue) + Y-direction (red) ───────────────
        ax_xy = self._fig.add_subplot(1, 3, 1)
        x_parts, y_parts = [], []

        avg_x = _average_fits([p.x_fit for p in valid])
        if avg_x is not None:
            xc, xm, xs = avg_x
            ax_xy.plot(xc, xm, color='steelblue', lw=1.5, label='X')
            ax_xy.fill_between(xc, xm - xs, xm + xs, alpha=0.2, color='steelblue')
            x_fwhms = [p.x_fwhm for p in valid if p.x_fwhm is not None]
            if x_fwhms:
                mx, sx = np.mean(x_fwhms), np.std(x_fwhms)
                x_parts.append(f"X: {mx:.3f}±{sx:.3f}")
                summary.append(f"X: {mx:.3f}±{sx:.3f} µm")

        avg_y = _average_fits([p.y_fit for p in valid])
        if avg_y is not None:
            xc, ym, ys = avg_y
            ax_xy.plot(xc, ym, color='tomato', lw=1.5, label='Y')
            ax_xy.fill_between(xc, ym - ys, ym + ys, alpha=0.2, color='tomato')
            y_fwhms = [p.y_fwhm for p in valid if p.y_fwhm is not None]
            if y_fwhms:
                my, sy = np.mean(y_fwhms), np.std(y_fwhms)
                y_parts.append(f"Y: {my:.3f}±{sy:.3f}")
                summary.append(f"Y: {my:.3f}±{sy:.3f} µm")

        ax_xy.set_title("XY  " + "  ".join(x_parts + y_parts) + " µm", fontsize=8)
        ax_xy.set_xlabel("Distance (µm)", fontsize=7)
        ax_xy.set_ylabel("Intensity", fontsize=7)
        ax_xy.tick_params(labelsize=6)
        if x_parts or y_parts:
            ax_xy.legend(fontsize=6)

        # ── XZ and YZ subplots ────────────────────────────────────────────────
        for col, (fit_attr, fwhm_attr, lbl) in enumerate([
            ('xz_fit', 'xz_fwhm', 'XZ'),
            ('yz_fit', 'yz_fwhm', 'YZ'),
        ]):
            ax = self._fig.add_subplot(1, 3, col + 2)
            avg = _average_fits([getattr(p, fit_attr) for p in valid])
            if avg is not None:
                xc, ym, ys = avg
                ax.plot(xc, ym, color='steelblue', lw=1.5)
                ax.fill_between(xc, ym - ys, ym + ys, alpha=0.2, color='steelblue')
            fwhm_list = [getattr(p, fwhm_attr) for p in valid if getattr(p, fwhm_attr) is not None]
            if fwhm_list:
                mf, sf = np.mean(fwhm_list), np.std(fwhm_list)
                ax.set_title(f"{lbl}  {mf:.3f}±{sf:.3f} µm", fontsize=8)
                summary.append(f"{lbl}: {mf:.3f}±{sf:.3f} µm")
            else:
                ax.set_title(f"{lbl}  (no fits)", fontsize=8)
            ax.set_xlabel("Distance (µm)", fontsize=7)
            ax.set_ylabel("Intensity", fontsize=7)
            ax.tick_params(labelsize=6)

        self._info_label.setText("  ".join(summary))
        self._fig.tight_layout(pad=0.5)

        # Apply user-defined axis limits when locked
        if self._lock_cb.isChecked():
            lat_xlim = (self._lat_xmin.value(), self._lat_xmax.value())
            ax_xlim  = (self._ax_xmin.value(),  self._ax_xmax.value())
            ylim     = (self._ymin.value(),      self._ymax.value())
            for i, ax in enumerate(self._fig.axes):
                ax.set_xlim(lat_xlim if i == 0 else ax_xlim)
                ax.set_ylim(ylim)

        self._canvas.draw()

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Averaged PSF Plot", "results/",
            "TIFF (*.tiff);;SVG (*.svg);;JPEG (*.jpg)"
        )
        if path:
            fmt = 'tiff' if path.endswith('.tiff') else 'svg' if path.endswith('.svg') else 'jpeg'
            self._fig.savefig(path, format=fmt, dpi=150, bbox_inches='tight')
