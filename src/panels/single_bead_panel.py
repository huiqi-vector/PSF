import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QCheckBox, QFileDialog, QDoubleSpinBox
)
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class SingleBeadPanel(QWidget):
    """Right upper panel: 2×3 plots for a single selected bead."""

    proj_mode_changed = pyqtSignal(bool)   # True = max-projection, False = single-slice

    def __init__(self, parent=None):
        super().__init__(parent)
        self._psf_results: dict = {}
        self._cmap = 'gray'
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        ctrl = QHBoxLayout()
        ctrl.addWidget(QLabel("Bead:"))
        self._combo = QComboBox()
        self._combo.setMinimumWidth(80)
        self._combo.currentIndexChanged.connect(self._refresh_plot)
        ctrl.addWidget(self._combo)
        self._pos_label = QLabel("Position: —")
        ctrl.addWidget(self._pos_label)
        ctrl.addStretch()

        self._max_proj_cb = QCheckBox("Max-projection (XY)")
        self._max_proj_cb.setChecked(True)
        self._max_proj_cb.setToolTip(
            "Checked: XY patch = max Z-projection (invariant to Z-slice).\n"
            "Unchecked: XY patch = currently active Z-slice."
        )
        self._max_proj_cb.toggled.connect(self.proj_mode_changed.emit)
        ctrl.addWidget(self._max_proj_cb)

        ctrl.addWidget(QLabel("Colormap:"))
        self._cmap_combo = QComboBox()
        for name in ('gray', 'hot', 'viridis', 'plasma', 'inferno', 'magma', 'green', 'cyan'):
            self._cmap_combo.addItem(name)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap_changed)
        ctrl.addWidget(self._cmap_combo)

        save_btn = QPushButton("Save Plot")
        save_btn.clicked.connect(self._save_plot)
        ctrl.addWidget(save_btn)
        layout.addLayout(ctrl)

        # ── Row 2: axis limit controls ────────────────────────────────────────
        lim_row = QHBoxLayout()
        self._lock_cb = QCheckBox("Lock Axes")
        self._lock_cb.setToolTip("When checked, applies the xlim/ylim values below to all fit plots.")
        self._lock_cb.toggled.connect(self._refresh_plot)
        lim_row.addWidget(self._lock_cb)
        lim_row.addSpacing(10)

        def _spinbox(lo, hi, val, step, dec):
            sb = QDoubleSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setSingleStep(step)
            sb.setDecimals(dec)
            sb.setFixedWidth(72)
            sb.valueChanged.connect(self._refresh_plot)
            return sb

        lim_row.addWidget(QLabel("Lat X:"))
        self._lat_xmin = _spinbox(-999, 999, -3.0, 0.5, 2)
        self._lat_xmax = _spinbox(-999, 999,  3.0, 0.5, 2)
        lim_row.addWidget(self._lat_xmin)
        lim_row.addWidget(QLabel("to"))
        lim_row.addWidget(self._lat_xmax)

        lim_row.addSpacing(10)
        lim_row.addWidget(QLabel("Axial X:"))
        self._ax_xmin = _spinbox(-9999, 9999, -20.0, 1.0, 1)
        self._ax_xmax = _spinbox(-9999, 9999,  20.0, 1.0, 1)
        lim_row.addWidget(self._ax_xmin)
        lim_row.addWidget(QLabel("to"))
        lim_row.addWidget(self._ax_xmax)

        lim_row.addSpacing(10)
        lim_row.addWidget(QLabel("Y:"))
        self._ymin = _spinbox(-1e7, 1e7,     0, 500, 0)
        self._ymax = _spinbox(-1e7, 1e7, 65535, 500, 0)
        lim_row.addWidget(self._ymin)
        lim_row.addWidget(QLabel("to"))
        lim_row.addWidget(self._ymax)

        lim_row.addStretch()
        layout.addLayout(lim_row)

        self._fig = Figure(figsize=(8, 4), tight_layout=True)
        self._canvas = FigureCanvas(self._fig)
        layout.addWidget(self._canvas, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def update_beads(self, psf_results: dict):
        self._psf_results = psf_results
        current_text = self._combo.currentText()
        self._combo.blockSignals(True)
        self._combo.clear()
        for bid in sorted(psf_results.keys()):
            self._combo.addItem(str(bid))
        idx = self._combo.findText(current_text)
        self._combo.setCurrentIndex(idx if idx >= 0 else (0 if self._combo.count() > 0 else -1))
        self._combo.blockSignals(False)
        self._refresh_plot()

    def show_bead(self, bead_id: int):
        idx = self._combo.findText(str(bead_id))
        if idx >= 0:
            self._combo.setCurrentIndex(idx)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _on_cmap_changed(self, name: str):
        self._cmap = name
        self._refresh_plot()

    def _refresh_plot(self):
        self._fig.clear()
        text = self._combo.currentText()
        if not text:
            self._canvas.draw()
            return
        psf = self._psf_results.get(int(text))
        if psf is None:
            self._canvas.draw()
            return

        self._pos_label.setText(f"Position: ({psf.x_pix}, {psf.y_pix}) px")

        # Row 0: image patches
        for col, (patch, title) in enumerate([
            (psf.xy_patch, 'XY plane'),
            (psf.xz_patch, 'XZ plane'),
            (psf.yz_patch, 'YZ plane'),
        ]):
            ax = self._fig.add_subplot(2, 3, col + 1)
            if patch is not None:
                ax.imshow(patch, cmap=self._cmap, aspect='auto', origin='upper')
            ax.set_title(title, fontsize=8)
            ax.axis('off')

        # Row 1: fitting curves
        # Subplot 4: XY — X-profile (blue) and Y-profile (green) on same axes
        ax_xy = self._fig.add_subplot(2, 3, 4)
        if psf.x_data and len(psf.x_data) == 2:
            ax_xy.scatter(psf.x_data[0], psf.x_data[1], s=6, alpha=0.5, color='steelblue')
        if psf.x_fit and len(psf.x_fit) == 2:
            ax_xy.plot(psf.x_fit[0], psf.x_fit[1], color='steelblue', lw=1.5,
                       label=f"X: {psf.x_fwhm:.3f} µm" if psf.x_fwhm else "X: —")
        if psf.y_data and len(psf.y_data) == 2:
            ax_xy.scatter(psf.y_data[0], psf.y_data[1], s=6, alpha=0.5, color='tomato')
        if psf.y_fit and len(psf.y_fit) == 2:
            ax_xy.plot(psf.y_fit[0], psf.y_fit[1], color='tomato', lw=1.5,
                       label=f"Y: {psf.y_fwhm:.3f} µm" if psf.y_fwhm else "Y: —")
        x_str = f"{psf.x_fwhm:.3f}" if psf.x_fwhm else "—"
        y_str = f"{psf.y_fwhm:.3f}" if psf.y_fwhm else "—"
        ax_xy.set_title(f"XY  X={x_str} µm  Y={y_str} µm", fontsize=8)
        ax_xy.set_xlabel("Distance (µm)", fontsize=7)
        ax_xy.set_ylabel("Intensity", fontsize=7)
        ax_xy.tick_params(labelsize=6)
        ax_xy.legend(fontsize=6, loc='upper right')

        # Subplots 5 & 6: XZ and YZ
        for col, (raw, fit, fwhm, lbl) in enumerate([
            (psf.xz_data, psf.xz_fit, psf.xz_fwhm, 'XZ'),
            (psf.yz_data, psf.yz_fit, psf.yz_fwhm, 'YZ'),
        ]):
            ax = self._fig.add_subplot(2, 3, col + 5)
            if raw and len(raw) == 2:
                ax.scatter(raw[0], raw[1], s=6, alpha=0.5, color='steelblue')
            if fit and len(fit) == 2:
                ax.plot(fit[0], fit[1], color='tomato', lw=1.5)
            fwhm_str = f"{fwhm:.3f} µm" if fwhm else "(fit failed)"
            ax.set_title(f"{lbl}  FWHM = {fwhm_str}", fontsize=8)
            ax.set_xlabel("Distance (µm)", fontsize=7)
            ax.set_ylabel("Intensity", fontsize=7)
            ax.tick_params(labelsize=6)

        self._fig.tight_layout(pad=0.5)

        # Apply user-defined axis limits when locked
        if self._lock_cb.isChecked():
            lat_xlim  = (self._lat_xmin.value(), self._lat_xmax.value())
            ax_xlim   = (self._ax_xmin.value(),  self._ax_xmax.value())
            ylim      = (self._ymin.value(),      self._ymax.value())
            for i, ax in enumerate(self._fig.axes):
                if i >= 3:  # fit subplots only (row 2)
                    xlim = lat_xlim if i == 3 else ax_xlim
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

        self._canvas.draw()

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Single Bead Plot", "results/",
            "TIFF (*.tiff);;SVG (*.svg);;JPEG (*.jpg)"
        )
        if path:
            fmt = 'tiff' if path.endswith('.tiff') else 'svg' if path.endswith('.svg') else 'jpeg'
            self._fig.savefig(path, format=fmt, dpi=150, bbox_inches='tight')
