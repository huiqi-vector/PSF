import os
import sys
import tifffile
import numpy as np
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStatusBar, QFileDialog, QDialog, QFormLayout,
    QDoubleSpinBox, QSpinBox, QDialogButtonBox, QMessageBox
)
from PyQt5.QtCore import Qt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.metadata import load_metadata, find_metadata_file, NoMetadataError
from core.psf_calculator import compute_psf, BeadPSF
from core.bead_detector import auto_detect_beads
from panels.image_panel import ImagePanel
from panels.single_bead_panel import SingleBeadPanel
from panels.averaged_panel import AveragedPanel


class ManualPixelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Enter Pixel Sizes")
        layout = QFormLayout(self)
        self._x = QDoubleSpinBox(); self._x.setRange(0.001, 100); self._x.setValue(1.0); self._x.setSuffix(" µm")
        self._y = QDoubleSpinBox(); self._y.setRange(0.001, 100); self._y.setValue(1.0); self._y.setSuffix(" µm")
        self._z = QDoubleSpinBox(); self._z.setRange(0.001, 100); self._z.setValue(1.0); self._z.setSuffix(" µm")
        layout.addRow("X pixel size:", self._x)
        layout.addRow("Y pixel size:", self._y)
        layout.addRow("Z step size:", self._z)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def values(self):
        return self._x.value(), self._y.value(), self._z.value()


class AutoSelectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Select Beads")
        layout = QFormLayout(self)
        self._n = QSpinBox(); self._n.setRange(1, 200); self._n.setValue(5)
        layout.addRow("Number of beads:", self._n)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept); btns.rejected.connect(self.reject)
        layout.addRow(btns)

    def n_beads(self):
        return self._n.value()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PSF Analyzer — no file loaded")
        self.resize(1400, 900)

        self._stack: np.ndarray | None = None
        self._meta: dict = {'x_um': 1.0, 'y_um': 1.0, 'z_um': 1.0}
        self._beads: list = []              # (bead_id, x_pix, y_pix)
        self._psf_results: dict = {}        # bead_id -> BeadPSF
        self._next_id = 1
        self._use_max_proj = True           # max-proj vs current Z-slice for XY
        self._half_window = 15              # pixels around bead for PSF extraction

        self._build_ui()
        self._update_info_label()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()
        load_btn = QPushButton("Load TIFF + Metadata")
        load_btn.clicked.connect(self._load_data)
        auto_btn = QPushButton("Auto Select Beads")
        auto_btn.clicked.connect(self._auto_select)
        clear_btn = QPushButton("Clear Beads")
        clear_btn.clicked.connect(self._clear_beads)
        self._draw_roi_btn = QPushButton("Draw ROI")
        self._draw_roi_btn.setCheckable(True)
        self._draw_roi_btn.setToolTip("Left-click-drag on image to draw a region that constrains auto-detection.")
        self._draw_roi_btn.toggled.connect(self._on_draw_roi_toggled)
        clear_roi_btn = QPushButton("Clear ROIs")
        clear_roi_btn.clicked.connect(self._clear_rois)

        toolbar.addWidget(load_btn)
        toolbar.addWidget(auto_btn)
        toolbar.addWidget(clear_btn)
        toolbar.addWidget(self._draw_roi_btn)
        toolbar.addWidget(clear_roi_btn)
        toolbar.addSpacing(16)
        toolbar.addWidget(QLabel("Window (px):"))
        self._hw_spin = QSpinBox()
        self._hw_spin.setRange(5, 80)
        self._hw_spin.setValue(self._half_window)
        self._hw_spin.setToolTip(
            "Half-window in pixels around each bead used for PSF extraction.\n"
            "Smaller = less chance of including neighbouring beads.\n"
            "Change triggers recomputation of all current beads."
        )
        self._hw_spin.valueChanged.connect(self._on_half_window_changed)
        toolbar.addWidget(self._hw_spin)
        toolbar.addStretch()
        self._info_label = QLabel()
        toolbar.addWidget(self._info_label)
        main_layout.addLayout(toolbar)

        # Main split
        h_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(h_splitter, stretch=1)

        self._image_panel = ImagePanel()
        self._image_panel.bead_selected.connect(self._on_bead_selected)
        self._image_panel.bead_appended.connect(self._on_bead_appended)
        self._image_panel.roi_added.connect(lambda *_: self._draw_roi_btn.setChecked(False))
        h_splitter.addWidget(self._image_panel)

        right_splitter = QSplitter(Qt.Vertical)
        self._single_bead_panel = SingleBeadPanel()
        self._single_bead_panel.proj_mode_changed.connect(self._on_proj_mode_changed)
        self._averaged_panel = AveragedPanel()
        right_splitter.addWidget(self._single_bead_panel)
        right_splitter.addWidget(self._averaged_panel)
        right_splitter.setSizes([550, 350])
        h_splitter.addWidget(right_splitter)
        h_splitter.setSizes([700, 700])

        self.setStatusBar(QStatusBar())

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        tiff_path, _ = QFileDialog.getOpenFileName(
            self, "Open TIFF File", "", "TIFF files (*.tif *.tiff);;All files (*)"
        )
        if not tiff_path:
            return
        try:
            stack = tifffile.imread(tiff_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load TIFF:\n{e}")
            return

        if stack.ndim == 2:
            stack = stack[np.newaxis, ...]
        elif stack.ndim == 4:
            stack = stack[:, 0, :, :]
        elif stack.ndim != 3:
            QMessageBox.critical(self, "Error", f"Unexpected TIFF shape: {stack.shape}")
            return

        meta_path = find_metadata_file(tiff_path)
        meta = None
        if meta_path:
            try:
                meta = load_metadata(meta_path)
                meta['n_planes'] = stack.shape[0]
            except Exception as e:
                self.statusBar().showMessage(f"Metadata error: {e} — enter manually")

        if meta is None:
            dlg = ManualPixelDialog(self)
            if dlg.exec_() != QDialog.Accepted:
                return
            x_um, y_um, z_um = dlg.values()
            meta = {'x_um': x_um, 'y_um': y_um, 'z_um': z_um,
                    'width': stack.shape[2], 'height': stack.shape[1],
                    'n_planes': stack.shape[0]}

        self._stack = stack
        self._meta = meta
        self._beads.clear()
        self._psf_results.clear()
        self._next_id = 1

        self._image_panel.load_stack(stack, meta['x_um'], meta['y_um'])
        self._update_beads_display()
        self._update_info_label()
        fname = os.path.basename(tiff_path)
        self.setWindowTitle(f"PSF Analyzer — {fname}")
        self.statusBar().showMessage(f"Loaded: {tiff_path}")

    def _update_info_label(self):
        if self._stack is None:
            self._info_label.setText("No data loaded")
            return
        m = self._meta
        nz, ny, nx = self._stack.shape
        self._info_label.setText(
            f"FOV: {nx}×{ny} px  ({nx*m['x_um']:.1f}×{ny*m['y_um']:.1f} µm)  |  "
            f"Z: {nz} planes  z-step: {m['z_um']:.3f} µm  |  "
            f"pixel: x={m['x_um']:.4f} y={m['y_um']:.4f} µm"
        )

    # ── Bead management ───────────────────────────────────────────────────────

    def _on_bead_selected(self, x, y):
        if self._stack is None:
            return
        self._beads.clear()
        self._psf_results.clear()
        self._next_id = 1
        self._add_bead(x, y)

    def _on_bead_appended(self, x, y):
        if self._stack is None:
            return
        self._add_bead(x, y)

    def _add_bead(self, x, y):
        bead_id = self._next_id
        self._next_id += 1
        self._beads.append((bead_id, x, y))
        current_z = self._image_panel._current_z if not self._use_max_proj else None
        psf = compute_psf(
            self._stack, bead_id, x, y,
            self._meta['x_um'], self._meta['y_um'], self._meta['z_um'],
            use_max_projection=self._use_max_proj,
            current_z=current_z,
            half_window=self._half_window,
        )
        self._psf_results[bead_id] = psf
        if psf.error:
            self.statusBar().showMessage(f"Bead {bead_id}: {psf.error}")
        self._update_beads_display()

    def _auto_select(self):
        if self._stack is None:
            QMessageBox.information(self, "No data", "Load a TIFF file first.")
            return
        dlg = AutoSelectDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
        n = dlg.n_beads()
        rois = self._image_panel.get_rois() or None
        positions = auto_detect_beads(self._stack, n, rois)
        if not positions:
            QMessageBox.information(self, "Auto Select", "No beads found. Try a different ROI or threshold.")
            return
        for x, y in positions:
            self._add_bead(x, y)
        self.statusBar().showMessage(f"Auto-selected {len(positions)} bead(s)")

    def _clear_beads(self):
        self._beads.clear()
        self._psf_results.clear()
        self._next_id = 1
        self._update_beads_display()

    def _on_draw_roi_toggled(self, checked: bool):
        self._image_panel.set_roi_mode(checked)
        self._draw_roi_btn.setText("Drawing ROI..." if checked else "Draw ROI")

    def _clear_rois(self):
        self._image_panel.clear_rois()
        self.statusBar().showMessage("ROIs cleared")

    def _recompute_all_beads(self):
        if self._stack is None:
            return
        current_z = self._image_panel._current_z if not self._use_max_proj else None
        new_results = {}
        for bead_id, x, y in self._beads:
            psf = compute_psf(
                self._stack, bead_id, x, y,
                self._meta['x_um'], self._meta['y_um'], self._meta['z_um'],
                use_max_projection=self._use_max_proj,
                current_z=current_z,
                half_window=self._half_window,
            )
            new_results[bead_id] = psf
        self._psf_results = new_results
        self._update_beads_display()

    def _on_proj_mode_changed(self, use_max_proj: bool):
        self._use_max_proj = use_max_proj
        self._recompute_all_beads()

    def _on_half_window_changed(self, val: int):
        self._half_window = val
        self._recompute_all_beads()

    def _update_beads_display(self):
        self._image_panel.set_beads(self._beads)
        self._single_bead_panel.update_beads(self._psf_results)
        self._averaged_panel.update_beads(self._psf_results)
        if self._beads:
            self._single_bead_panel.show_bead(self._beads[-1][0])
