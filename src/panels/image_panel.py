import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QPushButton,
    QGroupBox, QFileDialog, QSizePolicy
)
from PyQt5.QtCore import Qt, QPoint, QRect, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QFont

_DRAG_THRESHOLD = 4  # pixels before a click becomes a drag/pan


class ImageCanvas(QWidget):
    """Custom widget that renders the microscope image with bead and ROI overlays."""

    bead_clicked = pyqtSignal(int, int)          # single click: x_pix, y_pix
    bead_added = pyqtSignal(int, int)            # ctrl+click: x_pix, y_pix
    roi_drawn = pyqtSignal(int, int, int, int)   # completed ROI: x1, y1, x2, y2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        self._image_np = None
        self._pixmap = None
        self._zoom = 1.0
        self._offset = QPoint(0, 0)

        self._beads: list = []              # (bead_id, x_pix, y_pix)
        self._completed_rois: list = []     # (x1, y1, x2, y2) in image coords

        # Pan state — resolved at mouseRelease to distinguish click vs drag
        self._pan_start: QPoint | None = None
        self._offset_at_pan_start = QPoint()
        self._is_panning = False

        # ROI drawing state
        self._roi_mode = False
        self._drawing_roi = False
        self._roi_start = QPoint()
        self._roi_current = QPoint()

    # ── Public API ────────────────────────────────────────────────────────────

    def set_image(self, image_np: np.ndarray):
        self._image_np = image_np.astype(np.float32)
        self._rebuild_pixmap()
        self.update()

    def set_beads(self, beads: list):
        self._beads = beads
        self.update()

    def set_completed_rois(self, rois: list):
        self._completed_rois = rois
        self.update()

    def set_roi_mode(self, enabled: bool):
        self._roi_mode = enabled
        self.setCursor(Qt.CrossCursor if enabled else Qt.ArrowCursor)

    def set_zoom(self, zoom: float):
        self._zoom = max(0.1, min(zoom, 20.0))
        self._clamp_offset()
        self.update()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _rebuild_pixmap(self):
        if self._image_np is None:
            return
        img = self._image_np
        lo, hi = img.min(), img.max()
        img8 = ((img - lo) / (hi - lo) * 255).astype(np.uint8) if hi > lo else np.zeros_like(img, dtype=np.uint8)
        h, w = img8.shape
        qimg = QImage(img8.data, w, h, w, QImage.Format_Grayscale8)
        self._pixmap = QPixmap.fromImage(qimg)

    def _image_to_widget(self, ix, iy):
        return ix * self._zoom + self._offset.x(), iy * self._zoom + self._offset.y()

    def _widget_to_image(self, wx, wy):
        return int(round((wx - self._offset.x()) / self._zoom)), \
               int(round((wy - self._offset.y()) / self._zoom))

    def _clamp_offset(self):
        if self._pixmap is None:
            return
        iw = self._pixmap.width() * self._zoom
        ih = self._pixmap.height() * self._zoom
        ww, wh = self.width(), self.height()
        mx, my = ww // 2, wh // 2
        self._offset = QPoint(
            int(max(-iw + mx, min(self._offset.x(), ww - mx))),
            int(max(-ih + my, min(self._offset.y(), wh - my)))
        )

    # ── Qt overrides ──────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(40, 40, 40))

        if self._pixmap is None:
            painter.setPen(Qt.white)
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        iw = int(self._pixmap.width() * self._zoom)
        ih = int(self._pixmap.height() * self._zoom)
        painter.drawPixmap(self._offset.x(), self._offset.y(), iw, ih, self._pixmap)

        # Completed ROI rectangles (cyan)
        roi_pen = QPen(QColor(80, 200, 255), 1, Qt.DashLine)
        painter.setPen(roi_pen)
        for x1, y1, x2, y2 in self._completed_rois:
            wx1, wy1 = self._image_to_widget(x1, y1)
            wx2, wy2 = self._image_to_widget(x2, y2)
            painter.drawRect(QRect(int(wx1), int(wy1), int(wx2 - wx1), int(wy2 - wy1)))

        # Bead circles and labels (red)
        bead_pen = QPen(QColor(255, 100, 100), 2)
        painter.setPen(bead_pen)
        font = QFont()
        font.setPointSize(9)
        painter.setFont(font)
        r = max(8, int(10 * self._zoom))
        for bead_id, bx, by in self._beads:
            wx, wy = self._image_to_widget(bx, by)
            painter.setPen(bead_pen)
            painter.drawEllipse(int(wx) - r, int(wy) - r, r * 2, r * 2)
            painter.setPen(QPen(QColor(255, 220, 80), 1))
            painter.drawText(int(wx) + r + 2, int(wy) - 4, str(bead_id))

        # ROI being drawn (live feedback)
        if self._drawing_roi:
            painter.setPen(QPen(QColor(80, 200, 255), 1, Qt.DashLine))
            painter.drawRect(QRect(self._roi_start, self._roi_current).normalized())

    def wheelEvent(self, event):
        if self._image_np is None:
            return
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15
        mp = event.pos()
        ox = mp.x() - (mp.x() - self._offset.x()) * factor
        oy = mp.y() - (mp.y() - self._offset.y()) * factor
        self._zoom = max(0.1, min(self._zoom * factor, 20.0))
        self._offset = QPoint(int(ox), int(oy))
        self._clamp_offset()
        self.update()

    def mousePressEvent(self, event):
        if self._image_np is None:
            return
        if event.button() == Qt.LeftButton:
            if self._roi_mode:
                self._drawing_roi = True
                self._roi_start = event.pos()
                self._roi_current = event.pos()
            else:
                # start potential pan; resolve click vs drag at release
                self._pan_start = event.pos()
                self._offset_at_pan_start = QPoint(self._offset)
                self._is_panning = False

    def mouseMoveEvent(self, event):
        if self._drawing_roi:
            self._roi_current = event.pos()
            self.update()
        elif self._pan_start is not None and (event.buttons() & Qt.LeftButton):
            delta = event.pos() - self._pan_start
            if not self._is_panning and (abs(delta.x()) > _DRAG_THRESHOLD or abs(delta.y()) > _DRAG_THRESHOLD):
                self._is_panning = True
            if self._is_panning:
                self._offset = self._offset_at_pan_start + delta
                self._clamp_offset()
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self._drawing_roi:
                self._drawing_roi = False
                x1, y1 = self._widget_to_image(self._roi_start.x(), self._roi_start.y())
                x2, y2 = self._widget_to_image(event.pos().x(), event.pos().y())
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                if x2 > x1 and y2 > y1:
                    self.roi_drawn.emit(x1, y1, x2, y2)
                self.update()
            elif not self._is_panning and self._pan_start is not None:
                ix, iy = self._widget_to_image(event.pos().x(), event.pos().y())
                h, w = self._image_np.shape
                if 0 <= ix < w and 0 <= iy < h:
                    mods = event.modifiers()
                    if mods & (Qt.ControlModifier | Qt.MetaModifier):
                        self.bead_added.emit(ix, iy)
                    else:
                        self.bead_clicked.emit(ix, iy)
            self._pan_start = None
            self._is_panning = False


class ImagePanel(QWidget):
    """Left panel: image viewer with controls and bead selection."""

    bead_selected = pyqtSignal(int, int)
    bead_appended = pyqtSignal(int, int)
    roi_added = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stack = None
        self._current_z = 0
        self._beads: list = []
        self._rois: list = []
        self._x_um = 1.0
        self._y_um = 1.0
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # ── Row 1: Z-slice + zoom buttons ────────────────────────────────────
        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Z:"))
        self._z_slider = QSlider(Qt.Horizontal)
        self._z_slider.setRange(0, 0)
        self._z_slider.valueChanged.connect(self._on_z_changed)
        row1.addWidget(self._z_slider, stretch=1)
        self._z_label = QLabel("0 / 0")
        self._z_label.setFixedWidth(52)
        row1.addWidget(self._z_label)
        row1.addSpacing(8)
        zoom_out_btn = QPushButton("−")
        zoom_out_btn.setFixedWidth(28)
        zoom_out_btn.clicked.connect(lambda: self._canvas.set_zoom(self._canvas._zoom / 1.25))
        zoom_in_btn = QPushButton("+")
        zoom_in_btn.setFixedWidth(28)
        zoom_in_btn.clicked.connect(lambda: self._canvas.set_zoom(self._canvas._zoom * 1.25))
        zoom_reset_btn = QPushButton("Fit")
        zoom_reset_btn.setFixedWidth(36)
        zoom_reset_btn.clicked.connect(self._fit_zoom)
        row1.addWidget(zoom_out_btn)
        row1.addWidget(zoom_in_btn)
        row1.addWidget(zoom_reset_btn)
        layout.addLayout(row1)

        # ── Row 2: Min/Max display sliders ────────────────────────────────────
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Min:"))
        self._min_slider = QSlider(Qt.Horizontal)
        self._min_slider.setRange(0, 1000)
        self._min_slider.setValue(0)
        self._min_slider.setToolTip("Display minimum (visualization only — does not affect PSF calculation)")
        self._min_slider.valueChanged.connect(self._on_bc_changed)
        row2.addWidget(self._min_slider, stretch=1)
        row2.addWidget(QLabel("Max:"))
        self._max_slider = QSlider(Qt.Horizontal)
        self._max_slider.setRange(0, 1000)
        self._max_slider.setValue(1000)
        self._max_slider.setToolTip("Display maximum (visualization only — does not affect PSF calculation)")
        self._max_slider.valueChanged.connect(self._on_bc_changed)
        row2.addWidget(self._max_slider, stretch=1)
        save_btn = QPushButton("Save Image")
        save_btn.clicked.connect(self._save_image)
        row2.addWidget(save_btn)
        layout.addLayout(row2)

        # ── Canvas ────────────────────────────────────────────────────────────
        self._canvas = ImageCanvas()
        self._canvas.bead_clicked.connect(self._on_bead_clicked)
        self._canvas.bead_added.connect(self._on_bead_added)
        self._canvas.roi_drawn.connect(self._on_roi_drawn)
        layout.addWidget(self._canvas, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_stack(self, stack: np.ndarray, x_um=1.0, y_um=1.0):
        self._stack = stack.astype(np.float32)
        self._x_um = x_um
        self._y_um = y_um
        self._current_z = 0
        nz = stack.shape[0]
        self._z_slider.setRange(0, nz - 1)
        self._z_slider.setValue(0)
        self._z_label.setText(f"1 / {nz}")
        self._update_display()
        self._fit_zoom()

    def set_beads(self, beads: list):
        self._beads = beads
        self._canvas.set_beads(beads)

    def get_rois(self):
        return list(self._rois)

    def clear_rois(self):
        self._rois.clear()
        self._canvas.set_completed_rois([])

    # ── Handlers ──────────────────────────────────────────────────────────────

    def _on_z_changed(self, val):
        self._current_z = val
        nz = self._stack.shape[0] if self._stack is not None else 0
        self._z_label.setText(f"{val + 1} / {nz}")
        self._update_display()

    def _on_bc_changed(self):
        self._update_display()

    def _on_roi_mode_toggled(self, checked: bool):
        self._canvas.set_roi_mode(checked)

    def set_roi_mode(self, enabled: bool):
        self._canvas.set_roi_mode(enabled)

    def _update_display(self):
        if self._stack is None:
            return
        frame = self._stack[self._current_z]
        lo_frac = self._min_slider.value() / 1000.0
        hi_frac = self._max_slider.value() / 1000.0
        if hi_frac <= lo_frac:
            hi_frac = lo_frac + 0.001
        gmin, gmax = self._stack.min(), self._stack.max()
        clip_lo = gmin + lo_frac * (gmax - gmin)
        clip_hi = gmin + hi_frac * (gmax - gmin)
        self._canvas.set_image(np.clip(frame, clip_lo, clip_hi))

    def _on_bead_clicked(self, x, y):
        self.bead_selected.emit(x, y)

    def _on_bead_added(self, x, y):
        self.bead_appended.emit(x, y)

    def _on_roi_drawn(self, x1, y1, x2, y2):
        self._rois.append((x1, y1, x2, y2))
        self._canvas.set_completed_rois(self._rois)
        self.roi_added.emit(x1, y1, x2, y2)

    def _fit_zoom(self):
        if self._canvas._pixmap is None:
            return
        cw, ch = self._canvas.width(), self._canvas.height()
        iw, ih = self._canvas._pixmap.width(), self._canvas._pixmap.height()
        if iw == 0 or ih == 0:
            return
        zoom = min(cw / iw, ch / ih)
        self._canvas._zoom = zoom
        self._canvas._offset = QPoint(int((cw - iw * zoom) / 2), int((ch - ih * zoom) / 2))
        self._canvas.update()

    def _save_image(self):
        if self._stack is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "results/",
            "TIFF (*.tiff);;JPEG (*.jpg);;PNG (*.png)"
        )
        if not path:
            return
        pixmap = self._canvas.grab()
        if path.endswith('.tiff') or path.endswith('.tif'):
            pixmap.save(path, 'TIFF')
        elif path.endswith('.png'):
            pixmap.save(path, 'PNG')
        else:
            pixmap.save(path, 'JPEG')
