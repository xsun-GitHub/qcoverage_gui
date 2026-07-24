#!/usr/bin/env python3
"""QCoverage Version 3 (PyQt6) — standalone SAXS/WAXS detector coverage planner.

Run directly:
    python qcoverage_version3.py

Dependencies:
    python -m pip install pyFAI matplotlib PyQt6 numpy

Coordinate conventions
----------------------
* Detector beam centre X/Y is measured from detector pixel (0, 0).
* X follows detector columns; Y follows detector rows.
* The beamstop is centred on the direct beam.
* Beamstop position is entered as its distance from the Detector 1 plane.
  0 mm means approximately at Detector 1; larger values move it toward the sample.
* The beamstop shadow is projected separately onto each detector plane.
* Detectors are flat and normal to the incident beam.

Version 2 features
------------------
* Independent beam centres for two detectors.
* Configurable beamstop diameter and longitudinal position.
* Detector-plane geometry view.

Version 3 features
------------------
* Sampled two-dimensional qx-qy coverage.
* Beamstop shadow included in the sampled coverage.
* Radial q intervals derived from accessible detector pixels.
* CSV and PNG export.
"""
from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

HC_KEV_ANGSTROM = 12.398419843320026
BEAMSTOP_DIAMETERS_MM = (4.5, 6.0, 9.0)
BAR_COLOURS = ("tab:orange", "tab:blue")
PREFERRED_DETECTORS: dict[str, tuple[str, ...]] = {
    "DECTRIS Eiger2 4M": ("eiger2_4m", "eiger4m", "Eiger2 4M", "Eiger2_4M"),
    "DECTRIS Eiger2 9M": ("eiger2_9m", "eiger9m", "Eiger2 9M", "Eiger2_9M"),
    "PerkinElmer XRD 1621": ("perkin", "perkinelmer", "PerkinElmer", "Perkin"),
}


@dataclass(frozen=True)
class DetectorInfo:
    label: str
    factory_name: str
    class_name: str
    rows: int
    columns: int
    pixel_y_m: float
    pixel_x_m: float

    @property
    def width_mm(self) -> float:
        return self.columns * self.pixel_x_m * 1e3

    @property
    def height_mm(self) -> float:
        return self.rows * self.pixel_y_m * 1e3


@dataclass(frozen=True)
class GeometryInput:
    energy_kev: float
    distance_mm: float
    beam_x: float
    beam_y: float
    beam_unit: str
    beamstop_diameter_mm: float
    beamstop_to_detector1_mm: float
    detector1_distance_mm: float


@dataclass(frozen=True)
class CoverageResult:
    q_min: float
    q_max: float
    r_min_mm: float
    r_max_mm: float
    two_theta_min_deg: float
    two_theta_max_deg: float
    wavelength_angstrom: float
    beam_x_mm: float
    beam_y_mm: float
    beamstop_shadow_diameter_mm: float
    beamstop_sample_distance_mm: float
    qx: np.ndarray
    qy: np.ndarray
    q_abs: np.ndarray
    detector_x_mm: np.ndarray
    detector_y_mm: np.ndarray


def _detector_factory():
    try:
        from pyFAI.detectors import detector_factory
    except ImportError as exc:
        raise RuntimeError(
            "pyFAI is not installed. Run:\n"
            "python -m pip install pyFAI matplotlib PyQt6 numpy"
        ) from exc
    return detector_factory


def detector_from_aliases(label: str, aliases: tuple[str, ...]) -> DetectorInfo:
    detector_factory = _detector_factory()
    errors: list[str] = []
    for alias in aliases:
        try:
            detector = detector_factory(alias)
            shape = getattr(detector, "max_shape", None) or getattr(detector, "shape", None)
            if not shape or len(shape) != 2:
                raise ValueError("No usable two-dimensional detector shape")
            pixel1 = float(detector.pixel1)
            pixel2 = float(detector.pixel2)
            if pixel1 <= 0 or pixel2 <= 0:
                raise ValueError("Invalid pixel size")
            return DetectorInfo(
                label=label,
                factory_name=alias,
                class_name=type(detector).__name__,
                rows=int(shape[0]),
                columns=int(shape[1]),
                pixel_y_m=pixel1,
                pixel_x_m=pixel2,
            )
        except Exception as exc:
            errors.append(f"{alias}: {exc}")
    raise RuntimeError(f"Could not load {label} from pyFAI:\n" + "\n".join(errors))


def load_preferred_detectors() -> tuple[list[DetectorInfo], list[str]]:
    loaded: list[DetectorInfo] = []
    warnings: list[str] = []
    for label, aliases in PREFERRED_DETECTORS.items():
        try:
            loaded.append(detector_from_aliases(label, aliases))
        except RuntimeError as exc:
            warnings.append(str(exc))
    if not loaded:
        raise RuntimeError("No configured detector could be loaded.\n\n" + "\n\n".join(warnings))
    return loaded, warnings


def beam_position_mm(detector: DetectorInfo, geometry: GeometryInput) -> tuple[float, float]:
    if geometry.beam_unit == "pixels":
        return (
            geometry.beam_x * detector.pixel_x_m * 1e3,
            geometry.beam_y * detector.pixel_y_m * 1e3,
        )
    if geometry.beam_unit == "mm":
        return geometry.beam_x, geometry.beam_y
    raise ValueError("Beam-centre unit must be pixels or mm.")


def radius_to_q(radius_mm: np.ndarray | float, distance_mm: float, energy_kev: float):
    wavelength_a = HC_KEV_ANGSTROM / energy_kev
    two_theta = np.arctan2(radius_mm, distance_mm)
    theta = 0.5 * two_theta
    return 4.0 * np.pi * np.sin(theta) / wavelength_a


def calculate_coverage(
    detector: DetectorInfo,
    geometry: GeometryInput,
    samples_x: int = 330,
    samples_y: int = 330,
) -> CoverageResult:
    """Sample accessible detector pixels and calculate radial and 2D q coverage."""
    if geometry.energy_kev <= 0:
        raise ValueError("X-ray energy must be greater than zero.")
    if geometry.distance_mm <= 0:
        raise ValueError("Sample-detector distance must be greater than zero.")
    if geometry.beamstop_diameter_mm < 0:
        raise ValueError("Beamstop diameter cannot be negative.")

    beam_x_mm, beam_y_mm = beam_position_mm(detector, geometry)
    if geometry.detector1_distance_mm <= 0:
        raise ValueError("Detector 1 sample-detector distance must be greater than zero.")
    if geometry.beamstop_to_detector1_mm < 0:
        raise ValueError("Beamstop-to-Detector-1 distance cannot be negative.")
    if geometry.beamstop_to_detector1_mm >= geometry.detector1_distance_mm:
        raise ValueError(
            "Beamstop-to-Detector-1 distance must be smaller than the Detector 1 "
            "sample-detector distance; at the sample position the projection diverges."
        )

    # Common physical beamstop position measured from the sample.
    # s = 0 puts the stop in the Detector 1 plane; increasing s moves it toward sample.
    beamstop_sample_distance_mm = (
        geometry.detector1_distance_mm - geometry.beamstop_to_detector1_mm
    )
    # A detector upstream of the beamstop still has valid q coverage; the
    # downstream beamstop simply cannot cast a shadow onto that detector.
    # Equality is valid and gives a 1:1 shadow in the detector plane.
    beamstop_is_upstream = geometry.distance_mm >= beamstop_sample_distance_mm
    if beamstop_is_upstream:
        # Point-projection magnification from the sample to the detector plane.
        shadow_scale = geometry.distance_mm / beamstop_sample_distance_mm
        shadow_diameter_mm = geometry.beamstop_diameter_mm * shadow_scale
    else:
        shadow_diameter_mm = 0.0
    stop_x_mm = beam_x_mm
    stop_y_mm = beam_y_mm
    stop_radius_mm = shadow_diameter_mm / 2.0

    # Include rectangle boundaries; sampling is used for beamstop clipping and q-space view.
    x = np.linspace(0.0, detector.width_mm, max(40, samples_x), dtype=float)
    y = np.linspace(0.0, detector.height_mm, max(40, samples_y), dtype=float)
    xx, yy = np.meshgrid(x, y)

    blocked = (xx - stop_x_mm) ** 2 + (yy - stop_y_mm) ** 2 <= stop_radius_mm**2
    accessible = ~blocked
    if not np.any(accessible):
        raise ValueError(f"{detector.label}: the beamstop covers the sampled detector area.")

    dx = xx - beam_x_mm
    dy = yy - beam_y_mm
    radius = np.hypot(dx, dy)

    # Scattering-vector components for a flat detector.
    wavelength_a = HC_KEV_ANGSTROM / geometry.energy_kev
    k = 2.0 * np.pi / wavelength_a
    ray_length = np.sqrt(geometry.distance_mm**2 + radius**2)
    qx_all = k * dx / ray_length
    qy_all = k * dy / ray_length
    qz_all = k * (geometry.distance_mm / ray_length - 1.0)
    q_abs_all = np.sqrt(qx_all**2 + qy_all**2 + qz_all**2)

    r_values = radius[accessible]
    q_values = q_abs_all[accessible]
    r_min = float(np.min(r_values))
    r_max = float(np.max(r_values))
    q_min = float(np.min(q_values))
    q_max = float(np.max(q_values))

    return CoverageResult(
        q_min=q_min,
        q_max=q_max,
        r_min_mm=r_min,
        r_max_mm=r_max,
        two_theta_min_deg=math.degrees(math.atan2(r_min, geometry.distance_mm)),
        two_theta_max_deg=math.degrees(math.atan2(r_max, geometry.distance_mm)),
        wavelength_angstrom=wavelength_a,
        beam_x_mm=beam_x_mm,
        beam_y_mm=beam_y_mm,
        beamstop_shadow_diameter_mm=shadow_diameter_mm,
        beamstop_sample_distance_mm=beamstop_sample_distance_mm,
        qx=qx_all[accessible],
        qy=qy_all[accessible],
        q_abs=q_values,
        detector_x_mm=xx[accessible],
        detector_y_mm=yy[accessible],
    )


try:
    from PyQt6 import QtCore, QtWidgets
except ImportError as exc:
    raise SystemExit(
        "PyQt6 is required. Install dependencies with:\n"
        "python -m pip install pyFAI matplotlib PyQt6 numpy"
    ) from exc

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Circle, Rectangle


class DetectorPanel(QtWidgets.QGroupBox):
    changed = QtCore.pyqtSignal()

    def __init__(self, title: str, detectors: list[DetectorInfo], optional: bool):
        super().__init__(title)
        layout = QtWidgets.QFormLayout(self)
        self.enabled = QtWidgets.QCheckBox("Use this detector")
        # Start with both detector panels active.  Detector 2 remains optional
        # because its checkbox can still be cleared by the user.
        self.enabled.setChecked(True)
        self.enabled.setEnabled(optional)
        layout.addRow(self.enabled)

        self.detector_combo = QtWidgets.QComboBox()
        for detector in detectors:
            self.detector_combo.addItem(detector.label, detector)
        layout.addRow("Detector:", self.detector_combo)

        self.distance = self._spin(0.01, 100000.0, 1000.0, 3)
        self.distance.setSuffix(" mm")
        self.beam_x = self._spin(-1_000_000, 1_000_000, 1500, 3)
        self.beam_y = self._spin(-1_000_000, 1_000_000, 1500, 3)
        self.beam_unit = QtWidgets.QComboBox()
        self.beam_unit.addItems(("pixels", "mm"))
        layout.addRow("Sample-detector distance:", self.distance)
        layout.addRow("Beam centre X:", self.beam_x)
        layout.addRow("Beam centre Y:", self.beam_y)
        layout.addRow("Coordinate unit:", self.beam_unit)

        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addRow("pyFAI definition:", self.details)

        self.enabled.toggled.connect(self._set_enabled)
        self.enabled.toggled.connect(self.changed.emit)
        self.detector_combo.currentIndexChanged.connect(self._update_details)
        self.detector_combo.currentIndexChanged.connect(self.changed.emit)
        self.distance.valueChanged.connect(self.changed.emit)
        # Also recalculate when a typed value is committed.  This makes the
        # update reliable even when Qt considers the rounded value unchanged
        # until the editor loses focus.
        self.distance.editingFinished.connect(self.changed.emit)
        self.beam_x.valueChanged.connect(self.changed.emit)
        self.beam_y.valueChanged.connect(self.changed.emit)
        self.beam_unit.currentIndexChanged.connect(self.changed.emit)
        self._set_enabled(self.enabled.isChecked())
        self._update_details()

    @staticmethod
    def _spin(minimum, maximum, value, decimals):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        return widget

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (self.detector_combo, self.distance, self.beam_x, self.beam_y, self.beam_unit, self.details):
            widget.setEnabled(enabled)

    def _update_details(self) -> None:
        detector = self.detector_info(ignore_enabled=True)
        if detector is None:
            return
        self.details.setText(
            f"pyFAI class: {detector.class_name}\n"
            f"Factory alias: {detector.factory_name}\n"
            f"Pixels: {detector.columns} × {detector.rows}\n"
            f"Pixel size X/Y: {detector.pixel_x_m * 1e6:.3f} / "
            f"{detector.pixel_y_m * 1e6:.3f} µm\n"
            f"Envelope W/H: {detector.width_mm:.3f} / {detector.height_mm:.3f} mm"
        )

    def detector_info(self, ignore_enabled: bool = False) -> Optional[DetectorInfo]:
        if not ignore_enabled and not self.enabled.isChecked():
            return None
        return self.detector_combo.currentData()

    def geometry(self, energy, stop_diameter, stop_to_detector1, detector1_distance) -> GeometryInput:
        return GeometryInput(
            energy_kev=energy,
            distance_mm=self.distance.value(),
            beam_x=self.beam_x.value(),
            beam_y=self.beam_y.value(),
            beam_unit=self.beam_unit.currentText(),
            beamstop_diameter_mm=stop_diameter,
            beamstop_to_detector1_mm=stop_to_detector1,
            detector1_distance_mm=detector1_distance,
        )


class PlotCanvas(FigureCanvas):
    def __init__(self, figsize=(8, 4)):
        self.figure = Figure(figsize=figsize, constrained_layout=True)
        super().__init__(self.figure)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QCoverage Version 3.1 — PyQt6 detector planner")
        self.resize(1400, 1000)
        self._last_results: list[tuple[DetectorInfo, CoverageResult, DetectorPanel]] = []

        detectors, warnings = load_preferred_detectors()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)

        detector_row = QtWidgets.QHBoxLayout()
        self.panel_1 = DetectorPanel("Detector 1", detectors, optional=False)
        self.panel_2 = DetectorPanel("Detector 2", detectors, optional=True)

        detector_1_index = self.panel_1.detector_combo.findText("DECTRIS Eiger2 9M")
        if detector_1_index >= 0:
            self.panel_1.detector_combo.setCurrentIndex(detector_1_index)
        self.panel_1.distance.setValue(2000.0)
        self.panel_1.beam_x.setValue(1554.0)
        self.panel_1.beam_y.setValue(1631.0)

        detector_2_index = self.panel_2.detector_combo.findText("DECTRIS Eiger2 4M")
        if detector_2_index >= 0:
            self.panel_2.detector_combo.setCurrentIndex(detector_2_index)
        self.panel_2.distance.setValue(300.0)
        self.panel_2.beam_x.setValue(1034.0)
        self.panel_2.beam_y.setValue(-400.0)

        detector_row.addWidget(self.panel_1)
        detector_row.addWidget(self.panel_2)
        root.addLayout(detector_row)

        geometry_box = QtWidgets.QGroupBox("Experimental geometry and beamstop")
        grid = QtWidgets.QGridLayout(geometry_box)
        self.energy = self._spin(0.1, 1000, 12.0, 4, " keV")
        self.beamstop = QtWidgets.QComboBox()
        for diameter in BEAMSTOP_DIAMETERS_MM:
            self.beamstop.addItem(f"{diameter:g} mm", diameter)
        self.custom_stop = self._spin(0, 1000, 6.0, 3, " mm")
        self.custom_stop.setEnabled(False)
        self.beamstop.addItem("Custom", None)
        self.stop_to_detector1 = self._spin(0.0, 99999.0, 0.0, 3, " mm")

        grid.addWidget(QtWidgets.QLabel("X-ray energy:"), 0, 0)
        grid.addWidget(self.energy, 0, 1)
        grid.addWidget(QtWidgets.QLabel("Beamstop diameter:"), 0, 2)
        grid.addWidget(self.beamstop, 0, 3)
        grid.addWidget(self.custom_stop, 0, 4)
        grid.addWidget(QtWidgets.QLabel("Beamstop to Detector 1:"), 1, 0)
        grid.addWidget(self.stop_to_detector1, 1, 1)
        note = QtWidgets.QLabel(
            "0 mm = beamstop approximately in the Detector 1 plane; increasing the value "
            "moves it toward the sample. It must remain smaller than Detector 1 distance."
        )
        note.setWordWrap(True)
        grid.addWidget(note, 1, 2, 1, 4)
        root.addWidget(geometry_box)

        buttons = QtWidgets.QHBoxLayout()
        self.calculate_button = QtWidgets.QPushButton("Calculate / Update")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_png_button = QtWidgets.QPushButton("Export all plots PNG")
        buttons.addWidget(self.calculate_button)
        buttons.addWidget(self.export_csv_button)
        buttons.addWidget(self.export_png_button)
        buttons.addStretch(1)
        root.addLayout(buttons)

        # Left column: plots 1 and 2 with equal width and a 1:2 height ratio.
        # Right column: plot 3 spans both rows and is half the left-column width.
        self.plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QGridLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setHorizontalSpacing(6)
        plot_layout.setVerticalSpacing(6)

        self.range_canvas = PlotCanvas((8, 2.0))
        self.geometry_canvas = PlotCanvas((8, 4))
        self.qspace_canvas = PlotCanvas((4, 6))
        for canvas in (self.range_canvas, self.geometry_canvas, self.qspace_canvas):
            canvas.setMinimumSize(0, 0)
            canvas.setSizePolicy(
                QtWidgets.QSizePolicy.Policy.Expanding,
                QtWidgets.QSizePolicy.Policy.Expanding,
            )
        plot_layout.addWidget(self.range_canvas, 0, 0)
        plot_layout.addWidget(self.geometry_canvas, 1, 0)
        plot_layout.addWidget(self.qspace_canvas, 0, 1, 2, 1)
        plot_layout.setRowStretch(0, 1)
        plot_layout.setRowStretch(1, 2)
        plot_layout.setColumnStretch(0, 2)
        plot_layout.setColumnStretch(1, 1)
        root.addWidget(self.plot_container, stretch=1)

        self.table = QtWidgets.QTableWidget(0, 14)
        self.table.setHorizontalHeaderLabels((
            "Detector", "q min (Å⁻¹)", "q max (Å⁻¹)", "r min (mm)", "r max (mm)",
            "2θ min (°)", "2θ max (°)", "λ (Å)", "Beam X (mm)", "Beam Y (mm)",
            "Detector distance (mm)", "Stop-sample distance (mm)",
            "Stop shadow diameter (mm)", "Sample points",
        ))
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        three_row_height = (
            self.table.horizontalHeader().sizeHint().height()
            + 3 * self.table.verticalHeader().defaultSectionSize()
            + 2 * self.table.frameWidth()
            + 4
        )
        self.table.setFixedHeight(three_row_height)
        root.addWidget(self.table)

        self.calculate_button.clicked.connect(self.calculate)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_png_button.clicked.connect(self.export_png)
        self.panel_1.changed.connect(self._detector1_geometry_changed)
        self.panel_2.changed.connect(self.calculate)
        for widget in (self.energy, self.stop_to_detector1, self.custom_stop):
            widget.valueChanged.connect(self.calculate)
        self.beamstop.currentIndexChanged.connect(self._beamstop_changed)

        if warnings:
            self.statusBar().showMessage("Some configured pyFAI detectors were unavailable.")
        self._detector1_geometry_changed()

    @staticmethod
    def _spin(minimum, maximum, value, decimals, suffix=""):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setSuffix(suffix)
        widget.setKeyboardTracking(False)
        return widget

    def _detector1_geometry_changed(self) -> None:
        # The exact sample position is singular, so keep a small positive clearance.
        maximum = max(0.0, self.panel_1.distance.value() - 0.001)
        self.stop_to_detector1.setMaximum(maximum)
        self.calculate()

    def _beamstop_changed(self) -> None:
        self.custom_stop.setEnabled(self.beamstop.currentData() is None)
        self.calculate()

    def beamstop_diameter(self) -> float:
        data = self.beamstop.currentData()
        return self.custom_stop.value() if data is None else float(data)

    def calculate(self) -> None:
        panels = [p for p in (self.panel_1, self.panel_2) if p.detector_info() is not None]
        if not panels:
            return
        results: list[tuple[DetectorInfo, CoverageResult, DetectorPanel]] = []
        errors: list[str] = []
        for panel in panels:
            try:
                detector = panel.detector_info()
                assert detector is not None
                detector1_distance = self.panel_1.distance.value()
                geometry = panel.geometry(
                    self.energy.value(), self.beamstop_diameter(),
                    self.stop_to_detector1.value(), detector1_distance,
                )
                result = calculate_coverage(detector, geometry)
                results.append((detector, result, panel))
            except ValueError as exc:
                errors.append(str(exc))

        if not results:
            # Never leave a stale plot on screen: it looks as though changing
            # an input had no effect.
            self._last_results = []
            for canvas in (self.range_canvas, self.geometry_canvas, self.qspace_canvas):
                canvas.figure.clear()
                ax = canvas.figure.add_subplot(111)
                ax.text(
                    0.5, 0.5, "\n".join(errors),
                    ha="center", va="center", wrap=True,
                    transform=ax.transAxes, color="tab:red",
                )
                ax.set_axis_off()
                canvas.draw()
            self.table.setRowCount(0)
            self.statusBar().showMessage(" | ".join(errors))
            return

        self._last_results = results
        self._plot_ranges(results)
        self._plot_geometry(results)
        self._plot_qspace(results)
        self._update_table(results)
        if errors:
            self.statusBar().showMessage(
                "Updated valid detector(s). Invalid input: " + " | ".join(errors)
            )
        else:
            self.statusBar().showMessage("Version 3 coverage updated.")

    def _plot_ranges(self, results):
        fig = self.range_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        for i, (detector, result, _) in enumerate(results):
            ax.barh(i, result.q_max-result.q_min, left=result.q_min, height=0.5,
                    color=BAR_COLOURS[i], label=detector.label)
            ax.text(result.q_min, i+0.31, f"{result.q_min:.5g}", ha="left", fontsize=9)
            ax.text(result.q_max, i+0.31, f"{result.q_max:.5g} Å⁻¹", ha="right", fontsize=9)
        ax.set_yticks(range(len(results)), [d.label for d, _, _ in results])
        ax.set_xlabel("Scattering vector q (Å⁻¹)")
        ax.grid(axis="x", alpha=0.3)
        ax.set_ylim(-0.65, max(0.8, len(results)-0.2))
        self.range_canvas.draw()

    def _plot_geometry(self, results):
        fig = self.geometry_canvas.figure
        fig.clear()
        axes = fig.subplots(1, len(results), squeeze=False)[0]
        for i, ((detector, result, _), ax) in enumerate(zip(results, axes)):
            ax.add_patch(Rectangle((0, 0), detector.width_mm, detector.height_mm,
                                   fill=False, linewidth=1.8, edgecolor=BAR_COLOURS[i]))
            ax.add_patch(Circle((result.beam_x_mm, result.beam_y_mm),
                                result.beamstop_shadow_diameter_mm/2,
                                alpha=0.35, color="black"))
            ax.scatter([result.beam_x_mm], [result.beam_y_mm], marker="x", s=75,
                       color=BAR_COLOURS[i], label="Direct beam")
            ax.scatter(result.detector_x_mm[::80], result.detector_y_mm[::80], s=1,
                       alpha=0.16, color=BAR_COLOURS[i])
            ax.set_aspect("equal", adjustable="box")
            margin = max(detector.width_mm, detector.height_mm)*0.08
            ax.set_xlim(-margin, detector.width_mm+margin)
            ax.set_ylim(-margin, detector.height_mm+margin)
            ax.set_xlabel("Detector X (mm)")
            ax.set_ylabel("Detector Y (mm)")
            ax.grid(alpha=0.2)
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                borderaxespad=0,
                fontsize=8,
            )
        self.geometry_canvas.draw()

    def _plot_qspace(self, results):
        fig = self.qspace_canvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        for i, (detector, result, _) in enumerate(results):
            stride = max(1, result.qx.size // 30000)
            ax.scatter(result.qx[::stride], result.qy[::stride], s=2, alpha=0.28,
                       color=BAR_COLOURS[i], label=detector.label, rasterized=True)
        ax.axhline(0, linewidth=0.7, color="black", alpha=0.5)
        ax.axvline(0, linewidth=0.7, color="black", alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("qx (Å⁻¹)")
        ax.set_ylabel("qy (Å⁻¹)")
        ax.grid(alpha=0.2)
        ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            borderaxespad=0,
        )
        self.qspace_canvas.draw()

    def _update_table(self, results):
        self.table.setRowCount(len(results))
        diameter = self.beamstop_diameter()
        for row, (detector, result, panel) in enumerate(results):
            values = (
                detector.label, f"{result.q_min:.8g}", f"{result.q_max:.8g}",
                f"{result.r_min_mm:.7g}", f"{result.r_max_mm:.7g}",
                f"{result.two_theta_min_deg:.7g}", f"{result.two_theta_max_deg:.7g}",
                f"{result.wavelength_angstrom:.7g}", f"{result.beam_x_mm:.7g}",
                f"{result.beam_y_mm:.7g}", f"{panel.distance.value():.7g}",
                f"{result.beamstop_sample_distance_mm:.7g}",
                f"{result.beamstop_shadow_diameter_mm:.7g}", str(result.qx.size),
            )
            for col, value in enumerate(values):
                self.table.setItem(row, col, QtWidgets.QTableWidgetItem(value))

    def export_csv(self) -> None:
        if not self._last_results:
            return
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export Q coverage", "qcoverage_results.csv", "CSV files (*.csv)"
        )
        if not filename:
            return
        path = Path(filename)
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(("detector", "q_min_A-1", "q_max_A-1", "r_min_mm", "r_max_mm",
                             "two_theta_min_deg", "two_theta_max_deg", "wavelength_A",
                             "beam_x_mm", "beam_y_mm", "detector_distance_mm",
                             "beamstop_to_detector1_mm", "beamstop_sample_distance_mm",
                             "beamstop_physical_diameter_mm", "beamstop_shadow_diameter_mm"))
            for detector, result, panel in self._last_results:
                writer.writerow((detector.label, result.q_min, result.q_max, result.r_min_mm,
                                 result.r_max_mm, result.two_theta_min_deg,
                                 result.two_theta_max_deg, result.wavelength_angstrom,
                                 result.beam_x_mm, result.beam_y_mm, panel.distance.value(),
                                 self.stop_to_detector1.value(),
                                 result.beamstop_sample_distance_mm,
                                 self.beamstop_diameter(),
                                 result.beamstop_shadow_diameter_mm))
        self.statusBar().showMessage(f"Exported {path}")

    def export_png(self) -> None:
        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export all plots", "qcoverage_plots.png", "PNG files (*.png)"
        )
        if filename:
            image = self.plot_container.grab()
            image.save(filename, "PNG")
            self.statusBar().showMessage(f"Exported {filename}")


def main() -> int:
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("QCoverage Version 3.1")
    try:
        window = MainWindow()
    except RuntimeError as exc:
        QtWidgets.QMessageBox.critical(None, "Startup error", str(exc))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
