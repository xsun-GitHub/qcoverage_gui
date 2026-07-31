#!/usr/bin/env python3
"""QCoverage Version 4 (PyQt6) — standalone SAXS/WAXS detector coverage planner.

Run directly:
    python qcoverage_version4.py

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

Version 4 features
--------------------
* Drag either direct-beam marker in the detector-geometry plot.
* Beam-centre input fields and all q-coverage plots update in real time.
* Q-range design from a requested q min/max to Energy, Distance, Beamstop,
  and Detector recommendations.
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
CALIBRANT_COLOURS = ("tab:green", "tab:purple")
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


def load_calibrant_names() -> list[str]:
    """Return calibrants available from the installed pyFAI version."""
    try:
        from pyFAI.calibrant import CALIBRANT_FACTORY
    except ImportError as exc:
        raise RuntimeError("Could not load pyFAI calibrants.") from exc
    return sorted(CALIBRANT_FACTORY.keys(), key=str.casefold)


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

    # Radial scattering magnitude for a flat detector.
    wavelength_a = HC_KEV_ANGSTROM / geometry.energy_kev
    two_theta = np.arctan2(radius, geometry.distance_mm)
    q_abs_all = 4.0 * np.pi * np.sin(0.5 * two_theta) / wavelength_a

    # Map every sampled detector-plane point into q-space using its original
    # azimuth around the direct beam. Thus the right plot is the point-by-point
    # q transform of the detector shapes shown in the geometry plot, and its
    # radius is exactly the q reported by the radial-range plot.
    qx_all = np.divide(
        q_abs_all * dx, radius, out=np.zeros_like(radius), where=radius > 0
    )
    qy_all = np.divide(
        q_abs_all * dy, radius, out=np.zeros_like(radius), where=radius > 0
    )

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

    def __init__(
        self,
        title: str,
        detectors: list[DetectorInfo],
        calibrant_names: list[str],
        optional: bool,
    ):
        super().__init__(title)
        layout = QtWidgets.QGridLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setVerticalSpacing(3)
        self.enabled = QtWidgets.QCheckBox("Use this detector")
        # Start with both detector panels active.  Detector 2 remains optional
        # because its checkbox can still be cleared by the user.
        self.enabled.setChecked(True)
        self.enabled.setEnabled(optional)

        self.detector_combo = QtWidgets.QComboBox()
        for detector in detectors:
            self.detector_combo.addItem(detector.label, detector)
        self.detector_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.detector_combo.setFixedWidth(210)

        self.inverse_change_detector = QtWidgets.QCheckBox()
        self.inverse_change_detector.setToolTip("Allow Inverse Calc to change Detector")
        # Detector type is selected explicitly and is not changed by Inverse Calc.
        self.inverse_change_detector.setChecked(False)

        self.distance = self._spin(0.01, 100000.0, 1000.0, 3)
        self.distance.setSuffix(" mm")
        self.distance.setFixedWidth(150)
        self.inverse_change_distance = QtWidgets.QCheckBox()
        self.inverse_change_distance.setToolTip(
            "Allow Inverse Calc to change sample-detector distance"
        )
        self.inverse_change_distance.setChecked(True)
        self.inverse_constrain_distance = QtWidgets.QCheckBox()
        self.inverse_constrain_distance.setToolTip(
            "Constrain sample-detector distance during Inverse Calc"
        )
        self.beam_x = self._spin(0.0, 1_000_000, 1500, 3)
        self.beam_y = self._spin(0.0, 1_000_000, 1500, 3)
        self.beam_x.setFixedWidth(150)
        self.beam_y.setFixedWidth(150)
        self.inverse_change_beam_x = QtWidgets.QCheckBox()
        self.inverse_change_beam_x.setToolTip(
            "Allow Inverse Calc to change Beam centre X"
        )
        self.inverse_change_beam_x.setChecked(True)
        self.inverse_constrain_beam_x = QtWidgets.QCheckBox()
        self.inverse_constrain_beam_x.setToolTip(
            "Constrain Beam centre X during Inverse Calc"
        )
        self.inverse_change_beam_y = QtWidgets.QCheckBox()
        self.inverse_change_beam_y.setToolTip(
            "Allow Inverse Calc to change Beam centre Y"
        )
        self.inverse_change_beam_y.setChecked(True)
        self.inverse_constrain_beam_y = QtWidgets.QCheckBox()
        self.inverse_constrain_beam_y.setToolTip(
            "Constrain Beam centre Y during Inverse Calc"
        )
        self.beam_unit = QtWidgets.QComboBox()
        self.beam_unit.addItems(("pixels", "mm"))
        self.beam_unit.setFixedWidth(110)
        self._previous_beam_unit = "pixels"

        self.calibrant_combo = QtWidgets.QComboBox()
        self.calibrant_combo.addItems(calibrant_names)
        self.calibrant_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.calibrant_combo.setFixedWidth(150)

        self.details = QtWidgets.QLabel()
        self.details.setWordWrap(True)
        self.details.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)

        fit_header_2 = QtWidgets.QLabel("Fit")
        constraint_header_2 = QtWidgets.QLabel("Constraint")
        for header in (fit_header_2, constraint_header_2):
            header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        fit_header_2.setFixedWidth(32)
        constraint_header_2.setFixedWidth(72)
        constraint_header_2.setStyleSheet("font-size: 8pt;")

        layout.addWidget(self.enabled, 0, 0)
        layout.addWidget(fit_header_2, 0, 2)
        layout.addWidget(constraint_header_2, 0, 3)

        layout.addWidget(QtWidgets.QLabel("Detector:"), 1, 0)
        layout.addWidget(QtWidgets.QLabel("Sample-detector distance:"), 1, 1)
        layout.addWidget(self.detector_combo, 2, 0)
        layout.addWidget(self.distance, 2, 1)
        layout.addWidget(
            self.inverse_change_distance, 2, 2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(
            self.inverse_constrain_distance, 2, 3,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        layout.addWidget(QtWidgets.QLabel("Coordinate unit:"), 3, 0)
        layout.addWidget(QtWidgets.QLabel("Beam centre X:"), 3, 1)
        layout.addWidget(self.beam_unit, 4, 0)
        layout.addWidget(self.beam_x, 4, 1)
        layout.addWidget(
            self.inverse_change_beam_x, 4, 2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(
            self.inverse_constrain_beam_x, 4, 3,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        layout.addWidget(QtWidgets.QLabel("Calibrant:"), 5, 0)
        layout.addWidget(QtWidgets.QLabel("Beam centre Y:"), 5, 1)
        layout.addWidget(self.calibrant_combo, 6, 0)
        layout.addWidget(self.beam_y, 6, 1)
        layout.addWidget(
            self.inverse_change_beam_y, 6, 2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        layout.addWidget(
            self.inverse_constrain_beam_y, 6, 3,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )

        layout.addWidget(QtWidgets.QLabel("pyFAI definition:"), 7, 0)
        layout.addWidget(self.details, 8, 0, 1, 4)
        layout.setHorizontalSpacing(6)
        for column in range(4):
            layout.setColumnStretch(column, 0)

        self.enabled.toggled.connect(self._set_enabled)
        self.enabled.toggled.connect(self.changed.emit)
        self.detector_combo.currentIndexChanged.connect(self._update_details)
        self.detector_combo.currentIndexChanged.connect(self._update_beam_ranges)
        self.detector_combo.currentIndexChanged.connect(self.changed.emit)
        self.distance.valueChanged.connect(self.changed.emit)
        # Also recalculate when a typed value is committed.  This makes the
        # update reliable even when Qt considers the rounded value unchanged
        # until the editor loses focus.
        self.distance.editingFinished.connect(self.changed.emit)
        self.beam_x.valueChanged.connect(self.changed.emit)
        self.beam_y.valueChanged.connect(self.changed.emit)
        self.beam_unit.currentIndexChanged.connect(self._beam_unit_changed)
        self.beam_unit.currentIndexChanged.connect(self.changed.emit)
        self.calibrant_combo.currentIndexChanged.connect(self.changed.emit)
        self._set_enabled(self.enabled.isChecked())
        self._update_details()
        self._update_beam_ranges()

    @staticmethod
    def _spin(minimum, maximum, value, decimals):
        widget = QtWidgets.QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setDecimals(decimals)
        widget.setValue(value)
        widget.setKeyboardTracking(False)
        return widget

    def _set_enabled(self, enabled: bool) -> None:
        for widget in (
            self.detector_combo, self.distance, self.beam_x, self.beam_y,
            self.beam_unit, self.calibrant_combo, self.details,
            self.inverse_change_detector, self.inverse_change_distance,
            self.inverse_change_beam_x, self.inverse_change_beam_y,
            self.inverse_constrain_distance, self.inverse_constrain_beam_x,
            self.inverse_constrain_beam_y,
        ):
            widget.setEnabled(enabled)

    def _beam_unit_changed(self) -> None:
        detector = self.detector_info(ignore_enabled=True)
        new_unit = self.beam_unit.currentText()
        if detector is not None and new_unit != self._previous_beam_unit:
            old_x = self.beam_x.value()
            old_y = self.beam_y.value()
            self._previous_beam_unit = new_unit
            self._update_beam_ranges()
            blockers = (
                QtCore.QSignalBlocker(self.beam_x),
                QtCore.QSignalBlocker(self.beam_y),
            )
            if new_unit == "mm":
                self.beam_x.setValue(
                    old_x * detector.pixel_x_m * 1e3
                )
                self.beam_y.setValue(
                    old_y * detector.pixel_y_m * 1e3
                )
            else:
                self.beam_x.setValue(
                    old_x / (detector.pixel_x_m * 1e3)
                )
                self.beam_y.setValue(
                    old_y / (detector.pixel_y_m * 1e3)
                )
            del blockers
        else:
            self._previous_beam_unit = new_unit
        self._update_beam_ranges()

    def _update_beam_ranges(self) -> None:
        detector = self.detector_info(ignore_enabled=True)
        if detector is None:
            return
        if self.beam_unit.currentText() == "pixels":
            maximum_x = float(detector.columns)
            maximum_y = float(detector.rows)
        else:
            maximum_x = detector.width_mm
            maximum_y = detector.height_mm
        self.beam_x.setRange(-1_000_000.0, maximum_x)
        self.beam_y.setRange(-1_000_000.0, maximum_y)

    def _update_details(self) -> None:
        detector = self.detector_info(ignore_enabled=True)
        if detector is None:
            return
        self.details.setText(
            f"pyFAI class: {detector.class_name}\n"
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
        # Qt can temporarily resize an embedded canvas to a very small area
        # while tabs and layouts are being arranged. Matplotlib's constrained
        # layout then tries to fit every decoration into that transient size
        # and emits "axes sizes collapsed to zero". Use explicit, stable
        # margins for each plot instead.
        self.figure = Figure(figsize=figsize, constrained_layout=False)
        super().__init__(self.figure)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QCoverage Version 4 — interactive and inverse detector planner")
        self.resize(1460, 1100)
        self._last_results: list[tuple[DetectorInfo, CoverageResult, DetectorPanel]] = []
        self._inverse_recommendation: Optional[dict] = None
        self._calibrant_dspacing_cache: dict[str, np.ndarray] = {}
        self._geometry_axes: dict[
            DetectorPanel, tuple[object, DetectorInfo, CoverageResult]
        ] = {}
        self._dragging_panel: Optional[DetectorPanel] = None
        self._drag_refresh_timer = QtCore.QElapsedTimer()
        self._drag_refresh_timer.start()

        detectors, warnings = load_preferred_detectors()
        calibrant_names = load_calibrant_names()
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        central_layout = QtWidgets.QVBoxLayout(central)
        central_layout.setContentsMargins(6, 6, 6, 6)
        central_layout.setSpacing(4)
        self.design_tabs = QtWidgets.QTabWidget()
        self.forward_page = QtWidgets.QWidget()
        self.inverse_page = QtWidgets.QWidget()
        self.design_tabs.addTab(self.forward_page, "Forward Calc")
        self.design_tabs.addTab(self.inverse_page, "Inverse Calc")
        root = QtWidgets.QVBoxLayout(self.forward_page)
        inverse_page_layout = QtWidgets.QVBoxLayout(self.inverse_page)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(4)
        inverse_page_layout.setContentsMargins(6, 6, 6, 6)
        inverse_page_layout.setSpacing(4)

        detector_row = QtWidgets.QHBoxLayout()
        detector_row.setSpacing(6)
        self.panel_1 = DetectorPanel(
            "Detector 1", detectors, calibrant_names, optional=False
        )
        self.panel_2 = DetectorPanel(
            "Detector 2", detectors, calibrant_names, optional=True
        )
        self.inverse_change_detector_1 = self.panel_1.inverse_change_detector
        self.inverse_change_distance_1 = self.panel_1.inverse_change_distance
        self.inverse_change_beam_x_1 = self.panel_1.inverse_change_beam_x
        self.inverse_change_beam_y_1 = self.panel_1.inverse_change_beam_y
        self.inverse_change_detector_2 = self.panel_2.inverse_change_detector
        self.inverse_change_distance_2 = self.panel_2.inverse_change_distance
        self.inverse_change_beam_x_2 = self.panel_2.inverse_change_beam_x
        self.inverse_change_beam_y_2 = self.panel_2.inverse_change_beam_y

        detector_1_index = self.panel_1.detector_combo.findText("DECTRIS Eiger2 9M")
        if detector_1_index >= 0:
            self.panel_1.detector_combo.setCurrentIndex(detector_1_index)
        self.panel_1.distance.setValue(2000.0)
        self.panel_1.beam_x.setValue(1554.0)
        self.panel_1.beam_y.setValue(1631.0)
        agbh_index = self.panel_1.calibrant_combo.findText("AgBh")
        if agbh_index >= 0:
            self.panel_1.calibrant_combo.setCurrentIndex(agbh_index)

        detector_2_index = self.panel_2.detector_combo.findText("DECTRIS Eiger2 4M")
        if detector_2_index >= 0:
            self.panel_2.detector_combo.setCurrentIndex(detector_2_index)
        self.panel_2.distance.setValue(300.0)
        self.panel_2.beam_x.setValue(1034.0)
        self.panel_2.beam_y.setValue(-400.0)
        self.inverse_change_beam_x_2.setChecked(False)
        self.inverse_change_beam_y_2.setChecked(False)
        lab6_index = self.panel_2.calibrant_combo.findText("LaB6")
        if lab6_index >= 0:
            self.panel_2.calibrant_combo.setCurrentIndex(lab6_index)

        detector_row.addWidget(self.panel_1)
        detector_row.addWidget(self.panel_2)
        central_layout.addLayout(detector_row)
        central_layout.addWidget(self.design_tabs, stretch=0)

        geometry_box = QtWidgets.QGroupBox("Experimental geometry and beamstop")
        grid = QtWidgets.QGridLayout(geometry_box)
        grid.setContentsMargins(6, 6, 6, 6)
        grid.setVerticalSpacing(4)
        self.energy = self._spin(0.1, 1000, 12.0, 4, " keV")
        self.inverse_change_energy = QtWidgets.QCheckBox()
        self.inverse_change_energy.setToolTip("Allow Inverse Calc to change Energy")
        self.inverse_change_energy.setChecked(True)
        self.inverse_constrain_energy = QtWidgets.QCheckBox()
        self.inverse_constrain_energy.setToolTip(
            "Constrain Energy during Inverse Calc"
        )
        self.beamstop = QtWidgets.QComboBox()
        self.inverse_change_beamstop = QtWidgets.QCheckBox()
        self.inverse_change_beamstop.setToolTip(
            "Allow Inverse Calc to change Beamstop diameter"
        )
        self.inverse_change_beamstop.setChecked(False)
        for diameter in BEAMSTOP_DIAMETERS_MM:
            self.beamstop.addItem(f"{diameter:g} mm", diameter)
        self.custom_stop = self._spin(0, 1000, 6.0, 3, " mm")
        self.custom_stop.setEnabled(False)
        self.beamstop.addItem("Custom", None)
        self.stop_to_detector1 = self._spin(0.0, 99999.0, 0.0, 3, " mm")
        self.inverse_change_stop_to_detector1 = QtWidgets.QCheckBox()
        self.inverse_change_stop_to_detector1.setToolTip(
            "Allow Inverse Calc to change Beamstop-to-Detector-1 distance"
        )
        self.inverse_change_stop_to_detector1.setChecked(False)
        self.inverse_constrain_stop_to_detector1 = QtWidgets.QCheckBox()
        self.inverse_constrain_stop_to_detector1.setToolTip(
            "Constrain Beamstop-to-Detector-1 distance during Inverse Calc"
        )
        beamstop_field = QtWidgets.QWidget()
        beamstop_field_layout = QtWidgets.QHBoxLayout(beamstop_field)
        beamstop_field_layout.setContentsMargins(0, 0, 0, 0)
        beamstop_field_layout.addWidget(self.beamstop)
        beamstop_field_layout.addWidget(self.custom_stop)
        fit_header = QtWidgets.QLabel("Fit")
        fit_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        fit_header.setFixedWidth(32)
        constraint_header = QtWidgets.QLabel("Constraint")
        constraint_header.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        constraint_header.setFixedWidth(72)
        constraint_header.setStyleSheet("font-size: 8pt;")

        grid.addWidget(fit_header, 0, 2)
        grid.addWidget(constraint_header, 0, 3)
        grid.addWidget(QtWidgets.QLabel("X-ray energy:"), 1, 0)
        grid.addWidget(self.energy, 1, 1)
        grid.addWidget(
            self.inverse_change_energy, 1, 2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        grid.addWidget(
            self.inverse_constrain_energy, 1, 3,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        grid.addWidget(QtWidgets.QLabel("Beamstop diameter:"), 2, 0)
        grid.addWidget(beamstop_field, 2, 1)
        grid.addWidget(
            self.inverse_change_beamstop, 2, 2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        grid.addWidget(QtWidgets.QLabel("Beamstop to Detector 1:"), 3, 0)
        grid.addWidget(self.stop_to_detector1, 3, 1)
        grid.addWidget(
            self.inverse_change_stop_to_detector1, 3, 2,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        grid.addWidget(
            self.inverse_constrain_stop_to_detector1, 3, 3,
            alignment=QtCore.Qt.AlignmentFlag.AlignCenter,
        )
        note = QtWidgets.QLabel(
            "0 mm = beamstop approximately in the Detector 1 plane; increasing the value "
            "moves it toward the sample. It must remain smaller than Detector 1 distance."
        )
        note.setWordWrap(True)
        grid.addWidget(note, 1, 4, 3, 1)
        grid.setColumnStretch(4, 1)
        root.addWidget(geometry_box)

        inverse_box = QtWidgets.QGroupBox("Q Range")
        inverse_layout = QtWidgets.QGridLayout(inverse_box)
        self.target_q_min = self._spin(0.000001, 100.0, 0.003, 6, " Å⁻¹")
        self.target_q_max = self._spin(0.000001, 100.0, 7.0, 6, " Å⁻¹")
        self.inverse_search_button = QtWidgets.QPushButton("Find setup")
        self.inverse_apply_button = QtWidgets.QPushButton("Apply recommendation")
        self.inverse_defaults_button = QtWidgets.QPushButton("Restore defaults")
        self.inverse_apply_button.setEnabled(False)
        self.inverse_result = QtWidgets.QLabel(
            "Enter a target q range and click Find setup."
        )
        self.inverse_result.setWordWrap(True)
        self.inverse_result.setTextInteractionFlags(
            QtCore.Qt.TextInteractionFlag.TextSelectableByMouse
        )
        inverse_layout.addWidget(QtWidgets.QLabel("Q min:"), 0, 0)
        inverse_layout.addWidget(self.target_q_min, 0, 1)
        inverse_layout.addWidget(QtWidgets.QLabel("Q max:"), 0, 2)
        inverse_layout.addWidget(self.target_q_max, 0, 3)
        inverse_layout.addWidget(self.inverse_search_button, 0, 4)
        inverse_layout.addWidget(self.inverse_apply_button, 0, 5)
        inverse_layout.addWidget(self.inverse_defaults_button, 0, 6)
        inverse_layout.addWidget(self.inverse_result, 1, 0, 1, 8)
        inverse_layout.setColumnStretch(7, 1)

        # Constraint editors are opened from the separate Constraint checkbox
        # beside each numeric parameter, so the inverse page stays compact.
        self.inverse_constraints = {}
        constraint_specs = (
            ("energy", "Energy (keV)", 0.1, 1000.0, 5.0, 40.0, 4),
            ("distance_1", "Detector 1 distance (mm)", 0.01, 100000.0, 25.0, 10000.0, 3),
            ("beam_x_1", "Detector 1 Beam X", -1000000.0, 1000000.0, 0.0, self.panel_1.beam_x.maximum(), 3),
            ("beam_y_1", "Detector 1 Beam Y", -1000000.0, 1000000.0, 0.0, self.panel_1.beam_y.maximum(), 3),
            ("distance_2", "Detector 2 distance (mm)", 0.01, 100000.0, 25.0, 10000.0, 3),
            ("beam_x_2", "Detector 2 Beam X", -1000000.0, 1000000.0, 0.0, self.panel_2.beam_x.maximum(), 3),
            ("beam_y_2", "Detector 2 Beam Y", -1000000.0, 1000000.0, 0.0, self.panel_2.beam_y.maximum(), 3),
            ("stop_to_detector1", "Beamstop to Detector 1 (mm)", 0.0, 99999.0, 0.0, 0.0, 3),
        )
        constraint_checkboxes = {
            "energy": self.inverse_constrain_energy,
            "distance_1": self.panel_1.inverse_constrain_distance,
            "beam_x_1": self.panel_1.inverse_constrain_beam_x,
            "beam_y_1": self.panel_1.inverse_constrain_beam_y,
            "distance_2": self.panel_2.inverse_constrain_distance,
            "beam_x_2": self.panel_2.inverse_constrain_beam_x,
            "beam_y_2": self.panel_2.inverse_constrain_beam_y,
            "stop_to_detector1": self.inverse_constrain_stop_to_detector1,
        }
        self._inverse_constraint_labels = {}
        for key, label, low, high, default_min, default_max, decimals in constraint_specs:
            enabled = constraint_checkboxes[key]
            minimum = self._spin(low, high, default_min, decimals)
            maximum = self._spin(low, high, default_max, decimals)
            self.inverse_constraints[key] = (enabled, minimum, maximum)
            self._inverse_constraint_labels[key] = label
        for key, checkbox in constraint_checkboxes.items():
            checkbox.clicked.connect(
                lambda checked, constraint_key=key: (
                    self._edit_inverse_constraint(constraint_key)
                    if checked else None
                )
            )
        self._inverse_change_options = (
            self.inverse_change_energy,
            self.inverse_change_beamstop,
            self.inverse_change_distance_1,
            self.inverse_change_beam_x_1,
            self.inverse_change_beam_y_1,
            self.inverse_change_distance_2,
            self.inverse_change_beam_x_2,
            self.inverse_change_beam_y_2,
            self.inverse_change_stop_to_detector1,
        )
        inverse_note = QtWidgets.QLabel(
            "Use the Fit checkbox beside a parameter to include it in Inverse Calc. "
            "Use Constraint to set an optional minimum and maximum."
        )
        inverse_note.setWordWrap(True)
        inverse_page_layout.addWidget(inverse_box)
        inverse_page_layout.addWidget(inverse_note)

        buttons = QtWidgets.QHBoxLayout()
        self.calculate_button = QtWidgets.QPushButton("Calculate / Update")
        self.export_csv_button = QtWidgets.QPushButton("Export CSV")
        self.export_png_button = QtWidgets.QPushButton("Export all plots PNG")
        buttons.addWidget(self.calculate_button)
        buttons.addWidget(self.export_csv_button)
        buttons.addWidget(self.export_png_button)
        buttons.addStretch(1)
        central_layout.addLayout(buttons)

        # Left column: plots 1 and 2 with equal width and a 1:2 height ratio.
        # Right column: plot 3 spans both rows and is half the left-column width.
        self.plot_container = QtWidgets.QWidget()
        plot_layout = QtWidgets.QGridLayout(self.plot_container)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        plot_layout.setHorizontalSpacing(6)
        plot_layout.setVerticalSpacing(6)

        self.range_canvas = PlotCanvas((12, 3.3))
        self.geometry_canvas = PlotCanvas((12, 6.6))
        self.qspace_canvas = PlotCanvas((6, 9.9))
        self.geometry_canvas.setToolTip(
            "Drag a coloured × direct-beam marker to move that detector's beam centre."
        )
        self.geometry_canvas.mpl_connect(
            "button_press_event", self._geometry_drag_press
        )
        self.geometry_canvas.mpl_connect(
            "motion_notify_event", self._geometry_drag_motion
        )
        self.geometry_canvas.mpl_connect(
            "button_release_event", self._geometry_drag_release
        )
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
        plot_layout.setColumnStretch(0, 3)
        plot_layout.setColumnStretch(1, 2)
        central_layout.addWidget(self.plot_container, stretch=1)

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
        two_row_height = (
            self.table.horizontalHeader().sizeHint().height()
            + 2 * self.table.verticalHeader().defaultSectionSize()
            + 2 * self.table.frameWidth()
            + 4
        )
        self.table.setFixedHeight(two_row_height)
        central_layout.addWidget(self.table)

        self.calculate_button.clicked.connect(self.calculate)
        self.export_csv_button.clicked.connect(self.export_csv)
        self.export_png_button.clicked.connect(self.export_png)
        self.inverse_search_button.clicked.connect(self.find_inverse_design)
        self.inverse_apply_button.clicked.connect(self.apply_inverse_design)
        self.inverse_defaults_button.clicked.connect(self.restore_inverse_defaults)
        self.panel_1.changed.connect(self._detector1_geometry_changed)
        self.panel_2.changed.connect(self.calculate)
        self.panel_1.detector_combo.currentIndexChanged.connect(
            lambda: self._sync_beam_constraint_defaults(self.panel_1, "1")
        )
        self.panel_1.beam_unit.currentIndexChanged.connect(
            lambda: self._sync_beam_constraint_defaults(self.panel_1, "1")
        )
        self.panel_2.detector_combo.currentIndexChanged.connect(
            lambda: self._sync_beam_constraint_defaults(self.panel_2, "2")
        )
        self.panel_2.beam_unit.currentIndexChanged.connect(
            lambda: self._sync_beam_constraint_defaults(self.panel_2, "2")
        )
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

    def _sync_beam_constraint_defaults(
        self, panel: DetectorPanel, panel_number: str
    ) -> None:
        """Keep inactive beam constraints matched to detector coordinates."""
        for axis, maximum in (
            ("x", panel.beam_x.maximum()),
            ("y", panel.beam_y.maximum()),
        ):
            enabled, minimum, maximum_widget = self.inverse_constraints[
                f"beam_{axis}_{panel_number}"
            ]
            if not enabled.isChecked():
                minimum.setValue(0.0)
                maximum_widget.setValue(maximum)

    def _edit_inverse_constraint(self, key: str) -> None:
        """Open the range editor associated with a Constraint checkbox."""
        enabled, stored_minimum, stored_maximum = self.inverse_constraints[key]
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(f"Inverse Calc — {self._inverse_constraint_labels[key]}")
        layout = QtWidgets.QFormLayout(dialog)

        minimum = self._spin(
            stored_minimum.minimum(), stored_minimum.maximum(),
            stored_minimum.value(), stored_minimum.decimals(),
            stored_minimum.suffix(),
        )
        maximum = self._spin(
            stored_maximum.minimum(), stored_maximum.maximum(),
            stored_maximum.value(), stored_maximum.decimals(),
            stored_maximum.suffix(),
        )

        error_label = QtWidgets.QLabel()
        error_label.setStyleSheet("color: #b00020;")
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
        )

        def accept_if_valid() -> None:
            if maximum.value() < minimum.value():
                error_label.setText(
                    "Maximum must be greater than or equal to minimum."
                )
                return
            dialog.accept()

        buttons.accepted.connect(accept_if_valid)
        buttons.rejected.connect(dialog.reject)
        layout.addRow("Minimum:", minimum)
        layout.addRow("Maximum:", maximum)
        layout.addRow(error_label)
        layout.addRow(buttons)

        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            stored_minimum.setValue(minimum.value())
            stored_maximum.setValue(maximum.value())
        else:
            enabled.setChecked(False)

    def _constraint_range(
        self, key: str, fallback_min: float, fallback_max: float
    ) -> tuple[float, float]:
        enabled, minimum, maximum = self.inverse_constraints[key]
        if not enabled.isChecked():
            return fallback_min, fallback_max
        low, high = minimum.value(), maximum.value()
        if high < low:
            raise ValueError(
                f"{key.replace('_', ' ').title()}: maximum constraint must be "
                "greater than or equal to minimum."
            )
        return low, high

    def restore_inverse_defaults(self) -> None:
        """Restore the shipped forward geometry and inverse-search controls."""
        widgets = (
            self.panel_1.detector_combo, self.panel_1.distance,
            self.panel_1.beam_x, self.panel_1.beam_y, self.panel_1.beam_unit,
            self.panel_1.calibrant_combo,
            self.panel_2.enabled, self.panel_2.detector_combo,
            self.panel_2.distance, self.panel_2.beam_x, self.panel_2.beam_y,
            self.panel_2.beam_unit, self.panel_2.calibrant_combo,
            self.energy, self.beamstop, self.stop_to_detector1,
            self.target_q_min, self.target_q_max,
        )
        blockers = tuple(QtCore.QSignalBlocker(widget) for widget in widgets)
        detector_1_index = self.panel_1.detector_combo.findText("DECTRIS Eiger2 9M")
        detector_2_index = self.panel_2.detector_combo.findText("DECTRIS Eiger2 4M")
        if detector_1_index >= 0:
            self.panel_1.detector_combo.setCurrentIndex(detector_1_index)
        if detector_2_index >= 0:
            self.panel_2.detector_combo.setCurrentIndex(detector_2_index)
        self.panel_1.distance.setValue(2000.0)
        self.panel_1.beam_x.setValue(1554.0)
        self.panel_1.beam_y.setValue(1631.0)
        self.panel_1.beam_unit.setCurrentText("pixels")
        self.panel_1.calibrant_combo.setCurrentText("AgBh")
        self.panel_2.enabled.setChecked(True)
        self.panel_2._set_enabled(True)
        self.panel_2.distance.setValue(300.0)
        self.panel_2.beam_x.setValue(1034.0)
        self.panel_2.beam_y.setValue(-400.0)
        self.panel_2.inverse_change_beam_x.setChecked(False)
        self.panel_2.inverse_change_beam_y.setChecked(False)
        self.panel_2.beam_unit.setCurrentText("pixels")
        self.panel_2.calibrant_combo.setCurrentText("LaB6")
        self.energy.setValue(12.0)
        default_stop_index = self.beamstop.findData(6.0)
        if default_stop_index >= 0:
            self.beamstop.setCurrentIndex(default_stop_index)
        self.stop_to_detector1.setValue(0.0)
        self.target_q_min.setValue(0.003)
        self.target_q_max.setValue(7.0)
        for checkbox in self._inverse_change_options:
            checkbox.setChecked(True)
        self.inverse_change_beamstop.setChecked(False)
        self.inverse_change_stop_to_detector1.setChecked(False)
        self.inverse_change_beam_x_2.setChecked(False)
        self.inverse_change_beam_y_2.setChecked(False)
        constraint_defaults = {
            "energy": (5.0, 40.0),
            "distance_1": (25.0, 10000.0),
            "beam_x_1": (0.0, self.panel_1.beam_x.maximum()),
            "beam_y_1": (0.0, self.panel_1.beam_y.maximum()),
            "distance_2": (25.0, 10000.0),
            "beam_x_2": (0.0, self.panel_2.beam_x.maximum()),
            "beam_y_2": (0.0, self.panel_2.beam_y.maximum()),
            "stop_to_detector1": (0.0, 0.0),
        }
        for key, (enabled, minimum, maximum) in self.inverse_constraints.items():
            enabled.setChecked(False)
            minimum.setValue(constraint_defaults[key][0])
            maximum.setValue(constraint_defaults[key][1])
        del blockers
        self._inverse_recommendation = None
        self.inverse_apply_button.setEnabled(False)
        self.inverse_result.setText("Defaults restored. Click Find setup.")
        self.panel_1._update_details()
        self.panel_2._update_details()
        self._detector1_geometry_changed()
        self.design_tabs.setCurrentWidget(self.inverse_page)
        self.statusBar().showMessage("Default settings restored.")

    @staticmethod
    def _inverse_interval(
        detector: DetectorInfo,
        energy_kev: float,
        distance_mm: float,
        beamstop_diameter_mm: float,
        beam_x_mm: float,
        beam_y_mm: float,
    ) -> tuple[float, float]:
        """Fast rectangular-detector interval estimate used by the search."""
        dx_min = max(0.0, -beam_x_mm, beam_x_mm - detector.width_mm)
        dy_min = max(0.0, -beam_y_mm, beam_y_mm - detector.height_mm)
        geometric_r_min = math.hypot(dx_min, dy_min)
        r_min = max(beamstop_diameter_mm / 2.0, geometric_r_min)
        r_max = max(
            math.hypot(x - beam_x_mm, y - beam_y_mm)
            for x in (0.0, detector.width_mm)
            for y in (0.0, detector.height_mm)
        )
        return (
            float(radius_to_q(r_min, distance_mm, energy_kev)),
            float(radius_to_q(r_max, distance_mm, energy_kev)),
        )

    @staticmethod
    def _inverse_beam_position(
        panel: DetectorPanel,
        detector: DetectorInfo,
        fit_x: bool,
        fit_y: bool,
        x_limits: Optional[tuple[float, float]] = None,
        y_limits: Optional[tuple[float, float]] = None,
        toward_origin: bool = False,
    ) -> tuple[float, float, float, float]:
        """Return proposed beam position in mm plus input-field values."""
        if panel.beam_unit.currentText() == "pixels":
            current_x_mm = panel.beam_x.value() * detector.pixel_x_m * 1e3
            current_y_mm = panel.beam_y.value() * detector.pixel_y_m * 1e3
            proposed_x = (
                (0.0 if toward_origin else detector.columns / 2.0)
                if fit_x else panel.beam_x.value()
            )
            proposed_y = (
                (0.0 if toward_origin else detector.rows / 2.0)
                if fit_y else panel.beam_y.value()
            )
        else:
            current_x_mm = panel.beam_x.value()
            current_y_mm = panel.beam_y.value()
            proposed_x = (
                (0.0 if toward_origin else detector.width_mm / 2.0)
                if fit_x else panel.beam_x.value()
            )
            proposed_y = (
                (0.0 if toward_origin else detector.height_mm / 2.0)
                if fit_y else panel.beam_y.value()
            )
        if fit_x and x_limits is not None:
            proposed_x = min(max(proposed_x, x_limits[0]), x_limits[1])
        if fit_y and y_limits is not None:
            proposed_y = min(max(proposed_y, y_limits[0]), y_limits[1])
        if panel.beam_unit.currentText() == "pixels":
            beam_x_mm = proposed_x * detector.pixel_x_m * 1e3 if fit_x else current_x_mm
            beam_y_mm = proposed_y * detector.pixel_y_m * 1e3 if fit_y else current_y_mm
        else:
            beam_x_mm = proposed_x if fit_x else current_x_mm
            beam_y_mm = proposed_y if fit_y else current_y_mm
        return beam_x_mm, beam_y_mm, proposed_x, proposed_y

    def find_inverse_design(self) -> None:
        target_min = self.target_q_min.value()
        target_max = self.target_q_max.value()
        use_detector_2 = self.panel_2.enabled.isChecked()
        if target_min <= 0 or target_max <= target_min:
            self.inverse_result.setText(
                "Invalid target: Q max must be greater than Q min, and both must be positive."
            )
            self.inverse_apply_button.setEnabled(False)
            return

        self.inverse_search_button.setEnabled(False)
        self.inverse_result.setText("Searching engineering parameter space…")
        QtWidgets.QApplication.processEvents()

        # Practical default search envelope. A logarithmic distance grid gives
        # comparable resolution for short WAXS and long SAXS geometries.
        # Both proposed settings use the same energy and beamstop diameter.
        try:
            energy_limits = self._constraint_range("energy", 5.0, 40.0)
            distance_1_limits = self._constraint_range("distance_1", 25.0, 10000.0)
            distance_2_limits = self._constraint_range("distance_2", 25.0, 10000.0)
            beam_x_1_limits = self._constraint_range(
                "beam_x_1", 0.0, self.panel_1.beam_x.maximum()
            )
            beam_y_1_limits = self._constraint_range(
                "beam_y_1", 0.0, self.panel_1.beam_y.maximum()
            )
            beam_x_2_limits = self._constraint_range(
                "beam_x_2", 0.0, self.panel_2.beam_x.maximum()
            )
            beam_y_2_limits = self._constraint_range(
                "beam_y_2", 0.0, self.panel_2.beam_y.maximum()
            )
            stop_position_limits = self._constraint_range(
                "stop_to_detector1",
                self.stop_to_detector1.value(),
                self.stop_to_detector1.value(),
            )
        except ValueError as exc:
            self.inverse_search_button.setEnabled(True)
            self.inverse_apply_button.setEnabled(False)
            self.inverse_result.setText(str(exc))
            return

        all_detectors = [
            self.panel_1.detector_combo.itemData(i)
            for i in range(self.panel_1.detector_combo.count())
        ]
        energies = (
            np.linspace(energy_limits[0], energy_limits[1], 71)
            if self.inverse_change_energy.isChecked()
            else np.array([self.energy.value()])
        )
        stop_diameters = (
            BEAMSTOP_DIAMETERS_MM
            if self.inverse_change_beamstop.isChecked()
            else (self.beamstop_diameter(),)
        )
        detectors_1 = (
            all_detectors
            if self.inverse_change_detector_1.isChecked()
            else [self.panel_1.detector_info(ignore_enabled=True)]
        )
        detectors_2 = (
            all_detectors
            if self.inverse_change_detector_2.isChecked()
            else [self.panel_2.detector_info(ignore_enabled=True)]
        )
        distances_1 = (
            np.geomspace(distance_1_limits[0], distance_1_limits[1], 120)
            if self.inverse_change_distance_1.isChecked()
            else np.array([self.panel_1.distance.value()])
        )
        distances_2 = (
            np.geomspace(distance_2_limits[0], distance_2_limits[1], 120)
            if self.inverse_change_distance_2.isChecked()
            else np.array([self.panel_2.distance.value()])
        )
        best: Optional[dict] = None

        log_target_min = math.log(target_min)
        log_target_max = math.log(target_max)
        for stop_diameter in stop_diameters:
            for energy in energies:
                candidates_1: list[dict] = []
                candidates_2: list[dict] = []
                for detector in detectors_1:
                    beam_positions = {
                        self._inverse_beam_position(
                            self.panel_1,
                            detector,
                            self.inverse_change_beam_x_1.isChecked(),
                            self.inverse_change_beam_y_1.isChecked(),
                            beam_x_1_limits,
                            beam_y_1_limits,
                            toward_origin=toward_origin,
                        )
                        for toward_origin in (False, True)
                    }
                    for beam_x_mm, beam_y_mm, beam_x, beam_y in beam_positions:
                        for distance in distances_1:
                            q_min, q_max = self._inverse_interval(
                                detector, float(energy), float(distance), stop_diameter,
                                beam_x_mm, beam_y_mm,
                            )
                            candidates_1.append({
                                "detector": detector,
                                "distance": float(distance),
                                "beam_x": beam_x,
                                "beam_y": beam_y,
                                "q_min": q_min,
                                "q_max": q_max,
                            })
                if not use_detector_2:
                    q_mins = np.array([item["q_min"] for item in candidates_1])
                    q_maxs = np.array([item["q_max"] for item in candidates_1])
                    low_missing = np.maximum(
                        0.0, np.log(q_mins) - log_target_min
                    )
                    high_missing = np.maximum(
                        0.0, log_target_max - np.log(q_maxs)
                    )
                    endpoint_error = (
                        np.abs(np.log(q_mins) - log_target_min)
                        + np.abs(np.log(q_maxs) - log_target_max)
                    )
                    scores = 40.0 * (low_missing + high_missing) + endpoint_error
                    candidate_index = int(np.argmin(scores))
                    score = float(scores[candidate_index])
                    if best is None or score < best["score"]:
                        setup_1 = candidates_1[candidate_index]
                        best = {
                            "score": score,
                            "setup_1": setup_1,
                            "setup_2": None,
                            "energy": float(energy),
                            "beamstop": float(stop_diameter),
                            "stop_to_detector1": float(stop_position_limits[0]),
                            "q_min": setup_1["q_min"],
                            "q_max": setup_1["q_max"],
                            "gap": 0.0,
                            "target_min": target_min,
                            "target_max": target_max,
                        }
                    continue
                for detector in detectors_2:
                    beam_positions = {
                        self._inverse_beam_position(
                            self.panel_2,
                            detector,
                            self.inverse_change_beam_x_2.isChecked(),
                            self.inverse_change_beam_y_2.isChecked(),
                            beam_x_2_limits,
                            beam_y_2_limits,
                            toward_origin=toward_origin,
                        )
                        for toward_origin in (False, True)
                    }
                    for beam_x_mm, beam_y_mm, beam_x, beam_y in beam_positions:
                        for distance in distances_2:
                            q_min, q_max = self._inverse_interval(
                                detector, float(energy), float(distance), stop_diameter,
                                beam_x_mm, beam_y_mm,
                            )
                            candidates_2.append({
                                "detector": detector,
                                "distance": float(distance),
                                "beam_x": beam_x,
                                "beam_y": beam_y,
                                "q_min": q_min,
                                "q_max": q_max,
                            })

                q_mins_1 = np.array([item["q_min"] for item in candidates_1])
                q_maxs_1 = np.array([item["q_max"] for item in candidates_1])
                q_mins_2 = np.array([item["q_min"] for item in candidates_2])
                q_maxs_2 = np.array([item["q_max"] for item in candidates_2])

                combined_min = np.minimum(q_mins_1[:, None], q_mins_2[None, :])
                combined_max = np.maximum(q_maxs_1[:, None], q_maxs_2[None, :])
                first_is_low = q_mins_1[:, None] <= q_mins_2[None, :]
                first_end = np.where(
                    first_is_low, q_maxs_1[:, None], q_maxs_2[None, :]
                )
                second_start = np.where(
                    first_is_low, q_mins_2[None, :], q_mins_1[:, None]
                )
                log_gap = np.maximum(0.0, np.log(second_start / first_end))

                low_missing = np.maximum(
                    0.0, np.log(combined_min) - log_target_min
                )
                high_missing = np.maximum(
                    0.0, log_target_max - np.log(combined_max)
                )
                endpoint_error = (
                    np.abs(np.log(combined_min) - log_target_min)
                    + np.abs(np.log(combined_max) - log_target_max)
                )
                scores = (
                    40.0 * (low_missing + high_missing)
                    + 50.0 * log_gap
                    + endpoint_error
                )
                flat_index = int(np.argmin(scores))
                first_index, second_index = np.unravel_index(
                    flat_index, scores.shape
                )
                score = float(scores[first_index, second_index])
                if best is None or score < best["score"]:
                    setup_1 = candidates_1[first_index]
                    setup_2 = candidates_2[second_index]
                    low, high = sorted(
                        (setup_1, setup_2), key=lambda item: item["q_min"]
                    )
                    gap = max(
                        0.0,
                        high["q_min"] - low["q_max"],
                    )
                    best = {
                        "score": score,
                        "setup_1": setup_1,
                        "setup_2": setup_2,
                        "energy": float(energy),
                        "beamstop": float(stop_diameter),
                        "stop_to_detector1": float(stop_position_limits[0]),
                        "q_min": min(setup_1["q_min"], setup_2["q_min"]),
                        "q_max": max(setup_1["q_max"], setup_2["q_max"]),
                        "gap": gap,
                        "target_min": target_min,
                        "target_max": target_max,
                    }

        self.inverse_search_button.setEnabled(True)
        if best is None:
            self.inverse_result.setText("No recommendation could be calculated.")
            self.inverse_apply_button.setEnabled(False)
            return

        self._inverse_recommendation = best
        complete = (
            best["q_min"] <= target_min
            and best["q_max"] >= target_max
            and best["gap"] <= 0
        )
        if complete:
            verdict = "Full target covered"
        elif use_detector_2:
            verdict = "Closest achievable two-detector design"
        else:
            verdict = "Closest achievable single-detector design"
        setup_1 = best["setup_1"]
        setup_2 = best["setup_2"]
        gap_text = (
            "no predicted q gap"
            if best["gap"] <= 0
            else f"predicted gap width {best['gap']:.5g} Å⁻¹"
        )
        detector_2_text = (
            f"Detector 2: {setup_2['detector'].label} at {setup_2['distance']:.1f} mm "
            f"with beam ({setup_2['beam_x']:.1f}, {setup_2['beam_y']:.1f}) "
            f"({setup_2['q_min']:.5g}–{setup_2['q_max']:.5g} Å⁻¹); "
            if setup_2 is not None else ""
        )
        self.inverse_result.setText(
            f"{verdict} — Energy: {best['energy']:.2f} keV; "
            f"Beamstop: {best['beamstop']:.1f} mm; "
            f"Detector 1: {setup_1['detector'].label} at {setup_1['distance']:.1f} mm "
            f"with beam ({setup_1['beam_x']:.1f}, {setup_1['beam_y']:.1f}) "
            f"({setup_1['q_min']:.5g}–{setup_1['q_max']:.5g} Å⁻¹); "
            f"{detector_2_text}"
            f"{gap_text}. "
            "Beam coordinates use each detector panel's selected unit."
        )
        self.inverse_apply_button.setEnabled(True)

    def apply_inverse_design(self) -> None:
        recommendation = self._inverse_recommendation
        if recommendation is None:
            return
        setup_1 = recommendation["setup_1"]
        setup_2 = recommendation["setup_2"]
        detector_1 = setup_1["detector"]
        detector_1_index = self.panel_1.detector_combo.findText(detector_1.label)
        stop_index = self.beamstop.findData(recommendation["beamstop"])

        widgets = (
            self.panel_1.detector_combo,
            self.panel_1.distance,
            self.panel_1.beam_x,
            self.panel_1.beam_y,
            self.panel_1.beam_unit,
            self.panel_2.enabled,
            self.panel_2.detector_combo,
            self.panel_2.distance,
            self.panel_2.beam_x,
            self.panel_2.beam_y,
            self.panel_2.beam_unit,
            self.energy,
            self.beamstop,
            self.stop_to_detector1,
        )
        blockers = tuple(QtCore.QSignalBlocker(widget) for widget in widgets)
        if self.inverse_change_detector_1.isChecked() and detector_1_index >= 0:
            self.panel_1.detector_combo.setCurrentIndex(detector_1_index)
        if self.inverse_change_distance_1.isChecked():
            self.panel_1.distance.setValue(setup_1["distance"])
        if self.inverse_change_beam_x_1.isChecked():
            self.panel_1.beam_x.setValue(setup_1["beam_x"])
        if self.inverse_change_beam_y_1.isChecked():
            self.panel_1.beam_y.setValue(setup_1["beam_y"])
        if setup_2 is not None:
            if self.inverse_change_distance_2.isChecked():
                self.panel_2.distance.setValue(setup_2["distance"])
            if self.inverse_change_beam_x_2.isChecked():
                self.panel_2.beam_x.setValue(setup_2["beam_x"])
            if self.inverse_change_beam_y_2.isChecked():
                self.panel_2.beam_y.setValue(setup_2["beam_y"])
        if self.inverse_change_energy.isChecked():
            self.energy.setValue(recommendation["energy"])
        if self.inverse_change_beamstop.isChecked() and stop_index >= 0:
            self.beamstop.setCurrentIndex(stop_index)
        if self.inverse_change_stop_to_detector1.isChecked():
            self.stop_to_detector1.setValue(
                min(
                    recommendation["stop_to_detector1"],
                    max(0.0, self.panel_1.distance.value() - 0.001),
                )
            )
        del blockers

        self.panel_1._update_details()
        self.panel_2._update_details()
        self._detector1_geometry_changed()
        self.design_tabs.setCurrentWidget(self.inverse_page)
        self.statusBar().showMessage("Inverse-design recommendation applied.")

    def _geometry_drag_press(self, event) -> None:
        """Begin dragging when the left mouse button is near a beam marker."""
        if event.button != 1 or event.x is None or event.y is None:
            return
        for panel, (ax, _detector, result) in self._geometry_axes.items():
            if event.inaxes is not ax:
                continue
            marker_x, marker_y = ax.transData.transform(
                (result.beam_x_mm, result.beam_y_mm)
            )
            if math.hypot(event.x - marker_x, event.y - marker_y) <= 18:
                self._dragging_panel = panel
                self.geometry_canvas.setCursor(
                    QtCore.Qt.CursorShape.ClosedHandCursor
                )
                self.statusBar().showMessage(
                    f"Dragging {panel.title()} beam centre…"
                )
                return

    def _geometry_drag_motion(self, event) -> None:
        if self._dragging_panel is None:
            return
        # Limit full three-plot redraws to about 25 frames per second.
        if self._drag_refresh_timer.elapsed() < 40:
            return
        self._drag_refresh_timer.restart()
        self._set_beam_center_from_mouse(event, self._dragging_panel)

    def _geometry_drag_release(self, event) -> None:
        panel = self._dragging_panel
        if panel is None:
            return
        self._set_beam_center_from_mouse(event, panel)
        self._dragging_panel = None
        self.geometry_canvas.unsetCursor()

    def _set_beam_center_from_mouse(self, event, panel: DetectorPanel) -> None:
        """Convert canvas coordinates to the panel's selected beam-centre unit."""
        mapping = self._geometry_axes.get(panel)
        if mapping is None or event.x is None or event.y is None:
            return
        ax, detector, _result = mapping
        beam_x_mm, beam_y_mm = ax.transData.inverted().transform(
            (event.x, event.y)
        )

        if panel.beam_unit.currentText() == "pixels":
            beam_x = beam_x_mm / (detector.pixel_x_m * 1e3)
            beam_y = beam_y_mm / (detector.pixel_y_m * 1e3)
        else:
            beam_x = beam_x_mm
            beam_y = beam_y_mm

        # Block the two individual valueChanged signals so one mouse event
        # causes exactly one complete q-coverage recalculation.
        blockers = (
            QtCore.QSignalBlocker(panel.beam_x),
            QtCore.QSignalBlocker(panel.beam_y),
        )
        panel.beam_x.setValue(beam_x)
        panel.beam_y.setValue(beam_y)
        del blockers
        self.calculate()

        result = next(
            (
                coverage
                for _detector, coverage, result_panel in self._last_results
                if result_panel is panel
            ),
            None,
        )
        if result is not None:
            self.statusBar().showMessage(
                f"{panel.title()}: beam centre "
                f"({panel.beam_x.value():.3f}, {panel.beam_y.value():.3f}) "
                f"{panel.beam_unit.currentText()}; "
                f"q = {result.q_min:.5g}–{result.q_max:.5g} Å⁻¹"
            )

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
        fig.subplots_adjust(left=0.16, right=0.97, bottom=0.25, top=0.92)
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

    def _calibrant_ring_radii(
        self,
        calibrant_name: str,
        energy_kev: float,
        distance_mm: float,
        r_min_mm: float,
        r_max_mm: float,
    ) -> list[float]:
        """Calculate visible powder-ring radii from pyFAI d-spacings."""
        if calibrant_name not in self._calibrant_dspacing_cache:
            from pyFAI.calibrant import get_calibrant

            calibrant = get_calibrant(calibrant_name)
            self._calibrant_dspacing_cache[calibrant_name] = np.asarray(
                calibrant.dspacing, dtype=float
            )

        wavelength_a = HC_KEV_ANGSTROM / energy_kev
        radii: list[float] = []
        for d_spacing_a in self._calibrant_dspacing_cache[calibrant_name]:
            q_value = 2.0 * math.pi / float(d_spacing_a)
            sin_theta = q_value * wavelength_a / (4.0 * math.pi)
            if not 0.0 < sin_theta < 1.0:
                continue
            two_theta = 2.0 * math.asin(sin_theta)
            radius_mm = distance_mm * math.tan(two_theta)
            if r_min_mm <= radius_mm <= r_max_mm:
                radii.append(radius_mm)
        # A hard upper bound keeps rendering responsive for dense calibrants.
        return sorted(radii)[:80]

    def _plot_geometry(self, results):
        fig = self.geometry_canvas.figure
        fig.clear()
        fig.subplots_adjust(left=0.09, right=0.78, bottom=0.16, top=0.94, wspace=0.30)
        self._geometry_axes.clear()
        axes = fig.subplots(1, len(results), squeeze=False)[0]
        for i, ((detector, result, panel), ax) in enumerate(zip(results, axes)):
            self._geometry_axes[panel] = (ax, detector, result)
            detector_outline = Rectangle(
                (0, 0), detector.width_mm, detector.height_mm,
                fill=False, linewidth=1.8, edgecolor=BAR_COLOURS[i],
            )
            ax.add_patch(detector_outline)
            ring_radii = self._calibrant_ring_radii(
                panel.calibrant_combo.currentText(),
                self.energy.value(),
                panel.distance.value(),
                result.r_min_mm,
                result.r_max_mm,
            )
            for ring_index, ring_radius in enumerate(ring_radii):
                ring = Circle(
                    (result.beam_x_mm, result.beam_y_mm),
                    ring_radius,
                    fill=False,
                    linewidth=0.85,
                    alpha=0.60,
                    color=CALIBRANT_COLOURS[i],
                    label=(
                        panel.calibrant_combo.currentText()
                        if ring_index == 0 else None
                    ),
                )
                ring.set_clip_path(
                    detector_outline.get_path(),
                    detector_outline.get_transform(),
                )
                ax.add_patch(ring)
            ax.add_patch(Circle((result.beam_x_mm, result.beam_y_mm),
                                result.beamstop_shadow_diameter_mm/2,
                                alpha=0.35, color="black"))
            ax.scatter([result.beam_x_mm], [result.beam_y_mm], marker="x", s=95,
                       linewidths=2.2, color=BAR_COLOURS[i],
                       label=f"{panel.title()} beam centre")
            ax.scatter(result.detector_x_mm[::80], result.detector_y_mm[::80], s=1,
                       alpha=0.16, color=BAR_COLOURS[i])
            ax.set_aspect("equal", adjustable="box")
            margin = max(detector.width_mm, detector.height_mm)*0.08
            # Keep an off-detector beam centre visible and draggable. The
            # limits continue expanding if the marker is dragged farther out.
            ax.set_xlim(
                min(-margin, result.beam_x_mm - margin),
                max(detector.width_mm + margin, result.beam_x_mm + margin),
            )
            ax.set_ylim(
                min(-margin, result.beam_y_mm - margin),
                max(detector.height_mm + margin, result.beam_y_mm + margin),
            )
            ax.set_xlabel("Detector X (mm)")
            ax.set_ylabel("Detector Y (mm)")
            ax.grid(alpha=0.2)
        legend_entries = {}
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            for handle, label in zip(handles, labels):
                legend_entries.setdefault(label, handle)
        legend_order = [
            f"{panel.title()} beam centre"
            for _detector, _result, panel in results
        ] + [
            panel.calibrant_combo.currentText()
            for _detector, _result, panel in results
        ]
        ordered_labels = [
            label for label in legend_order if label in legend_entries
        ]
        fig.legend(
            [legend_entries[label] for label in ordered_labels], ordered_labels,
            loc="center right", bbox_to_anchor=(0.99, 0.5), fontsize=8,
        )
        self.geometry_canvas.draw()

    def _plot_qspace(self, results):
        fig = self.qspace_canvas.figure
        fig.clear()
        fig.subplots_adjust(left=0.19, right=0.76, bottom=0.12, top=0.96)
        ax = fig.add_subplot(111)
        for i, (detector, result, _) in enumerate(results):
            # A fixed stride aliases with the flattened detector grid and can
            # create false vertical gaps, especially when the beam is at
            # (0, 0). Evenly spaced indices vary the step and preserve the
            # detector's actual q-space outline.
            sample_count = min(result.qx.size, 40000)
            sample_indices = np.linspace(
                0, result.qx.size - 1, sample_count, dtype=int
            )
            # Always retain the true radial and Cartesian extrema. Otherwise
            # display downsampling can omit the far detector corner: the
            # radial-range plot then reports the correct qmax while the
            # q-space drawing appears to stop short of it.
            extrema_indices = np.array(
                [
                    np.argmin(result.q_abs),
                    np.argmax(result.q_abs),
                    np.argmin(result.qx),
                    np.argmax(result.qx),
                    np.argmin(result.qy),
                    np.argmax(result.qy),
                ],
                dtype=int,
            )
            sample_indices = np.unique(
                np.concatenate((sample_indices, extrema_indices))
            )
            ax.scatter(result.qx[sample_indices], result.qy[sample_indices],
                       s=2, alpha=0.28,
                       color=BAR_COLOURS[i], label=detector.label, rasterized=True)
        ax.axhline(0, linewidth=0.7, color="black", alpha=0.5)
        ax.axvline(0, linewidth=0.7, color="black", alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("qx (Å⁻¹)")
        ax.set_ylabel("qy (Å⁻¹)")
        ax.grid(alpha=0.2)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles, labels,
            loc="center right",
            bbox_to_anchor=(0.99, 0.5),
            fontsize=8,
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
    app.setApplicationName("QCoverage Version 4")
    try:
        window = MainWindow()
    except RuntimeError as exc:
        QtWidgets.QMessageBox.critical(None, "Startup error", str(exc))
        return 1
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
