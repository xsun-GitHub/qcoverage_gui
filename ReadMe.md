# QCoverage GUI

PyQt6 GUI for calculating and displaying SAXS/WAXS detector q coverage.

The default parameters are configured for the SAXS/WAXS detectors at the P62 beamline at DESY. 

The program can also compare the q ranges of two detectors or two different detector settings.

Supported detector presets include Eiger2 4M, Eiger2 9M, and PerkinElmer.

## Version 4: interactive beam center

In the detector-geometry plot:

1. Move the mouse onto a coloured `×` direct-beam marker.
2. Hold the left mouse button.
3. Drag the marker to the desired position.

While dragging, the Beam centre X/Y fields, radial q range, detector geometry,
qx-qy coverage, and results table update in real time. Both `pixels` and `mm`
coordinate units are supported, and the beam centre may be dragged outside the
detector boundary.

## Requirements

- Python 3.10 or newer
- PyQt6
- NumPy
- Matplotlib
- pyFAI

## Installation

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python qcoverage_v3_2.py
```
