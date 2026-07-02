# py_wv.py

*[日本語版 README](doc/README_jp.md) / [English README](doc/README.md)*

A **single-file** waveform viewer for ngspice **binary `.raw`** files: load,
overlay, split into stacked panes, and take two-point measurements. Built with
Python + pyqtgraph (PyQt6).

`py_wv.py` is a self-contained build of the `waveview` package — the rawfile
reader, measurement primitives, and the Qt GUI combined into one script. It
needs no `waveview/` package alongside it; only `numpy` and `pyqtgraph` are
required.

## Repository layout

```
py_wv/
  ├─ source/
  │    └─ py_wv.py        # the viewer (single self-contained script)
  ├─ doc/
  │    ├─ README.md       # full English documentation
  │    └─ README_jp.md    # 日本語ドキュメント
  └─ README.md            # this file
```

The script itself is organised into three sections, mirroring the original
package modules:

```
py_wv.py
  ├─ rawfile   # binary .raw parser + data model (real/complex)
  ├─ measure   # edge/level crossing detection + interpolation
  └─ app       # GUI (pyqtgraph) + main() entry point
```

## Requirements

- Python 3
- `numpy`
- `pyqtgraph`
- a Qt binding (`PyQt6`)

```bash
pip install numpy pyqtgraph PyQt6
```

## Launch

```bash
python3 source/py_wv.py                 # start empty, then load from the menu
python3 source/py_wv.py outputfile.raw  # start with a file
# or, if executable:
chmod +x source/py_wv.py
./source/py_wv.py outputfile.raw
```

> The UI starts in **English** by default. Switch to Japanese from
> **Option → Language** at any time.

## Features

- **Load ngspice binary `.raw` files** (real & complex sweeps). The same
  filename can be loaded any number of times; each load gets a unique `#N` id.
- **Overlay / split** traces into stacked panes with a shared, linked X axis.
- **SI-prefixed X-axis ticks** in 10³ steps (…, ps, ns, µs, ms, …) with the
  unit shown on every pane (e.g. `60ns`).
- **Flexible zoom / pan**: Fit All, zoom in/out, rectangle zoom, zoom-to-cursor,
  per-axis mouse actions, an on-screen X-axis navigation bar, and arrow-key
  scrolling.
- **Two-point measurement** (cursors A / B) with `Cursor` / `Level` / `Rise` /
  `Fall` reference modes; reports **Δt**, **1/Δt**, and **ΔV / ΔI**. Crossings
  are linearly interpolated, so resolution is finer than the simulation
  timestep.
- **Bilingual UI** (English / 日本語) and adjustable font size.
- **Per-trace colour and line style** (`Solid / Dash / Dot / DashDot /
  DashDotDot`).

For the full operation guide (menus, shortcuts, measurement details), see the
documentation:

- **English:** [doc/README.md](doc/README.md)
- **日本語:** [doc/README_jp.md](doc/README_jp.md)
