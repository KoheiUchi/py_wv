# py_wv.py

*[日本語版 README](README_jp.md)*

A **single-file** waveform viewer for ngspice **binary `.raw`** files: load,
overlay, split into stacked panes, and take two-point measurements. Built with
Python + pyqtgraph (PyQt6).

`py_wv.py` is a self-contained build of the `waveview` package — the rawfile
reader, measurement primitives, and the Qt GUI combined into one script. It
needs no `waveview/` package alongside it; only `numpy` and `pyqtgraph` are
required.

## Launch

```bash
python3 source/py_wv.py                 # start empty, then load from the menu
python3 source/py_wv.py outputfile.raw  # start with a file
# or, if executable:
chmod +x source/py_wv.py
./source/py_wv.py outputfile.raw
```

Dependencies: `numpy`, `pyqtgraph`, and a Qt binding (`PyQt6` is used).
Install with `pip install numpy pyqtgraph PyQt6`.

> The UI starts in **English** by default. Switch to Japanese from
> **Option → Language** at any time.

## Features

### Loading files
- Use the toolbar **Open .raw…** to select one or more files at once.
- **The same filename can be loaded any number of times.** Each load gets a
  unique `#N` id so it is distinguishable in the **Sources** tree and in
  measurement targets (e.g. `outputfile.raw #1`, `#2`).

### File menu
- **Open .raw… (Ctrl+O)** — load `.raw` file(s).
- **Close .raw (Ctrl+W)** — close (unload) the `.raw` selected in **Sources**.
  Selecting either the file node or one of its signals closes that file; its
  traces and measurement targets are removed automatically. Multi-select is
  supported.
- **Exit (Ctrl+Q)** — quit the program.

### Displaying waveforms
- Check a signal's checkbox in the left **Sources** tree to add it as a trace.
- `time` (the sweep axis) automatically becomes the X axis.
- **X-axis ticks use SI prefixes in 10³ steps (…, ps, ns, µs, ms, …) plus the
  unit** (e.g. `60ns`). Every stacked pane uses the same format, and the prefix
  switches automatically with the visible range.

### Overlay / split (stacked panes)
- Every trace has a **Pane** number (in the left **Traces** table).
  - **Same Pane number = overlaid** in one plot.
  - **Different Pane numbers = stacked top/bottom** (X axes stay linked).
- Select traces in the table, then:
  - Toolbar **Split selected ▼** — move the selection to a new pane.
  - Toolbar **Merge selected ▲** — overlay the selection into one pane.
  - Or change the **Pane** spinbox per trace.
- Table columns are resizable — drag the **Trace** column border to widen it.

### View / zoom (View menu)
- **Fit All (Ctrl+0)** — reset every pane to fit all data.
- **Zoom In / Zoom Out (Ctrl++ / Ctrl+-)** — zoom about the centre. X is shared
  across panes; Y is per-pane.
- **Rectangle Zoom (Ctrl+R)** — when ON, **left-drag a rectangle (any two
  points) to zoom into that region**. Turn OFF to return to panning.
- **Zoom X to Cursors A–B (Ctrl+B)** — fit the X range to the A–B cursor span.
- **Panels** — show/hide the Sources / Traces / Measure docks. A panel closed
  with its ✕ can be reopened here.
- **X-axis navigation bar** — a row of buttons below the plot operates the
  shared X axis with the mouse (mirrors the keyboard controls):
  - **◀◀ / ◀** — pan left (50% / 10% of the view width).
  - **－ / ＋** — zoom the X axis out / in about the view centre (Y is left
    unchanged).
  - **▶ / ▶▶** — pan right (10% / 50%).
- **Arrow-key horizontal scroll** — click the plot to focus it, then use
  **← / →** to scroll the waveform left/right (10% of the view width).
  **Shift+← / →** scrolls farther (50%). This only acts while the plot has
  focus, so it never interferes with spin boxes or tree navigation.
- **Per-axis mouse actions** (always on, no menu needed):
  - Drag / wheel **over the X axis** → zoom/pan **X only**.
  - Drag / wheel **over the Y axis** → zoom/pan **Y only**.
  - Wheel inside the plot → zoom both; right-drag → zoom; left-drag → pan
    (when Rectangle Zoom is OFF).

### Options (Option menu)
- **Language** — switch menu/label text between **English / 日本語** (applied
  immediately). The startup default is English.
- **Font** — change the UI font size:
  - **Enlarge (Ctrl+Shift+.)**
  - **Shrink (Ctrl+Shift+,)**
  - **Default (Ctrl+Shift+0)**
  - (`.` `,` `0` are used because `=` `+` `-` are Shift-composed on Japanese
    keyboards and do not map reliably as shortcut keys.)

### Line colour / pattern
- Pick a colour with the **Color** button in the **Traces** table.
- Choose a **Style**: `Solid / Dash / Dot / DashDot / DashDotDot`.

### Two-point measurement
Configure Point A / Point B in the right **Measure** panel.
- **Trace**: the waveform to measure (file + signal). A and B may use different
  waveforms.
- **Ref** (reference):
  - `Cursor` — the trace value at the cursor's X position.
  - `Level` — a level crossing (rising or falling) within the visible range.
  - `Rise` — a level crossing on a rising edge within the visible range.
  - `Fall` — a level crossing on a falling edge within the visible range.
  - For `Level` / `Rise` / `Fall`, only crossings inside the **currently
    displayed X range** are considered; when several match, the **rightmost**
    (largest X) one is chosen. Panning/zooming re-picks the point automatically.
- **Level**: the crossing threshold. The **½·max** button sets it to half the
  selected trace's maximum (e.g. Vdd = 5 V → 2.5 V).
- **Drag the vertical cursors A (yellow) / B (cyan)** on the plot; for `Cursor`
  mode the point follows the cursor, while `Rise`/`Fall`/`Level` snap to the
  rightmost matching crossing in view, and a marker (●) is shown.
- Results show **Δt**, **1/Δt**, and **ΔV / ΔI** (chosen automatically from the
  two points' quantities). Crossings are linearly interpolated between samples,
  so resolution is finer than the simulation timestep.

## Layout

`py_wv.py` is a single script organised into three sections, mirroring the
original package modules:

```
py_wv.py
  ├─ rawfile   # binary .raw parser + data model (real/complex)
  ├─ measure   # edge/level crossing detection + interpolation
  └─ app       # GUI (pyqtgraph) + main() entry point
```
