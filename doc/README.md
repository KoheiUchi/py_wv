# py_wv.py

*[日本語版 README](README_jp.md)*

A waveform viewer for ngspice **binary `.raw`** files: load,
overlay, split into stacked panes, and take two-point measurements. Built with
Python + pyqtgraph (PyQt6).

`py_wv.py` require `numpy` and `pyqtgraph`.

## Launch

```bash
cd /home/uchida/work/design/TR1um/wave
python3 py_wv.py                 # start empty, then load from the menu
python3 py_wv.py outputfile.raw  # start with a file
# or, if executable:
./py_wv.py outputfile.raw
```

Dependencies: `numpy`, `pyqtgraph`, and a Qt binding (`PyQt6` is used). 

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
Configure Point A / Point B in the right **Measure** panel. Each point is shown
as a circle marker (●) on its trace; the vertical cursor lines A (yellow) and
B (cyan) are used to position the points.
- **Trace**: the waveform to measure (file + signal). A and B may use different
  waveforms.
- **Ref** (reference):
  - `Cursor` — the point sits at its own X position on the trace (see below).
  - `Level` — the crossing of the level (rising or falling) **nearest the
    cursor line**.
  - `Rise` — the rising crossing of the level nearest the cursor line.
  - `Fall` — the falling crossing of the level nearest the cursor line.
  - For `Level` / `Rise` / `Fall` the whole trace is searched and the crossing
    **closest to the vertical cursor line** is chosen — the line acts as a
    seed: drop it near the edge of interest and the point snaps to that edge.
    Moving the line re-picks the point live.
- **Level**: the crossing threshold. The **½·max** button sets it to half the
  selected trace's maximum (e.g. Vdd = 5 V → 2.5 V).
- **Placing a point in `Cursor` mode** (two ways):
  - **Drag the circle marker (●) directly** — it slides along the waveform
    (the dragged X is used; Y snaps onto the trace).
  - **Move the vertical cursor line, then press the point's
    "Set to cursor A/B" (決定) button** — the circle jumps to where the line
    crosses the trace. In `Cursor` mode the circle does **not** follow the
    line automatically; press the button to commit a new position.
  - Dragging the circle or pressing the button switches **Ref** to `Cursor`
    automatically.
- **Bring line A/B into view** — when a cursor's vertical line has been panned
  or zoomed off-screen, its **"Bring line A/B into view"** button (below the
  決定 button) recentres the line on the currently visible X window. The button
  is **enabled only while that line is off-screen**, so it signals when the
  cursor has drifted out of view.
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
