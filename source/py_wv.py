#!/usr/bin/env python3
"""waveview (single-file build) - ngspice binary .raw waveform viewer.

This is a self-contained build of the ``waveview`` package combined into one
script. It bundles the rawfile reader, measurement primitives and the Qt GUI.

Usage:
    ./py_wv.py [file1.raw file2.raw ...]
"""

from __future__ import annotations

import enum
import itertools
import os
import re
import sys
from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets


# ===========================================================================
# rawfile - ngspice binary .raw reader and data model
# ===========================================================================
#
# Supports the binary rawfile format produced by ngspice (real and complex
# flags). Multiple files - including ones that share the same filename - can be
# loaded; each load produces an independent :class:`RawFile` with a unique id.

_uid_counter = itertools.count(1)


@dataclass
class Variable:
    """A single output variable (column) of a rawfile."""

    index: int
    name: str
    quantity: str  # "time", "voltage", "current", "frequency", ...

    @property
    def is_current(self) -> bool:
        return self.quantity == "current"

    @property
    def is_voltage(self) -> bool:
        return self.quantity == "voltage"


@dataclass
class RawFile:
    """A parsed ngspice rawfile.

    Attributes
    ----------
    uid:        process-unique id, lets the same path be loaded many times.
    path:       absolute filesystem path.
    label:      display label (filename, with a suffix when duplicated).
    title/date/plotname/command: header metadata.
    variables:  list of :class:`Variable`.
    data:       ``(npoints, nvars)`` float array. For complex sweeps this is
                the magnitude; ``cdata`` holds the raw complex values.
    """

    uid: int
    path: str
    label: str
    title: str
    date: str
    plotname: str
    command: str
    flags: str
    variables: list[Variable]
    data: np.ndarray
    cdata: np.ndarray | None = None
    _by_name: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_name = {v.name.lower(): v.index for v in self.variables}

    # -- axis helpers --------------------------------------------------
    @property
    def npoints(self) -> int:
        return self.data.shape[0]

    @property
    def x_variable(self) -> Variable:
        """The sweep axis (column 0): time / frequency / etc."""
        return self.variables[0]

    @property
    def x(self) -> np.ndarray:
        return self.data[:, 0]

    def index_of(self, name: str) -> int | None:
        return self._by_name.get(name.lower())

    def column(self, index: int) -> np.ndarray:
        return self.data[:, index]


def load_raw(path: str) -> RawFile:
    """Parse a binary (or ascii) ngspice rawfile into a :class:`RawFile`."""
    path = os.path.abspath(path)
    with open(path, "rb") as fh:
        blob = fh.read()

    # The header is plain text terminated by either "Binary:\n" or "Values:\n".
    bin_marker = b"Binary:\n"
    val_marker = b"Values:\n"
    bidx = blob.find(bin_marker)
    vidx = blob.find(val_marker)
    if bidx == -1 and vidx == -1:
        raise ValueError(f"{path}: not a recognisable ngspice rawfile "
                         "(no 'Binary:' or 'Values:' marker)")

    binary = bidx != -1 and (vidx == -1 or bidx < vidx)
    marker_idx = bidx if binary else vidx
    marker = bin_marker if binary else val_marker
    header_text = blob[:marker_idx].decode("latin-1")
    meta = _parse_header(header_text)

    nvars = meta["nvars"]
    npts = meta["npoints"]
    complex_ = "complex" in meta["flags"].lower()
    data_start = marker_idx + len(marker)

    if binary:
        if complex_:
            raw = np.frombuffer(blob, dtype="<c16", count=nvars * npts,
                                offset=data_start)
        else:
            raw = np.frombuffer(blob, dtype="<f8", count=nvars * npts,
                                offset=data_start)
        raw = raw.reshape(npts, nvars)
    else:
        raw = _parse_ascii_values(blob[data_start:].decode("latin-1"),
                                  nvars, npts, complex_)

    if complex_:
        cdata = raw
        data = np.abs(raw)
        # keep the sweep axis (frequency) real-valued for plotting
        data[:, 0] = raw[:, 0].real
    else:
        cdata = None
        data = raw.astype(np.float64, copy=True)

    return RawFile(
        uid=next(_uid_counter),
        path=path,
        label=os.path.basename(path),
        title=meta["title"],
        date=meta["date"],
        plotname=meta["plotname"],
        command=meta["command"],
        flags=meta["flags"],
        variables=meta["variables"],
        data=data,
        cdata=cdata,
    )


def _parse_header(text: str) -> dict:
    def grab(key: str, default: str = "") -> str:
        m = re.search(rf"^{key}:\s*(.*)$", text, re.MULTILINE)
        return m.group(1).strip() if m else default

    nvars = int(grab("No\\. Variables", "0"))
    npts = int(grab("No\\. Points", "0"))
    if nvars == 0 or npts == 0:
        raise ValueError("rawfile header missing variable/point counts")

    # Variable table: lines after "Variables:" up to the data marker.
    vstart = text.find("Variables:")
    var_block = text[vstart + len("Variables:"):]
    variables: list[Variable] = []
    for line in var_block.splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        variables.append(Variable(index=idx, name=parts[1], quantity=parts[2]))
        if len(variables) == nvars:
            break
    if len(variables) != nvars:
        raise ValueError(
            f"expected {nvars} variables, parsed {len(variables)}")

    return {
        "title": grab("Title"),
        "date": grab("Date"),
        "plotname": grab("Plotname"),
        "command": grab("Command"),
        "flags": grab("Flags"),
        "nvars": nvars,
        "npoints": npts,
        "variables": variables,
    }


def _parse_ascii_values(text: str, nvars: int, npts: int,
                        complex_: bool) -> np.ndarray:
    """Parse an ascii 'Values:' block. Fallback path; binary is the norm.

    Each point begins with an integer index followed by ``nvars`` values.
    """
    tokens = text.split()
    out = (np.zeros((npts, nvars), dtype="c16") if complex_
           else np.zeros((npts, nvars), dtype="f8"))
    it = iter(tokens)
    for p in range(npts):
        next(it)  # discard the leading point index
        for v in range(nvars):
            tok = next(it)
            if complex_:
                re_s, _, im_s = tok.partition(",")
                out[p, v] = complex(float(re_s), float(im_s or 0.0))
            else:
                out[p, v] = float(tok)
    return out


# ===========================================================================
# measure - locate a point on a trace by edge/level rules
# ===========================================================================
#
# A measurement point is found on a waveform ``(x, y)`` according to a
# :class:`RefMode`:
#
# * ``CURSOR``  - sample the trace value at the cursor's x position.
# * ``LEVEL``   - rightmost crossing of ``level`` (rising or falling) in view.
# * ``RISE``    - rightmost rising crossing of ``level`` in view.
# * ``FALL``    - rightmost falling crossing of ``level`` in view.
#
# For the level/edge modes the search is restricted to the currently visible
# x range (``xlim``); when several crossings fall inside the view the rightmost
# (largest x) one is chosen. Crossings are linearly interpolated between samples
# so sub-sample resolution is available regardless of the simulation timestep.

class RefMode(enum.Enum):
    CURSOR = "Cursor"
    LEVEL = "Level"
    RISE = "Rise"
    FALL = "Fall"

    @property
    def label(self) -> str:
        return self.value


@dataclass
class MeasurePoint:
    """Result of resolving a measurement reference on a trace."""

    x: float
    y: float
    ok: bool
    detail: str = ""


def value_at(x: np.ndarray, y: np.ndarray, xq: float) -> float:
    """Linearly interpolate ``y`` at sweep position ``xq``."""
    return float(np.interp(xq, x, y))


def _all_crossings(x: np.ndarray, y: np.ndarray, level: float,
                  edge: str) -> np.ndarray:
    """Return interpolated x positions where ``y`` crosses ``level``.

    ``edge`` is one of ``"rise"``, ``"fall"`` or ``"any"``.
    """
    d = y - level
    # sign changes between consecutive samples => a crossing in that segment
    s = np.signbit(d)
    change = np.where(s[:-1] != s[1:])[0]
    if change.size == 0:
        return np.empty(0)

    rising = d[change + 1] > d[change]
    if edge == "rise":
        change = change[rising]
    elif edge == "fall":
        change = change[~rising]
    if change.size == 0:
        return np.empty(0)

    y0 = d[change]
    y1 = d[change + 1]
    x0 = x[change]
    x1 = x[change + 1]
    denom = (y1 - y0)
    # guard against exact-equal samples (denom == 0)
    frac = np.where(denom != 0, -y0 / denom, 0.0)
    return x0 + frac * (x1 - x0)


def resolve(x: np.ndarray, y: np.ndarray, mode: RefMode,
            cursor_x: float, level: float = 0.0,
            xlim: tuple[float, float] | None = None) -> MeasurePoint:
    """Resolve a measurement point on ``(x, y)`` for ``mode``.

    ``CURSOR`` samples the trace at ``cursor_x``. For the level/edge modes the
    crossings are restricted to the visible x range ``xlim`` (a ``(lo, hi)``
    tuple, or ``None`` for the whole trace) and the rightmost one is returned.
    """
    if mode is RefMode.CURSOR:
        return MeasurePoint(cursor_x, value_at(x, y, cursor_x), True,
                            "cursor")

    edge = {"Level": "any", "Rise": "rise", "Fall": "fall"}[mode.value]
    xs = _all_crossings(x, y, level, edge)
    if xlim is not None:
        lo, hi = sorted(xlim)
        xs = xs[(xs >= lo) & (xs <= hi)]
    if xs.size == 0:
        where = " in view" if xlim is not None else ""
        return MeasurePoint(cursor_x, value_at(x, y, cursor_x), False,
                            f"no {mode.value.lower()} crossing of {level:g}{where}")

    # pick the rightmost crossing within the visible range
    pick = float(xs.max())
    return MeasurePoint(pick, level, True,
                        f"{mode.value} @ {level:g}")


# ===========================================================================
# app - waveview GUI
# ===========================================================================
#
# Load ngspice .raw files, plot, overlay/split, and measure.

Qt = QtCore.Qt

# ---- line-style catalogue -------------------------------------------------
LINE_STYLES = [
    ("Solid", Qt.PenStyle.SolidLine),
    ("Dash", Qt.PenStyle.DashLine),
    ("Dot", Qt.PenStyle.DotLine),
    ("DashDot", Qt.PenStyle.DashDotLine),
    ("DashDotDot", Qt.PenStyle.DashDotDotLine),
]

# A pleasant, high-contrast default colour cycle.
COLOR_CYCLE = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
    "#9a6324", "#fffac8", "#800000", "#aaffc3", "#808000",
]

_trace_uid = itertools.count(1)

# UI string table: key -> (English, 日本語). Used by MainWindow.t().
TR = {
    "menu.file":     ("&File", "ファイル(&F)"),
    "menu.view":     ("&View", "表示(&V)"),
    "menu.option":   ("&Option", "オプション(&O)"),
    "file.open":     ("Open .raw…", "raw を開く…"),
    "file.close":    ("Close .raw", "raw を閉じる"),
    "file.exit":     ("Exit", "終了"),
    "view.fit":      ("Fit All", "全体表示"),
    "view.zin":      ("Zoom In", "拡大"),
    "view.zout":     ("Zoom Out", "縮小"),
    "view.rect":     ("Rectangle Zoom", "範囲選択ズーム"),
    "view.cursors":  ("Zoom X to Cursors A–B", "カーソル A–B 間にズーム"),
    "view.panels":   ("Panels", "パネル表示"),
    "opt.language":  ("Language", "言語"),
    "opt.font":      ("Font", "フォント"),
    "opt.font_inc":  ("Enlarge", "拡大"),
    "opt.font_dec":  ("Shrink", "縮小"),
    "opt.font_def":  ("Default", "デフォルトに戻す"),
    "panel.sources": ("Sources", "信号一覧"),
    "panel.traces":  ("Traces", "トレース"),
    "panel.measure": ("Measure", "測定"),
    "tb.split":      ("Split selected ▼", "選択を分離 ▼"),
    "tb.merge":      ("Merge selected ▲", "選択を重ねる ▲"),
    "tb.remove":     ("Remove trace", "トレース削除"),
    "tree.signal":   ("Signal", "信号"),
    "tree.type":     ("Type", "種別"),
    "tbl.show":      ("Show", "表示"),
    "tbl.trace":     ("Trace", "トレース"),
    "tbl.color":     ("Color", "色"),
    "tbl.style":     ("Style", "線種"),
    "tbl.pane":      ("Pane", "ペイン"),
    "meas.point":    ("Point {}", "ポイント {}"),
    "meas.trace":    ("Trace", "対象"),
    "meas.ref":      ("Ref", "基準"),
    "meas.level":    ("Level", "レベル"),
    "tip.close":     ("Close the .raw selected in Sources",
                      "Sources で選択した .raw を閉じる"),
    "tip.rect":      ("Drag to zoom into a rectangle (two points)",
                      "左ドラッグで囲んだ任意の2点の範囲にズーム"),
    "tip.split":     ("Move selected traces to a new pane (stacked)",
                      "選択トレースを新規ペインに分離（上下表示）"),
    "tip.merge":     ("Overlay selected traces in one pane",
                      "選択トレースを1つのペインに重ねる"),
    "tip.half":      ("Set level to ½ of the trace's max (e.g. Vdd/2)",
                      "選択トレース最大値の1/2をレベルに設定 (例: Vdd/2)"),
    "status.start":  ("Open one or more .raw files to begin.",
                      "1つ以上の .raw ファイルを開いてください。"),
    "xnav.jleft":    ("Jump left (large step)", "大きく左へ移動"),
    "xnav.left":     ("Pan left", "左へ移動"),
    "xnav.zout":     ("Zoom out X", "X軸を縮小"),
    "xnav.zin":      ("Zoom in X", "X軸を拡大"),
    "xnav.right":    ("Pan right", "右へ移動"),
    "xnav.jright":   ("Jump right (large step)", "大きく右へ移動"),
}


@dataclass
class Trace:
    """A waveform selected for display: a (file, variable) pair plus style."""

    uid: int
    raw: RawFile
    var: Variable
    color: QtGui.QColor
    style: Qt.PenStyle = Qt.PenStyle.SolidLine
    width: float = 1.5
    pane: int = 0
    visible: bool = True
    curve: pg.PlotDataItem | None = field(default=None, repr=False)

    @property
    def display(self) -> str:
        return f"{self.raw.label}#{self.raw.uid}:{self.var.name}"

    def pen(self) -> QtGui.QPen:
        pen = QtGui.QPen(self.color)
        pen.setStyle(self.style)
        pen.setWidthF(self.width)
        pen.setCosmetic(True)
        return pen

    def xy(self) -> tuple[np.ndarray, np.ndarray]:
        return self.raw.x, self.raw.column(self.var.index)


# ---------------------------------------------------------------------------
class Cursor:
    """A draggable vertical cursor mirrored across every stacked pane."""

    def __init__(self, name: str, color: str):
        self.name = name
        self.color = color
        self.lines: list[pg.InfiniteLine] = []
        self.marker: pg.ScatterPlotItem | None = None
        self.marker_host: pg.PlotItem | None = None
        self._x = 0.0
        self._syncing = False
        self.on_moved = None  # callback()

    def x(self) -> float:
        return self._x

    def set_x(self, x: float):
        self._x = x
        self._syncing = True
        for ln in self.lines:
            ln.setPos(x)
        self._syncing = False

    def _line_moved(self, line: pg.InfiniteLine):
        if self._syncing:
            return
        self._x = line.value()
        self._syncing = True
        for ln in self.lines:
            if ln is not line:
                ln.setPos(self._x)
        self._syncing = False
        if self.on_moved:
            self.on_moved()


SI_PREFIXES = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m",
               0: "", 3: "k", 6: "M", 9: "G", 12: "T"}


class SIAxisItem(pg.AxisItem):
    """Axis with SI-prefixed tick labels in 10^3 steps (…, ps, ns, µs, …).

    Each tick shows the value scaled by a single prefix chosen for the whole
    axis, with the unit appended, e.g. ``60ns``. Because the prefix lives on
    the ticks (not the label), every stacked pane reads consistently even
    when it has no axis label."""

    def __init__(self, orientation, unit="", **kwargs):
        super().__init__(orientation, **kwargs)
        self.unit = unit
        self.enableAutoSIPrefix(False)  # we do the prefixing ourselves

    def tickStrings(self, values, scale, spacing):
        if len(values) == 0:
            return []
        ref = max((abs(v) for v in values), default=0.0) or abs(spacing) or 1.0
        exp = int(np.floor(np.log10(ref) / 3.0) * 3)
        exp = max(-15, min(12, exp))
        div = 10.0 ** exp
        prefix = SI_PREFIXES.get(exp, "")
        sp = spacing / div if spacing else 0.0
        dec = max(0, -int(np.floor(np.log10(sp)))) if sp > 0 else 0
        dec = min(dec, 6)
        return [f"{v / div:.{dec}f}{prefix}{self.unit}" for v in values]


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("waveview - ngspice waveform viewer")
        self.resize(1400, 850)

        self.raws: list[RawFile] = []
        self.traces: list[Trace] = []
        self.panes: dict[int, pg.PlotItem] = {}
        self._color_iter = itertools.cycle(COLOR_CYCLE)
        self._suppress_tree_signal = False
        self.rect_zoom = False
        self.lang = "en"               # "ja" or "en"
        self._tr: list = []            # (setter, key, fmt-args) registry

        app = QtWidgets.QApplication.instance()
        base_pt = app.font().pointSizeF()
        self._base_pt = base_pt if base_pt > 0 else 10.0
        self._font_pt = self._base_pt

        pg.setConfigOptions(antialias=True, background="#1b1b1b",
                            foreground="#d0d0d0")

        self._build_ui()
        self.retranslate()
        self.statusBar().showMessage(self.t("status.start"))

    # -- i18n ----------------------------------------------------------
    def t(self, key: str, *args) -> str:
        s = TR[key][0 if self.lang == "en" else 1]
        return s.format(*args) if args else s

    def _reg(self, setter, key: str, *args):
        """Register a text setter so retranslate() can re-apply it."""
        self._tr.append((setter, key, args))

    def set_language(self, lang: str):
        self.lang = lang
        self.retranslate()

    def retranslate(self):
        for setter, key, args in self._tr:
            setter(self.t(key, *args))
        self.tree.setHeaderLabels([self.t("tree.signal"), self.t("tree.type")])
        self.table.setHorizontalHeaderLabels(
            [self.t("tbl.show"), self.t("tbl.trace"), self.t("tbl.color"),
             self.t("tbl.style"), self.t("tbl.pane"), ""])
        self.act_lang_en.setChecked(self.lang == "en")
        self.act_lang_ja.setChecked(self.lang == "ja")

    # -- font scaling --------------------------------------------------
    def _apply_font(self):
        app = QtWidgets.QApplication.instance()
        f = app.font()
        f.setPointSizeF(self._font_pt)
        app.setFont(f)
        self.setFont(f)
        for w in self.findChildren(QtWidgets.QWidget):
            w.setFont(f)
        self.statusBar().showMessage(f"Font: {self._font_pt:.0f} pt")

    def font_enlarge(self):
        self._font_pt += 1
        self._apply_font()

    def font_shrink(self):
        self._font_pt = max(5.0, self._font_pt - 1)
        self._apply_font()

    def font_reset(self):
        self._font_pt = self._base_pt
        self._apply_font()

    # -- UI scaffolding ------------------------------------------------
    def _build_ui(self):
        self.glw = pg.GraphicsLayoutWidget()
        self.glw.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        # Central area: the plot on top, an X-axis navigation bar underneath.
        central = QtWidgets.QWidget()
        vlay = QtWidgets.QVBoxLayout(central)
        vlay.setContentsMargins(0, 0, 0, 0)
        vlay.setSpacing(0)
        vlay.addWidget(self.glw, 1)
        vlay.addWidget(self._build_xnav_bar())
        self.setCentralWidget(central)

        self.cursorA = Cursor("A", "#ffd400")
        self.cursorB = Cursor("B", "#00d9ff")
        self.cursorA.on_moved = self._on_cursor_moved
        self.cursorB.on_moved = self._on_cursor_moved

        self._build_menu()
        self._build_toolbar()
        self._build_sources_dock()
        self._build_traces_dock()
        self._build_measure_dock()
        self._add_panel_toggles()
        self._build_option_menu()
        self._build_pan_shortcuts()
        self._rebuild_panes()

    def _build_pan_shortcuts(self):
        """Left/Right arrow keys scroll the waveform horizontally.

        Scoped to the plot widget (WidgetWithChildrenShortcut) so the arrow
        keys still edit spin boxes / navigate the trees when those have focus.
        Shift makes a larger jump."""
        for seq, frac in (("Left", -0.10), ("Right", 0.10),
                          ("Shift+Left", -0.50), ("Shift+Right", 0.50)):
            sc = QtGui.QShortcut(QtGui.QKeySequence(seq), self.glw)
            sc.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            sc.activated.connect(lambda f=frac: self.pan_x(f))

    def pan_x(self, frac: float):
        """Pan the (linked) X axis by ``frac`` of the visible width.

        Positive scrolls toward later x (Right arrow); negative toward
        earlier x (Left arrow)."""
        if not self.panes:
            return
        vb = next(iter(self.panes.values())).getViewBox()
        (x0, x1), _ = vb.viewRange()
        d = (x1 - x0) * frac
        vb.setXRange(x0 + d, x1 + d, padding=0)

    def _build_xnav_bar(self) -> QtWidgets.QWidget:
        """A row of buttons under the plot to zoom/pan the shared X axis.

        Mirrors the keyboard controls (arrow keys pan, Ctrl+± zoom) with
        on-screen buttons for mouse-only use. The X axis is linked across
        panes, so these act on all stacked panes at once."""
        bar = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(bar)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.setSpacing(4)
        # left spacer keeps the buttons roughly under the plot area, clear of
        # the left (Y) axis gutter.
        lay.addStretch(1)

        def add(symbol: str, tip_key: str, slot):
            btn = QtWidgets.QToolButton()
            btn.setText(symbol)
            btn.setAutoRaise(True)
            btn.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            btn.setMinimumWidth(38)
            btn.clicked.connect(slot)
            self._reg(btn.setToolTip, tip_key)
            lay.addWidget(btn)
            return btn

        add("◀◀", "xnav.jleft", lambda: self.pan_x(-0.50))
        add("◀", "xnav.left", lambda: self.pan_x(-0.10))
        lay.addSpacing(12)
        add("－", "xnav.zout", self.zoom_x_out)
        add("＋", "xnav.zin", self.zoom_x_in)
        lay.addSpacing(12)
        add("▶", "xnav.right", lambda: self.pan_x(0.10))
        add("▶▶", "xnav.jright", lambda: self.pan_x(0.50))
        lay.addStretch(1)
        return bar

    def _scale_x(self, factor: float):
        """Zoom the shared X axis about the view centre, leaving Y untouched.

        X is linked across panes, so scaling the first viewbox propagates to
        the rest through the link."""
        vbs = self._viewboxes()
        if not vbs:
            return
        vbs[0].scaleBy(x=factor, y=None)

    def zoom_x_in(self):
        self._scale_x(0.8)

    def zoom_x_out(self):
        self._scale_x(1.25)

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("")
        self._reg(file_menu.setTitle, "menu.file")
        a = file_menu.addAction("")
        a.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        a.triggered.connect(self.open_files)
        self._reg(a.setText, "file.open")
        a = file_menu.addAction("")
        a.setShortcut(QtGui.QKeySequence.StandardKey.Close)
        a.triggered.connect(self.close_selected_raw)
        self._reg(a.setText, "file.close")
        self._reg(a.setToolTip, "tip.close")
        file_menu.addSeparator()
        a = file_menu.addAction("")
        a.setShortcuts([QtGui.QKeySequence.StandardKey.Quit,
                        QtGui.QKeySequence("Ctrl+Q")])
        a.triggered.connect(self.close)
        self._reg(a.setText, "file.exit")

        view = mb.addMenu("")
        self.view_menu = view
        self._reg(view.setTitle, "menu.view")
        a = view.addAction("")
        a.setShortcut("Ctrl+0")
        a.triggered.connect(self.fit_all)
        self._reg(a.setText, "view.fit")
        a = view.addAction("")
        a.setShortcuts([QtGui.QKeySequence.StandardKey.ZoomIn,
                        QtGui.QKeySequence("Ctrl+=")])
        a.triggered.connect(self.zoom_in)
        self._reg(a.setText, "view.zin")
        a = view.addAction("")
        a.setShortcut(QtGui.QKeySequence.StandardKey.ZoomOut)
        a.triggered.connect(self.zoom_out)
        self._reg(a.setText, "view.zout")
        view.addSeparator()
        self.act_rect = view.addAction("")
        self.act_rect.setCheckable(True)
        self.act_rect.setShortcut("Ctrl+R")
        self.act_rect.toggled.connect(self.set_rect_zoom)
        self._reg(self.act_rect.setText, "view.rect")
        self._reg(self.act_rect.setToolTip, "tip.rect")
        view.addSeparator()
        a = view.addAction("")
        a.setShortcut("Ctrl+B")
        a.triggered.connect(self.zoom_to_cursors)
        self._reg(a.setText, "view.cursors")

    def _add_panel_toggles(self):
        """Append dock show/hide toggles to the View menu.

        Called after the docks exist. Each toggle reflects and controls its
        dock's visibility, so a panel closed with its ✕ can be reopened here."""
        self.view_menu.addSeparator()
        sub = self.view_menu.addMenu("")
        self._reg(sub.setTitle, "view.panels")
        for dock, key in ((self.dock_sources, "panel.sources"),
                          (self.dock_traces, "panel.traces"),
                          (self.dock_measure, "panel.measure")):
            act = dock.toggleViewAction()
            sub.addAction(act)
            self._reg(act.setText, key)
            self._reg(dock.setWindowTitle, key)

    def _build_option_menu(self):
        opt = self.menuBar().addMenu("")
        self.menu_option = opt
        self._reg(opt.setTitle, "menu.option")

        lang = opt.addMenu("")
        self._reg(lang.setTitle, "opt.language")
        grp = QtGui.QActionGroup(self)
        grp.setExclusive(True)
        self.act_lang_en = lang.addAction("English")
        self.act_lang_ja = lang.addAction("日本語")
        for act, code in ((self.act_lang_en, "en"), (self.act_lang_ja, "ja")):
            act.setCheckable(True)
            grp.addAction(act)
            act.triggered.connect(lambda _=False, c=code: self.set_language(c))

        opt.addSeparator()
        font = opt.addMenu("")
        self._reg(font.setTitle, "opt.font")
        # Use layout-stable keys ( . , 0 ): on JP keyboards '=' '+' '-' are
        # produced with Shift and do not map reliably as shortcut keys.
        a = font.addAction("")
        a.setShortcut("Ctrl+Shift+.")
        a.triggered.connect(self.font_enlarge)
        self._reg(a.setText, "opt.font_inc")
        a = font.addAction("")
        a.setShortcut("Ctrl+Shift+,")
        a.triggered.connect(self.font_shrink)
        self._reg(a.setText, "opt.font_dec")
        a = font.addAction("")
        a.setShortcut("Ctrl+Shift+0")
        a.triggered.connect(self.font_reset)
        self._reg(a.setText, "opt.font_def")

    def _build_toolbar(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        a = tb.addAction("")
        a.triggered.connect(self.open_files)
        self._reg(a.setText, "file.open")
        tb.addSeparator()
        a = tb.addAction("")
        a.triggered.connect(self.split_selected)
        self._reg(a.setText, "tb.split")
        self._reg(a.setToolTip, "tip.split")
        a = tb.addAction("")
        a.triggered.connect(self.merge_selected)
        self._reg(a.setText, "tb.merge")
        self._reg(a.setToolTip, "tip.merge")
        tb.addSeparator()
        a = tb.addAction("")
        a.triggered.connect(self.remove_selected)
        self._reg(a.setText, "tb.remove")

    def _build_sources_dock(self):
        dock = QtWidgets.QDockWidget("Sources", self)
        dock.setObjectName("sources")
        self.dock_sources = dock
        self.tree = QtWidgets.QTreeWidget()
        self.tree.setHeaderLabels(["Signal", "Type"])
        self.tree.setColumnWidth(0, 220)
        self.tree.itemChanged.connect(self._on_tree_item_changed)
        dock.setWidget(self.tree)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_traces_dock(self):
        dock = QtWidgets.QDockWidget("Traces", self)
        dock.setObjectName("traces")
        self.dock_traces = dock
        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Show", "Trace", "Color", "Style", "Pane", ""])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows)
        hdr = self.table.horizontalHeader()
        # Interactive (not Stretch) so every column - including "Trace" - can
        # be dragged wider. A Stretch section auto-fills and cannot be resized.
        hdr.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.Interactive)
        hdr.setStretchLastSection(False)
        for col, w in enumerate((44, 200, 52, 90, 52, 32)):
            self.table.setColumnWidth(col, w)
        dock.setWidget(self.table)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _build_measure_dock(self):
        dock = QtWidgets.QDockWidget("Measure", self)
        dock.setObjectName("measure")
        self.dock_measure = dock
        w = QtWidgets.QWidget()
        form = QtWidgets.QVBoxLayout(w)

        self.cursor_widgets = {}
        for tag, cur in (("A", self.cursorA), ("B", self.cursorB)):
            box = QtWidgets.QGroupBox()
            self._reg(box.setTitle, "meas.point", tag)
            box.setStyleSheet(f"QGroupBox{{color:{cur.color};font-weight:bold}}")
            g = QtWidgets.QGridLayout(box)
            trace_cb = QtWidgets.QComboBox()
            mode_cb = QtWidgets.QComboBox()
            for m in RefMode:
                mode_cb.addItem(m.label, m)
            level_sb = pg.SpinBox(value=2.5, step=0.1, decimals=6)
            half_btn = QtWidgets.QPushButton("½·max")
            self._reg(half_btn.setToolTip, "tip.half")
            lbl_t, lbl_r, lbl_l = (QtWidgets.QLabel() for _ in range(3))
            self._reg(lbl_t.setText, "meas.trace")
            self._reg(lbl_r.setText, "meas.ref")
            self._reg(lbl_l.setText, "meas.level")
            g.addWidget(lbl_t, 0, 0)
            g.addWidget(trace_cb, 0, 1, 1, 2)
            g.addWidget(lbl_r, 1, 0)
            g.addWidget(mode_cb, 1, 1, 1, 2)
            g.addWidget(lbl_l, 2, 0)
            g.addWidget(level_sb, 2, 1)
            g.addWidget(half_btn, 2, 2)
            form.addWidget(box)

            trace_cb.currentIndexChanged.connect(self._recompute_measure)
            mode_cb.currentIndexChanged.connect(self._recompute_measure)
            level_sb.sigValueChanged.connect(self._recompute_measure)
            half_btn.clicked.connect(
                lambda _=False, t=tag: self._set_half_level(t))
            self.cursor_widgets[tag] = dict(
                trace=trace_cb, mode=mode_cb, level=level_sb)

        self.result_lbl = QtWidgets.QLabel("—")
        self.result_lbl.setTextFormat(Qt.TextFormat.RichText)
        self.result_lbl.setStyleSheet(
            "font-family:monospace; padding:8px; "
            "background:#111; color:#e6e6e6;")
        self.result_lbl.setWordWrap(True)
        form.addWidget(self.result_lbl)
        form.addStretch(1)

        dock.setWidget(w)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    # -- file loading --------------------------------------------------
    def open_files(self, paths=None):
        if not paths:
            paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
                self, "Open ngspice rawfiles", "", "Rawfiles (*.raw);;All (*)")
        for p in paths or []:
            self.load_file(p)

    def load_file(self, path: str):
        try:
            raw = load_raw(path)
        except Exception as exc:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "Load error", str(exc))
            return
        self.raws.append(raw)
        self._add_source_to_tree(raw)
        self.statusBar().showMessage(
            f"Loaded {raw.label}#{raw.uid}: {len(raw.variables)-1} signals, "
            f"{raw.npoints} pts")

    def close_selected_raw(self):
        """Close (unload) the .raw file(s) selected in the Sources tree."""
        items = self.tree.selectedItems()
        uids = set()
        for it in items:
            meta = it.data(0, Qt.ItemDataRole.UserRole)
            if meta:
                uids.add(meta[1])  # uid for both 'file' and 'var' rows
        if not uids:
            QtWidgets.QMessageBox.information(
                self, "Close .raw",
                "Sources で閉じたい .raw（または信号）を選択してください。")
            return
        self._close_raws(uids)

    def _close_raws(self, uids: set[int]):
        labels = [f"{r.label}#{r.uid}" for r in self.raws if r.uid in uids]
        # drop traces belonging to these files, then the files themselves
        self.traces = [t for t in self.traces if t.raw.uid not in uids]
        self.raws = [r for r in self.raws if r.uid not in uids]
        # remove their tree nodes
        self._suppress_tree_signal = True
        for i in reversed(range(self.tree.topLevelItemCount())):
            root = self.tree.topLevelItem(i)
            meta = root.data(0, Qt.ItemDataRole.UserRole)
            if meta and meta[1] in uids:
                self.tree.takeTopLevelItem(i)
        self._suppress_tree_signal = False
        self._refresh_traces_table()
        self._rebuild_panes()
        self._refresh_measure_trace_combos()
        self.statusBar().showMessage("Closed " + ", ".join(labels))

    def _add_source_to_tree(self, raw: RawFile):
        self._suppress_tree_signal = True
        root = QtWidgets.QTreeWidgetItem(
            self.tree, [f"{raw.label}  #{raw.uid}", raw.plotname])
        root.setData(0, Qt.ItemDataRole.UserRole, ("file", raw.uid))
        root.setExpanded(True)
        f = root.font(0)
        f.setBold(True)
        root.setFont(0, f)
        for var in raw.variables[1:]:  # skip the sweep axis
            it = QtWidgets.QTreeWidgetItem(root, [var.name, var.quantity])
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(0, Qt.CheckState.Unchecked)
            it.setData(0, Qt.ItemDataRole.UserRole,
                       ("var", raw.uid, var.index))
        self._suppress_tree_signal = False

    def _on_tree_item_changed(self, item, column):
        if self._suppress_tree_signal or column != 0:
            return
        meta = item.data(0, Qt.ItemDataRole.UserRole)
        if not meta or meta[0] != "var":
            return
        _, uid, vidx = meta
        checked = item.checkState(0) == Qt.CheckState.Checked
        if checked:
            self._add_trace(uid, vidx)
        else:
            self._remove_trace_by_var(uid, vidx)

    # -- trace management ----------------------------------------------
    def _find_raw(self, uid: int) -> RawFile | None:
        return next((r for r in self.raws if r.uid == uid), None)

    def _add_trace(self, raw_uid: int, vidx: int):
        raw = self._find_raw(raw_uid)
        if raw is None:
            return
        color = QtGui.QColor(next(self._color_iter))
        tr = Trace(uid=next(_trace_uid), raw=raw, var=raw.variables[vidx],
                   color=color)
        self.traces.append(tr)
        self._refresh_traces_table()
        self._rebuild_panes()
        self._refresh_measure_trace_combos()

    def _remove_trace_by_var(self, raw_uid: int, vidx: int):
        before = len(self.traces)
        self.traces = [t for t in self.traces
                       if not (t.raw.uid == raw_uid and t.var.index == vidx)]
        if len(self.traces) != before:
            self._refresh_traces_table()
            self._rebuild_panes()
            self._refresh_measure_trace_combos()

    def selected_traces(self) -> list[Trace]:
        rows = {i.row() for i in self.table.selectedIndexes()}
        return [self.traces[r] for r in sorted(rows) if r < len(self.traces)]

    def split_selected(self):
        sel = self.selected_traces()
        if not sel:
            return
        new_pane = (max((t.pane for t in self.traces), default=-1) + 1)
        for t in sel:
            t.pane = new_pane
        self._refresh_traces_table()
        self._rebuild_panes()

    def merge_selected(self):
        sel = self.selected_traces()
        if len(sel) < 2:
            return
        target = min(t.pane for t in sel)
        for t in sel:
            t.pane = target
        self._refresh_traces_table()
        self._rebuild_panes()

    def remove_selected(self):
        sel = set(self.selected_traces())
        if not sel:
            return
        self.traces = [t for t in self.traces if t not in sel]
        # uncheck in tree
        self._sync_tree_checks()
        self._refresh_traces_table()
        self._rebuild_panes()
        self._refresh_measure_trace_combos()

    def _sync_tree_checks(self):
        active = {(t.raw.uid, t.var.index) for t in self.traces}
        self._suppress_tree_signal = True
        for i in range(self.tree.topLevelItemCount()):
            root = self.tree.topLevelItem(i)
            for j in range(root.childCount()):
                child = root.child(j)
                meta = child.data(0, Qt.ItemDataRole.UserRole)
                if meta and meta[0] == "var":
                    on = (meta[1], meta[2]) in active
                    child.setCheckState(
                        0, Qt.CheckState.Checked if on
                        else Qt.CheckState.Unchecked)
        self._suppress_tree_signal = False

    # -- traces table --------------------------------------------------
    def _refresh_traces_table(self):
        self.table.blockSignals(True)
        self.table.setRowCount(len(self.traces))
        for row, t in enumerate(self.traces):
            # show
            chk = QtWidgets.QCheckBox()
            chk.setChecked(t.visible)
            chk.toggled.connect(lambda v, tr=t: self._set_visible(tr, v))
            self.table.setCellWidget(row, 0, self._center(chk))
            # label
            self.table.setItem(row, 1, QtWidgets.QTableWidgetItem(t.display))
            # colour
            cbtn = QtWidgets.QPushButton()
            cbtn.setStyleSheet(f"background:{t.color.name()};")
            cbtn.clicked.connect(lambda _=False, tr=t: self._pick_color(tr))
            self.table.setCellWidget(row, 2, cbtn)
            # style
            scb = QtWidgets.QComboBox()
            for name, st in LINE_STYLES:
                scb.addItem(name, st)
            scb.setCurrentIndex(
                next(i for i, (_, st) in enumerate(LINE_STYLES)
                     if st == t.style))
            scb.currentIndexChanged.connect(
                lambda idx, tr=t, cb=scb: self._set_style(tr, cb.itemData(idx)))
            self.table.setCellWidget(row, 3, scb)
            # pane
            psb = QtWidgets.QSpinBox()
            psb.setRange(0, 99)
            psb.setValue(t.pane)
            psb.valueChanged.connect(lambda v, tr=t: self._set_pane(tr, v))
            self.table.setCellWidget(row, 4, psb)
            # remove
            rbtn = QtWidgets.QPushButton("✕")
            rbtn.clicked.connect(lambda _=False, tr=t: self._remove_one(tr))
            self.table.setCellWidget(row, 5, rbtn)
        self.table.blockSignals(False)

    @staticmethod
    def _center(widget):
        wrap = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addStretch(1)
        lay.addWidget(widget)
        lay.addStretch(1)
        return wrap

    def _remove_one(self, tr: Trace):
        self.traces = [t for t in self.traces if t is not tr]
        self._sync_tree_checks()
        self._refresh_traces_table()
        self._rebuild_panes()
        self._refresh_measure_trace_combos()

    def _set_visible(self, tr: Trace, v: bool):
        tr.visible = v
        self._rebuild_panes()

    def _set_pane(self, tr: Trace, v: int):
        tr.pane = v
        self._rebuild_panes()

    def _set_style(self, tr: Trace, style):
        tr.style = style
        if tr.curve is not None:
            tr.curve.setPen(tr.pen())

    def _pick_color(self, tr: Trace):
        col = QtWidgets.QColorDialog.getColor(tr.color, self, "Trace colour")
        if col.isValid():
            tr.color = col
            self._refresh_traces_table()
            if tr.curve is not None:
                tr.curve.setPen(tr.pen())

    # -- plotting ------------------------------------------------------
    def _rebuild_panes(self):
        self.glw.clear()
        self.panes.clear()
        for c in (self.cursorA, self.cursorB):
            c.lines = []
            c.marker = None
            c.marker_host = None

        visible = [t for t in self.traces if t.visible]
        pane_ids = sorted({t.pane for t in visible}) or [0]

        xv = self.raws[0].x_variable if self.raws else None
        xunit = _unit(xv.quantity) if xv is not None else ""

        first_vb = None
        for r, pid in enumerate(pane_ids):
            # SI-prefixed bottom axis on EVERY pane so all read e.g. "60ns"
            # (not "6e-8") regardless of whether they carry an axis label.
            plot = self.glw.addPlot(
                row=r, col=0,
                axisItems={"bottom": SIAxisItem("bottom", unit=xunit)})
            plot.showGrid(x=True, y=True, alpha=0.25)
            plot.addLegend(offset=(8, 8))
            plot.getViewBox().setMouseMode(
                pg.ViewBox.RectMode if self.rect_zoom else pg.ViewBox.PanMode)
            # Pin the left axis to a fixed width so every stacked pane has the
            # same plot geometry. pyqtgraph aligns linked X axes in *screen*
            # space, so unequal left margins would otherwise skew the data
            # ranges and misalign the panes.
            plot.getAxis("left").setWidth(72)
            if first_vb is None:
                first_vb = plot.getViewBox()
                # Re-resolve level/edge points against the new visible range
                # whenever the (linked) X axis is panned or zoomed.
                first_vb.sigXRangeChanged.connect(
                    lambda *_: self._recompute_measure())
            else:
                plot.setXLink(self.panes[pane_ids[0]])
            # only the bottom pane carries the axis name label; the unit is
            # already shown on every pane's ticks via SIAxisItem.
            if r == len(pane_ids) - 1 and xv is not None:
                plot.setLabel("bottom", xv.name)
            self.panes[pid] = plot

        for t in visible:
            x, y = t.xy()
            curve = self.panes[t.pane].plot(
                x, y, pen=t.pen(), name=t.var.name)
            t.curve = curve

        self._install_cursors(pane_ids)
        self._recompute_measure()

    # -- view / zoom ---------------------------------------------------
    def _viewboxes(self) -> list[pg.ViewBox]:
        return [p.getViewBox() for p in self.panes.values()]

    def fit_all(self):
        """Fit every pane to its visible data (全体表示).

        Y is auto-ranged per pane; X is shared via the link, so set it once
        from the data extent (per-pane autoRange would disagree because the
        measurement markers nudge each pane's auto X-bounds)."""
        vbs = self._viewboxes()
        if not vbs:
            return
        for vb in vbs:
            vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)
        xs = [t.xy()[0] for t in self.traces if t.visible]
        if xs:
            lo = min(float(x.min()) for x in xs)
            hi = max(float(x.max()) for x in xs)
            pad = (hi - lo) * 0.02 or 1.0
            # set on every pane explicitly so they are exactly equal,
            # rather than relying on async link propagation
            for vb in vbs:
                vb.setXRange(lo - pad, hi + pad, padding=0)

    def _scale_all(self, factor: float):
        """Scale every pane about its centre. X is linked across panes, so
        scale X only once to avoid compounding through the link."""
        vbs = self._viewboxes()
        if not vbs:
            return
        vbs[0].scaleBy(x=factor, y=factor)   # this pane drives the linked X
        for vb in vbs[1:]:
            vb.scaleBy(x=None, y=factor)     # remaining panes: Y only

    def zoom_in(self):
        self._scale_all(0.8)

    def zoom_out(self):
        self._scale_all(1.25)

    def set_rect_zoom(self, on: bool):
        self.rect_zoom = on
        mode = pg.ViewBox.RectMode if on else pg.ViewBox.PanMode
        for vb in self._viewboxes():
            vb.setMouseMode(mode)
        self.statusBar().showMessage(
            "Rectangle zoom: 左ドラッグで囲んだ範囲にズーム" if on else
            "Pan mode: 左ドラッグで移動 / 右ドラッグ・ホイールでズーム "
            "(軸上のドラッグ・ホイールはその軸のみズーム)")

    def zoom_to_cursors(self):
        """Set the X range to the span between cursors A and B."""
        if not self.panes:
            return
        x0, x1 = sorted((self.cursorA.x(), self.cursorB.x()))
        if x0 == x1:
            return
        pad = (x1 - x0) * 0.05
        self.panes[next(iter(self.panes))].getViewBox().setXRange(
            x0 - pad, x1 + pad, padding=0)

    def _install_cursors(self, pane_ids):
        if not self.traces:
            return
        # seed cursor positions inside the data range on first use
        xall = self.raws[0].x if self.raws else np.array([0.0, 1.0])
        if self.cursorA.x() == 0.0 and self.cursorB.x() == 0.0:
            self.cursorA.set_x(float(xall[len(xall) // 3]))
            self.cursorB.set_x(float(xall[2 * len(xall) // 3]))
        for cur in (self.cursorA, self.cursorB):
            for pid in pane_ids:
                line = pg.InfiniteLine(
                    pos=cur.x(), angle=90, movable=True,
                    pen=pg.mkPen(cur.color, width=1, style=Qt.PenStyle.DashLine),
                    label=cur.name,
                    labelOpts={"color": cur.color, "position": 0.05})
                line.sigPositionChanged.connect(
                    lambda ln, c=cur: c._line_moved(ln))
                self.panes[pid].addItem(line)
                cur.lines.append(line)
            mk = pg.ScatterPlotItem(
                size=12, symbol="o", pen=pg.mkPen("w"),
                brush=pg.mkBrush(cur.color))
            # marker lives on the pane of its configured trace; added lazily
            cur.marker = mk

    # -- measurement ---------------------------------------------------
    def _refresh_measure_trace_combos(self):
        for tag in ("A", "B"):
            cb = self.cursor_widgets[tag]["trace"]
            prev = cb.currentData()
            cb.blockSignals(True)
            cb.clear()
            for t in self.traces:
                cb.addItem(t.display, t.uid)
            # restore previous selection if still present
            idx = cb.findData(prev)
            if idx >= 0:
                cb.setCurrentIndex(idx)
            cb.blockSignals(False)
        self._recompute_measure()

    def _trace_by_uid(self, uid) -> Trace | None:
        return next((t for t in self.traces if t.uid == uid), None)

    def _set_half_level(self, tag: str):
        w = self.cursor_widgets[tag]
        tr = self._trace_by_uid(w["trace"].currentData())
        if tr is None:
            return
        _, y = tr.xy()
        w["level"].setValue(float(np.nanmax(y)) / 2.0)

    def _on_cursor_moved(self):
        self._recompute_measure()

    def _resolve_point(self, tag: str) -> tuple[Trace | None, MeasurePoint | None]:
        w = self.cursor_widgets[tag]
        tr = self._trace_by_uid(w["trace"].currentData())
        if tr is None:
            return None, None
        cur = self.cursorA if tag == "A" else self.cursorB
        mode = w["mode"].currentData()
        x, y = tr.xy()
        mp = resolve(x, y, mode, cur.x(), w["level"].value(),
                     xlim=self._visible_xrange())
        return tr, mp

    def _visible_xrange(self) -> tuple[float, float] | None:
        """The X range currently shown in the plot, or None if no panes."""
        if not self.panes:
            return None
        vb = next(iter(self.panes.values())).getViewBox()
        (x0, x1), _ = vb.viewRange()
        return (x0, x1)

    def _recompute_measure(self):
        if not self.traces:
            self.result_lbl.setText("—")
            return
        trA, mpA = self._resolve_point("A")
        trB, mpB = self._resolve_point("B")
        self._draw_marker(self.cursorA, trA, mpA)
        self._draw_marker(self.cursorB, trB, mpB)
        self.result_lbl.setText(self._format_results(trA, mpA, trB, mpB))

    def _draw_marker(self, cur: Cursor, tr: Trace | None, mp: MeasurePoint | None):
        if cur.marker is None:
            return
        # detach from whichever pane currently hosts it
        if cur.marker_host is not None:
            cur.marker_host.removeItem(cur.marker)
            cur.marker_host = None
        if tr is None or mp is None or not mp.ok or tr.pane not in self.panes:
            cur.marker.setData([], [])
            return
        host = self.panes[tr.pane]
        host.addItem(cur.marker)
        cur.marker_host = host
        cur.marker.setData([mp.x], [mp.y])

    def _format_results(self, trA, mpA, trB, mpB) -> str:
        def fmt_pt(tag, tr, mp):
            if tr is None:
                return f"<b>{tag}</b>: (no trace)"
            if mp is None or not mp.ok:
                return (f"<b>{tag}</b> {tr.var.name}: "
                        f"<span style='color:#f55'>{mp.detail if mp else ''}</span>")
            return (f"<b>{tag}</b> {tr.var.name} [{mp.detail}]<br>"
                    f"&nbsp;&nbsp;x = {_eng(mp.x)}s, "
                    f"y = {_eng(mp.y)}{_short_unit(tr.var.quantity)}")

        lines = [fmt_pt("A", trA, mpA), fmt_pt("B", trB, mpB), "<hr>"]
        if (mpA and mpB and mpA.ok and mpB.ok):
            dt = mpB.x - mpA.x
            dy = mpB.y - mpA.y
            lines.append(f"<b>Δt</b> = {_eng(dt)}s")
            if dt != 0:
                lines.append(f"<b>1/Δt</b> = {_eng(1.0/dt)}Hz")
            dlabel = _delta_label(trA, trB)
            lines.append(f"<b>{dlabel}</b> = {_eng(dy)}"
                         f"{_short_unit(trB.var.quantity)}")
        return "<br>".join(lines)


# ---- formatting helpers ---------------------------------------------------
def _unit(quantity: str) -> str:
    return {"time": "s", "voltage": "V", "current": "A",
            "frequency": "Hz"}.get(quantity, "")


def _short_unit(quantity: str) -> str:
    return {"voltage": "V", "current": "A", "time": "s",
            "frequency": "Hz"}.get(quantity, "")


def _delta_label(trA: Trace, trB: Trace) -> str:
    qa, qb = trA.var.quantity, trB.var.quantity
    if qa == qb == "voltage":
        return "ΔV"
    if qa == qb == "current":
        return "ΔI"
    return "Δy"


def _eng(value: float) -> str:
    """Engineering-notation formatter (e.g. 1.50n, 250m)."""
    if value == 0 or not np.isfinite(value):
        return "0"
    prefixes = {-15: "f", -12: "p", -9: "n", -6: "µ", -3: "m",
                0: "", 3: "k", 6: "M", 9: "G", 12: "T"}
    exp = int(np.floor(np.log10(abs(value)) / 3) * 3)
    exp = max(-15, min(12, exp))
    mant = value / (10 ** exp)
    return f"{mant:.4g}{prefixes[exp]}"


def main(argv=None):
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    win = MainWindow()
    win.show()
    # auto-load any .raw paths passed on the command line
    files = [a for a in argv[1:] if a.endswith(".raw") and os.path.exists(a)]
    if files:
        for f in files:
            win.load_file(f)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
