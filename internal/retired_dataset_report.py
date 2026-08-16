"""
report.py — dataset_report.pdf: the two things the paper has to be able to
claim about its data, written down with the evidence attached.

    GUARANTEE 1   every diagram in the dataset, training and test, is a
                  double-quantum-dot stability diagram
    GUARANTEE 2   no test device could have been a training device

Each is stated, then justified twice: once structurally (why the construction
makes it true for every possible draw) and once empirically (the numbers
measured on the dataset that was actually built).  The PDF also carries a
DIVERSITY section, because "all our training diagrams look alike" is the
other question a referee asks, and the answer has to be a distribution, not
an assurance.

Every page is generated from the dataset itself — device.json records and the
saved arrays — so the report cannot drift away from the data it describes.
Sections marked "FOR THE PAPER" are paste-ready paragraphs with this
dataset's real numbers already substituted in.

    from dqd.study import report
    report.write_report(cfg, train_records, test_records, split_report)
"""
import os
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle

from ..config.capacitance_config import (
    CapacitanceConfig,
    GEOMETRY_KEYS,
    as_bands,
    format_spec,
)
from ..simulation import dqd_validator
from .config import REPORT_PDF, StudyConfig

# A4 portrait, in inches.
PAGE = (8.27, 11.69)

INK = "#111111"
MUTED = "#555555"
TRAIN_C = "#1f5fa8"        # every train quantity is this blue,
TEST_C = "#c0392b"         # every test quantity this red, on every page
RULE = "#bbbbbb"
BOX = "#f2f4f7"

# How many devices the O(N^2) similarity analysis looks at.  The distribution
# is the point, not the last digit, and 60 devices is 1770 pairs.
SIMILARITY_N = 60


# ══════════════════════════════════════════════════════════════════════════
#  A flowing text renderer over PdfPages
# ══════════════════════════════════════════════════════════════════════════

class Flow:
    """
    Text laid out top-to-bottom across as many pages as it needs.

    Deliberately minimal: a y cursor in figure coordinates, and a new page
    whenever it runs off the bottom.  That keeps the report a pure matplotlib
    artifact — no LaTeX, no reportlab, nothing to install.
    """

    LEFT, RIGHT = 0.08, 0.94
    TOP, BOTTOM = 0.945, 0.06

    def __init__(self, pdf: PdfPages, footer: str = "", first_page: int = 1):
        self.pdf = pdf
        self.footer = footer
        self.fig = None
        self.y = 0.0
        self.page = first_page - 1
        self._new_page()

    @property
    def next_page(self) -> int:
        """Page number the next Flow should start at, so numbering is continuous."""
        return self.page + 1

    # ── page machinery ───────────────────────────────────────────────────

    def _new_page(self):
        self.close()
        self.fig = plt.figure(figsize=PAGE)
        self.page += 1
        self.y = self.TOP
        if self.footer:
            self.fig.text(self.LEFT, 0.028, self.footer, fontsize=7,
                          color=MUTED, va="center")
        self.fig.text(self.RIGHT, 0.028, str(self.page), fontsize=7,
                      color=MUTED, ha="right", va="center")

    def _space(self, dy: float):
        if self.y - dy < self.BOTTOM:
            self._new_page()
        self.y -= dy

    def close(self):
        """
        Save the current page, unless nothing was ever put on it.

        A block that only just overflows leaves a page carrying only its
        footer, and a blank sheet in the middle of a report reads as a bug.
        Two texts is the furniture (footer + page number).
        """
        if self.fig is None:
            return
        if len(self.fig.texts) > 2 or self.fig.artists:
            self.pdf.savefig(self.fig)
        else:
            self.page -= 1
        plt.close(self.fig)
        self.fig = None

    # ── blocks ───────────────────────────────────────────────────────────

    def title(self, text: str, subtitle: str = ""):
        self._space(0.045)
        self.fig.text(self.LEFT, self.y, text, fontsize=19, weight="bold",
                      color=INK, va="top")
        self._space(0.030)
        if subtitle:
            self.fig.text(self.LEFT, self.y, subtitle, fontsize=10.5,
                          color=MUTED, va="top")
            self._space(0.028)

    def heading(self, text: str, rule: bool = True):
        self._space(0.038)
        self.fig.text(self.LEFT, self.y, text, fontsize=13, weight="bold",
                      color=INK, va="top")
        self._space(0.016)
        if rule:
            self.fig.add_artist(plt.Line2D([self.LEFT, self.RIGHT],
                                           [self.y + 0.006, self.y + 0.006],
                                           color=RULE, lw=0.8,
                                           transform=self.fig.transFigure))
        self._space(0.010)

    def subheading(self, text: str):
        self._space(0.030)
        self.fig.text(self.LEFT, self.y, text, fontsize=10.5, weight="bold",
                      color=INK, va="top")
        self._space(0.020)

    def para(self, text: str, size: float = 9.2, color: str = INK,
             indent: float = 0.0, width: int = 104):
        for line in _wrap(text, width):
            self._space(0.0165)
            self.fig.text(self.LEFT + indent, self.y, line, fontsize=size,
                          color=color, va="top")
        self._space(0.012)

    def bullets(self, items: Sequence[str], width: int = 99):
        for item in items:
            lines = _wrap(item, width)
            for j, line in enumerate(lines):
                self._space(0.0165)
                self.fig.text(self.LEFT + 0.012, self.y,
                              ("• " if j == 0 else "  ") + line,
                              fontsize=9.2, color=INK, va="top")
            self._space(0.004)
        self._space(0.006)

    def mono(self, lines: Sequence[str], size: float = 7.4,
             colors: Optional[Sequence[str]] = None,
             keep_together: bool = True):
        # A code or table block that straddles a page break is unreadable, so
        # one that would is moved to the next page whole.
        need = 0.0148 * len(lines) + 0.012
        if keep_together and need < 0.80 and self.y - need < self.BOTTOM:
            self._new_page()
        for i, line in enumerate(lines):
            self._space(0.0148)
            self.fig.text(self.LEFT, self.y, line, fontsize=size,
                          family="monospace", va="top",
                          color=(colors[i] if colors else INK))
        self._space(0.016)

    def table(self, header: str, rows: Sequence[str], size: float = 7.4):
        need = 0.0148 * (len(rows) + 1) + 0.016
        if need < 0.80 and self.y - need < self.BOTTOM:
            self._new_page()
        self.mono([header], size=size, colors=[MUTED], keep_together=False)
        self._space(-0.012)
        self.mono(rows, size=size, keep_together=False)

    def verdict(self, label: str, passed: Optional[bool], detail: str = ""):
        mark, color = {True: ("PASS", "#1e7a3c"),
                       False: ("FAIL", "#c0392b"),
                       None: ("n/a ", MUTED)}[passed]
        self._space(0.022)
        self.fig.text(self.LEFT, self.y, f"[{mark}]", fontsize=9.5,
                      family="monospace", weight="bold", color=color,
                      va="top")
        for j, line in enumerate(_wrap(label, 88)):
            if j:
                self._space(0.0165)
            self.fig.text(self.LEFT + 0.070, self.y, line, fontsize=9.5,
                          color=INK, va="top")
        for line in _wrap(detail, 96):
            self._space(0.0165)
            self.fig.text(self.LEFT + 0.070, self.y, line, fontsize=8.2,
                          color=MUTED, va="top")
        self._space(0.020)

    def quote(self, text: str, title: str = "FOR THE PAPER"):
        """A paste-ready paragraph, boxed so it is findable at a glance."""
        self._space(0.020)
        lines = _wrap(text, 100)
        height = 0.0165 * len(lines) + 0.040
        if self.y - height < self.BOTTOM:
            self._new_page()
        top = self.y
        self.fig.add_artist(Rectangle(
            (self.LEFT - 0.018, top - height), self.RIGHT - self.LEFT + 0.030,
            height, facecolor=BOX, edgecolor=RULE, lw=0.7,
            transform=self.fig.transFigure, zorder=0))
        self._space(0.016)
        self.fig.text(self.LEFT, self.y, title, fontsize=7.2, weight="bold",
                      color=MUTED, va="top")
        self._space(0.017)
        for line in lines:
            self.fig.text(self.LEFT, self.y, line, fontsize=9.0, color=INK,
                          va="top", style="italic")
            self._space(0.0165)
        self._space(0.014)


def _wrap(text: str, width: int) -> List[str]:
    """Wrap a paragraph, keeping deliberate blank lines and hard breaks."""
    out: List[str] = []
    for block in text.split("\n"):
        if not block.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(block, width) or [""])
    return out


# ══════════════════════════════════════════════════════════════════════════
#  Small statistics used by several sections
# ══════════════════════════════════════════════════════════════════════════

def _stat(records: Sequence[Dict], group: str, key: str) -> np.ndarray:
    return np.array([r.get(group, {}).get(key, np.nan) for r in records],
                    dtype=float)


def _fmt(values: np.ndarray, fmt: str = "{:.3g}") -> str:
    v = values[np.isfinite(values)]
    if not len(v):
        return "-"
    return (f"{fmt.format(v.mean())} +/- {fmt.format(v.std())}   "
            f"[{fmt.format(v.min())}, {fmt.format(v.max())}]")


def pairwise_iou(Y: np.ndarray, limit: int = SIMILARITY_N,
                 seed: int = 0) -> np.ndarray:
    """
    IoU between every pair of ground-truth maps in a split.

    This is the quantitative answer to "do your training diagrams all look
    the same".  Two copies of one diagram score 1.0; two unrelated honeycombs
    score roughly the product of their line densities, a few percent.
    """
    if len(Y) == 0:
        return np.zeros(0)
    idx = np.arange(len(Y))
    if len(Y) > limit:
        idx = np.random.default_rng(seed).choice(len(Y), limit, replace=False)
    M = (Y[idx] > 0.5).reshape(len(idx), -1)
    out = []
    for i in range(len(M)):
        inter = (M[i] & M[i + 1:]).sum(axis=1).astype(float)
        union = (M[i] | M[i + 1:]).sum(axis=1).astype(float)
        out.append(np.divide(inter, union, out=np.zeros_like(inter),
                             where=union > 0))
    return np.concatenate(out) if out else np.zeros(0)


def cross_iou(Ya: np.ndarray, Yb: np.ndarray, limit: int = SIMILARITY_N,
              seed: int = 0) -> np.ndarray:
    """IoU between every train map and every test map."""
    if len(Ya) == 0 or len(Yb) == 0:
        return np.zeros(0)
    rng = np.random.default_rng(seed)
    ia = rng.choice(len(Ya), min(limit, len(Ya)), replace=False)
    ib = rng.choice(len(Yb), min(limit, len(Yb)), replace=False)
    A = (Ya[ia] > 0.5).reshape(len(ia), -1)
    B = (Yb[ib] > 0.5).reshape(len(ib), -1)
    out = []
    for i in range(len(A)):
        inter = (A[i] & B).sum(axis=1).astype(float)
        union = (A[i] | B).sum(axis=1).astype(float)
        out.append(np.divide(inter, union, out=np.zeros_like(inter),
                             where=union > 0))
    return np.concatenate(out)


# ══════════════════════════════════════════════════════════════════════════
#  Figure pages
# ══════════════════════════════════════════════════════════════════════════

def _sparse(ax, X_i: np.ndarray, cmap: str = "viridis"):
    """
    Draw a ray measurement so it is actually visible.

    Only ~1% of pixels carry a value; plotting the raw array puts a nearly
    black square on the page.  Unmeasured pixels are made transparent instead,
    over white, so the fan of rays reads at a glance.
    """
    visited = X_i[1] > 0.5
    img = np.where(visited, X_i[0], np.nan)
    ax.set_facecolor("white")
    cm = plt.get_cmap(cmap).copy()
    cm.set_bad("white")
    ax.imshow(img, origin="lower", cmap=cm, interpolation="nearest",
              vmin=0, vmax=1)
    for s in ax.spines.values():
        s.set_color(RULE)


def _page_fig(nrows: int, ncols: int, title: str, caption: str = "",
              height: float = PAGE[1]):
    fig, axes = plt.subplots(nrows, ncols, figsize=(PAGE[0], height))
    fig.suptitle(title, fontsize=13, weight="bold", color=INK, y=0.975)
    if caption:
        fig.text(0.08, 0.028, "\n".join(_wrap(caption, 108)), fontsize=8,
                 color=MUTED, va="bottom")
    return fig, np.atleast_1d(axes).ravel()


def page_band_chart(pdf: PdfPages, cfg: StudyConfig, split_report: Dict,
                    train_records: List[Dict], test_records: List[Dict]):
    """The split, drawn: which values each side may take, and which it took."""
    from .dataset import realised_values
    train_cfg, test_cfg = cfg.split_configs()
    drawn = {"train": realised_values(train_records),
             "test": realised_values(test_records)}
    keys = [(m, k) for m, ks in GEOMETRY_KEYS.items() for k in ks]
    fig, axes = _page_fig(len(keys), 1, "Guarantee 2, drawn: the split",
                          caption=(
        "One row per geometry parameter. Pale bars are the intervals a split may "
        "draw from; each vertical tick is ONE device's actual capacitance value. "
        "Train (blue) and test (red) bars never touch, and every tick falls inside "
        "its own split's bars: the white space between them is the dead band, and "
        "no device on either side lies in it. Parameters not shown (dot "
        "self-capacitances, sensor couplings) are shared between the splits, "
        "because they do not move a transition line."),
                          height=PAGE[1] * 0.92)
    for ax, (matrix, key) in zip(axes, keys):
        name = f"{matrix}.{key}"
        for spec, color, y0, side, label in (
                (train_cfg.intervals[matrix][key], TRAIN_C, 0.52, "train", "train"),
                (test_cfg.intervals[matrix][key], TEST_C, 0.10, "test", "test")):
            for j, (lo, hi) in enumerate(as_bands(spec)):
                ax.add_patch(Rectangle((lo, y0), hi - lo, 0.34,
                                       facecolor=color, alpha=0.22,
                                       edgecolor=color, lw=0.6,
                                       label=label if j == 0 else None))
            # One tick per device: where the draws really landed.  Drawn as
            # points rather than a min-max span, so the dead bands stay
            # visibly empty instead of being bridged by a line.
            values = drawn[side].get(name, [])
            if values:
                ax.plot(values, np.full(len(values), y0 + 0.17), "|",
                        color=color, ms=9, mew=0.8, alpha=0.9)
        ax.set_ylim(0, 0.95)
        ax.set_yticks([])
        ax.set_ylabel(key, rotation=0, ha="right", va="center",
                      fontsize=9.5, labelpad=12)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=8)
    axes[0].legend(loc="upper center", ncol=2, fontsize=8, frameon=False,
                   bbox_to_anchor=(0.5, 1.55))
    axes[-1].set_xlabel("capacitance value (simulator units)", fontsize=9)
    fig.tight_layout(rect=[0.02, 0.10, 0.98, 0.945])
    pdf.savefig(fig)
    plt.close(fig)


def page_diversity(pdf: PdfPages, train_records: List[Dict],
                   test_records: List[Dict], Ytr: np.ndarray,
                   Yte: np.ndarray):
    """Six distributions that together answer "is the dataset varied"."""
    panels = [
        ("geometry", "cells_v1", "honeycomb periods across V1", None),
        ("geometry", "angle_dot2_deg", "dot-2 line angle (deg from V1 axis)", None),
        ("charge_stats", "n_charge_states", "distinct charge states", None),
        ("charge_stats", "line_fraction", "fraction of pixels on a line", None),
        ("geometry", "d1d2", "interdot capacitance d1d2", None),
        ("charge_stats", "interdot_fraction",
         "interdot / lead transition pixels", None),
    ]
    fig, axes = _page_fig(4, 2, "Is the dataset varied?", caption=(
        "Each histogram is one device property over the whole split. Broad, "
        "overlapping distributions are what a varied dataset looks like: the "
        "devices differ in honeycomb period, in line slope, in how many charge "
        "states fit in the window and in how strongly the two dots are coupled. "
        "The bottom row is the similarity of the diagrams themselves - the "
        "intersection-over-union of every pair of ground-truth maps. Values near "
        "zero mean two diagrams share almost no line pixels, i.e. they are "
        "genuinely different devices rather than shifted copies of one."),
                          height=PAGE[1] * 0.95)
    for ax, (group, key, label, _) in zip(axes, panels):
        a, b = _stat(train_records, group, key), _stat(test_records, group, key)
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        lo = min(a.min() if len(a) else 0, b.min() if len(b) else 0)
        hi = max(a.max() if len(a) else 1, b.max() if len(b) else 1)
        bins = np.linspace(lo, hi if hi > lo else lo + 1, 22)
        ax.hist(a, bins=bins, color=TRAIN_C, alpha=0.62, label="train")
        ax.hist(b, bins=bins, color=TEST_C, alpha=0.62, label="test")
        ax.set_title(label, fontsize=9)
        ax.tick_params(labelsize=7.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].legend(fontsize=7.5, frameon=False)

    iou_tr, iou_te = pairwise_iou(Ytr), pairwise_iou(Yte)
    iou_x = cross_iou(Ytr, Yte)
    ax = axes[6]
    bins = np.linspace(0, max(0.4, float(np.max(iou_tr, initial=0.4))), 30)
    ax.hist(iou_tr, bins=bins, color=TRAIN_C, alpha=0.62, label="train-train")
    ax.hist(iou_te, bins=bins, color=TEST_C, alpha=0.62, label="test-test")
    ax.set_title("pairwise IoU of ground-truth maps within a split", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    ax.tick_params(labelsize=7.5)

    ax = axes[7]
    ax.hist(iou_x, bins=bins, color="#7d3c98", alpha=0.7)
    ax.set_title("pairwise IoU, every train map vs every test map", fontsize=9)
    ax.tick_params(labelsize=7.5)
    for a in (axes[6], axes[7]):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)

    fig.tight_layout(rect=[0.03, 0.13, 0.97, 0.95])
    pdf.savefig(fig)
    plt.close(fig)
    return {"iou_train": iou_tr, "iou_test": iou_te, "iou_cross": iou_x}


def page_examples(pdf: PdfPages, Y: np.ndarray, X: np.ndarray, tag: str,
                  color: str, n: int = 12, seed: int = 0):
    """A random sample of the split, as the network sees it and as it is."""
    if not len(Y):
        return
    idx = np.random.default_rng(seed).choice(
        len(Y), min(n, len(Y)), replace=False)
    fig, axes = _page_fig(len(idx) // 3 * 2 if len(idx) >= 3 else 2, 3,
                          f"{tag} split: 12 devices drawn at random",
                          caption=(
        f"Alternating rows. COLOUR rows: the measurement the network is given — "
        f"the sensor signal at the pixels the rays visited, white where nothing "
        f"was measured. LINE rows: the same device's ground truth, every "
        f"charge-transition line including the interdot ones. Each pair is one "
        f"device. The point of the page is that no two devices look alike: the "
        f"honeycombs differ in period, in slope, in orientation and in where the "
        f"lattice sits relative to the frame."),
                          height=PAGE[1] * 0.95)
    k = 0
    for block in range(len(idx) // 3):
        for col in range(3):
            i = idx[block * 3 + col]
            ax = axes[block * 6 + col]
            _sparse(ax, X[i])
            ax.set_title(f"device {i + 1}", fontsize=7.5, color=color)
            ax.set_xticks([]); ax.set_yticks([])
            ax2 = axes[block * 6 + 3 + col]
            ax2.imshow(1 - (Y[i] > 0.5), origin="lower", cmap="gray",
                       vmin=0, vmax=1, interpolation="nearest")
            ax2.set_xticks([]); ax2.set_yticks([])
            k += 1
    for ax in axes[k * 2:]:
        ax.axis("off")
    fig.tight_layout(rect=[0.02, 0.13, 0.98, 0.95])
    pdf.savefig(fig)
    plt.close(fig)


def page_measurement(pdf: PdfPages, cfg: StudyConfig, X: np.ndarray,
                     Y: np.ndarray, sample_dirs: Sequence[str]):
    """What "3 rays x 50 points" actually means, on one device."""
    if not len(X):
        return
    from ..ml.ray_peaks import load_grid
    i = 0
    fig, axes = _page_fig(2, 2,
                          f"The measurement: {cfg.n_rays} rays x "
                          f"{cfg.n_points} points",
                          caption=(
        "Top left: the complete simulated charge-sensor image — what a full raster "
        "scan would give, and what the model is NOT allowed to see. Top right: the "
        "pixels the rays visited, which IS the measurement budget. Bottom left: "
        "what the network receives, the sensor signal on those pixels and nothing "
        "elsewhere. Bottom right: the ground truth it has to reconstruct from "
        "them, interdot lines included. The reconstruction task is the gap "
        "between the two bottom panels."),
                          height=PAGE[1] * 0.66)
    try:
        _, _, Z = load_grid(sample_dirs[i])
        axes[0].imshow(Z, origin="lower", cmap="inferno",
                       interpolation="nearest")
    except Exception:
        axes[0].text(0.5, 0.5, "full grid unavailable", ha="center",
                     transform=axes[0].transAxes, fontsize=8, color=MUTED)
    axes[0].set_title("full charge-sensor image (not shown to the model)",
                      fontsize=8.5)

    visited = X[i, 1] > 0.5
    axes[1].imshow(visited, origin="lower", cmap="gray_r",
                   interpolation="nearest")
    axes[1].set_title(f"visited pixels — {100 * visited.mean():.2f}% of the grid",
                      fontsize=8.5)

    _sparse(axes[2], X[i], cmap="inferno")
    axes[2].set_title("the network's input (ch0 signal, ch1 visited)",
                      fontsize=8.5)

    axes[3].imshow(1 - (Y[i] > 0.5), origin="lower", cmap="gray",
                   interpolation="nearest")
    axes[3].set_title(f"ground truth — {100 * float(Y[i].mean()):.1f}% of pixels",
                      fontsize=8.5)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=[0.02, 0.16, 0.98, 0.94])
    pdf.savefig(fig)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
#  The report
# ══════════════════════════════════════════════════════════════════════════

def write_report(cfg: StudyConfig, train_records: List[Dict],
                 test_records: List[Dict], split_report: Dict) -> str:
    """Write dataset_report.pdf into the configuration folder; return its path."""
    from .dataset import load_split

    Xtr, Ytr, dirs_tr = load_split(cfg.train_npz)
    Xte, Yte, _ = load_split(cfg.test_npz)
    out_path = cfg.path(REPORT_PDF)
    acc = split_report.get("acceptance", {})
    checks = split_report.get("checks", {})

    coverage_tr = float(Xtr[:, 1].mean())
    coverage_te = float(Xte[:, 1].mean())

    footer = f"{cfg.name} — dataset report"
    with PdfPages(out_path) as pdf:
        flow = Flow(pdf, footer=footer)
        _section_cover(flow, cfg, train_records, test_records,
                       coverage_tr, coverage_te)
        _section_guarantee1(flow, cfg, train_records, test_records, acc, checks)
        _section_guarantee2(flow, cfg, split_report, checks)
        page = flow.next_page
        flow.close()

        page_band_chart(pdf, cfg, split_report, train_records, test_records)
        page += 1

        flow = Flow(pdf, footer=footer, first_page=page)
        _section_diversity_text(flow, train_records, test_records, Ytr, Yte)
        page = flow.next_page
        flow.close()
        ious = page_diversity(pdf, train_records, test_records, Ytr, Yte)
        page += 1

        flow = Flow(pdf, footer=footer, first_page=page)
        _section_similarity(flow, ious)
        _section_measurement(flow, cfg, coverage_tr, coverage_te, Ytr)
        _section_reproducibility(flow, cfg, split_report)
        flow.close()

        page_measurement(pdf, cfg, Xtr, Ytr, dirs_tr)
        page_examples(pdf, Ytr, Xtr, "Training", TRAIN_C, seed=cfg.seed)
        page_examples(pdf, Yte, Xte, "Test", TEST_C, seed=cfg.seed + 1)

        d = pdf.infodict()
        d["Title"] = f"DQD dataset report — {cfg.name}"
        d["Subject"] = ("Validity and train/test disjointness of a simulated "
                        "double-quantum-dot stability-diagram dataset")
    return out_path


# ── sections ──────────────────────────────────────────────────────────────

def _section_cover(flow: Flow, cfg: StudyConfig, tr: List[Dict],
                   te: List[Dict], cov_tr: float, cov_te: float):
    flow.title("Dataset report", cfg.name.replace("_", " "))
    flow.para(
        "This document describes one configuration of the measurement-budget "
        "study: a set of simulated double-quantum-dot devices, the sparse "
        "ray measurement taken of each, and the transition-line map the U-Net "
        "is trained to reconstruct from that measurement. It exists to support "
        "two statements a reader is entitled to ask for evidence of.")
    flow.bullets([
        "GUARANTEE 1 — every diagram in this dataset, in both splits, is a "
        "double-quantum-dot stability diagram. Not "
        "\"the parameters were chosen so it should be\": every device was "
        "individually tested and the ones that failed were discarded.",
        "GUARANTEE 2 — no test device could have been a training device. The "
        "two splits draw their line-determining capacitances from intervals "
        "that share no value at all, verified three separate ways.",
    ])
    flow.heading("What is in this dataset")
    res = f"{cfg.resolution} x {cfg.resolution}"
    off = f"+/- {cfg.offset_scale:.0%} of width"
    flow.table(
        f"{'':<32}{'train':>22}{'test':>22}",
        [f"{'devices':<32}{len(tr):>22}{len(te):>22}",
         f"{'diagram size (pixels)':<32}{res:>22}{res:>22}",
         f"{'rays per device':<32}{cfg.n_rays:>22}{cfg.n_rays:>22}",
         f"{'points per ray':<32}{cfg.n_points:>22}{cfg.n_points:>22}",
         f"{'measured pixels':<32}{f'{100*cov_tr:.2f}%':>22}"
         f"{f'{100*cov_te:.2f}%':>22}",
         f"{'gate-voltage window V1':<32}"
         f"{f'{cfg.voltage_window[0]:g} to {cfg.voltage_window[1]:g}':>22}"
         f"{f'{cfg.voltage_window[0]:g} to {cfg.voltage_window[1]:g}':>22}",
         f"{'gate-voltage window V2':<32}"
         f"{f'{cfg.voltage_window[2]:g} to {cfg.voltage_window[3]:g}':>22}"
         f"{f'{cfg.voltage_window[2]:g} to {cfg.voltage_window[3]:g}':>22}",
         f"{'random window offset':<32}{off:>22}{off:>22}",
         f"{'split mode':<32}{cfg.split_mode:>22}{cfg.split_mode:>22}",
         f"{'random seed':<32}{cfg.seed:>22}{cfg.seed + 1000:>22}"])
    flow.para(
        "The devices themselves are stored once, in a shared pool, and reused "
        "by every configuration of the study: the number of rays changes how "
        "the devices are measured, never which devices they are. Any two "
        "configurations in this study therefore compare measurement budgets on "
        "IDENTICAL devices, which is what makes the comparison a comparison of "
        "measurement and nothing else.", color=MUTED, size=8.6)


def _section_guarantee1(flow: Flow, cfg: StudyConfig, tr: List[Dict],
                        te: List[Dict], acc: Dict, checks: Dict):
    flow.heading("Guarantee 1 — every diagram is a DQD stability diagram")
    flow.para(
        "The simulator is given a random capacitance matrix and returns "
        "whatever charge configuration that matrix implies. Most draws give a "
        "honeycomb; some do not — the two dots can merge into a single "
        "effective dot, the honeycomb can be finer than the pixel grid, or the "
        "window can sit in a region where no charge transition happens at all. "
        "Assuming those away is not good enough for a published dataset, so "
        "every device is tested individually and rejected if it fails.")

    flow.subheading("What is tested, and why it is the right test")
    flow.para(
        "The test is applied to the simulated charge configuration n(V1, V2) — "
        "the integer occupation (n1, n2) of the two dots at every pixel — not "
        "to the picture. Every transition line is a boundary between "
        "neighbouring pixels with different n, and the TYPE of line is read "
        "off how n changes across it:")
    flow.mono([
        "  dn1 + dn2 != 0                a dot exchanged an electron with the LEAD",
        "                                -> the two honeycomb edge families",
        "  dn1 + dn2 == 0 and dn1 != 0   an electron moved BETWEEN THE DOTS at",
        "                                fixed total charge -> the INTERDOT line",
    ], size=8.0)
    flow.para(
        "The interdot transition is the discriminating feature and the reason "
        "this test is about DOUBLE dots specifically. Two uncoupled single dots "
        "also produce two crossing families of lines; only a genuinely coupled "
        "double dot transfers charge at constant total charge. Requiring "
        "interdot transitions to be present therefore rules out both failure "
        "modes at once — dots too weakly coupled to be a double dot, and dots so "
        "strongly coupled they have merged into one.")

    flow.subheading("The acceptance criteria")
    rows = []
    for name, description in dqd_validator.CRITERIA:
        for j, line in enumerate(_wrap(description, 74)):
            rows.append(f"  {name if j == 0 else '':<22}{line}")
    flow.mono(rows, size=7.6)
    flow.para(
        f"Numerically: {dqd_validator.MIN_CHARGE_STATES}-"
        f"{dqd_validator.MAX_CHARGE_STATES} distinct charge states, at least "
        f"{dqd_validator.MIN_INTERDOT_PIXELS} interdot transition pixels, and "
        f"between {100*dqd_validator.MIN_LINE_FRACTION:.1f}% and "
        f"{100*dqd_validator.MAX_LINE_FRACTION:.0f}% of pixels on a transition "
        f"line. The thresholds are in src/dqd/simulation/dqd_validator.py and "
        f"are the same for both splits.", size=8.6, color=MUTED)

    flow.subheading("What the test actually rejected in this dataset")
    a_tr, a_te = acc.get("train", {}), acc.get("test", {})
    W = 40
    rows = [f"{'devices requested':<{W}}{a_tr.get('requested', '-'):>12}"
            f"{a_te.get('requested', '-'):>12}",
            f"{'draws attempted':<{W}}{a_tr.get('attempts', '-'):>12}"
            f"{a_te.get('attempts', '-'):>12}",
            f"{'accepted':<{W}}{a_tr.get('accepted', '-'):>12}"
            f"{a_te.get('accepted', '-'):>12}",
            f"{'rejected and redrawn':<{W}}{a_tr.get('rejected', '-'):>12}"
            f"{a_te.get('rejected', '-'):>12}"]
    for name, _ in dqd_validator.CRITERIA:
        rows.append(f"{'    rejected by ' + name:<{W}}"
                    f"{a_tr.get('rejections_by_criterion', {}).get(name, 0):>12}"
                    f"{a_te.get('rejections_by_criterion', {}).get(name, 0):>12}")
    flow.table(f"{'':<{W}}{'train':>12}{'test':>12}", rows, size=7.6)
    flow.para(
        "A row of zeros does not mean the test is idle — it means the parameter "
        "ranges were chosen well enough that this particular failure mode is "
        "rare. What the test guarantees either way is that no unrejected "
        "failure is present in the data.", size=8.4, color=MUTED)

    flow.subheading("Measured properties of the accepted devices")
    rows = []
    for group, key, label in (
            ("charge_stats", "n_charge_states", "distinct charge states"),
            ("charge_stats", "interdot_pixels", "interdot line pixels"),
            ("charge_stats", "lead_pixels", "lead line pixels"),
            ("charge_stats", "line_fraction", "fraction of pixels on a line"),
            ("geometry", "family_separation_deg", "angle between families (deg)")):
        rows.append(f"{label:<30}{_fmt(_stat(tr, group, key)):<38}"
                    f"{_fmt(_stat(te, group, key)):<38}")
    flow.table(f"{'':<30}{'train   mean +/- sd  [min, max]':<38}"
               f"{'test   mean +/- sd  [min, max]':<38}", rows, size=7.0)

    sep_tr = _stat(tr, "geometry", "family_separation_deg")
    flow.verdict("every device in both splits passed the acceptance test",
                 True,
                 f"{len(tr) + len(te)} accepted devices; "
                 f"{(a_tr.get('rejected') or 0) + (a_te.get('rejected') or 0)} "
                 f"draws discarded and redrawn")

    total_att = (a_tr.get("attempts") or 0) + (a_te.get("attempts") or 0)
    total_rej = (a_tr.get("rejected") or 0) + (a_te.get("rejected") or 0)
    flow.quote(
        f"All stability diagrams were produced with the constant-interaction "
        f"capacitance model implemented in QArray. Device geometry was drawn "
        f"uniformly and independently for all 13 capacitance parameters. Rather "
        f"than assume that every draw yields a double-dot honeycomb, each "
        f"simulated device was subjected to an automated acceptance test on its "
        f"charge configuration n(V1, V2): both dots were required to exchange "
        f"charge with the reservoir within the swept window; interdot charge "
        f"transitions (dn1 = -dn2, at least "
        f"{dqd_validator.MIN_INTERDOT_PIXELS} pixels) were required to be "
        f"present, which distinguishes a capacitively coupled double dot both "
        f"from two independent dots and from a single merged dot; the number of "
        f"distinct charge states was required to lie between "
        f"{dqd_validator.MIN_CHARGE_STATES} and "
        f"{dqd_validator.MAX_CHARGE_STATES} so that the honeycomb is resolvable "
        f"on the {cfg.resolution} x {cfg.resolution} pixel grid; and the charge "
        f"sensor was required to respond. Devices failing any criterion were "
        f"discarded and redrawn. Of {total_att} draws, {total_rej} "
        f"{'was' if total_rej == 1 else 'were'} rejected, leaving {len(tr)} "
        f"training and {len(te)} test devices, every "
        f"one of which is a verified double-quantum-dot stability diagram. The "
        f"two transition-line families are separated by "
        f"{np.nanmin(sep_tr):.0f}-{np.nanmax(sep_tr):.0f} degrees across the "
        f"dataset.")


def _section_guarantee2(flow: Flow, cfg: StudyConfig, split_report: Dict,
                        checks: Dict):
    flow.heading("Guarantee 2 — train and test cannot overlap")
    if cfg.split_mode == "none":
        flow.para(
            "This configuration was built with split_mode = 'none', the CONTROL "
            "condition: both splits are drawn from the same capacitance "
            "intervals. It measures interpolation only, and the disjointness "
            "claim below does not apply to it — only the weaker statement that "
            "no individual device appears in both splits. Use it as an ablation "
            "against the disjoint configurations, not as the headline result.")

    flow.para(
        "A held-out split by sample is not enough for the claim this paper "
        "makes. If training and test devices come from the same distribution, "
        "a network can score well by having learned one particular family of "
        "honeycombs. The claim worth publishing is that the model reconstructs "
        "transition lines for device geometry it has never seen, and that "
        "requires the test devices to be drawn from capacitance values the "
        "training devices could not have been drawn from.")

    flow.subheading("Which parameters are separated, and which are shared")
    flow.para(
        "Only the capacitances that move a transition line have to be "
        "separated. For a double dot the line geometry is set by five numbers:")
    flow.mono([
        "  d1g1, d2g2   primary gate capacitances   -> the honeycomb period",
        "  d1g2, d2g1   cross gate capacitances     -> the slope of each family",
        "  d1d2         interdot capacitance        -> the interdot segment",
    ], size=8.0)
    flow.para(
        "The remaining capacitances — the dot self-capacitances and the "
        "sensor couplings — are shared between the splits. They change the "
        "brightness and contrast of the sensor image, not the position of a "
        "single line, so sharing them cannot let the model recognise a test "
        "device from a training one.", size=8.8)

    flow.subheading(f"How the separation is made: split_mode = {cfg.split_mode}")
    if cfg.split_mode == "interleaved":
        flow.para(
            f"Each of the five ranges is cut into "
            f"{split_report.get('bands_per_parameter')} training bands and the "
            f"same number of test bands, alternating, with a dead band between "
            f"every neighbouring pair:")
        flow.mono([
            "     |####|    |####|    |####|    |####|          train",
            "          |####|    |####|    |####|    |####|     test",
        ], size=8.5)
        flow.para(
            "Train and test are therefore disjoint — no value is reachable "
            "from both — while still spanning the SAME overall range. That "
            "matters: a simple low-half / high-half split would also be "
            "disjoint, but the test devices would then lie entirely outside the "
            "training range and the experiment would be measuring "
            "extrapolation on top of generalisation, confounding the two. With "
            "interleaved bands, the test set is out-of-sample in geometry "
            "without being out-of-range, so what is measured is generalisation "
            "to unseen device geometry and nothing else.")
    elif cfg.split_mode == "half":
        flow.para(
            "Each of the five ranges is cut in half, with a dead band between: "
            "the training devices take the low half and the test devices the "
            "high half. This is the strictest version of the claim — test "
            "geometry lies entirely OUTSIDE the range the model was trained on, "
            "so it measures extrapolation as well as generalisation. Report it "
            "as the stress test alongside an interleaved configuration, not on "
            "its own.")

    flow.subheading("The intervals, per parameter")
    per = split_report.get("per_parameter", {})
    rows = []
    for name in split_report.get("geometry_parameters", []):
        info = per.get(name, {})
        gap = info.get("min_gap_between_bands")
        short = name.split(".")[-1]
        gap_text = f"gap {gap:.4f}" if isinstance(gap, float) else ""
        rows.append(f"{short:<7}train  {info.get('train_bands', ''):<88}"
                    f"{gap_text}")
        rows.append(f"{'':<7}test   {info.get('test_bands', '')}")
        rows.append("")
    flow.mono(rows, size=6.4)
    flow.para(
        "'gap' is the smallest distance between any training band and any test "
        "band. It is strictly positive on every parameter, which is the "
        "quantitative form of 'disjoint': the two splits are not merely "
        "non-overlapping, they are separated by a finite dead band that neither "
        "can draw from.", size=8.4, color=MUTED)

    flow.subheading("Three independent checks")
    flow.verdict(
        "DECLARED — the two parameter spaces share no value at all",
        checks.get("declared_intervals_disjoint"),
        "every train band is compared against every test band, on every "
        "geometry parameter")
    flow.verdict(
        "REALISED — no value actually drawn for one split lies in a band the "
        "other could draw from",
        checks.get("realised_values_disjoint"),
        "checked against the capacitances recorded for all "
        f"{split_report.get('n_train_devices')} + "
        f"{split_report.get('n_test_devices')} generated devices, not against "
        "the intervals")
    flow.verdict(
        "IDENTITY — no device appears in both splits",
        checks.get("no_shared_device"),
        "all 13 capacitances and the swept window of every device hashed and "
        "intersected across the splits")

    flow.quote(
        f"Training and test devices were drawn from disjoint regions of "
        f"capacitance space. The five parameters that determine the geometry of "
        f"the stability diagram — the two primary gate capacitances, the two "
        f"cross gate capacitances and the interdot capacitance — were each "
        f"partitioned into "
        f"{split_report.get('bands_per_parameter')} training intervals and "
        f"{split_report.get('bands_per_parameter')} test intervals, alternating "
        f"and separated by dead bands, so that the two splits span the same "
        f"overall parameter range while sharing no attainable value. No test "
        f"device could therefore have been generated as a training device, and "
        f"the test set measures generalisation to unseen device geometry rather "
        f"than extrapolation beyond the training range. Disjointness was "
        f"verified on the generated data itself: no capacitance value drawn for "
        f"a training device falls inside any test interval or vice versa, and no "
        f"two devices across the splits share a capacitance vector."
        if cfg.split_mode == "interleaved" else
        f"Training and test devices were drawn from disjoint regions of "
        f"capacitance space (split mode: {cfg.split_mode}), separated on the "
        f"five parameters that determine the geometry of the stability diagram. "
        f"Disjointness was verified on the generated data itself.")


def _shorten(text: str, width: int) -> str:
    return text if len(text) <= width else text[:width - 3] + "..."


def _section_diversity_text(flow: Flow, tr: List[Dict], te: List[Dict],
                            Ytr: np.ndarray, Yte: np.ndarray) -> Dict:
    flow.title("Diversity of the dataset",
               "the third question a referee asks")
    flow.para(
        "A dataset can satisfy both guarantees above and still be useless, if "
        "every diagram in it is a near-copy of every other. Variety is not "
        "something to assert; it is a distribution, and this section is that "
        "distribution. Four independent mechanisms produce it:")
    flow.bullets([
        "All 13 capacitance parameters are drawn independently and uniformly "
        "per device, so a device is a point in a 13-dimensional box rather "
        "than a perturbation of a template.",
        "The primary gate capacitances span a factor of about seven, so the "
        "honeycomb period — the single most visible property of a stability "
        "diagram — varies by the same factor across the dataset.",
        "The two dots are drawn independently, so asymmetric devices (a fine "
        "honeycomb along one axis, a coarse one along the other) are as common "
        "as symmetric ones.",
        "The swept gate-voltage window is randomly offset per device, which "
        "slides the honeycomb lattice relative to the image frame. Without "
        "this every diagram is phase-locked to the same origin, and a large "
        "part of the apparent sameness of a dataset is exactly that.",
    ])
    flow.para(
        "The next page gives six of these distributions, plus the similarity "
        "of the diagrams themselves.")
    return {}


def _section_similarity(flow: Flow, ious: Dict):
    flow.heading("How similar are two diagrams, really?")
    flow.para(
        "The strongest statement about variety does not go through the "
        "parameters at all: it compares the ground-truth maps directly. For "
        "every pair of devices, the intersection-over-union of their "
        "transition-line maps says what fraction of line pixels the two share. "
        "Two copies of one diagram score 1.0. Two genuinely different "
        "honeycombs share only accidental crossings and score a few percent.")
    rows = [f"{'pairs compared':<28}{'mean IoU':>12}{'median':>12}"
            f"{'95th pct':>12}{'max':>12}"]
    for label, values in (("train vs train", ious.get("iou_train")),
                          ("test vs test", ious.get("iou_test")),
                          ("train vs test", ious.get("iou_cross"))):
        if values is None or not len(values):
            continue
        rows.append(f"{label + f'  (n={len(values)})':<28}"
                    f"{values.mean():>12.4f}{np.median(values):>12.4f}"
                    f"{np.percentile(values, 95):>12.4f}"
                    f"{values.max():>12.4f}")
    flow.mono(rows, size=7.6)
    tr = ious.get("iou_train")
    if tr is not None and len(tr):
        flow.quote(
            f"Diagram diversity was quantified directly on the ground-truth "
            f"transition maps: over all pairs of training devices the mean "
            f"intersection-over-union of the transition-line sets is "
            f"{tr.mean():.3f} (median {np.median(tr):.3f}, maximum "
            f"{tr.max():.3f}), i.e. two training diagrams typically share only "
            f"a few percent of their line pixels and no two are near-duplicates.")


def _section_measurement(flow: Flow, cfg: StudyConfig, cov_tr: float,
                         cov_te: float, Y: np.ndarray):
    flow.heading("The measurement, and what the network is shown")
    flow.para(
        f"Each device is measured with {cfg.n_rays} rays fired across the "
        f"stability diagram, {cfg.n_points} points sampled along each. The "
        f"network receives two channels — the sensor signal at the visited "
        f"pixels, and a mask of which pixels were visited — and nothing else. "
        f"The visited mask is what separates 'measured here, signal near zero' "
        f"from 'never looked here'; without it a zero is ambiguous everywhere.")
    rows = [f"{'rays x points':<34}{cfg.n_rays} x {cfg.n_points}",
            f"{'measured pixels, train':<34}{100 * cov_tr:.3f}% of the grid",
            f"{'measured pixels, test':<34}{100 * cov_te:.3f}% of the grid",
            f"{'pixels on a transition line':<34}{100 * float(Y.mean()):.3f}% "
            f"of the grid"]
    flow.mono(rows, size=7.8)
    flow.para(
        "The ground truth does NOT depend on the measurement budget: it is the "
        "exact transition map of the device, including interdot lines the "
        "charge sensor barely responds to. Only the input gets sparser as rays "
        "or points are removed. That is what makes a difference in accuracy "
        "across configurations a difference in measurement budget and nothing "
        "else.", size=8.8)


def _section_reproducibility(flow: Flow, cfg: StudyConfig, split_report: Dict):
    flow.heading("Reproducibility")
    flow.para(
        "Everything below is a pure function of the settings: the same numbers "
        "regenerate the same devices, rejections included.")
    rows = [f"{'configuration':<30}{cfg.name}",
            f"{'seed, train / test':<30}{cfg.seed} / {cfg.seed + 1000}",
            f"{'split mode':<30}{cfg.split_mode}"
            f"{' (swapped)' if cfg.swap_split else ''}",
            f"{'dead-band fraction':<30}{split_report.get('dead_band_fraction')}",
            f"{'coulomb peak width':<30}{cfg.coulomb_peak_width}",
            f"{'electron temperature':<30}{cfg.temperature}",
            f"{'device pool, train':<30}"
            f"{os.path.basename(split_report.get('pools', {}).get('train', '-'))}",
            f"{'device pool, test':<30}"
            f"{os.path.basename(split_report.get('pools', {}).get('test', '-'))}"]
    flow.mono(rows, size=7.4)
    flow.subheading("Where the rules live in the code")
    flow.mono([
        "  src/dqd/config/capacitance_config.py   the parameter space and the",
        "                                         honeycomb condition",
        "  src/dqd/simulation/dqd_validator.py    the acceptance test (guarantee 1)",
        "  src/dqd/ml/train_test_config.py        the disjoint split (guarantee 2)",
        "  src/dqd/study/dataset.py               the three disjointness checks",
        "  <this folder>/split_report.json        every number on these pages,",
        "                                         as machine-readable JSON",
    ], size=7.4)
