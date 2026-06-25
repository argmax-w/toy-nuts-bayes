"""A single plotting convention, so every figure reads the same way.

Each visual channel carries one meaning and never collides with another: black is
reserved for the observed data, hue carries parameter identity in parameter-space
plots and role (prior or posterior) in data-space plots, the predictive ribbons
use a cool map for the prior and a warm map for the posterior, prior against
posterior of one parameter is told apart by linestyle and alpha, and the
green/amber/red diagnostic status is its own reserved axis. Importing this module
everywhere keeps the convention in one place rather than per figure.

The module holds only constants and small matplotlib helpers, so it pulls in
matplotlib but nothing from the sampler.
"""

from __future__ import annotations

# Categorical palette for parameter identity (Okabe-Ito). The warm vermillion
# (#D55E00) and reddish-purple (#CC79A7) sit at the tail because they clash with
# the predictive-ribbon families; the leading entries are safe to use anywhere.
PARAM_COLOURS = [
    "#0072B2",  # blue
    "#56B4E9",  # sky blue
    "#009E73",  # green
    "#999999",  # grey
    "#E69F00",  # orange
    "#F0E442",  # yellow
    "#D55E00",  # vermillion (clashes with the posterior ribbon)
    "#CC79A7",  # reddish-purple (clashes with the prior ribbon)
]

# Fixed palette for chain identity in per-chain plots (trace, rank). Chain id is
# orthogonal to parameter id, so it draws from a separate set.
CHAIN_COLOURS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]

# The observed data is the achromatic axis: black markers with a casing in the
# background colour so points separate even over the darkest ribbon peak.
DATA_COLOUR = "#000000"
DATA_EDGE = "#ffffff"
DATA_EDGE_WIDTH = 1.5

# Role -> linestyle and alpha for prior against posterior of the same parameter
# (same colour, different style), so contraction reads even in greyscale.
PRIOR_STYLE = dict(linestyle="--", alpha=0.55, linewidth=1.6)
POSTERIOR_STYLE = dict(linestyle="-", alpha=1.0, linewidth=1.9)

# Sequential maps for the Rao-Blackwellised density ribbon: cool for the prior,
# warm for the posterior, both chosen for contrast on white.
PRIOR_CMAP = "Purples"
POSTERIOR_CMAP = "OrRd"
RIBBON_ALPHA = 0.85

# The central predictive interval, two dotted lines from inverting the averaged
# CDF, drawn over the ribbon in its deepest colour with a thin white casing.
CI_QUANTILES = (0.025, 0.975)
CI_LINE_STYLE = ":"
BAND_ALPHA = 0.20
PRIOR_DEEP = "#3f007d"  # deepest stop of Purples
POSTERIOR_DEEP = "#7f0000"  # deepest stop of OrRd

# Diagnostic traffic lights: a separate semantic axis, never reused above.
STATUS_COLOURS = dict(green="#1a9850", amber="#fee08b", red="#d73027")

# Acceptance bands carry the same status meaning, drawn pale and faint so the
# data reads on top: the region is green where the curve or points sit inside it
# and red where they escape. Reserved for calibration bands, never the ribbons.
BAND_FILL_ALPHA = 0.13


def param_colour(i: int) -> str:
    """Colour for the ``i``-th parameter, consistent across every panel."""
    return PARAM_COLOURS[i % len(PARAM_COLOURS)]


def cmap_for(kind: str) -> str:
    """Ribbon colour map for ``"Prior"`` (cool) or ``"Posterior"`` (warm)."""
    return PRIOR_CMAP if kind.lower() == "prior" else POSTERIOR_CMAP


def deep_for(kind: str) -> str:
    """Deepest ribbon colour for the interval lines, matched to ``cmap_for``."""
    return PRIOR_DEEP if kind.lower() == "prior" else POSTERIOR_DEEP


def role_colour(kind: str) -> str:
    """Role hue for data-space lines and histograms (cool prior, warm posterior)."""
    return PRIOR_DEEP if kind.lower() == "prior" else POSTERIOR_DEEP


def status_colour(status: str) -> str:
    """Background colour for a green, amber or red diagnostic status."""
    return STATUS_COLOURS[status]


def scatter_data(ax, x, y, label: str = "observed data", zorder: int = 4, size: float = 16):
    """Scatter the observed data as black markers with a background-colour casing.

    The casing keeps the points legible where they cross the darkest ribbon peak.

    Args:
        ax: Target axes.
        x: Marker x positions.
        y: Marker y positions.
        label: Legend label.
        zorder: Drawing order, above the ribbon by default.
        size: Marker size.

    Returns:
        The matplotlib path collection.
    """
    return ax.scatter(
        x, y, s=size, color=DATA_COLOUR, edgecolor=DATA_EDGE,
        linewidth=DATA_EDGE_WIDTH * 0.4, zorder=zorder, label=label,
    )


def traffic_band(ax, lo, hi, ylim=None, alpha=BAND_FILL_ALPHA, label="95% band", margin=0.6):
    """Shade a pale-green acceptance band with pale-red rejection zones around it.

    The band ``[lo, hi]`` is the good region; everything outside it, up to the axis
    limits, is the bad region and is always shaded red so the reader sees where
    failure would land even when the curve never goes there. The y-limits are
    widened by ``margin`` of the band width if they do not already leave room for
    the red zones; pass ``ylim`` to fix them instead (for count axes that should
    not go negative). Call after plotting the curve so it sits on top.

    Args:
        ax: Target axes.
        lo: Lower edge of the acceptance band.
        hi: Upper edge of the acceptance band.
        ylim: Optional ``(bottom, top)`` to fix the axis extent.
        alpha: Fill alpha, kept low so the zones stay pale.
        label: Legend label for the green acceptance band.
        margin: Fraction of the band width to leave for the red zones when
            ``ylim`` is not given.
    """
    if ylim is None:
        bottom, top = ax.get_ylim()
        span = hi - lo
        bottom = min(bottom, lo - margin * span)
        top = max(top, hi + margin * span)
    else:
        bottom, top = ylim
    ax.set_ylim(bottom, top)
    ax.axhspan(lo, hi, color=STATUS_COLOURS["green"], alpha=alpha, zorder=0, label=label)
    if bottom < lo:
        ax.axhspan(bottom, lo, color=STATUS_COLOURS["red"], alpha=alpha, zorder=0)
    if top > hi:
        ax.axhspan(hi, top, color=STATUS_COLOURS["red"], alpha=alpha, zorder=0)
    ax.set_ylim(bottom, top)


def traffic_threshold(ax, threshold, alpha=BAND_FILL_ALPHA, good_below=True):
    """Shade the zones either side of a threshold green (good) and red (bad).

    Used for a one-sided check such as the Pareto ``k`` reliability line: the
    acceptance zone is shaded green and the rejection zone red, keeping the axis
    limits unchanged.

    Args:
        ax: Target axes, already populated so its y-limits are set.
        threshold: The boundary value on the y-axis.
        alpha: Fill alpha, kept low so the zones stay pale.
        good_below: Whether values below the threshold are the good ones.
    """
    lo, hi = ax.get_ylim()
    good = (lo, threshold) if good_below else (threshold, hi)
    bad = (threshold, hi) if good_below else (lo, threshold)
    ax.axhspan(*good, color=STATUS_COLOURS["green"], alpha=alpha)
    ax.axhspan(*bad, color=STATUS_COLOURS["red"], alpha=alpha)
    ax.set_ylim(lo, hi)


def style_table(report):
    """Colour a diagnostic report by its status columns for notebook display.

    Paints each value cell with the reserved green, amber or red taken from the
    matching ``<value>_status`` column, then hides the status columns. Returns a
    pandas Styler, or the plain frame if styling is unavailable.

    Args:
        report: A frame from ``diagnostics.diagnostic_report`` with value columns
            (``r_hat``, ``ess_bulk``, ``ess_tail``) each beside a ``*_status``
            column naming the status per row.

    Returns:
        A pandas Styler, or the frame itself on failure.
    """
    status_cols = [c for c in report.columns if str(c).endswith("_status")]
    pairs = [(c[: -len("_status")], c) for c in status_cols]
    try:
        styler = report.style
        for value_col, status_col in pairs:
            if value_col in report.columns:
                styler = styler.apply(
                    lambda _c, sc=status_col: [
                        f"background-color: {STATUS_COLOURS[s]}" for s in report[sc]
                    ],
                    subset=[value_col],
                )
        return styler.hide(axis="columns", subset=status_cols)
    except Exception:  # pragma: no cover - display convenience only
        return report
