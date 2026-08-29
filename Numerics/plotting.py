"""Plotting and display helpers for the thesis numerical appendix.

This module is deliberately presentation-only.  It accepts already prepared
NumPy arrays, pandas tables, packet records, contour records and output paths.
It does not assemble transfer matrices, construct interval enclosures, compute
singular values, or certify contours.

The public functions centralise the colour palette, Matplotlib configuration,
figure saving, notebook display, and reusable Phase 1--4 renderers used by the
Blaschke-deformation certifier.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
import ast
import math
import re
import warnings

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, SymLogNorm, to_hex
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np
import pandas as pd


THESIS_PALETTE: dict[str, str] = {
    "real": "dodgerblue",
    "imag": "deeppink",
    "main": "indigo",
    "ref": "darkgrey",
    "bg": "#f0f2f5",
}

THESIS_EXTRA_PALETTE: dict[str, str] = {
    "green": "#009E73",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "vermillion": "#D55E00",
    "purple": "#7E57C2",
    "teal": "#008B8B",
}

THESIS_LINE_COLOURS: tuple[str, ...] = (
    THESIS_PALETTE["real"],
    THESIS_PALETTE["imag"],
    THESIS_PALETTE["main"],
    THESIS_EXTRA_PALETTE["green"],
    THESIS_EXTRA_PALETTE["orange"],
    THESIS_EXTRA_PALETTE["purple"],
    THESIS_EXTRA_PALETTE["teal"],
    THESIS_EXTRA_PALETTE["vermillion"],
    THESIS_PALETTE["ref"],
)

PHASE1_GRADIENT_CMAP = LinearSegmentedColormap.from_list(
    "phase1_indigo_blue_pink",
    (
        (0.00, THESIS_PALETTE.get("main")),
        (0.52, THESIS_PALETTE.get("real")),
        (1.00, THESIS_PALETTE.get("imag")),
    ),
)

PHASE4_FAMILY_COLOURS: dict[str, str] = {
    "alpha": THESIS_PALETTE["real"],
    "alpha packet": THESIS_PALETTE["real"],
    "mu": THESIS_PALETTE["imag"],
    "mu packet": THESIS_PALETTE["imag"],
    "fixed": THESIS_PALETTE["main"],
    "fixed point": THESIS_PALETTE["main"],
}


def make_phase1_error_cmap() -> LinearSegmentedColormap:
    """Return the pre-polishing Phase 1 raw-error colour map."""

    return LinearSegmentedColormap.from_list(
        "thesis_error_cmap",
        (
            THESIS_PALETTE["main"],
            THESIS_PALETTE["real"],
            THESIS_PALETTE["bg"],
            "#f8c7e8",
            THESIS_PALETTE["imag"],
        ),
    )


def make_phase1_excess_cmap() -> LinearSegmentedColormap:
    """Return the pre-polishing Phase 1 finite-M excess colour map."""

    return LinearSegmentedColormap.from_list(
        "thesis_excess_cmap",
        (
            THESIS_PALETTE["main"],
            THESIS_PALETTE["real"],
            THESIS_PALETTE["bg"],
            THESIS_PALETTE["ref"],
            THESIS_PALETTE["imag"],
        ),
    )


def make_phase4_moat_cmap(name: str = "phase4_moat_surface") -> LinearSegmentedColormap:
    """Return the pre-polishing theorem-facing Phase 4 colour map."""

    return LinearSegmentedColormap.from_list(
        name,
        (
            THESIS_PALETTE["main"],
            THESIS_PALETTE["real"],
            THESIS_PALETTE["bg"],
            THESIS_PALETTE["imag"],
        ),
    )

PHASE1_FONT_SIZES: dict[str, float] = {
    "title": 28.0,
    "axis_label": 24.0,
    "tick_label": 22.0,
    "legend": 22.0,
    "annotation": 21.0,
    "heatmap_array_title": 26.0,
    "heatmap_subplot_title": 22.0,
    "heatmap_axis_label": 16.0,
    "heatmap_tick_label": 16.0,
    "colourbar_title": 22.0,
    "small_colourbar_title": 18.0,
    "colourbar_tick_label": 22.0,
}

DEFAULT_FIGURE_FORMATS: tuple[str, ...] = ("png",)


@dataclass(frozen=True)
class PlotResult:
    """Return value shared by the plotting helpers."""

    figure: Any
    axes: Any
    paths: tuple[Path, ...] = ()


__all__ = [
    "DEFAULT_FIGURE_FORMATS",
    "PHASE1_FONT_SIZES",
    "PHASE1_GRADIENT_CMAP",
    "PHASE4_FAMILY_COLOURS",
    "PlotResult",
    "THESIS_EXTRA_PALETTE",
    "THESIS_LINE_COLOURS",
    "THESIS_PALETTE",
    "configure_thesis_style",
    "create_phase4_interactive_dashboard",
    "display_saved_image",
    "display_table",
    "latex_target_label",
    "make_phase_palette",
    "phase1_gradient_palette",
    "make_phase1_error_cmap",
    "make_phase1_excess_cmap",
    "make_phase4_moat_cmap",
    "make_phase4_interactive_dashboard_renderer",
    "matplotlib_style",
    "parse_complex_for_plot",
    "plot_certification_ladder",
    "plot_eigenvalue_cloud",
    "plot_packet_contours",
    "plot_phase1_absolute_m_heatmaps",
    "plot_phase1_cluster_errors_vs_n",
    "plot_phase1_eigencloud_comparison",
    "plot_phase1_fixed_n_m_sweep",
    "plot_phase1_oversampling_displacement",
    "plot_phase1_oversampling_heatmaps",
    "plot_phase2_branch_geometry",
    "plot_phase2_certificate_comparison",
    "plot_phase2_certificate_components",
    "plot_phase2_certified_input_rows",
    "plot_phase2_tail_improvement_ratios",
    "plot_phase3_certificate_components",
    "plot_phase3_certified_row_extrapolation",
    "plot_phase3_empirical_deterministic_bridge",
    "plot_phase3_normalised_errors",
    "plot_phase3_rate_diagnostics",
    "plot_phase3_raw_errors_against_qstar",
    "plot_phase4_fragile_robustness",
    "plot_phase4_global_moat_surface",
    "plot_phase4_global_zoom",
    "plot_phase4_local_moat_surface",
    "plot_phase4_moat_heatmap",
    "plot_phase4_moat_minima",
    "plot_phase4_moat_minima_table",
    "plot_phase4_moat_profiles",
    "plot_phase4_sampled_validation",
    "plot_phase4_single_packet_profile",
    "plot_phase4_small_gain_values",
    "plot_status_panel",
    "positive_log_limits",
    "ratio_log_limits",
    "safe_tight_layout",
    "save_figure",
    "show_figure",
]


def configure_thesis_style(
    *,
    backend: str | None = None,
    use_tex: bool = False,
    serif: bool = False,
    font_scale: float = 1.0,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Install the common thesis Matplotlib style and return the applied map."""

    if backend:
        matplotlib.use(backend, force=True)
    rc: dict[str, Any] = {
        "text.usetex": bool(use_tex),
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "axes.titlesize": 20.0 * font_scale,
        "axes.labelsize": 17.0 * font_scale,
        "xtick.labelsize": 17.0 * font_scale,
        "ytick.labelsize": 17.0 * font_scale,
        "legend.fontsize": 17.0 * font_scale,
        "legend.frameon": True,
    }
    if serif:
        rc["font.family"] = "serif"
        rc["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
    if use_tex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    if extra:
        rc.update(dict(extra))
    plt.rcParams.update(rc)
    return rc


@contextmanager
def matplotlib_style(
    *,
    use_tex: bool = False,
    serif: bool = False,
    font_scale: float = 1.0,
    extra: Mapping[str, Any] | None = None,
):
    """Temporarily apply the common thesis style."""

    rc: dict[str, Any] = {
        "text.usetex": bool(use_tex),
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.35,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "axes.titlesize": 20.0 * font_scale,
        "axes.labelsize": 17.0 * font_scale,
        "xtick.labelsize": 17.0 * font_scale,
        "ytick.labelsize": 17.0 * font_scale,
        "legend.fontsize": 17.0 * font_scale,
    }
    if serif:
        rc["font.family"] = "serif"
        rc["font.serif"] = ["Computer Modern Roman", "DejaVu Serif"]
    if use_tex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}"
    if extra:
        rc.update(dict(extra))
    with plt.rc_context(rc):
        yield


def make_phase_palette(
    labels: Iterable[Any],
    *,
    palette: Mapping[str, str] = THESIS_PALETTE,
) -> dict[Any, str]:
    """Return the indigo--blue--pink gradient used throughout the notebook."""

    ordered = tuple(labels)
    cmap = LinearSegmentedColormap.from_list(
        "thesis_indigo_blue_pink",
        (
            (0.00, palette["main"]),
            (0.52, palette["real"]),
            (1.00, palette["imag"]),
        ),
    )
    denom = max(1, len(ordered) - 1)
    return {label: to_hex(cmap(i / denom)) for i, label in enumerate(ordered)}


def phase1_gradient_palette(labels: Iterable[Any]) -> dict[Any, str]:
    """Return the dedicated indigo--blue--pink Phase 1 curve gradient."""

    ordered = tuple(labels)
    denom = max(1, len(ordered) - 1)
    return {
        label: to_hex(PHASE1_GRADIENT_CMAP(index / denom))
        for index, label in enumerate(ordered)
    }


def latex_target_label(name: Any) -> str:
    """Return a TeX-safe label for the alpha and mu packet families."""

    text = str(name)
    if text.startswith("alpha^"):
        return rf"$\alpha^{{{text.split('^', 1)[1]}}}$"
    if text.startswith("mu^"):
        return rf"$\mu^{{{text.split('^', 1)[1]}}}$"
    return rf"${text}$"


def _reviewed_map_title(title: str) -> str:
    """Restore the mathematical Blaschke title used by the reviewed notebook."""

    return str(title).replace("Blaschke deformation mu=0.3", r"Blaschke $\mu=0.3$")


def parse_complex_for_plot(value: Any) -> complex:
    """Parse a stored scalar as binary64 complex data for plotting only."""

    if isinstance(value, complex):
        return value
    text = str(value).strip()
    compact = text.replace(" ", "").replace("I", "j").replace("i", "j")
    try:
        return complex(compact)
    except ValueError:
        pair = re.fullmatch(r"\(([^,]+),([^,]+)\)", text)
        if pair:
            return complex(float(pair.group(1)), float(pair.group(2)))
        raise ValueError(f"Could not parse complex plotting value: {text}")


def parse_selected_complex_values(value: Any) -> tuple[complex, ...]:
    """Parse a retained sequence of selected plotting values."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            parsed = (text,)
    else:
        parsed = value
    if isinstance(parsed, (str, bytes)) or not isinstance(parsed, Sequence):
        parsed = (parsed,)
    return tuple(parse_complex_for_plot(item) for item in parsed)


def _prime_computed_label(name: Any) -> str | None:
    """Return the reviewed primed label for a matched computed packet."""

    text = str(name)
    if text.startswith("alpha^"):
        return rf"$\alpha'_{{{text.split('^', 1)[1]}}}$"
    if text.startswith("mu^"):
        return rf"$\mu'_{{{text.split('^', 1)[1]}}}$"
    return None


def positive_log_limits(values: Iterable[Any], *, pad_decades: float = 0.65) -> tuple[float, float]:
    """Return positive logarithmic limits covering all finite values."""

    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0e-30, 1.0
    low = 10.0 ** np.floor(np.log10(vals.min()) - pad_decades)
    high = 10.0 ** np.ceil(np.log10(vals.max()) + pad_decades)
    return float(low), float(high)


def ratio_log_limits(values: Iterable[Any]) -> tuple[float, float]:
    """Return the ratio limits used in the Phase 2 comparison panel."""

    vals = np.asarray(list(values), dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return 1.0e-2, 1.05
    low = 10.0 ** np.floor(np.log10(vals.min() * 0.75))
    high = min(1.2, max(1.05, float(vals.max()) * 1.35))
    return float(low), float(high)


def safe_tight_layout(fig: Any, *args: Any, **kwargs: Any) -> bool:
    """Apply tight layout unless the figure contains a three-dimensional axis."""

    try:
        has_3d_axis = any(getattr(ax, "name", "") == "3d" for ax in fig.axes)
        if has_3d_axis:
            fig.subplots_adjust(left=0.055, right=0.965, bottom=0.115, top=0.84, wspace=0.16)
            return False
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="This figure includes Axes that are not compatible with tight_layout.*",
                category=UserWarning,
            )
            fig.tight_layout(*args, **kwargs)
        return True
    except Exception:
        try:
            fig.subplots_adjust(left=0.12, right=0.96, bottom=0.13, top=0.88)
        except Exception:
            pass
        return False


def save_figure(
    fig: Any,
    output_dir: str | Path,
    stem: str,
    *,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    dpi: int = 240,
    tight: bool = True,
    print_status: bool = True,
) -> tuple[Path, ...]:
    """Save one figure under a stable stem in every requested format."""

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        suffix = str(fmt).lower().lstrip(".")
        path = directory / f"{stem}.{suffix}"
        kwargs: dict[str, Any] = {}
        if suffix != "pdf":
            kwargs["dpi"] = int(dpi)
        if tight:
            kwargs["bbox_inches"] = "tight"
        try:
            fig.savefig(path, **kwargs)
        except Exception:
            if not tight:
                raise
            # Match the reviewed notebook's fail-soft save path: a renderer
            # that cannot compute a tight bounding box still writes the
            # figure at the requested resolution.
            kwargs.pop("bbox_inches", None)
            fig.savefig(path, **kwargs)
        paths.append(path)
        if print_status:
            print(f"Saved figure: {path}")
    return tuple(paths)


def show_figure(fig: Any | None = None, *, close: bool = False) -> None:
    """Display a Matplotlib figure and optionally close it afterwards."""

    if fig is not None:
        try:
            fig.canvas.draw_idle()
        except Exception:
            pass
    plt.show()
    if close and fig is not None:
        plt.close(fig)


def display_table(
    table: Any,
    *,
    columns: Sequence[str] | None = None,
    head: int | None = None,
) -> Any:
    """Display a table in Jupyter, falling back to plain text outside IPython."""

    shown = table
    if isinstance(shown, pd.DataFrame):
        if columns is not None:
            present = [column for column in columns if column in shown.columns]
            shown = shown.loc[:, present]
        if head is not None:
            shown = shown.head(int(head))
    try:
        from IPython.display import display

        display(shown)
    except Exception:
        print(shown)
    return shown


def display_saved_image(path: str | Path, *, width: int | None = None) -> Any:
    """Display an existing raster artefact without regenerating it."""

    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(image_path)
    try:
        from IPython.display import Image, display

        image = Image(filename=str(image_path), width=width)
        display(image)
        return image
    except Exception:
        print(image_path)
        return image_path


def _finish_plot(
    fig: Any,
    axes: Any,
    *,
    output_dir: str | Path | None,
    stem: str,
    formats: Sequence[str] = DEFAULT_FIGURE_FORMATS,
    dpi: int = 240,
    show: bool = True,
    tight: bool = True,
    bbox_tight: bool = True,
) -> PlotResult:
    if tight:
        safe_tight_layout(fig)
    paths = ()
    if output_dir is not None:
        paths = save_figure(
            fig, output_dir, stem, formats=formats, dpi=dpi, tight=bbox_tight
        )
    if show:
        show_figure(fig)
    return PlotResult(fig, axes, paths)


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _ellipse_boundary(radius: float, samples: int = 1600) -> np.ndarray:
    theta = np.linspace(0.0, 2.0 * np.pi, int(samples), endpoint=True)
    radius = float(radius)
    return 0.5 * (radius * np.exp(1j * theta) + radius ** -1 * np.exp(-1j * theta))


def plot_status_panel(
    *,
    cell_label: str,
    phase: str,
    message: str,
    map_label: str,
    status_table: pd.DataFrame | None = None,
    output_dir: str | Path | None = None,
    stem: str = "status",
    palette: Mapping[str, str] = THESIS_PALETTE,
    show: bool = True,
) -> PlotResult:
    """Render the explicit status panel formerly embedded in Cell 7."""

    fig, ax = plt.subplots(figsize=(8.8, 3.2), facecolor="white")
    ax.axis("off")
    lines = [cell_label, f"phase: {phase}", f"map: {map_label}", str(message)]
    ax.text(0.02, 0.78, "\n".join(lines), ha="left", va="top", fontsize=13,
            color=palette["main"], transform=ax.transAxes)
    if status_table is not None and len(status_table):
        text = status_table.to_string(index=False)
        ax.text(0.02, 0.10, text[:900], ha="left", va="bottom", fontsize=9.5,
                family="monospace", color="black", transform=ax.transAxes)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem, dpi=220, show=show)


def plot_eigenvalue_cloud(
    eigenvalues: Sequence[complex],
    *,
    targets: Sequence[Mapping[str, Any]] = (),
    title: str = "Finite eigenvalue cloud",
    output_dir: str | Path | None = None,
    stem: str = "finite_eigenvalue_cloud",
    palette: Mapping[str, str] = THESIS_PALETTE,
    show: bool = True,
) -> PlotResult:
    """Plot finite eigenvalues and optional labelled target centres."""

    eig = np.asarray(eigenvalues, dtype=np.complex128)
    fig, ax = plt.subplots(figsize=(8.4, 6.2), facecolor="white")
    if eig.size:
        ax.scatter(eig.real, eig.imag, s=26, color=palette["main"], alpha=0.78,
                   label="finite eigenvalues")
    for target in targets:
        centre = complex(target["centre"])
        name = target.get("name", "target")
        ax.scatter([centre.real], [centre.imag], s=70, facecolors="none",
                   edgecolors=palette["imag"], linewidths=1.5)
        ax.annotate(latex_target_label(name), (centre.real, centre.imag),
                    textcoords="offset points", xytext=(4, 4), fontsize=10,
                    color=palette["imag"])
    ax.axhline(0.0, color=palette["ref"], lw=0.9, alpha=0.7)
    ax.axvline(0.0, color=palette["ref"], lw=0.9, alpha=0.7)
    ax.set_xlabel("real part")
    ax.set_ylabel("imaginary part")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if eig.size:
        ax.legend(frameon=False)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem, show=show)


def plot_phase1_cluster_errors_vs_n(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    *,
    map_label: str,
    reference_curve: Mapping[str, Any] | None = None,
    error_column: str = "err_float",
    output_dir: str | Path | None = None,
    stem: str = "phase1_raw_cluster_errors_vs_N_latex",
    use_tex: bool = True,
    show: bool = True,
) -> PlotResult:
    """Plot prepared raw square-block cluster errors against Legendre dimension."""

    _require_columns(frame, ("N", "name", error_column), name="Phase 1 cluster table")
    rc = {
        "axes.titlesize": 22,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 11.5,
    }
    with matplotlib_style(use_tex=use_tex, serif=True, extra=rc):
        fig, ax = plt.subplots(figsize=(10.5, 6.2), facecolor="white")
        alpha_names = tuple(
            name for name in target_names if str(name).startswith("alpha^")
        )
        alpha_colours = phase1_gradient_palette(alpha_names)
        for index, name in enumerate(target_names):
            subset = frame.loc[frame["name"].astype(str) == str(name)].copy()
            y = pd.to_numeric(subset[error_column], errors="coerce")
            subset = subset.loc[np.isfinite(y) & (y > 0)].copy()
            if subset.empty:
                continue
            y = pd.to_numeric(subset[error_column], errors="coerce").to_numpy(dtype=float)
            if str(name) in {"mu^1", "mu^2"}:
                line_colour = THESIS_EXTRA_PALETTE["green"]
            else:
                line_colour = alpha_colours.get(
                    name,
                    THESIS_LINE_COLOURS[index % len(THESIS_LINE_COLOURS)],
                )
            ax.plot(subset["N"].astype(int), y, marker="o", markersize=5.4,
                    linewidth=2.0,
                    color=line_colour,
                    label=latex_target_label(name))
        if reference_curve is not None:
            ref_x = np.asarray(reference_curve["x"], dtype=float)
            ref_y = np.asarray(reference_curve["y"], dtype=float)
            ref_valid = np.isfinite(ref_x) & np.isfinite(ref_y) & (ref_y > 0)
            displayed = frame.loc[
                frame["name"].astype(str).isin(tuple(map(str, target_names)))
            ].copy()
            displayed["_plot_error"] = pd.to_numeric(
                displayed[error_column], errors="coerce"
            )
            displayed = displayed.loc[
                np.isfinite(displayed["_plot_error"])
                & (displayed["_plot_error"] > 0)
            ]
            if ref_valid.sum() >= 2 and not displayed.empty:
                slope = float(np.polyfit(ref_x[ref_valid], np.log(ref_y[ref_valid]), 1)[0])
                q_fit = float(np.exp(slope))
                ref_x = np.asarray(sorted(displayed["N"].astype(int).unique()), dtype=float)
                anchor_n = int(ref_x.min())
                anchor_error = float(
                    displayed.loc[
                        displayed["N"].astype(int).eq(anchor_n), "_plot_error"
                    ].max()
                )
                ref_y = anchor_error * q_fit ** (ref_x - anchor_n)
                reference_label = (
                    rf"fit $\Delta_{{\rm anch}}q_{{\rm fit}}^{{N-N_{{\rm anch}}}}$, "
                    rf"$q_{{\rm fit}}={q_fit:.3f}$"
                )
            else:
                reference_label = str(
                    reference_curve.get("label", "empirical exponential guide")
                )
            ax.plot(
                ref_x,
                ref_y,
                color=reference_curve.get("colour", "black"),
                linewidth=float(reference_curve.get("linewidth", 4.2)),
                linestyle=reference_curve.get("linestyle", (0, (10, 4))),
                label=reference_label,
            )
        ax.set_yscale("log")
        ax.set_xlabel(r"Legendre truncation $N$ with $M=N$")
        ax.set_ylabel(r"Raw cluster error $\Delta_\lambda(N,N)$")
        ax.set_title(f"{map_label.replace('_', ' ')}: square Legendre--Gauss blocks")
        ax.grid(True, which="both", linestyle="--", alpha=0.55)
        ax.legend(loc="lower left", ncol=2, frameon=True)
        return _finish_plot(
            fig, ax, output_dir=output_dir, stem=stem, formats=("pdf", "png"),
            dpi=240, show=show,
        )


def _plot_phase1_eigencloud_comparison_impl(
    blocks: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    stem: str = "phase1_eigencloud_leading_interval_comparison",
    xlim: tuple[float, float] = (-2.5e-2, 1.025),
    ylim: tuple[float, float] = (-5.0e-2, 5.0e-2),
    title: str = "Raw Legendre--Gauss eigenvalue clouds on the leading spectral interval",
    show: bool = True,
) -> PlotResult:
    """Plot prepared Phase 1 eigencloud blocks in vertically stacked panels."""

    if not blocks:
        raise ValueError("At least one eigencloud block is required")
    fig, axes = plt.subplots(len(blocks), 1, figsize=(17.0, 5.6 * len(blocks)),
                             facecolor="white", squeeze=False)
    axes_flat = axes.ravel()
    handles = labels = None
    for ax, block in zip(axes_flat, blocks):
        eig = np.asarray(block.get("eigenvalues", ()), dtype=np.complex128)
        targets = np.asarray(block.get("targets", ()), dtype=np.complex128)
        target_labels = tuple(block.get("target_labels", ()))
        targets_on_axis = targets.real.astype(np.complex128)
        matched = np.asarray(block.get("matched", ()), dtype=np.complex128)
        if not matched.size and eig.size and target_labels:
            unused = list(range(eig.size))
            selected: list[complex] = []
            for name, centre_value in target_labels:
                centre = complex(float(complex(centre_value).real), 0.0)
                multiplicity = 2 if str(name).startswith("mu^") else 1
                for _ in range(multiplicity):
                    if not unused:
                        break
                    chosen = min(unused, key=lambda index: abs(eig[index] - centre))
                    unused.remove(chosen)
                    selected.append(complex(eig[chosen]))
            matched = np.asarray(selected, dtype=np.complex128)
        ax.scatter(eig.real, eig.imag, s=95, color=THESIS_PALETTE["main"],
                   alpha=0.82, label="computed", zorder=3)
        ax.scatter(targets_on_axis.real, targets_on_axis.imag, s=150, facecolors="none",
                   edgecolors=THESIS_PALETTE["imag"], linewidths=2.3,
                   label="exact targets", zorder=4)
        if matched.size:
            ax.scatter(matched.real, matched.imag, s=120, facecolors="none",
                       edgecolors=THESIS_PALETTE["real"], linewidths=2.1,
                       label="matched approximations", zorder=5)
        for name, centre in target_labels:
            x_value = float(complex(centre).real)
            if xlim[0] <= x_value <= xlim[1]:
                ax.annotate(latex_target_label(name), (x_value, 0.0),
                            textcoords="offset points", xytext=(7, 8),
                            fontsize=PHASE1_FONT_SIZES["annotation"],
                            color=THESIS_PALETTE["ref"])
        ax.axhline(0.0, color=THESIS_PALETTE["ref"], linewidth=1.4, alpha=0.85)
        ax.axvline(0.0, color=THESIS_PALETTE["ref"], linewidth=1.4, alpha=0.65)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("real part")
        ax.set_ylabel("imaginary part" if ax is axes_flat[0] else "")
        ax.set_title(rf"$N={int(block['N'])}$, $M={int(block['M'])}$",
                     fontsize=PHASE1_FONT_SIZES["title"])
        ax.tick_params(axis="both", which="major",
                       labelsize=PHASE1_FONT_SIZES["tick_label"])
        ax.grid(True, linestyle="--", alpha=0.45)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
                   fontsize=PHASE1_FONT_SIZES["legend"],
                   bbox_to_anchor=(0.5, 0.035))
    fig.suptitle(title, fontsize=PHASE1_FONT_SIZES["title"], y=0.972)
    fig.subplots_adjust(top=0.90, bottom=0.23, left=0.085, right=0.985, hspace=0.42)
    return _finish_plot(fig, axes_flat, output_dir=output_dir, stem=stem,
                        dpi=320, show=show, tight=False)


def _plot_phase1_oversampling_displacement_impl(
    base_points: Sequence[complex],
    comparison_points: Sequence[complex],
    exact_targets: Sequence[complex],
    *,
    title: str,
    base_label: str,
    comparison_label: str,
    comparison_point_labels: Sequence[tuple[str, complex]] = (),
    magnification: float = 12.0,
    radius: float = 5.0e-2,
    output_dir: str | Path | None = None,
    stem: str = "phase1_oversampling_eigencloud_displacement",
    show: bool = True,
) -> PlotResult:
    """Show actual and magnified displacement between two prepared eigenclouds."""

    base = [complex(z) for z in base_points if abs(complex(z).real) <= radius and abs(complex(z).imag) <= radius]
    other = [complex(z) for z in comparison_points if abs(complex(z).real) <= radius and abs(complex(z).imag) <= radius]
    remaining = list(other)
    pairs: list[tuple[complex, complex]] = []
    for z in sorted(base, key=lambda value: (abs(value), value.real, value.imag)):
        if not remaining:
            break
        index = min(range(len(remaining)), key=lambda j: abs(remaining[j] - z))
        pairs.append((z, remaining.pop(index)))
    visible_labels: list[tuple[str, complex]] = []
    for name, point_value in comparison_point_labels:
        point = complex(point_value)
        label = _prime_computed_label(name)
        if label is not None and abs(point.real) <= radius and abs(point.imag) <= radius:
            visible_labels.append((label, point))
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 7.8), facecolor="white")
    targets = np.asarray(exact_targets, dtype=np.complex128)
    target_real = targets.real[np.abs(targets.real) <= radius]
    for ax in axes:
        ax.axhline(0.0, color=THESIS_PALETTE["ref"], linewidth=1.0, alpha=0.85)
        ax.axvline(0.0, color=THESIS_PALETTE["ref"], linewidth=1.0, alpha=0.65)
        ax.scatter(target_real, np.zeros_like(target_real), s=145, facecolors="none",
                   edgecolors=THESIS_PALETTE["imag"], linewidths=2.2,
                   label="exact targets", zorder=4)
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.set_xlabel("real part", fontsize=PHASE1_FONT_SIZES["axis_label"])
        ax.tick_params(axis="both", which="major",
                       labelsize=PHASE1_FONT_SIZES["tick_label"])
    axes[0].set_ylabel("imaginary part", fontsize=PHASE1_FONT_SIZES["axis_label"])
    axes[1].set_ylabel("")
    axes[0].scatter([z.real for z in base], [z.imag for z in base], s=105,
                    color=THESIS_PALETTE["main"], alpha=0.82, label=base_label)
    axes[0].scatter([z.real for z in other], [z.imag for z in other], s=95,
                    color=THESIS_PALETTE["real"], marker="x", linewidths=2.2,
                    alpha=0.82, label=comparison_label, zorder=6)
    for z, w in pairs:
        axes[0].annotate("", xy=(w.real, w.imag), xytext=(z.real, z.imag),
                         arrowprops={"arrowstyle": "->", "color": THESIS_PALETTE["real"],
                                     "lw": 2.0, "alpha": 0.78}, zorder=4)
    for index, (label, point) in enumerate(visible_labels):
        axes[0].annotate(label, (point.real, point.imag), textcoords="offset points",
                         xytext=(5, 5 + 4 * (index % 2)),
                         fontsize=PHASE1_FONT_SIZES["annotation"],
                         color=THESIS_PALETTE["real"], zorder=8)
    magnified = [z + magnification * (w - z) for z, w in pairs]
    axes[1].scatter([z.real for z in base], [z.imag for z in base], s=105,
                    color=THESIS_PALETTE["main"], alpha=0.82, label=base_label)
    axes[1].scatter([z.real for z in magnified], [z.imag for z in magnified], s=95,
                    color=THESIS_PALETTE["real"], marker="x", linewidths=2.2,
                    alpha=0.82, label=comparison_label, zorder=6)
    for (z, _), end in zip(pairs, magnified):
        axes[1].annotate("", xy=(end.real, end.imag), xytext=(z.real, z.imag),
                         arrowprops={"arrowstyle": "->", "color": THESIS_PALETTE["real"],
                                     "lw": 2.0, "alpha": 0.78}, zorder=4)
    magnified_labels: list[tuple[str, complex]] = []
    for label, comparison_point in visible_labels:
        if pairs:
            base_point, matched_point = min(
                pairs, key=lambda pair: abs(pair[1] - comparison_point)
            )
            magnified_labels.append(
                (label, base_point + magnification * (matched_point - base_point))
            )
    for index, (label, point) in enumerate(magnified_labels):
        axes[1].annotate(label, (point.real, point.imag), textcoords="offset points",
                         xytext=(5, 5 + 4 * (index % 2)),
                         fontsize=PHASE1_FONT_SIZES["annotation"],
                         color=THESIS_PALETTE["real"], zorder=8)
    axes[0].set_title("Actual displacement arrows", fontsize=PHASE1_FONT_SIZES["title"])
    axes[1].set_title(rf"After movement magnified by ${magnification:g}$",
                      fontsize=PHASE1_FONT_SIZES["title"])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=True,
               fontsize=PHASE1_FONT_SIZES["legend"],
               bbox_to_anchor=(0.5, 0.040))
    fig.suptitle(title, fontsize=PHASE1_FONT_SIZES["title"], y=0.965)
    fig.subplots_adjust(top=0.82, bottom=0.26, left=0.075, right=0.985, wspace=0.28)
    return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                        dpi=320, show=show, tight=False)


def _phase1_reviewed_style() -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "axes.unicode_minus": False,
        "axes.titlesize": PHASE1_FONT_SIZES["title"],
        "axes.labelsize": PHASE1_FONT_SIZES["axis_label"],
        "xtick.labelsize": PHASE1_FONT_SIZES["tick_label"],
        "ytick.labelsize": PHASE1_FONT_SIZES["tick_label"],
        "legend.fontsize": PHASE1_FONT_SIZES["legend"],
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
    }


def plot_phase1_eigencloud_comparison(
    blocks: Sequence[Mapping[str, Any]],
    *,
    output_dir: str | Path | None = None,
    stem: str = "phase1_eigencloud_leading_interval_comparison",
    xlim: tuple[float, float] = (-2.5e-2, 1.025),
    ylim: tuple[float, float] = (-5.0e-2, 5.0e-2),
    title: str = "Raw Legendre--Gauss eigenvalue clouds on the leading spectral interval",
    show: bool = True,
) -> PlotResult:
    """Render Phase 1 eigenclouds in the protected reviewed style."""

    with plt.rc_context(_phase1_reviewed_style()):
        return _plot_phase1_eigencloud_comparison_impl(
            blocks, output_dir=output_dir, stem=stem, xlim=xlim, ylim=ylim,
            title=title, show=show,
        )


def plot_phase1_oversampling_displacement(
    base_points: Sequence[complex],
    comparison_points: Sequence[complex],
    exact_targets: Sequence[complex],
    *,
    title: str,
    base_label: str,
    comparison_label: str,
    comparison_point_labels: Sequence[tuple[str, complex]] = (),
    magnification: float = 12.0,
    radius: float = 5.0e-2,
    output_dir: str | Path | None = None,
    stem: str = "phase1_oversampling_eigencloud_displacement",
    show: bool = True,
) -> PlotResult:
    """Render Phase 1 oversampling displacement in the reviewed style."""

    with plt.rc_context(_phase1_reviewed_style()):
        return _plot_phase1_oversampling_displacement_impl(
            base_points, comparison_points, exact_targets, title=title,
            base_label=base_label, comparison_label=comparison_label,
            comparison_point_labels=comparison_point_labels,
            magnification=magnification, radius=radius,
            output_dir=output_dir, stem=stem, show=show,
        )


def _plot_phase1_fixed_n_m_sweep_impl(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    *,
    n_value: int,
    guide: Mapping[str, Any] | None = None,
    map_label: str | None = None,
    output_dir: str | Path | None = None,
    stem: str | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot prepared raw cluster errors against quadrature order at fixed N."""

    _require_columns(frame, ("N", "M", "name"), name="Phase 1 M-sweep table")
    error_column = "error_float" if "error_float" in frame.columns else "error"
    colours = make_phase_palette(target_names)
    fig, ax = plt.subplots(figsize=(10.0, 6.0), facecolor="white")
    for target in target_names:
        subset = frame.loc[(frame["N"].astype(int) == int(n_value)) & (frame["name"].astype(str) == str(target))].sort_values("M")
        y = pd.to_numeric(subset[error_column], errors="coerce")
        valid = np.isfinite(y) & (y > 0)
        if not valid.any():
            continue
        ax.plot(subset.loc[valid, "M"].astype(int), y.loc[valid], marker="o",
                linewidth=1.9, markersize=4.7, color=colours[target],
                label=latex_target_label(target))
    if guide is None:
        guide_rows: list[pd.DataFrame] = []
        for target in target_names:
            subset = frame.loc[
                (frame["N"].astype(int) == int(n_value))
                & (frame["name"].astype(str) == str(target))
            ].copy()
            subset["_guide_error"] = pd.to_numeric(
                subset[error_column], errors="coerce"
            )
            guide_rows.append(subset[["M", "_guide_error"]])
        if guide_rows:
            guide_frame = pd.concat(guide_rows, ignore_index=True)
            guide_frame = guide_frame.loc[
                guide_frame["M"].astype(int).between(16, int(n_value))
                & np.isfinite(guide_frame["_guide_error"])
                & guide_frame["_guide_error"].between(1.0e-14, 1.0e-1)
            ]
            if len(guide_frame) >= 3:
                slope, intercept = np.polyfit(
                    guide_frame["M"].to_numpy(dtype=float),
                    np.log(guide_frame["_guide_error"].to_numpy(dtype=float)),
                    1,
                )
                line_x = np.arange(16, int(n_value) + 1)
                guide = {
                    "x": line_x,
                    "y": np.exp(intercept + slope * line_x),
                    "colour": "black",
                    "linestyle": "--",
                    "linewidth": 3.4,
                    "label": rf"local exp. guide, $q_M={np.exp(slope):.3f}$",
                }
    if guide is not None:
        ax.plot(guide["x"], guide["y"], color=guide.get("colour", "black"),
                linestyle=guide.get("linestyle", "--"), linewidth=guide.get("linewidth", 3.4),
                label=guide.get("label", "local exponential guide"))
    ax.axvline(int(n_value), color=THESIS_PALETTE["ref"], linestyle="--",
               linewidth=1.4, label=r"$M=N$")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M$")
    ax.set_ylabel("Raw cluster error")
    title = rf"Raw cluster errors versus $M$ at fixed $N={int(n_value)}$"
    if map_label:
        title += f" for {str(map_label).replace('_', ' ')}"
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", alpha=0.55)
    ax.legend(fontsize=11.5, ncol=3)
    return _finish_plot(fig, ax, output_dir=output_dir,
                        stem=stem or f"phase1_raw_cluster_errors_vs_M_N{int(n_value)}_latex",
                        dpi=320, show=show)


def plot_phase1_fixed_n_m_sweep(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    *,
    n_value: int,
    guide: Mapping[str, Any] | None = None,
    map_label: str | None = None,
    output_dir: str | Path | None = None,
    stem: str | None = None,
    show: bool = True,
) -> PlotResult:
    """Render the fixed-N Phase 1 sweep in the protected reviewed style."""

    with plt.rc_context(_phase1_reviewed_style()):
        return _plot_phase1_fixed_n_m_sweep_impl(
            frame,
            target_names,
            n_value=n_value,
            guide=guide,
            map_label=map_label,
            output_dir=output_dir,
            stem=stem,
            show=show,
        )


def _centres_to_edges(values: Sequence[float]) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        return np.asarray([vals[0] - 0.5, vals[0] + 0.5])
    mids = 0.5 * (vals[:-1] + vals[1:])
    return np.concatenate(([vals[0] - (mids[0] - vals[0])], mids,
                           [vals[-1] + (vals[-1] - mids[-1])]))


def _heatmap_grids(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    y_values: Sequence[int],
    x_values: Sequence[int],
    *,
    x_column: str,
    value_column: str,
) -> list[np.ndarray]:
    grids: list[np.ndarray] = []
    for target in target_names:
        subset = frame.loc[frame["name"].astype(str) == str(target)]
        lookup: dict[tuple[int, int], float] = {}
        for row in subset.itertuples(index=False):
            y = int(getattr(row, "N"))
            x = int(getattr(row, x_column))
            value = float(getattr(row, value_column))
            if value_column == "error":
                value = np.log10(value) if np.isfinite(value) and value > 0 else np.nan
            lookup[(y, x)] = value
        grid = np.full((len(y_values), len(x_values)), np.nan)
        for i, y in enumerate(y_values):
            for j, x in enumerate(x_values):
                grid[i, j] = lookup.get((int(y), int(x)), np.nan)
        grids.append(grid)
    return grids


def _plot_heatmap_array(
    grids: Sequence[np.ndarray],
    target_names: Sequence[str],
    y_values: Sequence[int],
    x_values: Sequence[int],
    *,
    x_label: str,
    title_prefix: str,
    array_title: str,
    cbar_title: str,
    cmap: Any,
    output_dir: str | Path | None,
    stem: str,
    ncols: int,
    mark_best: bool,
    show: bool,
) -> PlotResult:
    nplots = len(target_names)
    nrows = int(math.ceil(nplots / ncols))
    finite = [grid[np.isfinite(grid)] for grid in grids if np.isfinite(grid).any()]
    vmin = min(float(values.min()) for values in finite) if finite else None
    vmax = max(float(values.max()) for values in finite) if finite else None
    cmap_obj = cmap.copy()
    cmap_obj.set_bad("white")
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.3 * ncols, 4.15 * nrows + 0.7),
                             squeeze=False, facecolor="white", constrained_layout=True)
    active_axes: list[Any] = []
    last_mesh = None
    for index, (target, grid) in enumerate(zip(target_names, grids)):
        ax = axes[index // ncols, index % ncols]
        active_axes.append(ax)
        last_mesh = ax.pcolormesh(_centres_to_edges(x_values), _centres_to_edges(y_values),
                                  grid, shading="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax)
        ax.set_title(f"{title_prefix}{latex_target_label(target)}",
                     fontsize=PHASE1_FONT_SIZES["heatmap_subplot_title"])
        ax.set_xlabel(x_label, fontsize=PHASE1_FONT_SIZES["heatmap_axis_label"])
        ax.set_ylabel(r"$N$", fontsize=PHASE1_FONT_SIZES["heatmap_axis_label"])
        if x_label == r"$M$":
            preferred_ticks = (6, 10, 14, 18, 22, 26, 30, 36, 48)
            shown_ticks = [tick for tick in preferred_ticks if tick in set(x_values)]
        else:
            preferred_ticks = (0, 2, 4, 8, 12, 16, 24, 48)
            shown_ticks = [tick for tick in preferred_ticks if tick in set(x_values)]
        ax.set_xticks(shown_ticks)
        ax.set_yticks(y_values)
        ax.tick_params(axis="both", labelsize=PHASE1_FONT_SIZES["heatmap_tick_label"])
        ax.grid(True, linestyle="--", alpha=0.35)
        if mark_best and np.isfinite(grid).any():
            i, j = np.unravel_index(np.nanargmin(grid), grid.shape)
            ax.plot(x_values[j], y_values[i], marker="o", markersize=8,
                    markerfacecolor="none", markeredgecolor="white",
                    markeredgewidth=1.8)
            ax.plot(x_values[j], y_values[i], marker="o", markersize=9,
                    markerfacecolor="none", markeredgecolor=THESIS_PALETTE["main"],
                    markeredgewidth=0.9)
    for index in range(nplots, nrows * ncols):
        axes[index // ncols, index % ncols].set_visible(False)
    fig.suptitle(array_title, fontsize=PHASE1_FONT_SIZES["heatmap_array_title"], y=1.03)
    if last_mesh is not None:
        cbar = fig.colorbar(last_mesh, ax=active_axes, orientation="horizontal",
                            fraction=0.055, pad=0.075, aspect=45)
        cbar.ax.set_title(cbar_title, fontsize=PHASE1_FONT_SIZES["colourbar_title"], pad=8)
        cbar.ax.tick_params(labelsize=PHASE1_FONT_SIZES["colourbar_tick_label"])
    return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                        dpi=320, show=show, tight=False)


def plot_phase1_absolute_m_heatmaps(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    n_values: Sequence[int],
    m_values: Sequence[int],
    *,
    value_column: str,
    title_prefix: str,
    array_title: str,
    cbar_title: str,
    output_dir: str | Path | None = None,
    stem: str = "phase1_raw_NM_heatmap",
    cmap: Any | None = None,
    ncols: int = 3,
    mark_best: bool = True,
    map_label: str | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot Phase 1 heatmaps whose horizontal coordinate is absolute M."""

    _require_columns(frame, ("N", "M", "name", value_column), name="Phase 1 heatmap table")
    grids = _heatmap_grids(frame, target_names, n_values, m_values,
                           x_column="M", value_column=value_column)
    cmap = cmap or make_phase1_error_cmap()
    if map_label:
        array_title = f"{array_title}: {str(map_label).replace('_', ' ')}"
    with plt.rc_context(_phase1_reviewed_style()):
        return _plot_heatmap_array(
            grids, target_names, n_values, m_values, x_label=r"$M$",
            title_prefix=title_prefix, array_title=array_title, cbar_title=cbar_title,
            cmap=cmap, output_dir=output_dir, stem=stem, ncols=ncols,
            mark_best=mark_best, show=show,
        )


def plot_phase1_oversampling_heatmaps(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    n_values: Sequence[int],
    offsets: Sequence[int],
    *,
    value_column: str,
    title_prefix: str,
    array_title: str,
    cbar_title: str,
    output_dir: str | Path | None = None,
    stem: str = "phase1_raw_oversampling_heatmap",
    cmap: Any | None = None,
    ncols: int = 3,
    mark_best: bool = True,
    map_label: str | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot Phase 1 heatmaps whose horizontal coordinate is m equals M minus N."""

    work = frame.copy()
    if "m" not in work.columns:
        _require_columns(work, ("M", "N"), name="Phase 1 oversampling table")
        work["m"] = work["M"].astype(int) - work["N"].astype(int)
    _require_columns(work, ("N", "m", "name", value_column), name="Phase 1 oversampling table")
    grids = _heatmap_grids(work, target_names, n_values, offsets,
                           x_column="m", value_column=value_column)
    cmap = cmap or make_phase1_excess_cmap()
    if map_label:
        array_title = f"{array_title}: {str(map_label).replace('_', ' ')}"
    with plt.rc_context(_phase1_reviewed_style()):
        return _plot_heatmap_array(
            grids, target_names, n_values, offsets, x_label=r"$m=M-N$",
            title_prefix=title_prefix, array_title=array_title, cbar_title=cbar_title,
            cmap=cmap, output_dir=output_dir, stem=stem, ncols=ncols,
            mark_best=mark_best, show=show,
        )


def plot_phase2_branch_geometry(
    profile: pd.DataFrame,
    *,
    rho: float,
    r: float,
    branch_radius: float,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase2_geometry",
    show: bool = True,
) -> PlotResult:
    """Plot response, Hardy and branch-image ellipses with prepared branch traces."""

    _require_columns(profile, ("branch", "tau_re", "tau_im"), name="branch profile")
    branches = sorted(profile["branch"].unique())
    labels = ["response ellipse", "Hardy ellipse", "certified branch envelope"] + [f"branch {b}" for b in branches]
    colours = make_phase_palette(labels)
    fig, ax = plt.subplots(figsize=(8.2, 6.2), facecolor="white")
    for radius, label, style, linewidth, display_label in (
        (rho, "response ellipse", "-", 1.8, fr"$E_\rho$, $\rho={float(rho):.4g}$"),
        (r, "Hardy ellipse", "-", 1.8, fr"$E_r$, $r={float(r):.4g}$"),
        (branch_radius, "certified branch envelope", "--", 1.4,
         fr"$E_{{r_\tau}}$, $r_\tau={float(branch_radius):.4g}$"),
    ):
        curve = _ellipse_boundary(float(radius))
        ax.plot(curve.real, curve.imag, color=colours[label], lw=linewidth,
                ls=style, label=display_label)
    for branch in branches:
        subset = profile.loc[profile["branch"] == branch]
        ax.plot(subset["tau_re"], subset["tau_im"], color=colours[f"branch {branch}"],
                lw=1.2, label=f"branch {branch} image")
    ax.plot([-1, 1], [0, 0], color=THESIS_PALETTE["ref"], lw=2.0, alpha=0.75)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.set_title("Branch-image geometry")
    ax.legend(loc="best", fontsize=9)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem, show=show)


def plot_phase2_tail_improvement_ratios(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase2_tail_improvement_ratios",
    show: bool = True,
) -> PlotResult:
    """Plot the three Phase 2 branch-image refinement ratios."""

    specs = (
        ("branch-wise over certified global", "branchwise_over_certified_global", "o"),
        ("branch-image upper kernel over certified global", "kernel_over_certified_global", "s"),
        ("branch-image upper kernel over sampled global", "kernel_over_sampled_global", "D"),
    )
    _require_columns(frame, ("N",) + tuple(column for _, column, _ in specs), name="tail comparison")
    colours = make_phase_palette(label for label, _, _ in specs)
    fig, ax = plt.subplots(figsize=(8.2, 4.8), facecolor="white")
    for label, column, marker in specs:
        ax.semilogy(frame["N"], frame[column], marker=marker, color=colours[label], label=label)
    ax.axhline(1.0, color=THESIS_PALETTE["ref"], lw=1.0, ls="--")
    ax.set_xlabel("Legendre dimension N")
    ax.set_ylabel("ratio")
    ax.set_title("Input-tail refinement ratios")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(fontsize=9)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem, show=show)


def plot_phase2_certified_input_rows(
    profile: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    stem: str = "blaschke_deformation_certified_input_tail_rows",
    stride: int | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot certified branchwise, direct coherent and selected input rows."""

    columns = ("theta_mid", "branchwise_row_u", "combined_row_raw_u", "combined_row_u")
    _require_columns(profile, columns, name="certified input profile")
    step = int(stride or max(1, len(profile) // 4096))
    shown = profile.iloc[::step].copy()
    branchwise = pd.to_numeric(shown["branchwise_row_u"], errors="coerce")
    direct = pd.to_numeric(shown["combined_row_raw_u"], errors="coerce")
    selected = pd.to_numeric(shown["combined_row_u"], errors="coerce")
    with matplotlib_style(extra={
        "font.size": 10.5, "axes.titlesize": 14.5, "axes.labelsize": 13.5,
        "legend.fontsize": 14.0, "xtick.labelsize": 12.0, "ytick.labelsize": 12.0,
    }):
        fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.0), facecolor="white")
        axes[0].semilogy(shown["theta_mid"], direct, color=THESIS_PALETTE["real"],
                         lw=1.5, alpha=0.9, label="direct coherent-row enclosure")
        axes[0].semilogy(shown["theta_mid"], selected, color=THESIS_PALETTE["imag"],
                         lw=1.8, label="selected intersection = branchwise")
        axes[0].fill_between(shown["theta_mid"], selected, direct,
                             color=THESIS_PALETTE["real"], alpha=0.10,
                             label="certified endpoint gap")
        axes[0].set_xlabel(r"boundary parameter $\theta$")
        axes[0].set_ylabel("certified row upper bound")
        axes[0].set_title("Absolute certified unresolved-input rows")
        axes[0].grid(True, which="both", ls="--", alpha=0.28)
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), frameon=True)
        axes[1].axhline(0.0, color=THESIS_PALETTE["ref"], lw=1.0, linestyle=":")
        axes[1].plot(shown["theta_mid"], np.log10(direct / branchwise),
                     color=THESIS_PALETTE["real"], lw=1.5,
                     label="direct coherent endpoint factor")
        axes[1].plot(shown["theta_mid"], np.log10(selected / branchwise),
                     color=THESIS_PALETTE["main"], lw=1.5, ls="--",
                     label="selected minimum = branchwise")
        axes[1].set_xlabel(r"boundary parameter $\theta$")
        axes[1].set_ylabel(r"decimal factor $\log_{10}(U/U_{\rm br})$")
        axes[1].set_title("Relative separation from branchwise profile")
        axes[1].grid(True, which="both", ls="--", alpha=0.28)
        axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.34), frameon=True)
        fig.subplots_adjust(left=0.09, right=0.985, top=0.88, bottom=0.38, wspace=0.32)
        return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                            dpi=300, show=show, tight=False)


def plot_phase2_certificate_components(
    frame: pd.DataFrame,
    *,
    source_column: str = "source",
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase2_certificate_components",
    show: bool = True,
) -> PlotResult:
    """Compare Phase 2 certificate components across supplied certificate rows."""

    specs = (
        (r"$\mathcal{B}_{N,\mathrm{out}}^X$", "B_out"),
        (r"$\mathcal{B}_{N,\mathrm{in}}^X$", "B_in"),
        (r"$\widehat{\kappa}_N^{X\leftrightarrow r}\widehat B_{N,M}^{\mathrm{mat,best}}$", "collocation"),
        (r"$\widehat{\varepsilon}_{N,M}^{X,\mathrm{fin,best}}$", "epsilon"),
    )
    _require_columns(frame, (source_column,) + tuple(column for _, column in specs), name="certificate summary")
    source_labels = {
        "historical whole-ellipse certificate row": "old whole-ellipse",
        "branch-image balanced candidate": "branch-image",
        "branch-image plus response-prefactor row": "branch + response",
        "wide branch-image candidate": "wide branch-image",
    }
    labels = [source_labels.get(str(value), str(value)) for value in frame[source_column]]
    colours = make_phase_palette(labels)
    x = np.arange(len(specs), dtype=float)
    width = min(0.18, 0.72 / max(1, len(labels)))
    offsets = (np.arange(len(labels)) - (len(labels) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(11.0, 5.9), facecolor="white")
    values_all: list[float] = []
    for offset, label, (_, row) in zip(offsets, labels, frame.iterrows()):
        values = np.asarray([float(row[column]) for _, column in specs])
        values_all.extend(values.tolist())
        ax.bar(x + offset, values, width=width, label=label, color=colours[label],
               edgecolor="white", linewidth=0.75, alpha=0.93)
    ax.set_yscale("log")
    ax.set_ylim(*positive_log_limits(values_all))
    ax.set_xticks(x)
    ax.set_xticklabels([label for label, _ in specs])
    ax.set_xlabel("certificate component")
    ax.set_ylabel("bound component")
    ax.set_title("Old whole-ellipse versus promoted branch-image certificate")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    ax.legend(title="certificate row", fontsize=14, title_fontsize=14,
              ncols=2, loc="upper right", frameon=True)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem, show=show)


def plot_phase2_certificate_comparison(
    certificate_rows: pd.DataFrame,
    tail_ratios: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase2_certificate_array_1x2",
    show: bool = True,
) -> PlotResult:
    """Render the Phase 2 ratio panel and complete-certificate comparison."""

    ratio_specs = (
        ("branch-wise over certified global", "branchwise_over_certified_global"),
        ("branch-image upper kernel over certified global", "kernel_over_certified_global"),
        ("branch-image upper kernel over sampled global", "kernel_over_sampled_global"),
    )
    _require_columns(tail_ratios, ("N",) + tuple(column for _, column in ratio_specs), name="tail ratios")
    _require_columns(certificate_rows, ("source", "epsilon"), name="certificate rows")
    ratio_colours = make_phase_palette(label for label, _ in ratio_specs)
    source_labels = {
        "historical whole-ellipse certificate row": "old whole-ellipse",
        "branch-image balanced candidate": "branch-image",
        "branch-image plus response-prefactor row": "branch + response",
        "wide branch-image candidate": "wide branch-image",
    }
    row_labels = [source_labels.get(str(value), str(value)) for value in certificate_rows["source"]]
    row_colours = make_phase_palette(row_labels)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 5.1), facecolor="white", constrained_layout=True)
    ratios_all: list[float] = []
    for label, column in ratio_specs:
        values = pd.to_numeric(tail_ratios[column], errors="coerce").to_numpy(dtype=float)
        ratios_all.extend(values.tolist())
        axes[0].plot(tail_ratios["N"], values, marker="o", lw=2.2,
                     color=ratio_colours[label], label=label)
    axes[0].axhline(1.0, color=THESIS_PALETTE["ref"], lw=1.2, ls="--",
                    alpha=0.75, label="whole-ellipse reference")
    axes[0].set_yscale("log")
    axes[0].set_ylim(*ratio_log_limits(ratios_all))
    axes[0].set_xlabel("Legendre dimension N")
    axes[0].set_ylabel("ratio")
    axes[0].set_title("Input-tail refinement ratios")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.42)
    axes[0].legend(fontsize=18.6, loc="best", frameon=True)
    eps = pd.to_numeric(certificate_rows["epsilon"], errors="coerce").to_numpy(dtype=float)
    for index, (label, value) in enumerate(zip(row_labels, eps)):
        axes[1].bar(index, value, width=0.62, color=row_colours[label],
                    edgecolor="white", linewidth=0.85, alpha=0.93, label=label)
    axes[1].set_yscale("log")
    axes[1].set_ylim(*positive_log_limits(eps))
    axes[1].set_xticks([])
    axes[1].set_xlabel("certificate row")
    axes[1].set_ylabel("total perturbation radius")
    axes[1].set_title("Complete Phase 2 certificate")
    axes[1].grid(True, which="both", axis="y", linestyle="--", alpha=0.4)
    axes[1].legend(title="certificate row", fontsize=18.4,
                   title_fontsize=18.4, loc="upper right", frameon=True)
    return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                        show=show, tight=False)


def plot_phase3_raw_errors_against_qstar(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    *,
    reference_curve: Mapping[str, Any] | None = None,
    q_star: float | None = None,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase3_raw_cluster_errors_vs_branch-image_qstar",
    show: bool = True,
) -> PlotResult:
    """Plot prepared Phase 3 raw cluster curves and an optional q-star guide."""

    _require_columns(frame, ("N", "name", "error"), name="Phase 3 raw curves")
    colours = make_phase_palette(tuple(target_names) + ("qstar",))
    fig, ax = plt.subplots(figsize=(10.8, 5.8), facecolor="white", constrained_layout=True)
    for target in target_names:
        subset = frame.loc[frame["name"].astype(str) == str(target)].sort_values("N")
        if subset.empty:
            continue
        ax.semilogy(subset["N"], subset["error"], marker="o", linewidth=1.8,
                    color=colours[target], label=latex_target_label(target))
    if reference_curve is None and q_star is not None:
        anchor = frame.loc[
            (frame["name"].astype(str) == "alpha^1")
            & (pd.to_numeric(frame["error"], errors="coerce") > 1.0e-14)
        ].sort_values("N")
        if not anchor.empty:
            anchor_row = anchor.iloc[-1]
            n_line = np.arange(int(frame["N"].min()), int(frame["N"].max()) + 1)
            constant = float(anchor_row["error"]) / float(q_star) ** int(anchor_row["N"])
            reference_curve = {
                "x": n_line,
                "y": constant * float(q_star) ** n_line,
                "label": r"reference $Cq_*^N$",
            }
    if reference_curve is not None:
        ax.semilogy(reference_curve["x"], reference_curve["y"], linestyle="--",
                    linewidth=2.0, color=THESIS_PALETTE["ref"],
                    label=reference_curve.get("label", r"reference $Cq_*^N$"))
    ax.set_ylim(*positive_log_limits(frame["error"], pad_decades=0.7))
    ax.set_xlabel(r"Legendre dimension $N$")
    ax.set_ylabel("raw cluster error")
    ax.set_title(r"Raw square-block cluster errors versus branch-image $q_*^N$")
    ax.grid(True, which="both", linestyle="--", alpha=0.50)
    ax.legend(frameon=True, ncol=2)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=230, show=show, tight=False)


def plot_phase3_normalised_errors(
    frame: pd.DataFrame,
    target_names: Sequence[str],
    *,
    value_column: str = "error_over_qstar_power",
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase3_raw_cluster_errors_normalised_by_qstar",
    show: bool = True,
) -> PlotResult:
    """Plot raw errors after division by the prepared q-star powers."""

    _require_columns(frame, ("N", "name", value_column), name="normalised Phase 3 curves")
    colours = make_phase_palette(target_names)
    fig, ax = plt.subplots(figsize=(10.8, 5.4), facecolor="white", constrained_layout=True)
    for target in target_names:
        subset = frame.loc[frame["name"].astype(str) == str(target)].sort_values("N")
        if subset.empty:
            continue
        ax.semilogy(subset["N"], subset[value_column], marker="o", linewidth=1.8,
                    color=colours[target], label=latex_target_label(target))
    ax.set_xlabel(r"Legendre dimension $N$")
    ax.set_ylabel(r"raw error divided by $q_*^N$")
    ax.set_title(r"Raw cluster errors normalised by the branch-image single-space base")
    ax.set_ylim(*positive_log_limits(frame[value_column], pad_decades=0.7))
    ax.grid(True, which="both", linestyle="--", alpha=0.50)
    ax.legend(frameon=True, ncol=2)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=230, show=show, tight=False)


def plot_phase3_certificate_components(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase3_certificate_components",
    show: bool = True,
) -> PlotResult:
    """Plot deterministic components grouped by Phase 3 certificate row."""

    specs = (
        (r"$\mathcal{B}_{N,\mathrm{out}}^X$", "B_out"),
        (r"$\mathcal{B}_{N,\mathrm{in}}^X$", "B_in"),
        (r"$\widehat\kappa_N^{X\leftrightarrow r}\widehat B_{N,M}^{\mathrm{mat,best}}$", "collocation"),
        (r"$\widehat\varepsilon_{N,M}^{X,\mathrm{fin,best}}$", "epsilon"),
    )
    _require_columns(frame, ("short_source",) + tuple(column for _, column in specs),
                     name="Phase 3 certificate rows")
    symbol_map = {
        "old": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{old}}$",
        "old whole-ellipse": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{old}}$",
        "branch-image": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}}$",
        "branch": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}}$",
        "response": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}+\mathrm{resp}}$",
        "branch + response": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}+\mathrm{resp}}$",
        "wide": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{wide}}$",
        "wide branch-image": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{wide}}$",
    }
    row_names = frame["short_source"].astype(str).tolist()
    row_symbols = [symbol_map.get(name.strip().lower(), name) for name in row_names]
    component_colours = make_phase_palette(label for label, _ in specs)
    x = np.arange(len(frame), dtype=float)
    width = 0.18
    offsets = (np.arange(len(specs)) - (len(specs) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(11.2, 5.8), facecolor="white", constrained_layout=True)
    values_all: list[float] = []
    for offset, (label, column) in zip(offsets, specs):
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        values_all.extend(values.tolist())
        ax.bar(x + offset, values, width=width, color=component_colours[label],
               edgecolor="white", linewidth=0.7, alpha=0.92, label=label)
    ax.set_yscale("log")
    ax.set_ylim(*positive_log_limits(values_all))
    ax.set_xticks(x, row_symbols)
    ax.set_xlabel("certificate row")
    ax.set_ylabel("bound component")
    ax.set_title("Deterministic certificate components after branch-image refinement")
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.42)
    ax.legend(ncols=2, frameon=True, loc="upper right", fontsize=16)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=230, show=show, tight=False)


def plot_phase3_rate_diagnostics(
    fit_frame: pd.DataFrame,
    *,
    q_star: float,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase3_rate_diagnostics",
    show: bool = True,
) -> PlotResult:
    """Plot the normalised deterministic radius and effective finite-N base."""

    _require_columns(fit_frame, ("name", "fitted_base", "epsilon_over_qstar_power"),
                     name="Phase 3 rate diagnostics")
    labels = fit_frame["name"].astype(str).tolist()
    symbol_map = {
        "old": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{old}}$",
        "old whole-ellipse": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{old}}$",
        "branch-image": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}}$",
        "branch": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}}$",
        "response": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}+\mathrm{resp}}$",
        "branch + response": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}+\mathrm{resp}}$",
        "wide": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{wide}}$",
        "wide branch-image": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{wide}}$",
    }
    row_symbols = [symbol_map.get(label.strip().lower(), label) for label in labels]
    colours = make_phase_palette(labels)
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 5.1), facecolor="white", constrained_layout=True)
    x = np.arange(len(fit_frame))
    axes[0].bar(x, fit_frame["epsilon_over_qstar_power"], color=[colours[label] for label in labels], edgecolor="white")
    axes[0].set_yscale("log")
    axes[0].set_xticks(x, row_symbols)
    axes[0].set_ylabel(r"$\widehat\varepsilon_{N,M}^{X,\mathrm{fin,best}}/q_*^N$")
    axes[0].set_title("Dimension-normalised deterministic radius")
    axes[0].grid(True, which="both", axis="y", linestyle="--", alpha=0.42)
    axes[1].bar(x, fit_frame["fitted_base"], color=[colours[label] for label in labels], edgecolor="white")
    axes[1].axhline(float(q_star), color=THESIS_PALETTE["ref"], ls="--", lw=1.8, label=r"$q_*$")
    axes[1].set_xticks(x, row_symbols)
    axes[1].set_ylabel(r"$(\widehat\varepsilon_{N,M}^{X,\mathrm{fin,best}})^{1/N}$")
    axes[1].set_title(r"Effective finite-$N$ base")
    axes[1].grid(True, which="major", axis="y", linestyle="--", alpha=0.42)
    row_handles = [
        Line2D((0,), (0,), marker="s", linestyle="", markersize=9,
               markerfacecolor=colours[label], markeredgecolor="white",
               label=symbol)
        for label, symbol in zip(labels, row_symbols)
    ]
    axes[1].legend(handles=[axes[1].lines[0], *row_handles],
                   frameon=True, fontsize=15, loc="lower right")
    return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                        dpi=230, show=show, tight=False)


def plot_phase3_empirical_deterministic_bridge(
    raw_curves: pd.DataFrame,
    certificate_rows: pd.DataFrame,
    components: Mapping[str, float],
    *,
    target_names: Sequence[str],
    epsilon: float,
    q_star: float,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase3_empirical_vs_deterministic_bridge",
    show: bool = True,
) -> PlotResult:
    """Render the three-panel Phase 3 empirical-to-certificate bridge."""

    _require_columns(raw_curves, ("N", "name", "error"), name="Phase 3 raw curves")
    current_rows = certificate_rows.loc[
        certificate_rows.get(
            "short_source",
            pd.Series("", index=certificate_rows.index),
        ).astype(str).eq("response")
    ]
    current_row = current_rows.iloc[0] if len(current_rows) else (
        certificate_rows.iloc[-1] if len(certificate_rows) else None
    )
    colours = make_phase_palette(tuple(target_names) + ("epsilon",))
    for target in ("mu^1", "mu^2"):
        if target in colours:
            colours[target] = THESIS_EXTRA_PALETTE["green"]
    fig = plt.figure(figsize=(19.0, 5.8), facecolor="white", constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.55, 1.15, 1.10))
    axes = np.asarray([fig.add_subplot(grid[0, index]) for index in range(3)], dtype=object)
    for target in target_names:
        subset = raw_curves.loc[raw_curves["name"].astype(str) == str(target)].sort_values("N")
        axes[0].semilogy(subset["N"], subset["error"], marker="o", lw=1.5,
                         color=colours[target], label=latex_target_label(target))
    axes[0].axhline(float(epsilon), color=THESIS_PALETTE["imag"], ls="-.", lw=1.8,
                    label=r"$\varepsilon_X$")
    qstar_reference = (
        float(current_row["qstar_power"])
        if current_row is not None
        and "qstar_power" in current_row
        and np.isfinite(float(current_row["qstar_power"]))
        else float(q_star) ** int(raw_curves["N"].max())
    )
    axes[0].axhline(qstar_reference, color=THESIS_PALETTE["ref"],
                    ls="--", lw=1.8, label=r"$q_*^N$")
    axes[0].set_title(r"Raw spectra versus branch-image $X$-bound", fontsize=18)
    axes[0].set_xlabel(r"Legendre dimension $N$", fontsize=16)
    axes[0].set_ylabel("raw cluster error and certified scale", fontsize=16)
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend(fontsize=20.5, ncol=2, frameon=True)
    if not certificate_rows.empty:
        labels = certificate_rows.get("short_source", certificate_rows.get("source", pd.Series(range(len(certificate_rows))))).astype(str)
        values = pd.to_numeric(certificate_rows.get("epsilon", np.nan), errors="coerce")
        label_colours = make_phase_palette(labels)
        row_x = np.arange(len(values))
        axes[1].bar(row_x, values, width=0.62,
                    color=[label_colours[label] for label in labels],
                    edgecolor="white", label=r"$\varepsilon_X$")
        row_qstar = pd.to_numeric(certificate_rows.get("qstar_power", qstar_reference), errors="coerce")
        axes[1].scatter(row_x, row_qstar, marker="D", s=64,
                        color=THESIS_PALETTE["ref"], edgecolor="white",
                        zorder=4, label=r"$q_*^N$")
        for x_value, eps_value, base_value in zip(row_x, values, row_qstar):
            if np.isfinite(eps_value) and np.isfinite(base_value) and eps_value > 0 and base_value > 0:
                axes[1].annotate(rf"$\times {eps_value / base_value:.2g}$",
                                 xy=(x_value, eps_value), xytext=(0, 8),
                                 textcoords="offset points", ha="center", va="bottom",
                                 fontsize=9.5, color="#303030")
        axes[1].set_yscale("log")
        symbol_map = {
            "old": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{old}}$",
            "old whole-ellipse": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{old}}$",
            "branch-image": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}}$",
            "response": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}+\mathrm{resp}}$",
            "branch + response": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{br}+\mathrm{resp}}$",
            "wide": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{wide}}$",
            "wide branch-image": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{wide}}$",
        }
        axes[1].set_xticks(row_x, [symbol_map.get(label.lower(), label) for label in labels])
        axes[1].set_ylim(*positive_log_limits(
            list(values.to_numpy(dtype=float)) + list(row_qstar.to_numpy(dtype=float)),
            pad_decades=0.45,
        ))
        axes[1].grid(True, which="both", axis="y", linestyle="--", alpha=0.45)
        axes[1].legend(frameon=True, fontsize=20.5)
    axes[1].set_title("Certified rows at N=600", fontsize=18)
    axes[1].set_ylabel(r"$\widehat\varepsilon_{N,M}^{X,\mathrm{fin,best}}$ scale",
                       fontsize=16)
    component_keys = list(components)
    component_label_map = {
        "output leakage": r"$\mathcal{B}_{N,\mathrm{out}}^X$",
        "input leakage": r"$\mathcal{B}_{N,\mathrm{in}}^X$",
        "collocation defect": r"$\widehat\kappa_N^{X\leftrightarrow r}\widehat B_{N,M}^{\mathrm{mat,best}}$",
        "complete radius": r"$\widehat\varepsilon_{N,M}^{X,\mathrm{fin,best}}$",
    }
    component_labels = [component_label_map.get(key, key) for key in component_keys]
    component_values = [float(components[key]) for key in component_keys]
    component_colours = make_phase_palette(component_labels)
    axes[2].bar(np.arange(len(component_values)), component_values,
                width=0.62,
                color=[component_colours[label] for label in component_labels],
                edgecolor="white")
    axes[2].set_yscale("log")
    axes[2].axhline(qstar_reference, color=THESIS_PALETTE["ref"],
                    linestyle="--", linewidth=2.0, label=r"$q_*^N$")
    axes[2].set_ylim(*positive_log_limits(
        component_values + [qstar_reference],
        pad_decades=0.45,
    ))
    axes[2].set_xticks(np.arange(len(component_labels)), component_labels)
    axes[2].set_ylabel("component size", fontsize=16)
    axes[2].set_title("Promoted branch-image row", fontsize=18)
    axes[2].grid(True, which="both", axis="y", linestyle="--", alpha=0.45)
    axes[2].legend(frameon=True, fontsize=20.5)
    for axis in axes:
        axis.tick_params(axis="both", which="major", labelsize=21)
        axis.tick_params(axis="both", which="minor", labelsize=18)
    fig.suptitle("Phase 3 bridge: raw spectral diagnostics and deterministic certificate structure",
                 fontsize=20)
    return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                        dpi=230, show=show, tight=False)


def plot_phase3_certified_row_extrapolation(
    frame: pd.DataFrame,
    *,
    certified_n: int,
    certified_epsilon: float,
    value_columns: Sequence[str],
    output_dir: str | Path | None = None,
    stem: str = "branch_image_phase3_schur_components_vs_qstar",
    compatibility_stem: str | None = "branch_image_phase3_table_and_schur_components_vs_qstar",
    show: bool = True,
) -> PlotResult:
    """Plot diagnostic component extrapolations anchored at one certified row."""

    _require_columns(frame, ("N",) + tuple(value_columns), name="Phase 3 extrapolation table")
    label_map = {
        "B_out_extrapolated": "output geometric extrapolation",
        "B_in_extrapolated": "input geometric extrapolation",
        "collocation_extrapolated": "collocation geometric extrapolation",
        "epsilon_extrapolated": "RSS-plus-collocation extrapolation",
        "Ccert_qstar_power": r"anchored $C_{\mathrm{cert}}q_\ast^N$ guide",
    }
    labels = [label_map.get(column, column) for column in value_columns]
    colours = make_phase_palette(labels)
    fig, ax = plt.subplots(figsize=(9.8, 6.3), facecolor="white", constrained_layout=True)
    markers = ("o", "s", "D", "o", None)
    for index, column in enumerate(value_columns):
        label = labels[index]
        style = {"linestyle": "--", "linewidth": 2.0} if column == "Ccert_qstar_power" else {"linestyle": "-", "linewidth": 2.2 if column == "epsilon_extrapolated" else 1.8}
        colour = THESIS_PALETTE["imag"] if column == "epsilon_extrapolated" else THESIS_PALETTE["ref"] if column == "Ccert_qstar_power" else colours[label]
        ax.semilogy(frame["N"], frame[column], marker=markers[index],
                    color=colour, label=label, **style)
    ax.scatter([certified_n], [certified_epsilon], marker="*", s=850,
               color=THESIS_PALETTE["real"], edgecolor="white", linewidth=1.2,
               zorder=5, label=rf"only certified row: $N={int(certified_n)}$")
    ax.set_xlabel(r"Legendre dimension $N$")
    ax.set_ylabel("diagnostic extrapolated component")
    ax.set_title(rf"Diagnostic geometric extrapolation anchored at the certified $N={int(certified_n)}$ row")
    positive_values = np.concatenate([
        pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        for column in value_columns
    ])
    positive_values = positive_values[
        np.isfinite(positive_values) & (positive_values > 0)
    ]
    if positive_values.size:
        ax.set_ylim(
            10.0 ** (np.floor(np.log10(positive_values.min())) - 0.8),
            10.0 ** (np.ceil(np.log10(positive_values.max())) + 0.5),
        )
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(frameon=True, fontsize=15, loc="lower left")
    result = _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                          dpi=260, show=show, tight=False)
    if output_dir is not None and compatibility_stem:
        extra = save_figure(fig, output_dir, compatibility_stem, dpi=260)
        result = PlotResult(fig, ax, result.paths + extra)
    return result


def plot_phase4_sampled_validation(
    frame: pd.DataFrame,
    *,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_sampled_validation_1x2",
    show: bool = True,
) -> PlotResult:
    """Render the retained two-panel sampled Phase 4 diagnostic."""

    _require_columns(frame, ("target", "small_gain", "radius", "cluster_error"), name="sampled validation")
    ordered = frame.copy()
    labels = ordered["target"].astype(str).map(latex_target_label)
    families = (
        ordered["family"].astype(str)
        if "family" in ordered.columns
        else ordered["target"].astype(str).map(
            lambda value: "alpha packet" if value.startswith("alpha")
            else "mu packet" if value.startswith("mu") else "fixed point"
        )
    )
    family_colours = {
        "alpha packet": THESIS_PALETTE["real"],
        "mu packet": THESIS_PALETTE["imag"],
        "fixed point": THESIS_PALETTE["main"],
    }
    bar_colours = [family_colours.get(value, THESIS_PALETTE["main"]) for value in families]
    y = np.arange(len(frame))
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.6), facecolor="white", constrained_layout=True)
    axes[0].barh(y, ordered["small_gain"], color=bar_colours, edgecolor="white",
                 linewidth=0.7, alpha=0.94)
    axes[0].axvline(1.0, color=THESIS_PALETTE["ref"], linestyle="--",
                    linewidth=1.7, label="small-gain threshold")
    axes[0].set_xscale("log")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=12)
    axes[0].set_xlabel(r"$\varepsilon_X m_\Gamma^X$", fontsize=16)
    axes[0].set_title("Sampled small-gain factors", fontsize=20)
    axes[0].grid(True, which="both", axis="x", linestyle="--", alpha=0.45)
    axes[0].legend(fontsize=12, loc="lower right", frameon=True)
    axes[1].barh(y, ordered["cluster_error"], color=bar_colours, edgecolor="white",
                 linewidth=0.7, alpha=0.94, label="finite cluster error")
    axes[1].scatter(ordered["radius"], y, marker="D", s=38,
                    color=THESIS_PALETTE["main"], label="contour radius")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_xscale("log")
    axes[1].set_xlabel("error and radius scale", fontsize=16)
    axes[1].set_title("Finite cluster error versus contour radius", fontsize=20)
    axes[1].grid(True, which="both", axis="x", linestyle="--", alpha=0.45)
    axes[1].legend(fontsize=11, loc="lower right", frameon=True)
    return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                        dpi=240, show=show, tight=False)


def plot_phase4_small_gain_values(
    frame: pd.DataFrame,
    *,
    target_column: str = "target",
    value_column: str = "small_gain",
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_small_gain_horizontal",
    show: bool = True,
) -> PlotResult:
    """Plot sampled or certified small-gain values as horizontal bars."""

    _require_columns(frame, (target_column, value_column), name="small-gain table")
    families = (
        frame["family"].astype(str)
        if "family" in frame.columns
        else frame[target_column].astype(str).map(
            lambda value: "alpha packet" if value.startswith("alpha")
            else "mu packet" if value.startswith("mu") else "fixed point"
        )
    )
    family_colours = {
        "alpha packet": THESIS_PALETTE["real"],
        "mu packet": THESIS_PALETTE["imag"],
        "fixed point": THESIS_PALETTE["main"],
    }
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(9.6, 7.2), facecolor="white", constrained_layout=True)
    ax.barh(y, frame[value_column],
            color=[family_colours.get(value, THESIS_PALETTE["main"]) for value in families],
            edgecolor="white", linewidth=0.7, alpha=0.94)
    ax.axvline(1.0, color=THESIS_PALETTE["ref"], linestyle="--",
               linewidth=1.7, label="threshold")
    ax.set_xscale("log")
    ax.set_yticks(y)
    ax.set_yticklabels([latex_target_label(v) for v in frame[target_column]], fontsize=12)
    ax.set_xlabel(r"$\varepsilon_X m_\Gamma^X$", fontsize=16)
    ax.set_title("First fifteen sampled small-gain values", fontsize=20)
    ax.grid(True, which="both", axis="x", linestyle="--", alpha=0.45)
    ax.legend(fontsize=12, loc="lower right", frameon=True)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=240, show=show, tight=False)


def plot_phase4_moat_heatmap(
    profile_log10: np.ndarray,
    theta_values: Sequence[float],
    packet_labels: Sequence[str],
    *,
    theta_at_minimum: Sequence[float] | None = None,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_contour_moat_heatmap",
    show: bool = True,
) -> PlotResult:
    """Plot a prepared complete-contour moat profile array as a heatmap."""

    values = np.asarray(profile_log10, dtype=float)
    if values.ndim != 2:
        raise ValueError("profile_log10 must be a two-dimensional array")
    cmap = LinearSegmentedColormap.from_list(
        "phase4_moat_heatmap", (THESIS_PALETTE["imag"], THESIS_PALETTE["real"], THESIS_PALETTE["main"])
    )
    fig, ax = plt.subplots(figsize=(13.2, 7.4), facecolor="white", constrained_layout=True)
    theta = np.asarray(theta_values, dtype=float)
    image = ax.imshow(values, aspect="auto", origin="upper", cmap=cmap,
                      interpolation="nearest",
                      extent=(theta.min(), theta.max(), len(packet_labels) - 0.5, -0.5))
    ax.set_yticks(np.arange(len(packet_labels)))
    ax.set_yticklabels([latex_target_label(v) for v in packet_labels], fontsize=13)
    if theta_at_minimum is not None:
        for row, angle in enumerate(theta_at_minimum):
            ax.scatter(float(angle), row, s=24, color="white", edgecolor="black",
                       linewidth=0.35, zorder=3)
    ax.set_xlabel(r"contour parameter $\theta$", fontsize=18)
    ax.set_ylabel("spectral packet", fontsize=18)
    ax.set_title("Phase 4 contour-moat heatmap", fontsize=22)
    ax.tick_params(axis="x", labelsize=14)
    cbar = fig.colorbar(image, ax=ax, pad=0.015)
    cbar.set_label(r"$\log_{10}\sigma_{\min}(z(\theta)I-A_N^X)$", fontsize=16)
    cbar.ax.tick_params(labelsize=13)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=240, show=show, tight=False)


def plot_phase4_moat_profiles(
    theta: Sequence[float],
    profiles: Mapping[str, Sequence[float]],
    *,
    epsilon_line: float | None = None,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_contour_moat_profiles_lines",
    show: bool = True,
) -> PlotResult:
    """Plot complete-contour moat profiles supplied packet by packet."""

    colours = make_phase_palette(profiles)
    fig, ax = plt.subplots(figsize=(13.8, 7.8), facecolor="white", constrained_layout=True)
    for name, values in profiles.items():
        ax.plot(theta, values, lw=1.45, marker="o", markersize=2.2,
                alpha=0.90, color=colours[name], label=latex_target_label(name))
    if epsilon_line is not None:
        ax.axhline(float(epsilon_line), color=THESIS_PALETTE["ref"],
                   linestyle="--", linewidth=1.8, alpha=0.82,
                   label=r"$\varepsilon_X$")
    ax.set_yscale("log")
    theta_values = np.asarray(theta, dtype=float)
    if theta_values.size:
        ax.set_xlim(float(theta_values.min()), float(theta_values.max()))
    ax.set_xlabel(r"contour parameter $\theta$", fontsize=18)
    ax.set_ylabel(r"$\sigma_{\min}(z(\theta)I-A_N^X)$", fontsize=18)
    ax.set_title("Phase 4 first-fifteen Hardy-gauge contour moat profiles", fontsize=22)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid(True, which="both", linestyle="--", alpha=0.38)
    ax.legend(fontsize=10, ncol=4, loc="lower left", frameon=True)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=260, show=show, tight=False)


def plot_phase4_moat_minima_table(
    frame: pd.DataFrame,
    *,
    columns: Sequence[str],
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_contour_profile_minima_table",
    show: bool = True,
) -> PlotResult:
    """Render selected moat-minimum columns as a Matplotlib table figure."""

    _require_columns(frame, columns, name="moat minima table")
    fig, ax = plt.subplots(figsize=(12.8, 6.8), facecolor="white", constrained_layout=True)
    ax.axis("off")
    display_frame = frame.loc[:, columns].copy()
    if "target" in display_frame:
        display_frame["target"] = display_frame["target"].map(latex_target_label)
    if "theta_at_profile_min" in display_frame:
        display_frame["theta_at_profile_min"] = display_frame["theta_at_profile_min"].map(
            lambda value: f"{float(value):.3f}"
        )
    for column in ("profile_s_min", "epsilon_m_gamma"):
        if column in display_frame:
            display_frame[column] = display_frame[column].map(lambda value: f"{float(value):.3e}")
    headings = {
        "rank": "rank",
        "target": "target",
        "theta_at_profile_min": r"$\theta_{\min}$",
        "profile_s_min": r"$s_{\min}$",
        "epsilon_m_gamma": r"$\varepsilon_X m_\Gamma^X$",
    }
    table = ax.table(cellText=display_frame.astype(str).to_numpy(),
                     colLabels=[headings.get(column, column) for column in columns],
                     cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    table.scale(1.0, 1.35)
    ax.set_title("First-fifteen contour-moat minima", fontsize=20, pad=16)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=240, show=show, tight=False)


def plot_phase4_moat_minima(
    frame: pd.DataFrame,
    *,
    target_column: str = "target",
    moat_column: str = "s_min_gamma",
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_moat_minima_profile",
    show: bool = True,
) -> PlotResult:
    """Plot one certified or sampled moat minimum per packet."""

    _require_columns(frame, (target_column, moat_column), name="moat minima")
    labels = frame[target_column].astype(str).tolist()
    families = (
        frame["family"].astype(str)
        if "family" in frame.columns
        else frame[target_column].astype(str).map(
            lambda value: "alpha packet" if value.startswith("alpha")
            else "mu packet" if value.startswith("mu") else "fixed point"
        )
    )
    family_colours = {
        "alpha packet": THESIS_PALETTE["real"],
        "mu packet": THESIS_PALETTE["imag"],
        "fixed point": THESIS_PALETTE["main"],
    }
    colours = [family_colours.get(value, THESIS_PALETTE["main"]) for value in families]
    x = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(12.8, 6.6), facecolor="white", constrained_layout=True)
    ax.scatter(x, frame[moat_column], s=74, color=colours,
               edgecolor="white", linewidth=0.7, zorder=3)
    ax.plot(x, frame[moat_column], color=THESIS_PALETTE["main"],
            linewidth=1.4, alpha=0.55)
    ax.set_yscale("log")
    ax.set_xticks(x, [latex_target_label(label) for label in labels], rotation=35)
    ax.set_xticklabels([latex_target_label(label) for label in labels],
                       rotation=35, ha="right", fontsize=12)
    ax.set_ylabel("sampled moat minimum", fontsize=16)
    ax.set_title("First-fifteen sampled contour-moat minima", fontsize=20)
    ax.grid(True, which="both", axis="y", linestyle="--", alpha=0.42)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=240, show=show, tight=False)


def plot_phase4_single_packet_profile(
    theta: Sequence[float],
    sampled_profile: Sequence[float],
    *,
    packet_label: str,
    epsilon_x: float | None = None,
    d_gamma0: float | None = None,
    certified_lower: float | None = None,
    sampled_minimum: float | None = None,
    output_dir: str | Path | None = None,
    stem: str | None = None,
    show: bool = True,
) -> PlotResult:
    """Plot one packet's sampled moat profile with certified reference levels."""

    theta_values = np.asarray(theta, dtype=float)
    profile = np.asarray(sampled_profile, dtype=float)
    if profile.size == 0 or theta_values.size != profile.size:
        raise ValueError("theta and sampled_profile must be non-empty and have equal lengths")
    minimum_index = int(np.nanargmin(profile))
    minimum = float(profile[minimum_index] if sampled_minimum is None else sampled_minimum)
    minimum_theta = float(theta_values[minimum_index])
    epsilon = epsilon_x if epsilon_x is not None else certified_lower
    silent_distance = float(d_gamma0) if d_gamma0 is not None else math.inf
    lifted_resolvent = max(1.0 / minimum, 0.0 if math.isinf(silent_distance) else 1.0 / silent_distance)
    small_gain = None if epsilon is None else float(epsilon) * lifted_resolvent
    safe = str(packet_label).replace("^", "")
    rc = {
        "font.family": "DejaVu Sans", "mathtext.fontset": "cm",
        "axes.titlesize": 22, "axes.labelsize": 18,
        "xtick.labelsize": 15, "ytick.labelsize": 15,
        "legend.fontsize": 14,
    }
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(9.8, 5.3), facecolor="white", constrained_layout=True)
        ax.plot(theta_values, profile, color=THESIS_PALETTE["real"], linewidth=2.35,
                label=r"$\sigma_{\min}(z(\theta)I_N-A_{N,M}^{X})$")
        if epsilon is not None:
            ax.axhline(float(epsilon), color=THESIS_PALETTE["main"],
                       linestyle=(0, (2.2, 2.2)), linewidth=2.2,
                       label=r"$\widehat\varepsilon_{N,M}^{X,\mathrm{fin}}$")
        ax.scatter([minimum_theta], [minimum], s=190, color=THESIS_PALETTE["imag"],
                   edgecolor="white", linewidth=1.3, zorder=5,
                   label=r"$s_{\Gamma,N}^{\mathrm{H},J}$")
        ax.annotate(r"$s_{\Gamma,N}^{\mathrm{H},J}$", xy=(minimum_theta, minimum),
                    xytext=(18, 14), textcoords="offset points",
                    color=THESIS_PALETTE["imag"], fontsize=19,
                    arrowprops={"arrowstyle": "-", "color": THESIS_PALETTE["imag"], "linewidth": 1.2})
        if epsilon is not None:
            ax.annotate(r"$\widehat\varepsilon_{N,M}^{X,\mathrm{fin}}$",
                        xy=(74, float(epsilon)), xytext=(0, 13),
                        textcoords="offset points", color=THESIS_PALETTE["main"],
                        fontsize=17, ha="center")
        ax.set_yscale("log")
        positive_reference = minimum if epsilon is None else min(float(epsilon), minimum)
        y_min = max(positive_reference / 3.0, 1.0e-20)
        y_max = max(float(np.nanmax(profile)) * 3.0,
                    0.0 if epsilon is None else float(epsilon) * 4.0)
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, 360)
        ax.set_xticks([0, 90, 180, 270, 360])
        ax.set_xlabel(r"contour parameter $\theta$")
        ax.set_ylabel(r"$\sigma_{\min}(z(\theta)I_N-A_{N,M}^{X})$")
        ax.set_title(rf"Sampled Hardy-gauge moat profile for {latex_target_label(packet_label)}")
        ax.grid(True, which="major", linestyle="-", alpha=0.28)
        ax.grid(True, which="minor", linestyle="--", alpha=0.17)
        ax.legend(loc="upper left", frameon=False)
        if small_gain is not None:
            status_text = (
                rf"$\widehat m_{{\Gamma,N}}^{{\mathrm{{H}},J}}={lifted_resolvent:.3e}$"
                + "\n" +
                rf"$\widehat\varepsilon_{{N,M}}^{{X,\mathrm{{fin}}}}\widehat m_{{\Gamma,N}}^{{\mathrm{{H}},J}}={small_gain:.3e}$"
            )
            ax.text(0.985, 0.04, status_text, transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=13, color="#333333",
                    bbox={"boxstyle": "round,pad=0.35", "facecolor": "white",
                          "edgecolor": THESIS_PALETTE["ref"], "alpha": 0.88})
        return _finish_plot(
            fig, ax, output_dir=output_dir,
            stem=stem or f"branch_image_wide_candidate_sampled_hardy_moat_profile_{safe}_N600_M610",
            dpi=300, show=show, tight=False,
        )


def plot_phase4_fragile_robustness(
    frame: pd.DataFrame,
    *,
    sample_column: str = "J",
    value_column: str = "epsilon_m_gamma",
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_fragile_robustness",
    show: bool = True,
) -> PlotResult:
    """Plot the fragile-packet diagnostic against angular sample count."""

    _require_columns(frame, (sample_column, value_column), name="fragile robustness table")
    fig, ax = plt.subplots(figsize=(8.4, 4.8), facecolor="white", constrained_layout=True)
    target_column = "target" if "target" in frame.columns else None
    groups = frame.groupby(target_column, sort=False) if target_column else [("profile", frame)]
    for target, group in groups:
        colour = THESIS_PALETTE["real"] if str(target).startswith("alpha") else THESIS_PALETTE["imag"]
        ax.plot(group[sample_column], group[value_column], marker="o", linewidth=2.0,
                color=colour, label=latex_target_label(target))
    ax.axhline(1.0, color=THESIS_PALETTE["ref"], linestyle="--",
               linewidth=1.5, label="threshold")
    ax.set_yscale("log")
    ax.set_xlabel("angular samples J")
    ax.set_ylabel(r"$\varepsilon_X m_\Gamma^X$")
    ax.set_title("Fragile-packet sampled moat robustness")
    ax.grid(True, which="both", linestyle="--", alpha=0.45)
    ax.legend(frameon=True)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=240, show=show, tight=False)


def _normalise_contours(contours: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for item in contours:
        normalised.append({
            "centre": complex(item["centre"]),
            "radius": float(item["radius"]),
            "name": str(item.get("name", item.get("target", "packet"))),
            "family": str(item.get("family", "packet")),
            "colour": item.get("colour"),
        })
    return normalised


def _plot_moat_surface(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    moat_grid: np.ndarray,
    contours: Sequence[Mapping[str, Any]],
    *,
    eigenvalues: Sequence[complex] = (),
    title_3d: str,
    title_2d: str,
    output_dir: str | Path | None,
    stem: str,
    dpi: int,
    figsize: tuple[float, float],
    family_colours: Mapping[str, str] | None = None,
    default_contour_colour: str | None = None,
    respect_record_colours: bool = True,
    point_facecolour: str = "none",
    point_edgecolour: str | None = None,
    point_label_colour: str | None = None,
    eigenvalue_colour: str | None = None,
    show: bool = True,
) -> PlotResult:
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    moat = np.asarray(moat_grid, dtype=float)
    if x.shape != y.shape or x.shape != moat.shape:
        raise ValueError("x_grid, y_grid and moat_grid must have identical shapes")
    z_log = np.log10(np.maximum(moat, np.finfo(float).tiny))
    cmap = make_phase4_moat_cmap()
    fig = plt.figure(figsize=figsize, facecolor="white", constrained_layout=False)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.05, 0.035),
                            left=0.045, right=0.90, bottom=0.105,
                            top=0.845, wspace=0.22)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax2d = fig.add_subplot(grid[0, 1])
    cax = fig.add_subplot(grid[0, 2])
    ax3d.plot_surface(x, y, z_log, cmap=cmap, linewidth=0,
                      antialiased=True, alpha=0.82, shade=True)
    levels = np.linspace(float(np.nanmin(z_log)), float(np.nanmax(z_log)), 36)
    filled = ax2d.contourf(x, y, z_log, levels=levels, cmap=cmap)
    cbar = fig.colorbar(filled, cax=cax)
    cbar.set_label(r"$\log_{10}\sigma_{\min}$", fontsize=13)
    cbar.ax.tick_params(labelsize=11)
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    records = _normalise_contours(contours)
    family_names = list(dict.fromkeys(record["family"] for record in records))
    resolved_family_colours = dict(
        make_phase_palette(family_names)
        if family_colours is None
        else family_colours
    )
    for record in records:
        centre = record["centre"]
        record_colour = record["colour"] if respect_record_colours else None
        colour = (
            record_colour
            or resolved_family_colours.get(record["family"])
            or default_contour_colour
            or THESIS_PALETTE["main"]
        )
        cx = centre.real + record["radius"] * np.cos(theta)
        cy = centre.imag + record["radius"] * np.sin(theta)
        ax2d.plot(cx, cy, color=colour, lw=1.7)
        ax2d.scatter(
            [centre.real], [centre.imag], s=60,
            facecolors=point_facecolour,
            edgecolors=point_edgecolour or colour,
            lw=1.4, zorder=5,
        )
        ax2d.text(centre.real, centre.imag, latex_target_label(record["name"]),
                  fontsize=11, color=point_label_colour or colour,
                  ha="left", va="bottom")
        ax3d.plot(cx, cy, np.full_like(cx, np.nanmin(z_log)), color=colour,
                  lw=1.2, alpha=0.8)
    eig = np.asarray(eigenvalues, dtype=np.complex128)
    if eig.size:
        eig_colour = eigenvalue_colour or THESIS_PALETTE["main"]
        ax2d.scatter(eig.real, eig.imag, s=18, color=eig_colour,
                     alpha=0.70, label="finite eigenvalues")
        ax3d.scatter(eig.real, eig.imag, np.full(eig.size, np.nanmin(z_log)),
                     s=10, color=eig_colour, alpha=0.70)
    ax3d.set_xlabel(r"$\operatorname{Re}\zeta$")
    ax3d.set_ylabel(r"$\operatorname{Im}\zeta$")
    ax3d.set_zlabel(r"$\log_{10}\sigma_{\min}$")
    ax3d.set_title(title_3d)
    ax2d.set_xlabel(r"$\operatorname{Re}\zeta$")
    ax2d.set_ylabel(r"$\operatorname{Im}\zeta$")
    ax2d.set_title(title_2d)
    ax2d.set_aspect("equal", adjustable="box")
    if eig.size:
        ax2d.legend(frameon=False, loc="upper right")
    return _finish_plot(fig, (ax3d, ax2d, cax), output_dir=output_dir,
                        stem=stem, dpi=dpi, show=show, tight=False)


def plot_phase4_local_moat_surface(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    moat_grid: np.ndarray,
    contours: Sequence[Mapping[str, Any]],
    *,
    eigenvalues: Sequence[complex] = (),
    profile_theta: Sequence[float] | None = None,
    profile_moat: Sequence[float] | None = None,
    alpha: float = 0.65,
    mu: float = 0.3,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_mu2_hardy_moat_surface_3d_2d_N600_M610",
    show: bool = True,
) -> PlotResult:
    """Render the local mu-squared three-dimensional and top-down moat figure."""

    x_mesh = np.asarray(x_grid, dtype=float)
    y_mesh = np.asarray(y_grid, dtype=float)
    moat = np.asarray(moat_grid, dtype=float)
    if x_mesh.ndim == 1 and y_mesh.ndim == 1:
        x_axis, y_axis = x_mesh, y_mesh
        x_mesh, y_mesh = np.meshgrid(x_axis, y_axis)
    else:
        x_axis, y_axis = x_mesh[0, :], y_mesh[:, 0]
    if x_mesh.shape != y_mesh.shape or x_mesh.shape != moat.shape:
        raise ValueError("x_grid, y_grid and moat_grid must have identical surface shapes")
    if moat.size == 0:
        raise ValueError("moat_grid must be non-empty")
    if not np.all(np.isfinite(moat)):
        raise ValueError("moat_grid must contain only finite values")
    if np.any(moat <= 0.0):
        raise ValueError("moat_grid must contain only strictly positive values")
    records = _normalise_contours(contours)
    if not records:
        raise ValueError("The local moat figure requires one accepted contour")
    record = records[0]
    centre = record["centre"]
    radius = record["radius"]
    log_floor = min(1.0e-18, float(np.min(moat)))
    z_log = np.log10(np.maximum(moat, log_floor))
    z_floor = float(np.nanmin(z_log))
    xlim = (float(np.min(x_axis)), float(np.max(x_axis)))
    ylim = (float(np.min(y_axis)), float(np.max(y_axis)))
    theta = np.linspace(0.0, 2.0 * np.pi, 512)
    contour_x = centre.real + radius * np.cos(theta)
    contour_y = centre.imag + radius * np.sin(theta)
    if profile_theta is not None and profile_moat is not None:
        source_theta = np.asarray(profile_theta, dtype=float)
        source_log = np.log10(np.maximum(np.asarray(profile_moat, dtype=float), log_floor))
        order = np.argsort(source_theta)
        contour_z = np.interp(theta, source_theta[order], source_log[order], period=2.0 * np.pi)
    else:
        from scipy.interpolate import RegularGridInterpolator

        interpolator = RegularGridInterpolator(
            (y_axis, x_axis), z_log, bounds_error=False, fill_value=z_floor
        )
        contour_z = interpolator(np.column_stack((contour_y, contour_x)))
    eig = np.asarray(eigenvalues, dtype=np.complex128)
    if eig.size:
        eig = eig[
            (eig.real >= xlim[0]) & (eig.real <= xlim[1])
            & (eig.imag >= ylim[0]) & (eig.imag <= ylim[1])
        ]
    exact_targets: list[tuple[str, float]] = []
    for family, base, maximum in (("alpha", float(alpha), 17), ("mu", float(mu), 7)):
        for power in range(1, maximum + 1):
            value = base ** power
            if xlim[0] <= value <= xlim[1]:
                exact_targets.append((rf"$\{family}^{power}$", value))
    cmap = make_phase4_moat_cmap("phase4_branch_image_local_moat")
    fig = plt.figure(figsize=(15.5, 6.4), facecolor="white")
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)
    ax3d.plot_surface(x_mesh, y_mesh, z_log, cmap=cmap, linewidth=0.0,
                      antialiased=True, alpha=0.92)
    ax3d.plot(contour_x, contour_y, contour_z, color=THESIS_PALETTE["real"],
              linewidth=2.4, label=r"$\mu^2$ contour")
    if eig.size:
        ax3d.scatter(eig.real, eig.imag, np.full(eig.size, z_floor),
                     color=THESIS_PALETTE["real"], s=28, depthshade=False,
                     label="finite eigenvalues")
    for label, value in exact_targets:
        ax3d.scatter([value], [0.0], [z_floor], facecolors="none",
                     edgecolors=THESIS_PALETTE["real"], s=58, depthshade=False)
        ax3d.text(value, 0.0, z_floor, label, color=THESIS_PALETTE["main"], fontsize=18)
    ax3d.set_title(r"3D sampled Hardy-gauge moat near $\mu^2$", fontsize=18)
    ax3d.set_xlabel(r"$\operatorname{Re}\zeta$", fontsize=16)
    ax3d.set_ylabel(r"$\operatorname{Im}\zeta$", fontsize=16)
    ax3d.set_zlabel(r"$\log_{10}\sigma_{\min}(\zeta I-A_N^X)$", fontsize=16)
    ax3d.tick_params(axis="both", which="major", labelsize=14)
    ax3d.view_init(elev=28, azim=-58)
    ax3d.legend(loc="upper left", fontsize=14)
    levels = np.linspace(float(np.nanmin(z_log)), float(np.nanmax(z_log)), 26)
    filled = ax2d.contourf(x_mesh, y_mesh, z_log, levels=levels, cmap=cmap)
    ax2d.contour(x_mesh, y_mesh, z_log, levels=levels[::4],
                 colors=THESIS_PALETTE["ref"], linewidths=0.45, alpha=0.55)
    ax2d.plot(contour_x, contour_y, color=THESIS_PALETTE["real"],
              linewidth=2.0, label=r"$\mu^2$ contour")
    if eig.size:
        ax2d.scatter(eig.real, eig.imag, color=THESIS_PALETTE["real"], s=30,
                     label="finite eigenvalues", zorder=4)
    for label, value in exact_targets:
        ax2d.scatter([value], [0.0], facecolors="none",
                     edgecolors=THESIS_PALETTE["real"], s=68,
                     linewidths=1.4, zorder=5)
        ax2d.annotate(label, (value, 0.0), textcoords="offset points",
                      xytext=(4, 5), color=THESIS_PALETTE["main"], fontsize=18)
    ax2d.axhline(0.0, color=THESIS_PALETTE["ref"], linewidth=0.9, alpha=0.75)
    ax2d.set_aspect("equal", adjustable="box")
    ax2d.set_xlim(*xlim)
    ax2d.set_ylim(*ylim)
    ax2d.set_title("2D moat map and accepted contour", fontsize=18)
    ax2d.set_xlabel(r"$\operatorname{Re}\zeta$", fontsize=16)
    ax2d.set_ylabel(r"$\operatorname{Im}\zeta$", fontsize=16)
    ax2d.tick_params(axis="both", which="major", labelsize=14)
    ax2d.legend(loc="upper right", fontsize=14)
    colourbar = fig.colorbar(filled, ax=ax2d, fraction=0.046, pad=0.04)
    colourbar.set_label(r"$\log_{10}\sigma_{\min}(\zeta I-A_N^X)$", fontsize=16)
    colourbar.ax.tick_params(labelsize=12)
    fig.suptitle(r"Phase 4 sampled finite-section moat landscape near $\mu^2$", fontsize=18)
    fig.tight_layout()
    return _finish_plot(fig, (ax3d, ax2d), output_dir=output_dir, stem=stem,
                        dpi=220, show=show, tight=False)


def plot_phase4_global_moat_surface(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    moat_grid: np.ndarray,
    contours: Sequence[Mapping[str, Any]],
    *,
    zoom_surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    deep_zoom_surface: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    eigenvalues: Sequence[complex] = (),
    alpha: float = 0.65,
    mu: float = 0.3,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_global_hardy_moat_surface_3d_2d_N600_M610",
    show: bool = True,
) -> PlotResult:
    """Render the global three-dimensional and top-down moat figure."""

    from scipy.interpolate import RegularGridInterpolator

    def prepare_surface(surface_x: np.ndarray, surface_y: np.ndarray, surface_moat: np.ndarray):
        x_values = np.asarray(surface_x, dtype=float)
        y_values = np.asarray(surface_y, dtype=float)
        moat_values = np.asarray(surface_moat, dtype=float)
        if x_values.ndim == 1 and y_values.ndim == 1:
            x_axis, y_axis = x_values, y_values
            x_values, y_values = np.meshgrid(x_axis, y_axis)
        else:
            x_axis, y_axis = x_values[0, :], y_values[:, 0]
        if x_values.shape != y_values.shape or x_values.shape != moat_values.shape:
            raise ValueError("Every surface grid must have matching x, y and moat shapes")
        z_values = np.log10(np.maximum(moat_values, 1.0e-18))
        interpolator = RegularGridInterpolator(
            (y_axis, x_axis), z_values, bounds_error=False,
            fill_value=float(np.nanmin(z_values)),
        )
        return x_axis, y_axis, x_values, y_values, z_values, interpolator

    global_pack = prepare_surface(x_grid, y_grid, moat_grid)
    zoom_pack = prepare_surface(*(zoom_surface or (x_grid, y_grid, moat_grid)))
    deep_pack = prepare_surface(*(deep_zoom_surface or (x_grid, y_grid, moat_grid)))
    global_x, global_y, global_X, global_Y, global_Z, global_interp = global_pack
    records = _normalise_contours(contours)
    eig = np.asarray(eigenvalues, dtype=np.complex128)
    global_xlim = (float(global_x.min()), float(global_x.max()))
    global_ylim = (float(global_y.min()), float(global_y.max()))
    family_colours = {
        "alpha": THESIS_PALETTE["real"], "alpha packet": THESIS_PALETTE["real"],
        "mu": THESIS_PALETTE["imag"], "mu packet": THESIS_PALETTE["imag"],
        "fixed": THESIS_PALETTE["main"], "fixed point": THESIS_PALETTE["main"],
    }
    exact_targets: list[dict[str, Any]] = []
    for family, base, maximum in (("alpha packet", float(alpha), 19), ("mu packet", float(mu), 9)):
        symbol = "alpha" if family.startswith("alpha") else "mu"
        for power in range(1, maximum + 1):
            value = base ** power
            if global_xlim[0] <= value <= global_xlim[1]:
                exact_targets.append({"label": rf"$\{symbol}^{power}$", "value": value,
                                      "family": family, "power": power})
    cmap = make_phase4_moat_cmap("phase4_global_moat_palette")
    theta = np.linspace(0.0, 2.0 * np.pi, 384)
    fig = plt.figure(figsize=(15.8, 15.4), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.08, 1.0), hspace=0.28, wspace=0.16)
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax3d_deep = fig.add_subplot(grid[0, 1], projection="3d")
    ax2d = fig.add_subplot(grid[1, :])

    def plot_3d_panel(ax: Any, pack: tuple[Any, ...], title: str, *, zoomed: bool = True) -> None:
        x_axis, y_axis, X_values, Y_values, Z_values, interpolator = pack
        xlim = (float(x_axis.min()), float(x_axis.max()))
        ylim = (float(y_axis.min()), float(y_axis.max()))
        ax.plot_surface(X_values, Y_values, Z_values, cmap=cmap, linewidth=0.0,
                        antialiased=True, alpha=0.30)
        for record in records:
            centre, radius = record["centre"], record["radius"]
            if centre.real + radius < xlim[0] or centre.real - radius > xlim[1]:
                continue
            cx = centre.real + radius * np.cos(theta)
            cy = centre.imag + radius * np.sin(theta)
            inside = ((cx >= xlim[0]) & (cx <= xlim[1]) &
                      (cy >= ylim[0]) & (cy <= ylim[1]))
            if np.count_nonzero(inside) < 2:
                continue
            cz = interpolator(np.column_stack((cy, cx)))
            cx, cy, cz = cx.astype(float), cy.astype(float), np.asarray(cz, dtype=float)
            cx[~inside] = np.nan
            cy[~inside] = np.nan
            cz[~inside] = np.nan
            colour = family_colours.get(record["family"], THESIS_PALETTE["main"])
            ax.plot(cx, cy, cz, color=colour, linewidth=1.65 if zoomed else 1.6, alpha=0.9)
        if eig.size:
            visible = eig[(eig.real >= xlim[0]) & (eig.real <= xlim[1]) &
                          (eig.imag >= ylim[0]) & (eig.imag <= ylim[1])]
            if visible.size:
                ez = interpolator(np.column_stack((visible.imag, visible.real)))
                ax.scatter(visible.real, visible.imag, ez, color=THESIS_PALETTE["main"],
                           s=24 if zoomed else 18, depthshade=False,
                           label="finite eigenvalues")
        z_span = float(np.nanmax(Z_values) - np.nanmin(Z_values))
        for target in exact_targets:
            value = target["value"]
            if not xlim[0] <= value <= xlim[1]:
                continue
            point_z = float(interpolator([[0.0, value]])[0])
            ax.scatter([value], [0.0], [point_z], facecolors="#ff7a00",
                       edgecolors="#fff200", s=82 if zoomed else 68,
                       linewidths=1.7, depthshade=False)
            if target["family"] == "alpha packet" and target["power"] <= 9:
                ax.text(value, 0.0, point_z + 0.045 * z_span, target["label"],
                        color="#003366", fontsize=15)
        ax.set_title(title, fontsize=18)
        ax.set_xlabel(r"$\operatorname{Re}\zeta$", fontsize=15)
        ax.set_ylabel(r"$\operatorname{Im}\zeta$", fontsize=15)
        ax.set_zlabel(r"$\log_{10}\sigma_{\min}(\zeta I-A_N^X)$", fontsize=15)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.view_init(elev=29, azim=-58)
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="upper left", fontsize=10)

    plot_3d_panel(ax3d, zoom_pack, f"3D near-origin zoom, grid {len(zoom_pack[0])}")
    plot_3d_panel(ax3d_deep, deep_pack, f"3D cluster restriction, grid {len(deep_pack[0])}")
    levels = np.linspace(float(np.nanmin(global_Z)), float(np.nanmax(global_Z)), 30)
    filled = ax2d.contourf(global_X, global_Y, global_Z, levels=levels, cmap=cmap)
    ax2d.contour(global_X, global_Y, global_Z, levels=levels[::4],
                 colors=THESIS_PALETTE["ref"], linewidths=0.42, alpha=0.5)
    seen_families: set[str] = set()
    for record in records:
        centre, radius = record["centre"], record["radius"]
        colour = family_colours.get(record["family"], THESIS_PALETTE["main"])
        label = record["family"] if record["family"] not in seen_families else None
        ax2d.plot(centre.real + radius * np.cos(theta),
                  centre.imag + radius * np.sin(theta), color=colour,
                  linewidth=1.4, alpha=0.9, label=label)
        seen_families.add(record["family"])
    if eig.size:
        visible = eig[(eig.real >= global_xlim[0]) & (eig.real <= global_xlim[1]) &
                      (eig.imag >= global_ylim[0]) & (eig.imag <= global_ylim[1])]
        if visible.size:
            ax2d.scatter(visible.real, visible.imag, color=THESIS_PALETTE["main"],
                         s=18, label="finite eigenvalues", zorder=4)
    for index, target in enumerate(exact_targets):
        value = target["value"]
        ax2d.scatter([value], [0.0], facecolors="#ff7a00", edgecolors="#fff200",
                     s=60, linewidths=1.4, zorder=5)
        if target["family"] == "alpha packet" and target["power"] <= 9:
            ax2d.annotate(target["label"], (value, 0.0), textcoords="offset points",
                          xytext=(3, 5 if index % 2 == 0 else -12),
                          color="#003366", fontsize=15)
    ax2d.axhline(0.0, color=THESIS_PALETTE["ref"], linewidth=0.8, alpha=0.75)
    ax2d.set_aspect("auto")
    ax2d.set_xlim(*global_xlim)
    ax2d.set_ylim(*global_ylim)
    ax2d.set_title("2D global moat map and first-fifteen contours", fontsize=18)
    ax2d.set_xlabel(r"$\operatorname{Re}\zeta$", fontsize=15)
    ax2d.set_ylabel(r"$\operatorname{Im}\zeta$", fontsize=15)
    ax2d.tick_params(axis="both", which="major", labelsize=11)
    ax2d.legend(loc="upper right", fontsize=10)
    colourbar = fig.colorbar(filled, ax=ax2d, fraction=0.046, pad=0.04)
    colourbar.set_label(r"$\log_{10}\sigma_{\min}(\zeta I-A_N^X)$", fontsize=14)
    colourbar.ax.tick_params(labelsize=11)
    fig.suptitle("Global sampled Hardy-gauge moat landscape for first fifteen packets", fontsize=18)
    fig.tight_layout()
    return _finish_plot(fig, (ax3d, ax3d_deep, ax2d), output_dir=output_dir,
                        stem=stem, dpi=300, show=show, tight=False)


def plot_phase4_global_zoom(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    moat_grid: np.ndarray,
    contours: Sequence[Mapping[str, Any]],
    *,
    windows: Sequence[tuple[float, float, float, float]] | None = None,
    titles: Sequence[str] = ("Near-origin Euclidean zoom", "Global symlog-x diagnostic"),
    linthresh: float = 0.015,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_global_hardy_moat_zoom_symlog_N600_M610",
    show: bool = True,
) -> PlotResult:
    """Render the paired zoom and symmetric-log complete-contour map."""

    if len(titles) != 2:
        raise ValueError("Exactly two titles are required")
    x = np.asarray(x_grid, dtype=float)
    y = np.asarray(y_grid, dtype=float)
    moat = np.asarray(moat_grid, dtype=float)
    if x.ndim == 1 and y.ndim == 1:
        x_axis, y_axis = x, y
        x, y = np.meshgrid(x_axis, y_axis)
    else:
        x_axis, y_axis = x[0, :], y[:, 0]
    z_log = np.log10(np.maximum(moat, 1.0e-18))
    if windows is None:
        windows = (
            (max(float(x_axis.min()), -0.01), min(float(x_axis.max()), 0.18), -0.055, 0.055),
            (float(x_axis.min()), float(x_axis.max()), float(y_axis.min()), float(y_axis.max())),
        )
    if len(windows) != 2:
        raise ValueError("Exactly two windows are required")
    cmap = make_phase4_moat_cmap("phase4_global_zoom_moat_palette")
    fig, axes = plt.subplots(1, 2, figsize=(16.0, 6.7), facecolor="white", constrained_layout=True)
    records = _normalise_contours(contours)
    family_colours = {
        "alpha": THESIS_PALETTE["real"], "alpha packet": THESIS_PALETTE["real"],
        "mu": THESIS_PALETTE["imag"], "mu packet": THESIS_PALETTE["imag"],
        "fixed": THESIS_PALETTE["main"], "fixed point": THESIS_PALETTE["main"],
    }
    theta = np.linspace(0.0, 2.0 * np.pi, 720)
    levels = np.linspace(float(np.nanmin(z_log)), float(np.nanmax(z_log)), 30)
    filled = None
    for ax, window, title in zip(axes, windows, titles):
        filled = ax.contourf(x, y, z_log, levels=levels, cmap=cmap)
        ax.contour(x, y, z_log, levels=levels[::4], colors=THESIS_PALETTE["ref"],
                   linewidths=0.42, alpha=0.5)
        seen: set[str] = set()
        for record in records:
            centre = record["centre"]
            colour = family_colours.get(record["family"], THESIS_PALETTE["main"])
            cx = centre.real + record["radius"] * np.cos(theta)
            cy = centre.imag + record["radius"] * np.sin(theta)
            label = record["family"] if record["family"] not in seen else None
            ax.plot(cx, cy, color=colour, lw=1.3, alpha=0.9, label=label)
            seen.add(record["family"])
        for record in records:
            centre = record["centre"]
            if title.startswith("Near") and not window[0] <= centre.real <= window[1]:
                continue
            colour = family_colours.get(record["family"], THESIS_PALETTE["main"])
            ax.scatter([centre.real], [centre.imag], facecolors="none",
                       edgecolors=colour, s=42, linewidths=1.1, zorder=5)
            ax.annotate(latex_target_label(record["name"]),
                        (centre.real, centre.imag), textcoords="offset points",
                        xytext=(3, 5), color=colour, fontsize=10)
        ax.axhline(0.0, color=THESIS_PALETTE["ref"], linewidth=0.8, alpha=0.72)
        ax.set_xlim(window[0], window[1])
        ax.set_ylim(window[2], window[3])
        ax.set_xlabel(r"$\operatorname{Re}\zeta$")
        ax.set_ylabel(r"$\operatorname{Im}\zeta$")
        ax.set_title(title, fontsize=19)
        ax.set_xlabel(r"$\operatorname{Re}\zeta$", fontsize=16)
        ax.set_ylabel(r"$\operatorname{Im}\zeta$", fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=12)
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), loc="upper right", fontsize=10, frameon=True)
    axes[0].set_aspect("auto")
    axes[1].set_xscale("symlog", linthresh=float(linthresh))
    axes[1].set_aspect("auto")
    if filled is not None:
        cbar = fig.colorbar(filled, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02)
        cbar.set_label(r"$\log_{10}\sigma_{\min}(\zeta I-A_N^X)$", fontsize=14)
        cbar.ax.tick_params(labelsize=11)
    fig.suptitle("Readable views of the global first-fifteen Hardy-gauge moat", fontsize=19)
    return _finish_plot(fig, axes, output_dir=output_dir, stem=stem,
                        dpi=230, show=show, tight=False)


def make_phase4_interactive_dashboard_renderer(
    *,
    surface_paths: Mapping[str, str | Path],
    packet_frame: pd.DataFrame,
    eigenvalue_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    stem: str = "branch_image_wide_candidate_first15_interactive_dashboard_snapshot_N600_M610",
) -> Callable[..., PlotResult]:
    """Build the helper-owned renderer used by the Phase 4 dashboard.

    The returned callback contains all Matplotlib implementation.  Notebook
    cells provide only retained data paths and widget values.
    """

    from scipy.interpolate import RegularGridInterpolator

    surfaces: list[dict[str, Any]] = []
    for name, path_value in surface_paths.items():
        path = Path(path_value)
        if not path.exists():
            continue
        payload = np.load(path, allow_pickle=False)
        x_axis = np.asarray(payload["x"], dtype=float)
        y_axis = np.asarray(payload["y"], dtype=float)
        moat = np.asarray(payload["s_min"], dtype=float)
        z_log = np.log10(np.maximum(moat, 1.0e-18))
        surfaces.append({
            "name": str(name), "path": path, "x": x_axis, "y": y_axis,
            "Z": z_log,
            "interp": RegularGridInterpolator(
                (y_axis, x_axis), z_log, bounds_error=False,
                fill_value=float(np.nanmin(z_log)),
            ),
            "grid_score": int(x_axis.size * y_axis.size),
        })
    if not surfaces:
        raise FileNotFoundError("No stored Hardy-gauge surface cache is available")
    global_surface = next((item for item in surfaces if item["name"] == "global"), surfaces[0])
    packets = packet_frame.copy()
    if "target" not in packets.columns:
        raise ValueError("The dashboard packet table requires a target column")
    packets = packets.loc[packets["target"].astype(str) != "1"].copy()
    if "rank" in packets.columns:
        packets = packets.sort_values("rank").head(15)
    else:
        packets = packets.head(15)
        packets["rank"] = np.arange(1, len(packets) + 1)
    centre_column = "centre" if "centre" in packets.columns else "center"
    packets["centre_complex"] = packets[centre_column].map(parse_complex_for_plot)
    if "family" not in packets.columns:
        packets["family"] = packets["target"].astype(str).map(
            lambda value: "alpha packet" if value.startswith("alpha")
            else "mu packet" if value.startswith("mu") else "fixed point"
        )
    packets["centre_real"] = packets["centre_complex"].map(lambda value: float(value.real))
    packets["radius"] = pd.to_numeric(packets["radius"], errors="raise")
    rank_order = packets.sort_values("rank").reset_index(drop=True)
    origin_order = packets.sort_values(["centre_real", "rank"]).reset_index(drop=True)
    family_colours = {
        "alpha": THESIS_PALETTE["real"], "alpha packet": THESIS_PALETTE["real"],
        "mu": THESIS_PALETTE["imag"], "mu packet": THESIS_PALETTE["imag"],
        "fixed": THESIS_PALETTE["main"], "fixed point": THESIS_PALETTE["main"],
    }
    cmap = make_phase4_moat_cmap("phase4_mpl_indigo_blue_pink")
    eig = np.empty(0, dtype=np.complex128)
    if eigenvalue_path is not None and Path(eigenvalue_path).exists():
        eig_frame = pd.read_csv(eigenvalue_path)
        if {"real", "imag"}.issubset(eig_frame.columns):
            eig = eig_frame["real"].to_numpy(float) + 1j * eig_frame["imag"].to_numpy(float)
    saved_snapshot = False

    def contains(surface: Mapping[str, Any], x0: float, x1: float, y0: float, y1: float) -> bool:
        return (x0 >= float(np.min(surface["x"])) and x1 <= float(np.max(surface["x"]))
                and y0 >= float(np.min(surface["y"])) and y1 <= float(np.max(surface["y"])))

    def choose_surface(x0: float, x1: float, y0: float, y1: float) -> Mapping[str, Any]:
        candidates = [surface for surface in surfaces if contains(surface, x0, x1, y0, y1)]
        return max(candidates, key=lambda item: item["grid_score"]) if candidates else global_surface

    def renderer(
        packet_count: int,
        order_mode: str,
        opacity: float,
        azimuth: float,
        elevation: float,
        xy_scale: float,
        z_scale: float,
        zoom_factor: float,
        surface_resolution: int,
        show_eigenvalues: bool,
    ) -> PlotResult:
        nonlocal saved_snapshot
        table = origin_order if order_mode == "near-origin order" else rank_order
        selected = table.head(int(packet_count)).copy()
        centres = selected["centre_complex"].to_numpy()
        radii = selected["radius"].to_numpy(float)
        centre_re = np.asarray([value.real for value in centres], dtype=float)
        centre_im = np.asarray([value.imag for value in centres], dtype=float)
        left, right = centre_re - radii, centre_re + radii
        x_min, x_max = float(np.min(left)), float(np.max(right))
        y_extent = float(max(0.012, np.max(np.abs(centre_im) + radii)))
        width = max(x_max - x_min, 2.0 * y_extent)
        x_mid = 0.5 * (x_min + x_max)
        requested_x_half = 0.5 * (width + 0.32 * width) * float(xy_scale) / float(zoom_factor)
        requested_y_half = (y_extent + 0.16 * y_extent) * float(xy_scale) / float(zoom_factor)
        centre_margin = max(0.002, 0.55 * float(np.max(radii)))
        required_x_half = 0.5 * float(np.max(centre_re) - np.min(centre_re)) + centre_margin
        required_y_half = float(np.max(np.abs(centre_im))) + centre_margin
        x_half = max(requested_x_half, required_x_half)
        y_half = max(requested_y_half, required_y_half, 0.012)
        x0 = max(float(np.min(global_surface["x"])), x_mid - x_half)
        x1 = min(float(np.max(global_surface["x"])), x_mid + x_half)
        y0 = max(float(np.min(global_surface["y"])), -y_half)
        y1 = min(float(np.max(global_surface["y"])), y_half)
        active = choose_surface(x0, x1, y0, y1)
        resolution = int(surface_resolution)
        xs = np.linspace(x0, x1, resolution)
        ys = np.linspace(y0, y1, max(40, int(resolution * 0.72)))
        visible_centres = ((centre_re >= x0) & (centre_re <= x1)
                           & (centre_im >= y0) & (centre_im <= y1))
        if np.any(visible_centres):
            xs = np.unique(np.concatenate((xs, centre_re[visible_centres])))
            ys = np.unique(np.concatenate((ys, centre_im[visible_centres])))
            radius_edges = np.concatenate((centre_im[visible_centres] - radii[visible_centres],
                                           centre_im[visible_centres] + radii[visible_centres]))
            radius_edges = radius_edges[(radius_edges >= y0) & (radius_edges <= y1)]
            if radius_edges.size:
                ys = np.unique(np.concatenate((ys, radius_edges)))
        X_values, Y_values = np.meshgrid(xs, ys)
        raw_z = active["interp"](np.column_stack((Y_values.ravel(), X_values.ravel()))).reshape(X_values.shape)
        z_anchor = float(np.nanmedian(raw_z))
        visual_z = z_anchor + float(z_scale) * (raw_z - z_anchor)
        z_min, z_max = float(np.nanmin(visual_z)), float(np.nanmax(visual_z))
        z_pad = max(0.05, 0.06 * (z_max - z_min))
        fig = plt.figure(figsize=(19.2, 8.2), constrained_layout=False)
        grid = fig.add_gridspec(1, 3, width_ratios=(1.05, 1.05, 0.045),
                                left=0.045, right=0.975, bottom=0.105,
                                top=0.845, wspace=0.18)
        ax3d = fig.add_subplot(grid[0, 0], projection="3d")
        ax2d = fig.add_subplot(grid[0, 1])
        colour_axis = fig.add_subplot(grid[0, 2])
        ax3d.plot_surface(X_values, Y_values, visual_z, cmap=cmap, linewidth=0,
                          antialiased=True, alpha=float(opacity), shade=True)
        filled = ax2d.contourf(X_values, Y_values, raw_z, levels=32, cmap=cmap)
        colourbar = fig.colorbar(filled, cax=colour_axis)
        colourbar.set_label(r"$\log_{10}\sigma_{\min}$")
        colourbar.ax.tick_params(labelsize=11)
        theta = np.linspace(0.0, 2.0 * np.pi, 420)
        for _, row in selected.iterrows():
            centre = complex(row["centre_complex"])
            radius = float(row["radius"])
            colour = family_colours.get(str(row["family"]), THESIS_PALETTE["main"])
            cx = centre.real + radius * np.cos(theta)
            cy = centre.imag + radius * np.sin(theta)
            cz_raw = active["interp"](np.column_stack((cy, cx)))
            cz = z_anchor + float(z_scale) * (cz_raw - z_anchor)
            visible = ((cx >= x0) & (cx <= x1) & (cy >= y0) & (cy <= y1))
            cx3, cy3, cz3 = cx.copy(), cy.copy(), np.asarray(cz, dtype=float).copy()
            cx3[~visible] = np.nan
            cy3[~visible] = np.nan
            cz3[~visible] = np.nan
            ax3d.plot(cx3, cy3, cz3, color=colour, linewidth=2.1)
            ax2d.plot(cx3, cy3, color=colour, linewidth=1.7)
            if x0 <= centre.real <= x1 and y0 <= centre.imag <= y1:
                centre_raw = float(active["interp"]([[centre.imag, centre.real]])[0])
                centre_z = z_anchor + float(z_scale) * (centre_raw - z_anchor)
                ax3d.scatter([centre.real], [centre.imag], [centre_z], s=74,
                             facecolors="#ff7a00", edgecolors="#003366",
                             linewidths=1.8, depthshade=False)
                ax3d.plot([centre.real, centre.real], [centre.imag, centre.imag],
                          [centre_z, centre_z + z_pad * 0.20], color="#003366",
                          linewidth=1.1, alpha=0.95)
                ax2d.scatter([centre.real], [centre.imag], s=64,
                             facecolors="#ff7a00", edgecolors="#003366",
                             linewidths=1.6, zorder=6)
                label = latex_target_label(row["target"])
                ax3d.text(centre.real, centre.imag, centre_z + z_pad * 0.24,
                          label, fontsize=14, color="#003366")
                ax2d.text(centre.real, centre.imag, label, fontsize=14,
                          color="#003366", ha="left", va="bottom")
        if show_eigenvalues and eig.size:
            visible_eig = eig[(eig.real >= x0) & (eig.real <= x1)
                              & (eig.imag >= y0) & (eig.imag <= y1)]
            if visible_eig.size:
                eig_raw = active["interp"](np.column_stack((visible_eig.imag, visible_eig.real)))
                eig_z = z_anchor + float(z_scale) * (eig_raw - z_anchor)
                ax3d.scatter(visible_eig.real, visible_eig.imag, eig_z, s=10,
                             color=THESIS_PALETTE["main"], alpha=0.85,
                             label="finite eigenvalues")
                ax2d.scatter(visible_eig.real, visible_eig.imag, s=10,
                             color=THESIS_PALETTE["main"], alpha=0.82,
                             label="finite eigenvalues")
        alpha_handle, = ax2d.plot([], [], color=THESIS_PALETTE["real"], linewidth=2,
                                  label="alpha packet")
        mu_handle, = ax2d.plot([], [], color=THESIS_PALETTE["imag"], linewidth=2,
                               label="mu packet")
        ax2d.legend(handles=(alpha_handle, mu_handle), loc="upper right", fontsize=10)
        ax3d.set_xlim(x0, x1)
        ax3d.set_ylim(y0, y1)
        ax3d.set_zlim(z_min - z_pad, z_max + z_pad)
        ax3d.view_init(elev=float(elevation), azim=float(azimuth))
        ax3d.xaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        ax3d.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        ax3d.zaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))
        ax3d.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax3d.tick_params(axis="x", which="major", labelsize=9, pad=1)
        ax3d.tick_params(axis="y", which="major", labelsize=9, pad=1)
        ax3d.tick_params(axis="z", which="major", labelsize=9, pad=8)
        ax3d.set_xlabel(r"$\operatorname{Re}\zeta$", labelpad=7)
        ax3d.set_ylabel(r"$\operatorname{Im}\zeta$", labelpad=7)
        ax3d.set_zlabel(r"$\log_{10}\sigma_{\min}$", labelpad=18)
        ax3d.set_title("3D lifted contour landscape")
        try:
            ax3d.set_box_aspect((1.15, 1.0, 0.60), zoom=1.05)
        except TypeError:
            ax3d.set_box_aspect((1.15, 1.0, 0.60))
        ax3d.set_anchor("C")
        ax2d.set_anchor("C")
        ax2d.set_xlim(x0, x1)
        ax2d.set_ylim(y0, y1)
        ax2d.set_aspect("auto")
        ax2d.xaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax2d.yaxis.set_major_locator(MaxNLocator(nbins=6, prune=None))
        ax2d.tick_params(axis="both", which="major", labelsize=10, pad=3)
        ax2d.set_xlabel(r"$\operatorname{Re}\zeta$")
        ax2d.set_ylabel(r"$\operatorname{Im}\zeta$")
        ax2d.set_title("2D moat map with selected packets")
        fig.suptitle(
            f"Matplotlib Hardy-gauge moat dashboard: {len(selected)} packets, "
            f"{order_mode}, surface: {active['name']}", fontsize=16,
        )
        paths: tuple[Path, ...] = ()
        if output_dir is not None and not saved_snapshot:
            paths = save_figure(fig, output_dir, stem, dpi=220, tight=True)
            saved_snapshot = True
        show_figure(fig)
        return PlotResult(fig, (ax3d, ax2d, colour_axis), paths)

    return renderer


def create_phase4_interactive_dashboard(
    render_callback: Callable[..., Any],
    *,
    maximum_packets: int,
    default_packets: int = 7,
    display_dashboard: bool = True,
) -> Any:
    """Create the Matplotlib-only widget controls around a supplied renderer.

    The callback receives packet_count, order_mode, opacity, azimuth,
    elevation, xy_scale, z_scale, zoom_factor, surface_resolution and
    show_eigenvalues.  Surface generation and singular-value evaluation remain
    the caller's responsibility.
    """

    try:
        import ipywidgets as widgets
    except Exception as exc:
        raise ImportError("The interactive dashboard requires ipywidgets") from exc
    packet_count = widgets.IntSlider(
        value=min(default_packets, maximum_packets), min=1, max=maximum_packets,
        step=1, description="packets", continuous_update=False,
    )
    order_mode = widgets.ToggleButtons(
        options=("near-origin order", "rank order"),
        value="near-origin order", description="order",
    )
    opacity = widgets.FloatSlider(value=0.30, min=0.08, max=0.95, step=0.03,
                                  description="opacity", continuous_update=False)
    azimuth = widgets.IntSlider(value=306, min=0, max=360, step=2,
                                description="azimuth", continuous_update=False)
    elevation = widgets.IntSlider(value=24, min=5, max=80, step=1,
                                  description="elevation", continuous_update=False)
    xy_scale = widgets.FloatSlider(value=1.0, min=0.35, max=2.2, step=0.05,
                                   description="xy scale", continuous_update=False)
    z_scale = widgets.FloatSlider(value=0.34, min=0.06, max=1.4, step=0.04,
                                  description="z scale", continuous_update=False)
    zoom = widgets.FloatSlider(value=2.4, min=0.6, max=5.0, step=0.1,
                               description="zoom", continuous_update=False)
    resolution = widgets.IntSlider(value=150, min=70, max=260, step=10,
                                   description="surface px", continuous_update=False)
    eigenvalues = widgets.Checkbox(value=True, description="finite eigenvalues")
    controls = widgets.HBox((
        widgets.VBox((packet_count, order_mode, opacity, resolution, eigenvalues)),
        widgets.VBox((azimuth, elevation, xy_scale, z_scale, zoom)),
    ))
    output = widgets.interactive_output(render_callback, {
        "packet_count": packet_count,
        "order_mode": order_mode,
        "opacity": opacity,
        "azimuth": azimuth,
        "elevation": elevation,
        "xy_scale": xy_scale,
        "z_scale": z_scale,
        "zoom_factor": zoom,
        "surface_resolution": resolution,
        "show_eigenvalues": eigenvalues,
    })
    dashboard = widgets.VBox((controls, output))
    if display_dashboard:
        try:
            from IPython.display import display

            display(dashboard)
        except Exception:
            pass
    return dashboard


def plot_certification_ladder(
    frame: pd.DataFrame,
    *,
    title: str,
    status_colours: Mapping[str, str] | None = None,
    output_dir: str | Path | None = None,
    stem: str = "universal_certification_ladder",
    show: bool = True,
) -> PlotResult:
    """Render the universal certification-status ladder."""

    _require_columns(frame, ("audit_item", "status"), name="certification audit")
    colours = dict(status_colours or {
        "theorem_certified": THESIS_PALETTE["real"],
        "sampled_pass": THESIS_PALETTE["imag"],
        "sampled_not_interval_certified": THESIS_PALETTE["imag"],
        "sampled_not_theorem_certified": THESIS_PALETTE["imag"],
        "reference_diagnostic_not_exact": THESIS_PALETTE["ref"],
        "diagnostic_not_theorem_certified": THESIS_PALETTE["main"],
    })
    y = np.arange(len(frame))
    fig, ax = plt.subplots(figsize=(11.0, 5.8), facecolor="white")
    bar_colours = [colours.get(str(value), THESIS_PALETTE["ref"]) for value in frame["status"]]
    ax.barh(y, np.ones_like(y, dtype=float), color=bar_colours, alpha=0.86)
    ax.set_yticks(y, frame["audit_item"])
    ax.tick_params(axis="y", labelsize=17)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    for index, row in frame.reset_index(drop=True).iterrows():
        ax.text(0.02, index, str(row["status"]).replace("_", " "), va="center",
                color="white", fontsize=11, fontweight="bold")
    ax.set_title(_reviewed_map_title(title), fontsize=20)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=220, show=show)


def plot_packet_contours(
    packets: pd.DataFrame,
    *,
    eigenvalues: Sequence[complex] = (),
    centre_column: str = "packet_centre",
    radius_column: str = "radius",
    name_column: str = "name",
    title: str = "Nontrivial packet contours",
    output_dir: str | Path | None = None,
    stem: str = "first14_packet_audit",
    show: bool = True,
) -> PlotResult:
    """Plot packet centres, contour circles and optional finite eigenvalues."""

    _require_columns(packets, (centre_column, radius_column, name_column), name="packet table")
    labels = packets[name_column].astype(str).tolist()
    colours = make_phase_palette(labels)
    fig, ax = plt.subplots(figsize=(10.5, 7.0), facecolor="white")
    for row in packets.itertuples(index=False):
        record = row._asdict()
        centre = complex(record[centre_column])
        radius = float(record[radius_column]) if pd.notna(record[radius_column]) else 0.0
        name = str(record[name_column])
        colour = colours[name]
        ax.scatter(centre.real, centre.imag, s=70, color=colour, zorder=3)
        if radius > 0:
            ax.add_patch(plt.Circle((centre.real, centre.imag), radius, fill=False,
                                    color=colour, lw=1.4, alpha=0.8))
        ax.annotate(latex_target_label(name), (centre.real, centre.imag),
                    xytext=(4, 5), textcoords="offset points", fontsize=10,
                    color=colour)
    eig = np.asarray(eigenvalues, dtype=np.complex128)
    if eig.size:
        eig = eig[np.abs(eig - 1.0) > 1.0e-10]
        ax.scatter(eig.real, eig.imag, s=14, color=THESIS_PALETTE["main"],
                   alpha=0.55, label="finite eigenvalues")
    ax.axhline(0.0, color=THESIS_PALETTE["ref"], lw=0.8)
    ax.axvline(0.0, color=THESIS_PALETTE["ref"], lw=0.8)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$\Re z$")
    ax.set_ylabel(r"$\Im z$")
    ax.set_title(_reviewed_map_title(title))
    if eig.size:
        ax.legend(frameon=False, fontsize=16)
    ax.grid(True, alpha=0.2)
    return _finish_plot(fig, ax, output_dir=output_dir, stem=stem,
                        dpi=220, show=show)
