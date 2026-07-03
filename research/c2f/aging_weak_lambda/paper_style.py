"""Publication matplotlib style for weak-coupling aging figures.

Ported from third_party/cav-hoomd/plotting/:
  - plot_fictive_temperature.py (rcParams, COLORS, legend box style)
  - plot_potential_energy_components.py (classic grid, spine widths, energy colors)

LaTeX rendering uses, in order of preference:
  1. Standard ``latex`` + ``dvipng`` (when fully functional)
  2. ``tectonic`` + ``ghostscript`` (pixi global install on this cluster)
  3. Matplotlib mathtext with the Computer Modern (``cm``) fontset
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mimage
import matplotlib.pyplot as plt
import numpy as np

_LATEX_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{amssymb}
"""

# Sequential palette for coupling-indexed lines (plot_fictive_temperature.py).
COLORS: list[str] = [
    "#1B4F72",  # dark blue
    "#5DADE2",  # light blue
    "#F0B27A",  # light orange/salmon
    "#C0392B",  # dark red
    "#6C3483",  # purple
    "#1ABC9C",  # teal
    "#E74C3C",  # bright red
    "#2ECC71",  # green
]

# Fig 3b energy-component colors (plot_potential_energy_components.py).
COLOR_HARMONIC = "#1f77b4"
COLOR_LJ_COULOMB = "#d62728"
COLOR_TOTAL = "#2ca02c"
COLOR_KINETIC = "#2ca02c"

_STYLE_APPLIED = False
_TEXMANAGER_PATCHED = False
_LATEX_BACKEND = "mathtext"


def _test_latex_dvipng() -> bool:
    latex = shutil.which("latex")
    dvipng = shutil.which("dvipng")
    if not latex or not dvipng:
        return False
    try:
        with tempfile.TemporaryDirectory() as tmp:
            tex_path = Path(tmp) / "test.tex"
            tex_path.write_text(
                r"\documentclass{article}\begin{document}x\end{document}",
                encoding="utf-8",
            )
            subprocess.run(
                [latex, "-interaction=nonstopmode", "-halt-on-error", "test.tex"],
                cwd=tmp,
                capture_output=True,
                timeout=60,
                check=False,
            )
            dvi_path = Path(tmp) / "test.dvi"
            if not dvi_path.exists():
                return False
            subprocess.run(
                [dvipng, "-T", "tight", "-o", "test.png", str(dvi_path)],
                cwd=tmp,
                capture_output=True,
                timeout=60,
                check=True,
            )
            return (Path(tmp) / "test.png").exists()
    except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def _detect_latex_backend() -> str:
    if _test_latex_dvipng():
        return "latex"
    if shutil.which("tectonic") and shutil.which("gs"):
        return "tectonic"
    return "mathtext"


def _patch_texmanager_for_tectonic() -> None:
    """Route matplotlib usetex through tectonic + ghostscript."""
    global _TEXMANAGER_PATCHED
    if _TEXMANAGER_PATCHED:
        return

    from matplotlib.texmanager import TexManager

    tectonic = shutil.which("tectonic")
    gs = shutil.which("gs")
    if not tectonic or not gs:
        return

    def _sanitize_tex_for_tectonic(tex_src: str) -> str:
        drop_packages = ("type1cm", "type1ec", "courier", "helvet", "psfrag", "underscore")
        cleaned: list[str] = []
        for line in tex_src.splitlines():
            stripped = line.strip()
            if stripped.startswith("\\DeclareUnicodeCharacter"):
                continue
            if "\\usepackage[utf8]{inputenc}" in line:
                continue
            if any(f"\\usepackage{{{pkg}}}" in line for pkg in drop_packages):
                continue
            if any(f"\\usepackage[{pkg}" in line for pkg in drop_packages):
                continue
            if stripped.startswith("\\makeatletter") and "underscore" in line:
                continue
            if "\\usepackage[papersize=72in" in line:
                cleaned.append("\\usepackage[margin=1in]{geometry}")
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    @classmethod
    def _tectonic_make_png(cls, tex, fontsize, dpi):  # noqa: N805
        from tempfile import TemporaryDirectory

        pngfile = Path(cls.get_basefile(tex, fontsize, dpi)).with_suffix(".png")
        if not pngfile.exists():
            with TemporaryDirectory(dir=pngfile.parent) as tmpdir:
                tmp = Path(tmpdir)
                tex_src = _sanitize_tex_for_tectonic(cls._get_tex_source(tex, fontsize))
                (tmp / "file.tex").write_text(tex_src, encoding="utf-8")
                subprocess.check_output(
                    [tectonic, str(tmp / "file.tex")],
                    cwd=tmpdir,
                    stderr=subprocess.STDOUT,
                )
                subprocess.check_output(
                    [
                        gs,
                        "-dSAFER",
                        "-dBATCH",
                        "-dNOPAUSE",
                        "-sDEVICE=pngalpha",
                        f"-r{int(dpi)}",
                        "-dTextAlphaBits=4",
                        "-dGraphicsAlphaBits=4",
                        "-dEPSCrop",
                        f"-sOutputFile={tmp / 'file.png'}",
                        str(tmp / "file.pdf"),
                    ],
                    stderr=subprocess.STDOUT,
                )
                (tmp / "file.png").replace(pngfile)
        return str(pngfile)

    @classmethod
    def _tectonic_get_text_width_height_descent(cls, tex, fontsize, renderer=None):  # noqa: N805
        if tex.strip() == "":
            return 0, 0, 0
        if renderer is not None and hasattr(renderer, "dpi"):
            dpi = renderer.dpi
            dpi_fraction = renderer.points_to_pixels(1.0)
        else:
            dpi = mpl.rcParams["figure.dpi"]
            dpi_fraction = 1.0
        pngfile = cls.make_png(tex, fontsize, dpi)
        rgba = mimage.imread(pngfile)
        alpha = rgba[:, :, -1]
        rows = np.any(alpha > 0.05, axis=1)
        cols = np.any(alpha > 0.05, axis=0)
        if not rows.any() or not cols.any():
            return 0, 0, 0
        r0, r1 = int(np.where(rows)[0][0]), int(np.where(rows)[0][-1])
        c0, c1 = int(np.where(cols)[0][0]), int(np.where(cols)[0][-1])
        pt_per_px = 72.0 / dpi / dpi_fraction
        width = (c1 - c0 + 1) * pt_per_px
        height = (r1 - r0 + 1) * pt_per_px
        baseline_row = r0 + int(0.78 * (r1 - r0))
        descent = max((r1 - baseline_row) * pt_per_px, height * 0.1)
        return width, height, descent

    TexManager.make_png = _tectonic_make_png
    TexManager.get_text_width_height_descent = _tectonic_get_text_width_height_descent
    _TEXMANAGER_PATCHED = True


def apply_paper_style(*, use_latex: bool = True, grid: bool = False) -> None:
    """Apply shared rcParams for paper-matched figures."""
    global _STYLE_APPLIED, _LATEX_BACKEND
    if _STYLE_APPLIED:
        return

    if use_latex:
        _LATEX_BACKEND = _detect_latex_backend()
    else:
        _LATEX_BACKEND = "mathtext"

    use_usetex = _LATEX_BACKEND in {"latex", "tectonic"}
    if _LATEX_BACKEND == "tectonic":
        _patch_texmanager_for_tectonic()

    mpl.style.use("classic")
    params: dict[str, object] = {
        "text.usetex": use_usetex,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 14,
        "axes.linewidth": 1.2,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "axes.grid": grid,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "legend.framealpha": 0.9,
        "axes.unicode_minus": False,
    }
    if use_usetex:
        params.update(
            {
                "text.latex.preamble": _LATEX_PREAMBLE,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )
    plt.rcParams.update(params)
    _STYLE_APPLIED = True


def style_axes(ax, *, grid: bool | None = None) -> None:
    """Apply per-axis styling (spines, optional grid)."""
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
    if grid is not None:
        ax.grid(grid, alpha=0.3, linestyle="--")


def paper_legend(ax, **kwargs) -> None:
    """Legend box matching plot_fictive_temperature.py."""
    defaults = {
        "loc": "upper right",
        "frameon": True,
        "fancybox": False,
        "edgecolor": "black",
        "framealpha": 0.9,
    }
    defaults.update(kwargs)
    ax.legend(**defaults)


def save_figure(fig, stem: Path, *, png_dpi: int = 150) -> None:
    """Save PDF at 300 dpi and companion PNG."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = stem.parent / f"{stem.name}.pdf"
    png_path = stem.parent / f"{stem.name}.png"
    fig.savefig(
        png_path,
        dpi=png_dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    if _LATEX_BACKEND == "tectonic":
        from PIL import Image

        with Image.open(png_path) as image:
            rgb = image.convert("RGB")
            rgb.save(pdf_path, "PDF", resolution=png_dpi)
    else:
        fig.savefig(
            pdf_path,
            dpi=300,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
    print(f"Wrote {pdf_path}")
    print(f"Wrote {png_path}")


def latex_backend() -> str:
    """Return active LaTeX backend: ``latex``, ``tectonic``, or ``mathtext``."""
    return _LATEX_BACKEND
