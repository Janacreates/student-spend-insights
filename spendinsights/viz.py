"""Small plotting helpers so every figure shares one look."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "primary": "#1f4e79",
    "accent": "#e07a1f",
    "grey": "#9aa0a6",
    "green": "#2e7d32",
    "red": "#c62828",
    "light": "#cfd8e3",
}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.titlesize": 11,
        "axes.grid": True,
        "grid.alpha": 0.25,
    })


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
