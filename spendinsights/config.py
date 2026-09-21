"""Shared constants and output locations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = 42

CITIES = ["EDMONTON", "CALGARY", "RED DEER", "LETHBRIDGE", "ST ALBERT"]

# Categories a student can realistically steer with a budget.
VARIABLE_CATEGORIES = ["groceries", "dining", "transport", "entertainment", "shopping"]
# Large, lumpy, or contractual outflows that a budget nudge would not target.
FIXED_CATEGORIES = ["rent", "tuition"]


@dataclass(frozen=True)
class Paths:
    """All input/output locations, rooted so tests can run in a temp dir."""

    root: Path = ROOT

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw_csv(self) -> Path:
        return self.data / "raw" / "transactions_raw.csv"

    @property
    def truth_csv(self) -> Path:
        return self.data / "raw" / "ground_truth.csv"

    @property
    def students_csv(self) -> Path:
        return self.data / "raw" / "students.csv"

    @property
    def db(self) -> Path:
        return self.data / "spend.db"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def figures(self) -> Path:
        return self.reports / "figures"

    @property
    def sql(self) -> Path:
        # SQL files ship with the repo, independent of the output root.
        return ROOT / "sql"

    def ensure(self) -> None:
        for p in (self.data / "raw", self.figures):
            p.mkdir(parents=True, exist_ok=True)
