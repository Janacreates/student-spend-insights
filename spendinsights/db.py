"""SQLite layer: load cleaned data and run the analysis queries in sql/."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import Paths


def connect(paths: Paths) -> sqlite3.Connection:
    return sqlite3.connect(paths.db)


def build_db(paths: Paths, txns: pd.DataFrame, students: pd.DataFrame) -> None:
    """(Re)create the database with students and categorised transactions."""
    paths.data.mkdir(parents=True, exist_ok=True)
    if paths.db.exists():
        paths.db.unlink()
    t = txns.copy()
    t["date"] = t["date"].dt.strftime("%Y-%m-%d")
    t["month"] = t["month"].dt.strftime("%Y-%m-01")
    cols = ["txn_id", "student_id", "date", "month", "description_norm", "amount", "spend",
            "direction", "category", "confidence", "category_source"]
    with connect(paths) as con:
        students.to_sql("students", con, index=False)
        t[cols].to_sql("transactions", con, index=False)
        con.execute("CREATE INDEX idx_txn_student_month ON transactions(student_id, month)")
        con.execute("CREATE INDEX idx_txn_category ON transactions(category)")


def save_table(paths: Paths, name: str, df: pd.DataFrame) -> None:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    with connect(paths) as con:
        out.to_sql(name, con, index=False, if_exists="replace")


def query(paths: Paths, name: str, **params) -> pd.DataFrame:
    """Run sql/<name>.sql (with optional :named parameters) and return a DataFrame."""
    sql = (paths.sql / f"{name}.sql").read_text(encoding="utf-8")
    with connect(paths) as con:
        return pd.read_sql_query(sql, con, params=params or None)
