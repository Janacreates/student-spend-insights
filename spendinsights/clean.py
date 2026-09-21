"""Profiling and cleaning of the raw transaction export.

`clean()` returns the cleaned frame plus a profile dict that documents every issue found
and what was done about it (written to reports/data_quality_report.md).
"""
from __future__ import annotations

import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import CITIES, Paths

DATE_PATTERNS = [
    ("iso (YYYY-MM-DD)", r"^\d{4}-\d{2}-\d{2}$", "%Y-%m-%d"),
    ("us slash (MM/DD/YYYY)", r"^\d{2}/\d{2}/\d{4}$", "%m/%d/%Y"),
    ("DD-Mon-YYYY", r"^\d{2}-[A-Za-z]{3}-\d{4}$", "%d-%b-%Y"),
    ("Mon D, YYYY", r"^[A-Za-z]{3} \d{1,2}, \d{4}$", "%b %d, %Y"),
]

_PREFIX = re.compile(r"^(POS PURCHASE|PURCHASE|SQ ?\*|TST ?\*)\s*")
_STORE_NO = re.compile(r"#\s*\d+")
_TRAIL_CITY = re.compile(r"\s+(" + "|".join(CITIES) + r")(\s+AB)?$")


def normalize_description(text) -> str:
    """Strip processor prefixes, store numbers and trailing city so the same merchant matches."""
    if text is None or (isinstance(text, float) and np.isnan(text)) or not str(text).strip():
        return "UNKNOWN"
    t = str(text).upper().strip()
    t = _PREFIX.sub("", t)
    t = _STORE_NO.sub("", t)
    t = _TRAIL_CITY.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or "UNKNOWN"


def parse_dates(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse the mixed date formats; returns (dates, format label per row)."""
    s = series.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    labels = pd.Series("unparsed", index=s.index, dtype="object")
    for label, pattern, fmt in DATE_PATTERNS:
        mask = s.str.match(pattern).fillna(False).astype(bool)
        if mask.any():
            out.loc[mask] = pd.to_datetime(s[mask], format=fmt, errors="coerce")
            labels.loc[mask] = label
    return out, labels


def parse_amounts(series: pd.Series) -> Tuple[pd.Series, Dict[str, int]]:
    """Parse '$1,250.00', '-4.75', '(12.50)' into signed floats (negative = debit)."""
    s = series.astype("string").str.strip()
    paren = (s.str.startswith("(") & s.str.endswith(")")).fillna(False).astype(bool)
    neg = (s.str.startswith("-").fillna(False).astype(bool)) | paren
    num = s.str.replace(r"[^0-9.]", "", regex=True).astype(object)
    vals = pd.to_numeric(num, errors="coerce").astype(float)
    signed = pd.Series(np.where(neg, -vals, vals), index=series.index)
    styles = {
        "with_dollar_sign": int(s.str.contains(r"\$", regex=True).fillna(False).sum()),
        "with_thousands_separator": int(s.str.contains(",", regex=False).fillna(False).sum()),
        "accounting_parentheses": int(paren.sum()),
    }
    return signed, styles


def clean(raw: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    prof: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()

    dup = df.duplicated(keep="first")
    prof["exact_duplicate_rows_removed"] = int(dup.sum())
    df = df[~dup].copy()
    prof["duplicate_txn_ids_after_dedupe"] = int(df.txn_id.duplicated().sum())

    dates, fmt = parse_dates(df["posted_date"])
    prof["date_formats"] = fmt.value_counts().to_dict()
    prof["unparseable_dates"] = int(dates.isna().sum())

    amount, styles = parse_amounts(df["amount"])
    prof["amount_styles"] = styles
    prof["unparseable_amounts"] = int(amount.isna().sum())

    desc_missing = df["description"].isna() | (df["description"].astype("string").str.strip() == "")
    prof["missing_description"] = int(desc_missing.sum())

    given = df["txn_type"].astype("string").str.lower().fillna("")
    valid = given.isin(["debit", "credit"])
    inferred = np.where(amount < 0, "debit", "credit")
    direction = np.where(valid, given, inferred)
    prof["missing_txn_type_inferred_from_sign"] = int((~valid).sum())
    prof["txn_type_sign_conflicts"] = int((valid & (given.to_numpy() != inferred)).sum())

    keep = dates.notna() & amount.notna()
    prof["rows_dropped_unparseable"] = int((~keep).sum())

    out = pd.DataFrame({
        "txn_id": df.txn_id,
        "student_id": df.student_id,
        "date": dates,
        "description_raw": df.description,
        "description_norm": [normalize_description(x) for x in df.description],
        "amount": amount,
        "direction": direction,
    })[keep.to_numpy()]
    out["spend"] = np.where(out.direction == "debit", out.amount.abs(), 0.0)
    out["month"] = out.date.dt.to_period("M").dt.to_timestamp()
    out = out.sort_values(["date", "txn_id"]).reset_index(drop=True)

    prof["rows_clean"] = int(len(out))
    prof["students"] = int(out.student_id.nunique())
    prof["date_min"] = str(out.date.min().date())
    prof["date_max"] = str(out.date.max().date())
    prof["debit_rows"] = int((out.direction == "debit").sum())
    prof["credit_rows"] = int((out.direction == "credit").sum())
    prof["unique_normalized_descriptions"] = int(out.description_norm.nunique())
    return out, prof


def write_quality_report(prof: dict, path) -> None:
    lines = [
        "# Data quality report",
        "",
        "Generated by `spendinsights.clean`. Counts are for the raw export before cleaning.",
        "",
        f"- Raw rows: **{prof['rows_raw']:,}** -> clean rows: **{prof['rows_clean']:,}**"
        f" ({prof['students']} students, {prof['date_min']} to {prof['date_max']})",
        f"- Debits: {prof['debit_rows']:,}, credits: {prof['credit_rows']:,}",
        "",
        "| Issue | Count | Handling |",
        "|---|---:|---|",
        f"| Exact duplicate rows | {prof['exact_duplicate_rows_removed']:,} | dropped (first kept) |",
        f"| Duplicate `txn_id` after de-dupe | {prof['duplicate_txn_ids_after_dedupe']:,} | none expected |",
        f"| Missing merchant description | {prof['missing_description']:,} | kept, labelled `UNKNOWN`, never auto-categorised |",
        f"| Missing debit/credit flag | {prof['missing_txn_type_inferred_from_sign']:,} | inferred from amount sign |",
        f"| Flag/sign conflicts | {prof['txn_type_sign_conflicts']:,} | none expected |",
        f"| Unparseable dates | {prof['unparseable_dates']:,} | row dropped |",
        f"| Unparseable amounts | {prof['unparseable_amounts']:,} | row dropped |",
        "",
        "## Date formats found",
        "",
        "| Format | Rows |",
        "|---|---:|",
    ]
    lines += [f"| {k} | {v:,} |" for k, v in prof["date_formats"].items()]
    lines += [
        "",
        "Slash dates are parsed as MM/DD/YYYY. That is an assumption about the export; a real",
        "pipeline should confirm the convention with the data owner because DD/MM is ambiguous",
        "for days <= 12.",
        "",
        "## Amount styles found",
        "",
        "| Style | Rows |",
        "|---|---:|",
    ]
    lines += [f"| {k.replace('_', ' ')} | {v:,} |" for k, v in prof["amount_styles"].items()]
    lines += [
        "",
        f"Merchant normalisation (strip processor prefixes, store numbers, trailing city) reduced the "
        f"descriptions to **{prof['unique_normalized_descriptions']:,}** unique strings.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    p = Paths()
    cleaned, profile = clean(pd.read_csv(p.raw_csv, dtype=str, keep_default_na=False, na_values=[""]))
    p.reports.mkdir(parents=True, exist_ok=True)
    write_quality_report(profile, p.reports / "data_quality_report.md")
    print(profile)
