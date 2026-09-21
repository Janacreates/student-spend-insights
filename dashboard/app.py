"""Streamlit demo: a student sets budgets and sees where they stand mid-month.

Run the pipeline first (`make run`), then: streamlit run dashboard/app.py
All data shown is synthetic.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "spend.db"
VARIABLE = ["groceries", "dining", "transport", "entertainment", "shopping"]

st.set_page_config(page_title="Student Spend Insights", layout="wide")


@st.cache_data
def load() -> dict:
    con = sqlite3.connect(DB)
    out = {
        "txns": pd.read_sql_query(
            "SELECT student_id, date, month, description_norm AS merchant, spend, category "
            "FROM transactions WHERE direction = 'debit'", con, parse_dates=["date", "month"]),
        "forecasts": pd.read_sql_query("SELECT * FROM forecasts", con, parse_dates=["month"]),
        "flags": pd.read_sql_query("SELECT * FROM flags", con, parse_dates=["date"]),
        "recurring": pd.read_sql_query("SELECT * FROM recurring_charges", con),
    }
    con.close()
    return out


if not DB.exists():
    st.error("No database found. Run `make run` (or `python -m spendinsights.pipeline`) first.")
    st.stop()

data = load()
st.title("Student Spend Insights")
st.caption("Synthetic data. Budgets default to the model's forecast; overrides are not saved.")

students = sorted(data["forecasts"].student_id.unique())
months = sorted(data["forecasts"].month.unique())
c1, c2, c3 = st.columns(3)
student = c1.selectbox("Student", students)
month = pd.Timestamp(c2.selectbox("Month", months, index=len(months) - 1,
                                  format_func=lambda m: pd.Timestamp(m).strftime("%B %Y")))
days_in_month = month.days_in_month
day = c3.slider("As of day of month", 5, days_in_month, 15)

tx = data["txns"]
tx_m = tx[(tx.student_id == student) & (tx.month == month)]
mtd = tx_m[tx_m.date.dt.day <= day].groupby("category").spend.sum()
fc = data["forecasts"]
fc = fc[(fc.student_id == student) & (fc.month == month)].set_index("category")

st.subheader("Budget check")
rows = []
cols = st.columns(len(VARIABLE))
for col, cat in zip(cols, VARIABLE):
    p10, p50, p90 = (fc.loc[cat, k] if cat in fc.index else 0.0 for k in ("p10", "p50", "p90"))
    budget = col.number_input(f"{cat} budget ($)", min_value=0.0, value=float(round(p50 / 5) * 5), step=5.0,
                              key=f"{student}-{month}-{cat}")
    spent = float(mtd.get(cat, 0.0))
    projected = spent * days_in_month / day
    if budget <= 0:
        status = "no budget"
    elif projected > 1.15 * budget:
        status = "over pace"
    elif projected > budget:
        status = "watch"
    else:
        status = "on track"
    rows.append({"category": cat, "spent so far": spent, "projected month-end": projected, "budget": budget,
                 "forecast range (p10-p90)": f"${p10:,.0f} - ${p90:,.0f}", "status": status})
table = pd.DataFrame(rows)
st.dataframe(table.style.format({"spent so far": "${:,.0f}", "projected month-end": "${:,.0f}", "budget": "${:,.0f}"}),
             hide_index=True, use_container_width=True)
over = table[table.status == "over pace"]
if len(over):
    st.warning("Nudge: on pace to exceed budget in " + ", ".join(over.category) + ".")
else:
    st.success("No categories are on pace to exceed budget by more than 15%.")
st.bar_chart(table.set_index("category")[["spent so far", "projected month-end", "budget"]], stack=False)

left, right = st.columns(2)
with left:
    st.subheader("Where the money went this month")
    by_cat = tx_m.groupby("category").spend.sum().sort_values(ascending=False)
    st.bar_chart(by_cat)
with right:
    st.subheader("Subscriptions detected")
    rec = data["recurring"]
    rec = rec[rec.student_id == student][["merchant", "avg_charge", "months_charged"]]
    st.dataframe(rec, hide_index=True, use_container_width=True)
    st.caption(f"About ${rec.avg_charge.sum():,.2f} per month in recurring charges.")

st.subheader("Transactions flagged for review")
fl = data["flags"][data["flags"].student_id == student].sort_values("date", ascending=False).head(15)
st.dataframe(fl[["date", "description_norm", "category", "spend", "reason"]], hide_index=True,
             use_container_width=True)
