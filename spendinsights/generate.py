"""Synthetic student bank-transaction generator.

There is no public dataset of real student transactions, so this module simulates one.
Every parameter below is an assumption chosen to look plausible, not an estimate from
real data. Results computed on this data describe the *pipeline*, not real students.

Design choices that matter for the analysis:
  * Ground truth (true category, injected anomalies) is written to a separate file so
    the pipeline never sees it and can only be *evaluated* against it.
  * The raw export is deliberately messy (mixed date formats, amount styles, missing
    fields, duplicate rows, merchant strings with store numbers and typos) so the
    cleaning step has real work to do.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .config import CITIES, VARIABLE_CATEGORIES, Paths

MERCHANTS: Dict[str, List[str]] = {
    "groceries": [
        "SAFEWAY", "REAL CDN SUPERSTORE", "WALMART SUPERCENTRE", "SOBEYS", "COSTCO WHOLESALE",
        "FRESHCO", "NO FRILLS", "T&T SUPERMARKET", "CO-OP FOOD STORE", "LUCKY 8 MART",
        "GREEN LEAF MARKET", "SAVE ON FOODS", "ASIAN GROCERY HUB", "HARVEST FRESH",
    ],
    "dining": [
        "TIM HORTONS", "STARBUCKS", "MCDONALDS", "SUBWAY", "A&W", "BOSTON PIZZA", "JOES PIZZA",
        "MAPLE LEAF CAFE", "SUSHI KING", "THAI GARDEN", "CAMPUS CAFETERIA", "UBER *EATS",
        "DOORDASH", "SKIPTHEDISHES", "MOONLIGHT NOODLE HOUSE", "BLUE DOOR BISTRO",
        "PHO HOA RESTAURANT", "BUBBLE TEA HOUSE", "HARVEYS BURGERS", "EL SOL TACOS",
    ],
    "transport": [
        "ETS TRANSIT", "UBER *TRIP", "LYFT", "PETRO-CANADA", "SHELL", "ESSO", "CO-OP GAS BAR",
        "IMPARK PARKING", "CAMPUS PARKING", "VIA RAIL", "RED ARROW COACH", "CITY BIKE SHARE",
    ],
    "entertainment": [
        "CINEPLEX", "STEAM GAMES", "PLAYSTATION NETWORK", "TICKETMASTER", "BOWLERAMA",
        "ESCAPE ROOM CO", "LIVE NATION", "SPORTS PUB & GRILL", "THE BREWERY TAPROOM",
        "ARCADE PALACE", "CLIMBING GYM DROPIN", "UNIVERSITY THEATRE TICKETS",
    ],
    "shopping": [
        "AMAZON.CA", "ZARA", "H&M", "BEST BUY", "WINNERS", "UNIQLO", "IKEA", "DOLLARAMA",
        "LULULEMON", "SHEIN", "APPLE STORE", "THRIFT TREASURES", "MAPLE MALL KIOSK", "TECH DEPOT",
    ],
    "health": [
        "SHOPPERS DRUG MART", "LONDON DRUGS", "UNIVERSITY PHARMACY", "WELLNESS CLINIC",
        "DENTAL CENTRE", "LIFELABS", "OPTICAL BOUTIQUE",
    ],
    "education": [
        "CAMPUS BOOKSTORE", "CHEGG", "PEARSON", "COURSERA", "STAPLES", "CAMPUS PRINT SHOP",
        "STUDENT UNION STORE",
    ],
    "rent": [
        "E-TRANSFER RENT PAYMENT", "RENT - CAMPUS RESIDENCE", "PROPERTY MGMT RENT",
        "LANDLORD E-TRANSFER RENT",
    ],
    "tuition": ["UNIVERSITY TUITION PAYMENT", "STUDENT ACCOUNT TUITION", "TUITION FEES PAYMENT"],
    "income": [
        "PAYROLL DEPOSIT", "DIRECT DEPOSIT PAYROLL", "BURSARY DEPOSIT", "STUDENT AID DEPOSIT",
        "SCHOLARSHIP DEPOSIT", "E-TRANSFER FROM PARENT",
    ],
}

SUBSCRIPTION_PRICE = {
    "NETFLIX.COM": 16.49, "SPOTIFY": 11.99, "APPLE.COM/BILL": 4.99,
    "GOOGLE *YOUTUBE PREMIUM": 13.99, "AMAZON PRIME*MEMBERSHIP": 9.99, "DISNEY PLUS": 11.99,
    "ADOBE CREATIVE CLOUD": 27.99, "GOODLIFE FITNESS": 24.99, "PLANET FITNESS": 14.99,
    "NORDVPN": 6.99, "OPENAI *CHATGPT SUBSCR": 26.00, "CRUNCHYROLL": 9.99,
}
MERCHANTS["subscriptions"] = list(SUBSCRIPTION_PRICE)

# Merchants used only for injected "unknown merchant" anomalies (fictional names).
FRAUD_MERCHANTS = ["GLOBALSHOP HK", "QUICKCASH TOPUP", "PROMOSAVE CO", "GIFTCARD OUTLET", "LUXDEALS INTL"]

# category -> (expected transactions per month, mean amount in $, lognormal sigma)
CATEGORY_PARAMS = {
    "groceries": (7.0, 42.0, 0.55),
    "dining": (11.0, 16.0, 0.60),
    "transport": (9.0, 10.0, 0.60),
    "entertainment": (3.0, 24.0, 0.70),
    "shopping": (3.5, 38.0, 0.80),
    "health": (1.0, 22.0, 0.70),
    "education": (0.6, 30.0, 0.70),
}
IMPULSE_CATS = {"dining", "entertainment", "shopping"}
PLAIN_CATS = {"rent", "tuition", "income"}  # descriptions without store numbers / city
PREFIXES = ["", "", "", "", "POS PURCHASE ", "PURCHASE ", "SQ *", "TST* "]


def _seasonal(cat: str, month: int) -> float:
    """Multiplier on the monthly transaction rate (exam months, holidays, term starts)."""
    if cat == "dining":
        return 1.3 if month in (4, 12) else (0.85 if month in (7, 8) else 1.0)
    if cat == "shopping":
        return {9: 1.25, 11: 1.4, 12: 1.7}.get(month, 1.0)
    if cat == "education":
        return 6.0 if month in (9, 1) else 1.0
    if cat == "entertainment":
        return 1.2 if month in (5, 6, 12) else 1.0
    if cat == "transport":
        return 1.15 if month in (12, 1, 2) else 1.0
    return 1.0


def _sample_days(cat: str, month_start: pd.Timestamp, n: int, rng: np.random.Generator) -> pd.DatetimeIndex:
    days = pd.date_range(month_start, periods=month_start.days_in_month, freq="D")
    w = np.ones(len(days))
    if cat in ("dining", "entertainment"):
        w[days.dayofweek >= 5] = 1.7
    return days[rng.choice(len(days), size=n, p=w / w.sum())]


def _render(merchant: str, cat: str, city: str, rng: np.random.Generator) -> str:
    """Turn a clean merchant name into a messy statement descriptor."""
    if cat in PLAIN_CATS:
        return str(merchant)
    m = str(merchant)
    r = rng.random()
    if r < 0.03 and len(m) > 8:
        m = m[: -int(rng.integers(1, 4))]  # truncated by the processor
    elif r < 0.06:
        m = m.replace(" ", "")  # spaces dropped
    prefix = str(rng.choice(PREFIXES))
    store = f" #{int(rng.integers(100, 9999))}" if rng.random() < 0.45 else ""
    loc = str(rng.choice(["", "", f" {city}", f" {city} AB"]))
    return f"{prefix}{m}{store}{loc}"


def _make_students(n: int, rng: np.random.Generator) -> pd.DataFrame:
    housing = rng.choice(["residence", "shared", "family"], size=n, p=[0.25, 0.5, 0.25])
    rent = np.where(housing == "residence", rng.normal(1050, 120, n),
                    np.where(housing == "shared", rng.normal(750, 110, n), 0.0)).round(0)
    has_job = rng.random(n) < 0.65
    return pd.DataFrame({
        "student_id": [f"S{i:04d}" for i in range(n)],
        "housing": housing,
        "city": rng.choice(CITIES, n, p=[0.5, 0.25, 0.1, 0.1, 0.05]),
        "rent": rent,
        "job_monthly": np.where(has_job, rng.lognormal(np.log(900), 0.35, n), 0.0).round(2),
        "parent_support": np.where(rng.random(n) < 0.4, rng.choice([300, 400, 500, 600], n), 0),
        "tuition": rng.normal(3200, 500, n).clip(2000, 5000).round(2),
        "has_aid": rng.random(n) < 0.85,
    })


def _fmt_date(d: pd.Timestamp, rng: np.random.Generator) -> str:
    r = rng.random()
    if r < 0.70:
        return d.strftime("%Y-%m-%d")
    if r < 0.85:
        return d.strftime("%m/%d/%Y")  # MM/DD/YYYY convention
    if r < 0.95:
        return d.strftime("%d-%b-%Y")
    return f"{d.strftime('%b')} {d.day}, {d.year}"


def _fmt_amount(a: float, rng: np.random.Generator) -> str:
    """Signed amount rendered in one of several export styles (negative = debit)."""
    body = f"{abs(a):,.2f}" if rng.random() < 0.3 else f"{abs(a):.2f}"
    dollar = "$" if rng.random() < 0.25 else ""
    if a < 0:
        if rng.random() < 0.06:
            return f"({dollar}{body})"  # accounting-style negative
        return f"-{dollar}{body}"
    return f"{dollar}{body}"


def generate(
    n_students: int = 150, n_months: int = 24, seed: int = 42, start: str = "2023-09-01"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (raw_transactions, ground_truth, students)."""
    rng = np.random.default_rng(seed)
    months = pd.date_range(start, periods=n_months, freq="MS")
    students = _make_students(n_students, rng)
    rows: list = []

    def add(sid, date, merchant, cat, amount, city):
        rows.append((sid, date, _render(merchant, cat, city, rng), str(merchant), cat, round(float(amount), 2)))

    for s in students.itertuples(index=False):
        level = {c: rng.lognormal(0, 0.35) for c in CATEGORY_PARAMS}
        impulse = rng.lognormal(0, 0.25)
        fav = {c: list(rng.choice(MERCHANTS[c], size=min(5, len(MERCHANTS[c])), replace=False))
               for c in CATEGORY_PARAMS}
        subs = list(rng.choice(list(SUBSCRIPTION_PRICE), size=int(min(5, 1 + rng.poisson(1.5))), replace=False))
        sub_start = {m: int(rng.integers(0, 6)) for m in subs}
        sub_end = {m: (int(rng.integers(n_months // 3, n_months)) if rng.random() < 0.15 else n_months + 1) for m in subs}
        sub_day = {m: int(rng.integers(1, 28)) for m in subs}
        rent_merchant = str(rng.choice(MERCHANTS["rent"]))
        pays_summer_rent = rng.random() < 0.6

        for mi, month in enumerate(months):
            m = month.month
            drift = 1.003 ** mi  # mild inflation-like trend

            for cat, (rate, mean_amt, sigma) in CATEGORY_PARAMS.items():
                n = int(rng.poisson(rate * _seasonal(cat, m)))
                if n == 0:
                    continue
                factor = level[cat] * (impulse if cat in IMPULSE_CATS else 1.0) * drift
                amts = rng.lognormal(np.log(mean_amt * factor) - sigma ** 2 / 2, sigma, n).clip(min=0.75)
                for a, d in zip(amts, _sample_days(cat, month, n, rng)):
                    merch = rng.choice(fav[cat]) if rng.random() < 0.85 else rng.choice(MERCHANTS[cat])
                    add(s.student_id, d, merch, cat, -a, s.city)

            if s.rent > 0 and (m not in (5, 6, 7, 8) or pays_summer_rent):
                add(s.student_id, month + pd.Timedelta(days=int(rng.integers(0, 3))),
                    rent_merchant, "rent", -s.rent, s.city)
            if m in (9, 1) and rng.random() < 0.92:
                add(s.student_id, month + pd.Timedelta(days=int(rng.integers(4, 15))),
                    rng.choice(MERCHANTS["tuition"]), "tuition", -s.tuition, s.city)
                if s.has_aid:
                    add(s.student_id, month + pd.Timedelta(days=2),
                        rng.choice(["BURSARY DEPOSIT", "STUDENT AID DEPOSIT", "SCHOLARSHIP DEPOSIT"]),
                        "income", s.tuition * 0.8, s.city)
            if s.job_monthly > 0:
                for day in (0, 14):
                    add(s.student_id, month + pd.Timedelta(days=day),
                        rng.choice(["PAYROLL DEPOSIT", "DIRECT DEPOSIT PAYROLL"]), "income",
                        s.job_monthly / 2 * rng.uniform(0.9, 1.1), s.city)
            if s.parent_support > 0:
                add(s.student_id, month + pd.Timedelta(days=4), "E-TRANSFER FROM PARENT", "income",
                    s.parent_support, s.city)
            for sub in subs:
                if sub_start[sub] <= mi < sub_end[sub]:
                    add(s.student_id, month + pd.Timedelta(days=sub_day[sub] - 1), sub, "subscriptions",
                        -SUBSCRIPTION_PRICE[sub], s.city)

    df = pd.DataFrame(rows, columns=["student_id", "date", "description", "merchant", "category", "amount"])
    df["is_anomaly"] = False
    df["anomaly_type"] = ""

    # ---- inject labelled anomalies (used only to evaluate detection) ----
    eligible = df.index[(df.amount < 0) & df.category.isin(VARIABLE_CATEGORIES + ["health", "education"])]
    n_spike = int(0.004 * len(eligible))
    spike_idx = rng.choice(eligible, size=n_spike, replace=False)
    df.loc[spike_idx, "amount"] = (df.loc[spike_idx, "amount"] * rng.uniform(6, 15, n_spike)).round(2)
    df.loc[spike_idx, ["is_anomaly", "anomaly_type"]] = [True, "spike"]

    rest = np.setdiff1d(eligible, spike_idx)
    dup_idx = rng.choice(rest, size=int(0.002 * len(eligible)), replace=False)
    dups = df.loc[dup_idx].copy()
    dups["date"] = dups["date"] + pd.to_timedelta(rng.integers(0, 2, len(dups)), unit="D")
    dups[["is_anomaly", "anomaly_type"]] = [True, "duplicate"]

    n_fraud = max(5, int(0.001 * len(eligible)))
    span_days = (months[-1] + pd.offsets.MonthEnd(0) - months[0]).days
    fraud = pd.DataFrame({
        "student_id": rng.choice(students.student_id, n_fraud),
        "date": months[0] + pd.to_timedelta(rng.integers(0, span_days, n_fraud), unit="D"),
        "merchant": rng.choice(FRAUD_MERCHANTS, n_fraud),
        "category": "shopping",
        "amount": -rng.uniform(40, 400, n_fraud).round(2),
        "is_anomaly": True,
        "anomaly_type": "unknown_merchant",
    })
    city_of = students.set_index("student_id")["city"]
    fraud["description"] = [_render(mm, "shopping", city_of[sid], rng)
                            for mm, sid in zip(fraud.merchant, fraud.student_id)]

    df = pd.concat([df, dups, fraud], ignore_index=True)
    df = df.sort_values(["date", "student_id"], kind="stable").reset_index(drop=True)
    df["txn_id"] = [f"T{i:07d}" for i in range(len(df))]

    truth = df[["txn_id", "category", "merchant", "is_anomaly", "anomaly_type"]].rename(
        columns={"category": "true_category"})

    # ---- render the messy raw export ----
    raw = pd.DataFrame({
        "txn_id": df.txn_id,
        "student_id": df.student_id,
        "posted_date": [_fmt_date(d, rng) for d in df.date],
        "description": df.description,
        "amount": [_fmt_amount(a, rng) for a in df.amount],
        "txn_type": np.where(df.amount < 0, "DEBIT", "CREDIT"),
    })
    n = len(raw)
    raw.loc[rng.random(n) < 0.003, "description"] = ""  # missing merchant text
    lower = rng.random(n) < 0.03
    raw.loc[lower, "description"] = raw.loc[lower, "description"].str.lower()
    raw.loc[rng.random(n) < 0.01, "txn_type"] = ""  # missing debit/credit flag
    dup_rows = raw.sample(frac=0.002, random_state=seed)  # exact duplicate export rows
    raw = pd.concat([raw, dup_rows]).sort_index(kind="stable").reset_index(drop=True)

    return raw, truth, students[["student_id", "housing", "city"]]


def write_dataset(paths: Paths, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths.ensure()
    raw, truth, students = generate(**kwargs)
    raw.to_csv(paths.raw_csv, index=False)
    truth.to_csv(paths.truth_csv, index=False)
    students.to_csv(paths.students_csv, index=False)
    return raw, truth, students


if __name__ == "__main__":
    r, t, s = write_dataset(Paths())
    print(f"wrote {len(r):,} raw rows for {len(s)} students")
