import numpy as np
import pandas as pd
import pytest

from spendinsights.clean import clean, normalize_description, parse_amounts, parse_dates


@pytest.mark.parametrize("raw,expected", [
    ("POS PURCHASE TIM HORTONS #1234 EDMONTON AB", "TIM HORTONS"),
    ("SQ *JOES PIZZA", "JOES PIZZA"),
    ("TST* SUSHI KING #88 CALGARY", "SUSHI KING"),
    ("safeway", "SAFEWAY"),
    ("UBER *EATS", "UBER *EATS"),
    ("", "UNKNOWN"),
    (None, "UNKNOWN"),
    (float("nan"), "UNKNOWN"),
])
def test_normalize_description(raw, expected):
    assert normalize_description(raw) == expected


def test_parse_dates_handles_all_formats():
    s = pd.Series(["2024-03-05", "03/05/2024", "05-Mar-2024", "Mar 5, 2024", "garbage"])
    dates, labels = parse_dates(s)
    assert (dates.iloc[:4] == pd.Timestamp("2024-03-05")).all()
    assert pd.isna(dates.iloc[4])
    assert labels.iloc[4] == "unparsed"


def test_parse_amounts_styles_and_signs():
    s = pd.Series(["-12.50", "-$1,250.00", "(4.75)", "$60.00", "45"])
    vals, styles = parse_amounts(s)
    assert vals.tolist() == [-12.5, -1250.0, -4.75, 60.0, 45.0]
    assert styles["accounting_parentheses"] == 1
    assert styles["with_thousands_separator"] == 1


def _raw(rows):
    return pd.DataFrame(rows, columns=["txn_id", "student_id", "posted_date", "description", "amount", "txn_type"])


def test_clean_drops_exact_duplicates_and_infers_direction():
    raw = _raw([
        ["T1", "S1", "2024-01-05", "TIM HORTONS #1", "-4.50", "DEBIT"],
        ["T1", "S1", "2024-01-05", "TIM HORTONS #1", "-4.50", "DEBIT"],  # exact duplicate
        ["T2", "S1", "01/06/2024", "PAYROLL DEPOSIT", "500.00", ""],      # missing flag
        ["T3", "S1", "2024-01-07", "", "(9.99)", "DEBIT"],                # missing description
    ])
    out, prof = clean(raw)
    assert prof["exact_duplicate_rows_removed"] == 1
    assert len(out) == 3
    assert out.set_index("txn_id").loc["T2", "direction"] == "credit"
    assert out.set_index("txn_id").loc["T3", "description_norm"] == "UNKNOWN"
    assert out.set_index("txn_id").loc["T3", "spend"] == pytest.approx(9.99)
    assert prof["missing_txn_type_inferred_from_sign"] == 1
    assert prof["missing_description"] == 1
    assert np.isclose(out[out.direction == "credit"].spend.sum(), 0.0)  # credits are not spend
