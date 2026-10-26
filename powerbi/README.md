# Power BI report

A three-page Power BI report on the star schema from [`warehouse.py`](../spendinsights/warehouse.py). All data is synthetic (see the main README).

The finished report is [`student_spend_insights.pbix`](student_spend_insights.pbix), with a [PDF export](student_spend_insights.pdf) for viewing without Power BI. This guide rebuilds it from scratch.

## 1. Get the data

On the machine that runs the pipeline:

```bash
make run   # if data/spend.db does not exist yet
make bi    # writes data/powerbi/*.csv (7 files, about 11 MB)
```

Copy the `data/powerbi/` folder to the Windows machine with Power BI Desktop. In Power BI, use **Get data > Text/CSV** for each of the 7 files, with type detection based on the entire dataset. (**Get data > Folder** would combine them into one table, which is wrong here because each file has different columns.) Check these types:

| Column | Type |
|---|---|
| `dim_date[date]` | Date |
| all `*_key` columns | Whole number |
| `spend`, `amount`, `actual`, `forecast`, `lower_80`, `upper_80`, `abs_error` | Decimal number |
| `is_*`, `in_interval`, `budget_eligible`, `alert_day15` | Whole number (blank where not applicable) |

## 2. Model

In **Model view**, create these relationships: many-to-one, single direction, from each fact to the dimension.

| From (fact) | To (dimension) |
|---|---|
| `fact_transactions[date_key]` | `dim_date[date_key]` |
| `fact_transactions[student_key]` | `dim_student[student_key]` |
| `fact_transactions[merchant_key]` | `dim_merchant[merchant_key]` |
| `fact_transactions[category_key]` | `dim_category[category_key]` |
| `fact_forecast[month_key]` | `dim_date[date_key]` (the first day of each month) |
| `fact_forecast[student_key]` | `dim_student[student_key]` |
| `fact_forecast[category_key]` | `dim_category[category_key]` |
| `fact_flags[date_key]` | `dim_date[date_key]` |
| `fact_flags[student_key]`, `[merchant_key]`, `[category_key]` | the matching dimensions |

Then select `dim_date` and choose **Table tools > Mark as date table**, using the `date` column. Sort `dim_date[month_name]` by `dim_date[month]`.

## 3. Measures

Create an empty table called `_Measures` (**Enter data**, then delete its one column) and add:

```dax
Total Spend = CALCULATE ( SUM ( fact_transactions[spend] ), fact_transactions[direction] = "debit" )

Transactions = COUNTROWS ( fact_transactions )

Avg Monthly Spend per Student =
AVERAGEX (
    SUMMARIZE ( fact_transactions, dim_student[student_key], dim_date[month_start] ),
    [Total Spend]
)

Controllable Spend = CALCULATE ( [Total Spend], dim_category[category_group] = "controllable" )

Controllable Share = DIVIDE ( [Controllable Spend], [Total Spend] )

Spend Last Month = CALCULATE ( [Total Spend], DATEADD ( dim_date[date], -1, MONTH ) )

MoM Change = DIVIDE ( [Total Spend] - [Spend Last Month], [Spend Last Month] )

Actual (test months) = SUM ( fact_forecast[actual] )

Forecast (test months) = SUM ( fact_forecast[forecast] )

Forecast MAE = AVERAGE ( fact_forecast[abs_error] )

Interval Coverage = AVERAGE ( fact_forecast[in_interval] )

Overspend Rate = CALCULATE ( AVERAGE ( fact_forecast[is_material_overspend] ), fact_forecast[budget_eligible] = 1 )

Alerts Sent = SUM ( fact_forecast[alert_day15] )

Correct Alerts =
CALCULATE (
    COUNTROWS ( fact_forecast ),
    fact_forecast[alert_day15] = 1,
    fact_forecast[is_material_overspend] = 1
)

Alert Precision = DIVIDE ( [Correct Alerts], [Alerts Sent] )

Alert Recall =
DIVIDE ( [Correct Alerts], CALCULATE ( COUNTROWS ( fact_forecast ), fact_forecast[is_material_overspend] = 1 ) )

Flagged Transactions = COUNTROWS ( fact_flags )

Flagged Spend = SUM ( fact_flags[spend] )
```

Format the spend measures as currency, and `Controllable Share`, `MoM Change`, `Interval Coverage`, `Overspend Rate`, `Alert Precision` and `Alert Recall` as percentages.

**Check your model against the pipeline** (seed 42, no filters applied). If these match, the relationships and measures are right:

| Measure | Expected |
|---|---:|
| Total Spend | $7,495,878.47 |
| Avg Monthly Spend per Student | $2,082.19 |
| Controllable Share | 44.8% |
| Forecast MAE | 76.60 |
| Interval Coverage | 80.2% |
| Overspend Rate | 37.2% |
| Alerts Sent | 1,833 |
| Alert Precision / Recall (all six test months) | 66.7% / 73.0% |
| Flagged Transactions / Flagged Spend | 3,167 / $378,833.08 |

The main README's 67.8% and 74.2% score the alert on the held-out second half of the test months only (June to August 2025). Filter `dim_date[month_start]` to those months to reproduce them.

## 4. Pages

1. **Overview.** Cards: Total Spend, Avg Monthly Spend per Student, Controllable Share, Overspend Rate. A line chart of Total Spend by `dim_date[month_start]`, with `dim_category[category_group]` as the legend. A bar chart of Total Spend by `dim_category[category]`. Slicers: `dim_date[academic_year]`, `dim_student[housing]`, `dim_student[city]`.
2. **Segments and merchants.** A matrix of Avg Monthly Spend per Student by `housing` (rows) and `city` (columns), with conditional formatting. A bar chart of the top 10 `dim_merchant[merchant_name]` by Controllable Spend (Top N filter). A clustered column chart of Avg Monthly Spend per Student by `housing`, split by `dim_date[is_exam_month]` and filtered to dining. It uses the per-month average rather than Total Spend because there are only 4 exam months against 20 others.
3. **Forecasts and alerts.** Cards: Forecast MAE, Interval Coverage, Alert Precision, Alert Recall. A line chart of Actual and Forecast (test months) by month. A table by `dim_category[category]` with MAE, Interval Coverage, Overspend Rate, Alert Precision and Alert Recall. A bar chart of Flagged Transactions by `fact_flags[reason]`.

Give each page a title text box, followed by the line "Synthetic data: 150 simulated students over 24 months."

Older versions of Power BI Desktop label the chart wells **Axis** and **Values** instead of **X-axis** and **Y-axis**.

## 5. Save

- Save the report as `powerbi/student_spend_insights.pbix`.
- Use **File > Export > Export to PDF** for `powerbi/student_spend_insights.pdf`. **Print > Microsoft Print to PDF** adds margins and squeezes the pages.
- The README images `reports/figures/powerbi_overview.png`, `powerbi_segments.png` and `powerbi_forecasts.png` are the PDF's three pages, rendered at 1800 pixels wide.
