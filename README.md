# Trading KPI Dashboard

A local Streamlit dashboard for importing trade-log spreadsheets (Excel/CSV), calculating KPI metrics, and visualizing performance across time horizons.

## Features

- Multi-file Excel/CSV import
- Header row parsing controls (auto-detect or manual row/depth) for logs that keep stats above the table
- Flexible column mapping (works with different trade-log headers)
- Date-range filter for custom KPI windows
- UI modes:
  - Compact vs expanded layout density
  - Presentation mode for cleaner screenshots
- Setup-based filters from your trade-log columns (Trend, Grade, etc.)
- KPI breakdown by:
  - Month
  - Quarter
  - 6-month (H1/H2)
  - Year
- KPI baseline tracking for:
  - MTD
  - QTD
  - 6M
  - YTD
- Deviation table against trailing historical average
- Goal tracking and comparison for Net P&L, Win Rate, and Expectancy (R)
- Specific trade logging with persistent annotations:
  - Setup tags
  - Mistake type
  - Review notes
  - Chart link URL
  - Chart image URL/path
- Visuals:
  - Equity curve
  - Daily P&L calendar heatmap
  - P&L by period bar chart
  - Trade P&L and R-multiple distributions
  - Losses by mistake-type pie chart

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Expected inputs

At minimum, map:

- Exit/Close date
- P&L

Optional but recommended:

- Entry date
- R Multiple
- Risk ($)
- Asset/Symbol

## Notes

- The app parses values like `$1,250`, `-350`, and `(200)` automatically.
- KPI calculations are based on realized trade close date.
- If your workbook has multiple sheets, choose the correct sheet in the sidebar.
