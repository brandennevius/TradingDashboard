from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from kpi_engine import auto_match_columns, compute_kpis, prepare_trades

st.set_page_config(page_title="Actionable Review", layout="wide")

STANDARD_FIELDS = [
    "exit_date",
    "entry_date",
    "pnl",
    "r_multiple",
    "risk",
    "asset",
    "side",
    "notes",
    "mistake_type",
]


def normalize(value: object) -> str:
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


@st.cache_data(show_spinner=False)
def list_sheets(uploaded_file) -> List[str]:
    if not uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        return []
    return list(pd.ExcelFile(io.BytesIO(uploaded_file.getvalue())).sheet_names)


@st.cache_data(show_spinner=False)
def preview_workbook(uploaded_file, sheet_name: Optional[str]) -> pd.DataFrame:
    payload = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(payload), header=None, nrows=40)
    return pd.read_excel(io.BytesIO(payload), sheet_name=sheet_name, header=None, nrows=40)


@st.cache_data(show_spinner=False)
def load_workbook(uploaded_file, sheet_name: Optional[str], header_row: int) -> pd.DataFrame:
    payload = uploaded_file.getvalue()
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(payload), header=header_row)
    return pd.read_excel(io.BytesIO(payload), sheet_name=sheet_name, header=header_row)


def detect_header_row(preview: pd.DataFrame) -> int:
    best_row, best_score = 0, -1
    for idx, row in preview.iterrows():
        cells = {normalize(v) for v in row.dropna().tolist() if normalize(v)}
        score = sum(any(token in c for c in cells) for token in ["asset", "side", "entry", "exit", "pnl", "pl", "risk", "grade", "trend", "mistake"])
        if {"asset", "side"}.issubset(cells) and ("pnl" in cells or "pl" in cells):
            score += 10
        if score > best_score:
            best_row, best_score = int(idx), score
    return best_row


def fix_excel_serial_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["close_date", "entry_date"]:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        years = parsed.dropna().dt.year
        if not years.empty and years.median() < 1990:
            numeric = pd.to_numeric(out[col], errors="coerce")
            serial_dates = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            out[col] = serial_dates.where(serial_dates.notna(), parsed)
        else:
            out[col] = parsed
    return out


def is_categorical(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return False
    unique_count = non_null.nunique()
    return 2 <= unique_count <= min(80, max(8, int(len(non_null) * 0.6)))


def candidate_characteristics(raw: pd.DataFrame, mapped_cols: Iterable[str]) -> List[str]:
    excluded = {c for c in mapped_cols if c}
    return sorted([col for col in raw.columns if col not in excluded and is_categorical(raw[col])])


def attach_columns(trades: pd.DataFrame, raw: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = trades.copy()
    for col in cols:
        if col in raw.columns and col not in out.columns:
            out[col] = raw.loc[out.index, col].astype(str).replace({"nan": ""}).str.strip()
    return out


def fmt_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.0f}"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:.1f}%"


def segment_table(df: pd.DataFrame, group_cols: List[str], min_trades: int) -> pd.DataFrame:
    if df.empty or not group_cols:
        return pd.DataFrame()
    rows = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(chunk) < min_trades:
            continue
        k = compute_kpis(chunk)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update({
            "Trades": k["Trades"],
            "Net P&L": k["Net P&L"],
            "Win Rate": k["Win Rate"],
            "Expectancy (R)": k["Expectancy (R)"],
            "Profit Factor": k["Profit Factor"],
            "Payoff Ratio R": k["Payoff Ratio R"],
            "Max Drawdown R": k["Max Drawdown R"],
            "Worst R": k["Worst R"],
            "Avg Risk": k["Avg Risk"],
        })
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    sort_col = "Expectancy (R)" if table["Expectancy (R)"].notna().any() else "Net P&L"
    return table.sort_values(sort_col, ascending=False).reset_index(drop=True)


def mistake_impact(df: pd.DataFrame, col: str, min_trades: int) -> pd.DataFrame:
    if col not in df.columns:
        return pd.DataFrame()
    actual = compute_kpis(df)
    rows = []
    for label, chunk in df.groupby(col, dropna=False):
        if len(chunk) < min_trades:
            continue
        without = df.drop(chunk.index)
        k = compute_kpis(chunk)
        kw = compute_kpis(without)
        rows.append({
            "Mistake": str(label).strip() or "(blank)",
            "Trades": k["Trades"],
            "Net P&L": k["Net P&L"],
            "Expectancy (R)": k["Expectancy (R)"],
            "Worst R": k["Worst R"],
            "P&L Lift If Removed": kw["Net P&L"] - actual["Net P&L"],
            "R Exp Lift If Removed": kw["Expectancy (R)"] - actual["Expectancy (R)"],
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("P&L Lift If Removed", ascending=False)


def note_keyword_tags(df: pd.DataFrame) -> pd.DataFrame:
    if "notes" not in df.columns:
        return pd.DataFrame()
    patterns = {
        "Moved stop / stop management": r"move|moved|breakeven|break even|stop",
        "Too tight stop": r"tight|wicked|wick|stopped.*open",
        "Added too much / averaged badly": r"added|adding|average price|average cost|addign",
        "No setup / forced / revenge": r"no setup|forced|foreced|revenge|impulse|nothing",
        "Countertrend / day trade": r"day trade|counter trend|countertrend|lower timeframe|ltf",
        "Margin / liquidation pressure": r"margin|liquidated|liquidation|closed out|couldn.t afford|drawdown",
        "Early exit / fear": r"early|premature|fear|afraid|took.*off|sold.*early",
        "Correlation / too many positions": r"correlated|correlation|other positions|too many|overnight",
    }
    rows = []
    notes = df["notes"].fillna("").astype(str)
    for tag, pattern in patterns.items():
        mask = notes.str.contains(pattern, case=False, regex=True, na=False)
        chunk = df[mask]
        if chunk.empty:
            continue
        k = compute_kpis(chunk)
        rows.append({"Pattern": tag, "Trades": k["Trades"], "Net P&L": k["Net P&L"], "Expectancy (R)": k["Expectancy (R)"], "Worst R": k["Worst R"]})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Net P&L")


def style_table(df: pd.DataFrame):
    money_cols = [c for c in df.columns if "P&L" in c or c in {"Avg Risk"}]
    pct_cols = [c for c in df.columns if "Rate" in c]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in money_cols + pct_cols]
    return df.style.format({**{c: "${:,.0f}" for c in money_cols}, **{c: "{:.1%}" for c in pct_cols}, **{c: "{:,.2f}" for c in num_cols}})


st.markdown("""
<style>
.review-hero { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 20px; background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(235,247,239,.88)); padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10); }
.review-hero h1 { margin: 0; font-size: 2rem; }
.review-hero p { margin: .35rem 0 0 0; color: #395443; font-weight: 500; }
</style>
<div class="review-hero"><h1>Actionable Review</h1><p>Turn the trade log into decisions: what to keep, cut, size down, and review visually.</p></div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Upload trade log", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload your trade log to generate the review.")
    st.stop()

sheets = list_sheets(uploaded)
sheet_name = st.sidebar.selectbox("Sheet", sheets, index=sheets.index("Trade Log") if "Trade Log" in sheets else 0) if sheets else None
preview = preview_workbook(uploaded, sheet_name)
header_row = st.sidebar.number_input("Header row (0-indexed)", min_value=0, max_value=50, value=detect_header_row(preview), step=1)
raw = load_workbook(uploaded, sheet_name, int(header_row))
raw.columns = [str(c).strip() for c in raw.columns]
raw = raw.loc[:, [not str(c).startswith("Unnamed") for c in raw.columns]]

matches = auto_match_columns(raw.columns)
with st.sidebar.expander("Column mapping", expanded=False):
    mapping: Dict[str, Optional[str]] = {}
    options = [None] + list(raw.columns)
    for field in STANDARD_FIELDS:
        default = matches.get(field)
        index = options.index(default) if default in options else 0
        mapping[field] = st.selectbox(field, options, index=index, format_func=lambda x: "-" if x is None else str(x))

trades = fix_excel_serial_dates(prepare_trades(raw, mapping))
if trades.empty:
    st.error("Could not prepare trades. Map Exit and P&L.")
    st.stop()

characteristics = candidate_characteristics(raw, mapping.values())
default_chars = [c for c in characteristics if normalize(c) in {"grade", "trend", "freshness", "coverage", "cot", "valuation", "seasonality", "mistake"}]
selected_chars = st.sidebar.multiselect("Characteristics to analyze", characteristics, default=default_chars[:8])
trades = attach_columns(trades, raw, selected_chars)

filtered = trades.copy()
min_date = pd.Timestamp(trades["close_date"].min()).date()
max_date = pd.Timestamp(trades["close_date"].max()).date()
date_range = st.sidebar.date_input("Close date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.Timestamp(d) for d in date_range]
    filtered = filtered[(filtered["close_date"] >= start) & (filtered["close_date"] <= end)]

min_trades = st.sidebar.slider("Minimum trades per segment", 2, 30, 5)

k = compute_kpis(filtered)
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trades", k["Trades"])
c2.metric("Net P&L", fmt_money(k["Net P&L"]))
c3.metric("Expectancy R", fmt_num(k["Expectancy (R)"]))
c4.metric("Win Rate", fmt_pct(k["Win Rate"]))
c5.metric("Max DD R", fmt_num(k["Max Drawdown R"]))

st.markdown("## Decision Summary")
summary = []
if "mistake_type" in filtered.columns:
    mi = mistake_impact(filtered, "mistake_type", min_trades=2)
    if not mi.empty:
        top_leak = mi.iloc[0]
        summary.append(f"Biggest removable leak: **{top_leak['Mistake']}**. Estimated lift if removed: **{fmt_money(top_leak['P&L Lift If Removed'])}** across {int(top_leak['Trades'])} trades.")

combo_cols = [c for c in selected_chars if c in filtered.columns][:3]
combo = segment_table(filtered, combo_cols, min_trades=min_trades) if combo_cols else pd.DataFrame()
if not combo.empty:
    best = combo.iloc[0]
    worst = combo.sort_values("Expectancy (R)" if combo["Expectancy (R)"].notna().any() else "Net P&L").iloc[0]
    best_label = " | ".join(str(best[c]) for c in combo_cols)
    worst_label = " | ".join(str(worst[c]) for c in combo_cols)
    summary.append(f"Best repeatable segment: **{best_label}** with **{int(best['Trades'])} trades**, expectancy **{fmt_num(best['Expectancy (R)'])}R**, net **{fmt_money(best['Net P&L'])}**.")
    summary.append(f"Weakest segment: **{worst_label}** with **{int(worst['Trades'])} trades**, expectancy **{fmt_num(worst['Expectancy (R)'])}R**, net **{fmt_money(worst['Net P&L'])}**.")

risk_warnings = []
if "risk" in filtered.columns and "Grade" in filtered.columns:
    weak_big = filtered[(filtered["Grade"].astype(str).str.upper().isin(["C", "D", "F"])) & (filtered["risk"] > filtered["risk"].median())]
    if not weak_big.empty:
        risk_warnings.append(f"{len(weak_big)} weak-grade trades risked more than median risk. This is a rule-quality problem, not a setup problem.")
if risk_warnings:
    summary.extend(risk_warnings)

if not summary:
    st.info("Not enough signal yet. Add Grade, Trend, COT, Valuation, Mistake, and Notes tags for stronger review output.")
else:
    for item in summary:
        st.markdown(f"- {item}")

st.markdown("## What to keep / cut")
if combo.empty:
    st.info("Select at least one characteristic to rank setup combinations.")
else:
    left, right = st.columns(2)
    with left:
        st.markdown("### Keep / study")
        st.dataframe(style_table(combo.head(10)), use_container_width=True, hide_index=True)
    with right:
        st.markdown("### Cut / reduce")
        st.dataframe(style_table(combo.tail(10).sort_values("Expectancy (R)" if combo["Expectancy (R)"].notna().any() else "Net P&L")), use_container_width=True, hide_index=True)

    chart_df = combo.head(20).copy()
    chart_df["Segment"] = chart_df[combo_cols].astype(str).agg(" | ".join, axis=1)
    fig = px.bar(chart_df.sort_values("Expectancy (R)"), x="Expectancy (R)", y="Segment", orientation="h", title="Top setup segments by expectancy")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## Mistake impact")
if "mistake_type" in filtered.columns:
    mi = mistake_impact(filtered, "mistake_type", min_trades=2)
    if mi.empty:
        st.info("No mistake groups met the minimum sample size.")
    else:
        st.dataframe(style_table(mi), use_container_width=True, hide_index=True)
        fig = px.bar(mi.sort_values("P&L Lift If Removed"), x="P&L Lift If Removed", y="Mistake", orientation="h", title="Estimated P&L lift if mistake type disappeared")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Map the Mistake column to unlock mistake impact.")

st.markdown("## Notes pattern detector")
patterns = note_keyword_tags(filtered)
if patterns.empty:
    st.info("No note patterns detected. This improves when notes consistently mention things like moved stop, margin, added, no setup, tight stop, early exit, etc.")
else:
    st.dataframe(style_table(patterns), use_container_width=True, hide_index=True)
    fig = px.bar(patterns.sort_values("Net P&L"), x="Net P&L", y="Pattern", orientation="h", title="P&L by repeated behavior pattern found in notes")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## Oversized weak trades")
if "risk" in filtered.columns:
    risk_view = filtered.copy()
    if "Grade" in risk_view.columns:
        weak = risk_view[risk_view["Grade"].astype(str).str.upper().isin(["C", "D", "F"])]
    else:
        weak = risk_view
    weak = weak.sort_values("risk", ascending=False).head(20)
    cols = [c for c in ["close_date", "asset", "side", "Grade", "pnl", "r_multiple", "risk", "mistake_type", "notes"] if c in weak.columns]
    st.dataframe(weak[cols], use_container_width=True, hide_index=True)
else:
    st.info("Map Risk ($) to unlock risk discipline review.")
