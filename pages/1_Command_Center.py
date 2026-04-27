from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from kpi_engine import compute_equity_curve, compute_kpis, period_kpi_table
from shared_data import load_shared_trade_data, normalize

st.set_page_config(page_title="Command Center", layout="wide")


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


def is_categorical(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return False
    unique_count = non_null.nunique()
    return 2 <= unique_count <= min(60, max(8, int(len(non_null) * 0.5)))


def candidate_columns(raw: pd.DataFrame, mapped_cols: Iterable[str]) -> List[str]:
    excluded = {c for c in mapped_cols if c}
    return sorted([col for col in raw.columns if col not in excluded and is_categorical(raw[col])])


def attach_columns(trades: pd.DataFrame, raw: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = trades.copy()
    for col in cols:
        if col in raw.columns and col not in out.columns:
            out[col] = raw.loc[out.index, col].astype(str).replace({"nan": ""}).str.strip()
    return out


def style_table(df: pd.DataFrame):
    money_cols = [c for c in df.columns if "P&L" in c or "Drawdown" in c or "Risk" in c]
    pct_cols = [c for c in df.columns if "Rate" in c]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in money_cols + pct_cols]
    return df.style.format({**{c: "${:,.0f}" for c in money_cols}, **{c: "{:.1%}" for c in pct_cols}, **{c: "{:,.2f}" for c in num_cols}})


st.markdown("""
<style>
.command-hero { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 20px; background: linear-gradient(135deg, rgba(255,255,255,.94), rgba(235,247,239,.88)); padding: 1rem 1.2rem; margin-bottom: .9rem; box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10); }
.command-hero h1 { margin: 0; font-size: 2rem; }
.command-hero p { margin: .35rem 0 0 0; color: #395443; font-weight: 500; }
.nav-card { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 16px; padding: .85rem 1rem; background: rgba(255,255,255,.76); min-height: 128px; }
.nav-card h3 { margin: 0 0 .35rem 0; }
.nav-card p { color: #4a6253; margin: 0; }
</style>
<div class="command-hero"><h1>Command Center</h1><p>One-page control panel: performance, drift, current period health, and where to review next.</p></div>
""", unsafe_allow_html=True)

raw, trades, mapping, _ = load_shared_trade_data("command_center")

extra_cols = candidate_columns(raw, mapping.values())
default_cols = [c for c in extra_cols if normalize(c) in {"grade", "trend", "freshness", "coverage", "cot", "valuation", "seasonality", "mistake"}]
selected_cols = st.sidebar.multiselect("Optional grouping columns", extra_cols, default=default_cols[:6])
trades = attach_columns(trades, raw, selected_cols)

filtered = trades.copy()
min_date = pd.Timestamp(trades["close_date"].min()).date()
max_date = pd.Timestamp(trades["close_date"].max()).date()
date_range = st.sidebar.date_input("Close date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.Timestamp(d) for d in date_range]
    filtered = filtered[(filtered["close_date"] >= start) & (filtered["close_date"] <= end)]

k = compute_kpis(filtered)
cols = st.columns(6)
cols[0].metric("Trades", int(k["Trades"]))
cols[1].metric("Net P&L", fmt_money(k["Net P&L"]))
cols[2].metric("Expectancy R", fmt_num(k["Expectancy (R)"]))
cols[3].metric("Win Rate", fmt_pct(k["Win Rate"]))
cols[4].metric("Profit Factor", fmt_num(k["Profit Factor"]))
cols[5].metric("Max DD R", fmt_num(k["Max Drawdown R"]))

st.markdown("## Current read")
readouts = []
if pd.notna(k.get("Expectancy (R)")):
    if k["Expectancy (R)"] > 0:
        readouts.append(f"Positive R expectancy: **{fmt_num(k['Expectancy (R)'])}R** per trade in the filtered sample.")
    else:
        readouts.append(f"Negative R expectancy: **{fmt_num(k['Expectancy (R)'])}R** per trade. Do not add size until the leak is identified.")
if k["Trades"] < 30:
    readouts.append("Sample size is below 30 trades; treat all segment conclusions as clues, not proof.")
if pd.notna(k.get("Max Drawdown R")) and k["Max Drawdown R"] < -5:
    readouts.append(f"R drawdown is material at **{fmt_num(k['Max Drawdown R'])}R**. Check Risk Discipline before increasing size.")
if not readouts:
    readouts.append("Not enough mapped R/risk data for a strong read. Map R Multiple and Risk ($) for better diagnostics.")
for item in readouts:
    st.markdown(f"- {item}")

left, right = st.columns([1.1, .9])
with left:
    st.markdown("## Equity curve")
    curve = compute_equity_curve(filtered)
    if curve.empty:
        st.info("No equity curve available.")
    else:
        fig = px.line(curve, x="close_date", y="equity", title="Cumulative P&L")
        st.plotly_chart(fig, use_container_width=True)
with right:
    st.markdown("## Navigation")
    st.markdown("""
<div class="nav-card"><h3>Edge Intelligence</h3><p>Find setup traits and combinations that actually have positive expectancy.</p></div><br>
<div class="nav-card"><h3>Visual Replay</h3><p>Scan filtered TradingView snapshots to see what winners and losers look like.</p></div><br>
<div class="nav-card"><h3>Actionable Review</h3><p>Get the keep/cut list and highest-impact behavior changes.</p></div><br>
<div class="nav-card"><h3>Risk Discipline</h3><p>Find oversized trades, rule breaches, exposure pressure, and loss sequences.</p></div>
""", unsafe_allow_html=True)

st.markdown("## Period health")
period = st.radio("Period", ["Monthly", "Quarterly"], horizontal=True)
freq = "M" if period == "Monthly" else "Q"
ptable = period_kpi_table(filtered, freq)
if ptable.empty:
    st.info("No period table available.")
else:
    keep_cols = [c for c in ["Period", "Trades", "Net P&L", "Win Rate", "Expectancy (R)", "Profit Factor", "Max Drawdown R"] if c in ptable.columns]
    st.dataframe(style_table(ptable[keep_cols].tail(12)), use_container_width=True, hide_index=True)
    fig = px.bar(ptable.tail(12), x="Period", y="Net P&L", title=f"{period} net P&L")
    st.plotly_chart(fig, use_container_width=True)

if selected_cols:
    st.markdown("## Fast segment pulse")
    group_col = st.selectbox("Group by", selected_cols)
    rows = []
    for label, chunk in filtered.groupby(group_col, dropna=False):
        if len(chunk) < 2:
            continue
        kk = compute_kpis(chunk)
        rows.append({"Segment": str(label) or "(blank)", "Trades": kk["Trades"], "Net P&L": kk["Net P&L"], "Expectancy (R)": kk["Expectancy (R)"], "Win Rate": kk["Win Rate"], "Profit Factor": kk["Profit Factor"]})
    seg = pd.DataFrame(rows)
    if seg.empty:
        st.info("No segment met the minimum display size.")
    else:
        seg = seg.sort_values("Expectancy (R)", ascending=False)
        st.dataframe(style_table(seg), use_container_width=True, hide_index=True)
