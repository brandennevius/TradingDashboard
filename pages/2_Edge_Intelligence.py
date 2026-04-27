from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from kpi_engine import compute_kpis
from shared_data import load_shared_trade_data, normalize

st.set_page_config(page_title="Edge Intelligence", layout="wide")

DISPLAY_METRICS = ["Trades", "Net P&L", "Expectancy (R)", "Avg R", "Win Rate", "Profit Factor", "Payoff Ratio R", "Max Drawdown R", "System Quality Number"]


def fmt_money(value: float) -> str:
    if pd.isna(value): return "-"
    return f"${value:,.0f}"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value): return "-"
    return f"{value:,.{digits}f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value): return "-"
    return f"{value * 100:,.1f}%"


def display_metric_value(metric: str, value: float) -> str:
    if metric in {"Net P&L", "Best Trade", "Worst Trade", "Avg Risk", "Total Risk", "Largest Risk", "Max Drawdown"}: return fmt_money(value)
    if "Rate" in metric or metric.endswith("%"): return fmt_pct(value)
    return fmt_num(value)


def is_categorical(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty: return False
    unique_count = non_null.nunique()
    return 2 <= unique_count <= min(40, max(8, int(len(non_null) * 0.45)))


def candidate_characteristics(raw: pd.DataFrame, mapped_cols: Iterable[str]) -> List[str]:
    excluded = {c for c in mapped_cols if c}
    return sorted([col for col in raw.columns if col not in excluded and is_categorical(raw[col])])


def attach_characteristics(trades: pd.DataFrame, raw: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = trades.copy()
    for col in cols:
        if col in raw.columns and col not in out.columns:
            out[col] = raw.loc[out.index, col].astype(str).replace({"nan": ""}).str.strip()
    return out


def segment_table(df: pd.DataFrame, group_cols: List[str], min_trades: int) -> pd.DataFrame:
    if df.empty or not group_cols: return pd.DataFrame()
    rows = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple): keys = (keys,)
        if len(chunk) < min_trades: continue
        k = compute_kpis(chunk)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update({metric: k.get(metric, np.nan) for metric in DISPLAY_METRICS})
        row["Best Trade"] = k.get("Best Trade", np.nan)
        row["Worst Trade"] = k.get("Worst Trade", np.nan)
        row["Best R"] = k.get("Best R", np.nan)
        row["Worst R"] = k.get("Worst R", np.nan)
        row["Avg Risk"] = k.get("Avg Risk", np.nan)
        rows.append(row)
    if not rows: return pd.DataFrame()
    table = pd.DataFrame(rows)
    sort_col = "Expectancy (R)" if table["Expectancy (R)"].notna().any() else "Net P&L"
    return table.sort_values(sort_col, ascending=False).reset_index(drop=True)


def mistake_impact_table(df: pd.DataFrame, mistake_col: str, min_trades: int) -> pd.DataFrame:
    if df.empty or mistake_col not in df.columns: return pd.DataFrame()
    rows = []
    total_kpis = compute_kpis(df)
    actual_net = total_kpis.get("Net P&L", np.nan)
    actual_r = total_kpis.get("Expectancy (R)", np.nan)
    for mistake, chunk in df.groupby(mistake_col, dropna=False):
        label = str(mistake).strip() or "Unlabeled"
        if len(chunk) < min_trades: continue
        clean_kpis = compute_kpis(df.drop(chunk.index))
        mistake_kpis = compute_kpis(chunk)
        rows.append({"Mistake Type": label, "Trades": mistake_kpis.get("Trades", 0), "Net P&L": mistake_kpis.get("Net P&L", np.nan), "Expectancy (R)": mistake_kpis.get("Expectancy (R)", np.nan), "Win Rate": mistake_kpis.get("Win Rate", np.nan), "Worst R": mistake_kpis.get("Worst R", np.nan), "P&L Improvement If Removed": clean_kpis.get("Net P&L", np.nan) - actual_net, "R Expectancy Change": clean_kpis.get("Expectancy (R)", np.nan) - actual_r})
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("P&L Improvement If Removed", ascending=False).reset_index(drop=True)


def format_dataframe(df: pd.DataFrame):
    money_cols = [c for c in df.columns if "P&L" in c or c in {"Avg Loss", "Avg Risk", "Best Trade", "Worst Trade"}]
    pct_cols = [c for c in df.columns if "Rate" in c]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in money_cols + pct_cols]
    return df.style.format({**{c: "${:,.0f}" for c in money_cols}, **{c: "{:.1%}" for c in pct_cols}, **{c: "{:,.2f}" for c in num_cols}})


st.markdown("""
<style>.edge-hero { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 20px; background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(235,247,239,.88)); padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10); }.edge-hero h1 { margin: 0; font-size: 2rem; }.edge-hero p { margin: .35rem 0 0 0; color: #395443; font-weight: 500; }</style><div class="edge-hero"><h1>Edge Intelligence</h1><p>Find what works, what does not, which mistakes cost the most, and where exposure pressure shows up.</p></div>
""", unsafe_allow_html=True)

raw, trades, mapping, _ = load_shared_trade_data("edge_intelligence")

characteristics = candidate_characteristics(raw, mapping.values())
default_chars = [c for c in characteristics if normalize(c) in {"grade", "trend", "freshness", "coverage", "cot", "valuation", "seasonality", "mistake"}]
manual_chars = st.sidebar.multiselect("Characteristics to analyze", characteristics, default=default_chars[:8])
trades = attach_characteristics(trades, raw, manual_chars)

filtered = trades.copy()
min_date = pd.Timestamp(trades["close_date"].min()).date()
max_date = pd.Timestamp(trades["close_date"].max()).date()
date_range = st.sidebar.date_input("Close date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.Timestamp(d) for d in date_range]
    filtered = filtered[(filtered["close_date"] >= start) & (filtered["close_date"] <= end)]

for col in ["asset", "side"] + manual_chars:
    if col in filtered.columns and is_categorical(filtered[col]):
        values = sorted([v for v in filtered[col].fillna("__BLANK__").astype(str).str.strip().replace({"": "__BLANK__"}).unique()])
        labels = ["(blank)" if v == "__BLANK__" else v for v in values]
        selected_labels = st.sidebar.multiselect(f"Filter: {col}", labels, default=labels)
        selected = ["__BLANK__" if v == "(blank)" else v for v in selected_labels]
        if set(selected) != set(values):
            comparable = filtered[col].fillna("__BLANK__").astype(str).str.strip().replace({"": "__BLANK__"})
            filtered = filtered[comparable.isin(selected)]

min_trades = st.sidebar.slider("Minimum trades per segment", 1, 30, 3)
primary_metric = st.sidebar.selectbox("Rank segments by", ["Expectancy (R)", "Net P&L", "Profit Factor", "System Quality Number", "Win Rate"], index=0)

if filtered.empty:
    st.warning("No trades left after filters.")
    st.stop()

kpis = compute_kpis(filtered)
cols = st.columns(6)
for col, metric in zip(cols, ["Trades", "Net P&L", "Expectancy (R)", "Win Rate", "Payoff Ratio R", "Max Drawdown R"]):
    col.metric(metric, display_metric_value(metric, kpis.get(metric, np.nan)))

if len(filtered) < 30:
    st.warning("Sample size is below 30 trades after filters. Treat conclusions as clues, not proof.")

st.markdown("### Best and worst setup combinations")
best_setup_cols = [c for c in manual_chars if c in filtered.columns]
if best_setup_cols:
    combo_cols = st.multiselect("Build setup combinations from", best_setup_cols, default=best_setup_cols[: min(3, len(best_setup_cols))])
    combo_table = segment_table(filtered, combo_cols, min_trades=min_trades)
    if not combo_table.empty:
        combo_table = combo_table.sort_values(primary_metric, ascending=False)
        left, right = st.columns(2)
        with left:
            st.markdown("**Best combinations**")
            st.dataframe(format_dataframe(combo_table.head(15)), use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Worst combinations**")
            st.dataframe(format_dataframe(combo_table.tail(15).sort_values(primary_metric)), use_container_width=True, hide_index=True)
        chart_df = combo_table.head(20).copy()
        chart_df["Segment"] = chart_df[combo_cols].astype(str).agg(" | ".join, axis=1)
        fig = px.bar(chart_df.sort_values(primary_metric), x=primary_metric, y="Segment", orientation="h", title=f"Top segments by {primary_metric}", hover_data=["Trades", "Net P&L", "Win Rate", "Profit Factor"])
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Select characteristics to analyze setup combinations.")

st.markdown("### Mistake impact")
mistake_options = [c for c in ["mistake_type", "notes"] + manual_chars if c in filtered.columns]
if mistake_options:
    mistake_col = st.selectbox("Mistake / issue column", mistake_options, index=0)
    mistake_table = mistake_impact_table(filtered, mistake_col, min_trades=min_trades)
    if mistake_table.empty:
        st.info("No mistake categories met the minimum trade count.")
    else:
        st.dataframe(format_dataframe(mistake_table), use_container_width=True, hide_index=True)
        fig = px.bar(mistake_table.sort_values("P&L Improvement If Removed"), x="P&L Improvement If Removed", y="Mistake Type", orientation="h", title="Estimated P&L lift if removed")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Map or select a mistake column to unlock mistake impact.")

st.markdown("### Raw filtered trades")
visible_cols = [c for c in ["close_date", "entry_date", "asset", "side", "pnl", "r_multiple", "risk", "mistake_type"] + manual_chars if c in filtered.columns]
st.dataframe(filtered[visible_cols].sort_values("close_date", ascending=False), use_container_width=True, hide_index=True)
