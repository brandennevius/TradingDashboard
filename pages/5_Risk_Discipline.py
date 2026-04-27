from __future__ import annotations

import re
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from kpi_engine import compute_kpis
from shared_data import load_shared_trade_data, normalize

st.set_page_config(page_title="Risk Discipline", layout="wide")


def is_categorical(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return False
    unique_count = non_null.nunique()
    return 2 <= unique_count <= min(80, max(8, int(len(non_null) * 0.6)))


def candidate_columns(raw: pd.DataFrame, mapped_cols: Iterable[str]) -> List[str]:
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


def style_table(df: pd.DataFrame):
    money_cols = [c for c in df.columns if "P&L" in c or "Risk" in c]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in money_cols]
    return df.style.format({**{c: "${:,.0f}" for c in money_cols}, **{c: "{:,.2f}" for c in num_cols}})


def detect_rule_violations(df: pd.DataFrame, max_risk_by_grade: Dict[str, float]) -> pd.DataFrame:
    rows = []
    grade_col = "Grade" if "Grade" in df.columns else None
    notes = df.get("notes", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)
    mistake = df.get("mistake_type", pd.Series([""] * len(df), index=df.index)).fillna("").astype(str)

    for idx, trade in df.iterrows():
        grade = str(trade.get(grade_col, "")).strip().upper() if grade_col else ""
        risk = trade.get("risk", np.nan)
        pnl = trade.get("pnl", np.nan)
        r = trade.get("r_multiple", np.nan)
        text = f"{notes.loc[idx]} {mistake.loc[idx]}".lower()

        def add(rule, severity):
            rows.append({"Rule": rule, "Severity": severity, "P&L": pnl, "Risk": risk, "R": r})

        if grade in max_risk_by_grade and pd.notna(risk) and risk > max_risk_by_grade[grade]:
            add("Risk above grade limit", "High")

        if re.search(r"no setup|forced|revenge", text):
            add("No setup / revenge", "Critical")

        if re.search(r"margin|liquid", text):
            add("Margin pressure", "High")

        if re.search(r"added|adding", text):
            add("Added too much", "High")

        if re.search(r"move|stop", text):
            add("Stop management", "Medium")

    return pd.DataFrame(rows)


st.markdown("## Risk Discipline")

raw, trades, mapping, _ = load_shared_trade_data("risk")

extra_cols = candidate_columns(raw, mapping.values())
selected_cols = st.sidebar.multiselect("Extra columns", extra_cols)
trades = attach_columns(trades, raw, selected_cols)

filtered = trades.copy()

st.sidebar.markdown("### Risk limits")
limits = {
    "A+": st.sidebar.number_input("A+", value=4000.0),
    "A": st.sidebar.number_input("A", value=2000.0),
    "B": st.sidebar.number_input("B", value=1000.0),
    "C": st.sidebar.number_input("C", value=500.0),
    "D": st.sidebar.number_input("D/F", value=0.0),
    "F": 0.0,
}

violations = detect_rule_violations(filtered, limits)

st.metric("Trades", len(filtered))
st.metric("Violations", len(violations))

if not violations.empty:
    summary = violations.groupby("Rule").agg(Count=("Rule", "count"), Loss=("P&L", "sum")).reset_index()
    st.dataframe(summary)
    fig = px.bar(summary, x="Loss", y="Rule", orientation="h")
    st.plotly_chart(fig)
else:
    st.success("No violations detected")
