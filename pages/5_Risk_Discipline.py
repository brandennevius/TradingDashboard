from __future__ import annotations

import io
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from kpi_engine import auto_match_columns, compute_kpis, prepare_trades

st.set_page_config(page_title="Risk Discipline", layout="wide")

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
    money_cols = [c for c in df.columns if "P&L" in c or "Risk" in c or c in {"Cost", "Loss"}]
    pct_cols = [c for c in df.columns if "Rate" in c or c.endswith("%")]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in money_cols + pct_cols]
    return df.style.format({**{c: "${:,.0f}" for c in money_cols}, **{c: "{:.1%}" for c in pct_cols}, **{c: "{:,.2f}" for c in num_cols}})


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
        note = notes.loc[idx]
        mistake_val = mistake.loc[idx]

        def add_violation(rule: str, severity: str, detail: str):
            rows.append({
                "Rule": rule,
                "Severity": severity,
                "Date": trade.get("close_date"),
                "Asset": trade.get("asset", ""),
                "Side": trade.get("side", ""),
                "Grade": grade,
                "Risk": risk,
                "P&L": pnl,
                "R": r,
                "Mistake": mistake_val,
                "Detail": detail,
            })

        if grade in max_risk_by_grade and pd.notna(risk) and risk > max_risk_by_grade[grade]:
            add_violation("Risk above grade limit", "High", f"{grade} trade risked {fmt_money(risk)} vs limit {fmt_money(max_risk_by_grade[grade])}.")

        if grade in {"C", "D", "F"} and pd.notna(risk) and pd.notna(pnl) and pnl < 0:
            add_violation("Weak-grade loser", "Medium", f"Weak-grade trade lost {fmt_money(abs(pnl))}.")

        text = f"{note} {mistake_val}".lower()
        patterns = [
            ("Moved stop / poor stop management", "High", r"move|moved|breakeven|break even|stop too tight|tight stop|wicked"),
            ("Added too much", "High", r"added|adding|addign|average price|average cost|too much size"),
            ("No setup / forced / revenge", "Critical", r"no setup|forced|foreced|revenge|impulse|nothing"),
            ("Margin pressure", "High", r"margin|liquidated|liquidation|closed out|couldn.t afford|drawdown"),
            ("Lower timeframe / day trade", "Medium", r"day trade|lower timeframe|ltf|intraday"),
            ("Correlation / too many positions", "High", r"correlat|too many|other positions|overnight"),
            ("Early exit / fear", "Medium", r"early|premature|fear|afraid|took.*off|sold.*early"),
        ]
        for rule, severity, pattern in patterns:
            if re.search(pattern, text):
                add_violation(rule, severity, "Detected from mistake/notes text.")

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def exposure_by_day(df: pd.DataFrame) -> pd.DataFrame:
    if "entry_date" not in df.columns or df.empty:
        return pd.DataFrame()
    working = df.dropna(subset=["entry_date", "close_date"]).copy()
    if working.empty:
        return pd.DataFrame()
    days = pd.date_range(working["entry_date"].min().normalize(), working["close_date"].max().normalize(), freq="D")
    rows = []
    for day in days:
        open_trades = working[(working["entry_date"] <= day) & (working["close_date"] >= day)]
        if open_trades.empty:
            continue
        rows.append({
            "Date": day,
            "Open Trades": len(open_trades),
            "Open Risk": open_trades["risk"].sum() if "risk" in open_trades.columns else np.nan,
            "Assets": ", ".join(sorted(open_trades.get("asset", pd.Series(dtype=str)).dropna().astype(str).unique())[:10]),
        })
    return pd.DataFrame(rows)


def loss_sequences(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    working = df.sort_values("close_date").copy()
    rows = []
    seq_id = 0
    current = []
    for _, row in working.iterrows():
        if row["pnl"] < 0:
            current.append(row)
        else:
            if current:
                seq_id += 1
                chunk = pd.DataFrame(current)
                rows.append({
                    "Sequence": seq_id,
                    "Start": chunk["close_date"].min(),
                    "End": chunk["close_date"].max(),
                    "Loss Trades": len(chunk),
                    "Total Loss": chunk["pnl"].sum(),
                    "Total R": chunk["r_multiple"].sum() if "r_multiple" in chunk.columns else np.nan,
                    "Assets": ", ".join(chunk.get("asset", pd.Series(dtype=str)).dropna().astype(str).unique()[:8]),
                })
                current = []
    if current:
        seq_id += 1
        chunk = pd.DataFrame(current)
        rows.append({
            "Sequence": seq_id,
            "Start": chunk["close_date"].min(),
            "End": chunk["close_date"].max(),
            "Loss Trades": len(chunk),
            "Total Loss": chunk["pnl"].sum(),
            "Total R": chunk["r_multiple"].sum() if "r_multiple" in chunk.columns else np.nan,
            "Assets": ", ".join(chunk.get("asset", pd.Series(dtype=str)).dropna().astype(str).unique()[:8]),
        })
    return pd.DataFrame(rows).sort_values("Total Loss") if rows else pd.DataFrame()


st.markdown("""
<style>
.risk-hero { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 20px; background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(245,238,232,.88)); padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10); }
.risk-hero h1 { margin: 0; font-size: 2rem; }
.risk-hero p { margin: .35rem 0 0 0; color: #554339; font-weight: 500; }
</style>
<div class="risk-hero"><h1>Risk Discipline</h1><p>Find rule breaks, oversized weak trades, loss clusters, and exposure pressure.</p></div>
""", unsafe_allow_html=True)

uploaded = st.sidebar.file_uploader("Upload trade log", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload your trade log to inspect risk discipline.")
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

extra_cols = candidate_columns(raw, mapping.values())
default_extra = [c for c in extra_cols if normalize(c) in {"grade", "trend", "freshness", "coverage", "cot", "valuation", "seasonality", "mistake"}]
selected_cols = st.sidebar.multiselect("Extra columns", extra_cols, default=default_extra[:8])
trades = attach_columns(trades, raw, selected_cols)

filtered = trades.copy()
min_date = pd.Timestamp(trades["close_date"].min()).date()
max_date = pd.Timestamp(trades["close_date"].max()).date()
date_range = st.sidebar.date_input("Close date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.Timestamp(d) for d in date_range]
    filtered = filtered[(filtered["close_date"] >= start) & (filtered["close_date"] <= end)]

st.sidebar.markdown("### Risk limits by grade")
a_plus = st.sidebar.number_input("A+ max risk $", min_value=0.0, value=4000.0, step=100.0)
a = st.sidebar.number_input("A max risk $", min_value=0.0, value=2000.0, step=100.0)
b = st.sidebar.number_input("B max risk $", min_value=0.0, value=1000.0, step=100.0)
c = st.sidebar.number_input("C max risk $", min_value=0.0, value=500.0, step=100.0)
d = st.sidebar.number_input("D/F max risk $", min_value=0.0, value=0.0, step=100.0)
max_risk_by_grade = {"A+": a_plus, "A": a, "B": b, "C": c, "D": d, "F": d, "": d}

k = compute_kpis(filtered)
violations = detect_rule_violations(filtered, max_risk_by_grade)
critical = len(violations[violations["Severity"] == "Critical"]) if not violations.empty else 0
high = len(violations[violations["Severity"] == "High"]) if not violations.empty else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trades", k["Trades"])
c2.metric("Net P&L", fmt_money(k["Net P&L"]))
c3.metric("Expectancy R", fmt_num(k["Expectancy (R)"]))
c4.metric("Critical breaches", critical)
c5.metric("High breaches", high)

st.markdown("## Rule violations")
if violations.empty:
    st.success("No rule violations detected from current rules and notes patterns.")
else:
    summary = violations.groupby(["Rule", "Severity"], as_index=False).agg(Trades=("Rule", "count"), PnL=("P&L", "sum"), Avg_R=("R", "mean"), Avg_Risk=("Risk", "mean")).sort_values(["Severity", "PnL"])
    st.dataframe(style_table(summary), use_container_width=True, hide_index=True)
    fig = px.bar(summary.sort_values("PnL"), x="PnL", y="Rule", color="Severity", orientation="h", title="P&L tied to rule violations")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Individual breaches")
    st.dataframe(style_table(violations.sort_values(["Severity", "P&L"])), use_container_width=True, hide_index=True)

st.markdown("## Risk by grade")
if "Grade" in filtered.columns and "risk" in filtered.columns:
    by_grade = filtered.groupby("Grade", dropna=False).agg(Trades=("pnl", "count"), Net_PnL=("pnl", "sum"), Avg_Risk=("risk", "mean"), Max_Risk=("risk", "max"), Avg_R=("r_multiple", "mean")).reset_index()
    st.dataframe(style_table(by_grade), use_container_width=True, hide_index=True)
    fig = px.box(filtered, x="Grade", y="risk", points="all", title="Risk distribution by grade")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Map/select Grade and Risk ($) to unlock risk-by-grade review.")

st.markdown("## Exposure pressure")
exposure = exposure_by_day(filtered)
if exposure.empty:
    st.info("Map Entry date to unlock exposure pressure.")
else:
    e1, e2, e3 = st.columns(3)
    e1.metric("Max open trades", int(exposure["Open Trades"].max()))
    e2.metric("Avg open trades", fmt_num(exposure["Open Trades"].mean()))
    e3.metric("Max open risk", fmt_money(exposure["Open Risk"].max()) if "Open Risk" in exposure.columns else "-")
    fig = px.area(exposure, x="Date", y="Open Trades", hover_data=["Assets", "Open Risk"], title="Open trades over time")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(style_table(exposure.sort_values("Open Trades", ascending=False).head(20)), use_container_width=True, hide_index=True)

st.markdown("## Loss sequences")
seq = loss_sequences(filtered)
if seq.empty:
    st.info("No loss sequences detected.")
else:
    st.dataframe(style_table(seq.head(20)), use_container_width=True, hide_index=True)
    fig = px.bar(seq.head(20).sort_values("Total Loss"), x="Total Loss", y="Sequence", orientation="h", title="Worst consecutive loss sequences")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("## Practical takeaway")
if not violations.empty:
    worst_rule = summary.sort_values("PnL").iloc[0]
    st.warning(f"Highest-cost rule category: {worst_rule['Rule']} ({fmt_money(worst_rule['PnL'])}). Fix this before adding size or adding more setups.")
else:
    st.info("No obvious rule breach category found. Tighten your tags or lower your risk thresholds to stress-test behavior.")
