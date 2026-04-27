from __future__ import annotations

import io
import re
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import streamlit as st

from kpi_engine import auto_match_columns, prepare_trades

st.set_page_config(page_title="Visual Replay", layout="wide")

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
    "chart_link",
    "chart_image",
]


def normalize(value: object) -> str:
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


@st.cache_data(show_spinner=False)
def list_sheets(uploaded_file) -> List[str]:
    name = uploaded_file.name.lower()
    if not name.endswith((".xlsx", ".xls")):
        return []
    xls = pd.ExcelFile(io.BytesIO(uploaded_file.getvalue()))
    return list(xls.sheet_names)


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
    required_sets = [
        {"asset", "side", "pnl"},
        {"asset", "side", "pl"},
        {"entry", "exit", "pnl"},
        {"entry", "exit", "pl"},
    ]
    best_row = 0
    best_score = -1
    for idx, row in preview.iterrows():
        cells = {normalize(v) for v in row.dropna().tolist()}
        cells = {c for c in cells if c}
        if not cells:
            continue
        score = 0
        for token in ["asset", "side", "entry", "exit", "pnl", "pl", "risk", "grade", "trend", "mistake", "link"]:
            if token in cells or any(token in c for c in cells):
                score += 1
        if any(req.issubset(cells) for req in required_sets):
            score += 10
        if score > best_score:
            best_score = score
            best_row = int(idx)
    return best_row


def fix_excel_serial_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["close_date", "entry_date"]:
        if col not in out.columns:
            continue
        parsed = pd.to_datetime(out[col], errors="coerce")
        years = parsed.dropna().dt.year
        if years.empty:
            continue
        if years.median() < 1990:
            numeric = pd.to_numeric(out[col], errors="coerce")
            serial_dates = pd.to_datetime(numeric, unit="D", origin="1899-12-30", errors="coerce")
            if serial_dates.notna().sum() >= parsed.notna().sum() * 0.8:
                out[col] = serial_dates
            else:
                out[col] = parsed
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


def candidate_filter_columns(raw: pd.DataFrame, mapped_cols: Iterable[str]) -> List[str]:
    excluded = {c for c in mapped_cols if c}
    candidates: List[str] = []
    for col in raw.columns:
        if col in excluded:
            continue
        if is_categorical(raw[col]):
            candidates.append(col)
    return sorted(candidates)


def attach_columns(trades: pd.DataFrame, raw: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = trades.copy()
    for col in cols:
        if col in raw.columns and col not in out.columns:
            out[col] = raw.loc[out.index, col].astype(str).replace({"nan": ""}).str.strip()
    return out


def tradingview_snapshot_to_image(url: object) -> Optional[str]:
    if not isinstance(url, str):
        return None
    match = re.search(r"tradingview\.com/x/([A-Za-z0-9]+)/?", url)
    if not match:
        return None
    code = match.group(1)
    folder = code[0].lower()
    return f"https://s3.tradingview.com/snapshots/{folder}/{code}.png"


def build_chart_image_column(df: pd.DataFrame) -> pd.Series:
    if "chart_image" in df.columns:
        images = df["chart_image"].astype(str).replace({"nan": "", "None": ""}).str.strip()
        converted = images.where(images != "", None)
    else:
        converted = pd.Series([None] * len(df), index=df.index)

    if "chart_link" in df.columns:
        from_links = df["chart_link"].apply(tradingview_snapshot_to_image)
        converted = converted.where(converted.notna(), from_links)
    return converted


def fmt_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.0f}"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def render_chart_card(trade: pd.Series) -> None:
    img = trade.get("_chart_img")
    if isinstance(img, str) and img:
        st.image(img, use_container_width=True)
    else:
        st.caption("No image")

    meta = []
    if "asset" in trade:
        meta.append(str(trade.get("asset", "")))
    if "side" in trade:
        meta.append(str(trade.get("side", "")))
    if "pnl" in trade:
        meta.append(fmt_money(trade.get("pnl")))
    if "r_multiple" in trade and pd.notna(trade.get("r_multiple")):
        meta.append(f"{trade.get('r_multiple'):.2f}R")
    st.caption(" | ".join([m for m in meta if m and m != "nan"]))

    link = trade.get("chart_link")
    if isinstance(link, str) and link.startswith("http"):
        st.markdown(f"[Open TradingView]({link})")


def render_grid(df: pd.DataFrame, cols: int) -> None:
    for start in range(0, len(df), cols):
        row = df.iloc[start : start + cols]
        columns = st.columns(cols)
        for col, (_, trade) in zip(columns, row.iterrows()):
            with col:
                render_chart_card(trade)


st.markdown(
    """
<style>
.visual-hero { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 20px; background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(235,247,239,.88)); padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10); }
.visual-hero h1 { margin: 0; font-size: 2rem; }
.visual-hero p { margin: .35rem 0 0 0; color: #395443; font-weight: 500; }
</style>
<div class="visual-hero"><h1>Visual Replay</h1><p>Filter your trade log and review TradingView snapshots as a grid or large vertical replay.</p></div>
""",
    unsafe_allow_html=True,
)

uploaded = st.sidebar.file_uploader("Upload trade log", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload your trade log. TradingView snapshot links like https://www.tradingview.com/x/NyqItw8b/ will be converted to images automatically.")
    st.stop()

sheets = list_sheets(uploaded)
sheet_name = st.sidebar.selectbox("Sheet", sheets, index=sheets.index("Trade Log") if "Trade Log" in sheets else 0) if sheets else None
preview = preview_workbook(uploaded, sheet_name)
detected_header = detect_header_row(preview)
header_row = st.sidebar.number_input("Header row (0-indexed)", min_value=0, max_value=50, value=detected_header, step=1)

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
    st.error("Could not prepare trades. Use sheet 'Trade Log', header row 13, then map Exit and P&L.")
    st.write("Detected columns:", list(raw.columns))
    st.stop()

filter_cols = candidate_filter_columns(raw, mapping.values())
default_filters = [c for c in filter_cols if normalize(c) in {"grade", "trend", "freshness", "coverage", "cot", "valuation", "seasonality", "mistake"}]
selected_filter_cols = st.sidebar.multiselect("Filter columns", filter_cols, default=default_filters[:8])
trades = attach_columns(trades, raw, selected_filter_cols)
trades["_chart_img"] = build_chart_image_column(trades)

filtered = trades.copy()

st.sidebar.markdown("---")
outcome = st.sidebar.radio("Outcome", ["All", "Winners", "Losers", "Breakeven"], horizontal=False)
if outcome == "Winners":
    filtered = filtered[filtered["pnl"] > 0]
elif outcome == "Losers":
    filtered = filtered[filtered["pnl"] < 0]
elif outcome == "Breakeven":
    filtered = filtered[filtered["pnl"] == 0]

min_date = pd.Timestamp(trades["close_date"].min()).date()
max_date = pd.Timestamp(trades["close_date"].max()).date()
date_range = st.sidebar.date_input("Close date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.Timestamp(d) for d in date_range]
    filtered = filtered[(filtered["close_date"] >= start) & (filtered["close_date"] <= end)]

for col in ["asset", "side"] + selected_filter_cols:
    if col in filtered.columns and is_categorical(filtered[col]):
        values = sorted([v for v in filtered[col].dropna().astype(str).str.strip().unique() if v])
        chosen = st.sidebar.multiselect(f"Filter: {col}", values, default=values)
        if chosen:
            filtered = filtered[filtered[col].astype(str).str.strip().isin(chosen)]

only_with_images = st.sidebar.checkbox("Only rows with images", value=True)
if only_with_images:
    filtered = filtered[filtered["_chart_img"].notna()]

sort_by = st.sidebar.selectbox("Sort by", [c for c in ["close_date", "pnl", "r_multiple", "risk", "asset"] if c in filtered.columns], index=0 if "close_date" in filtered.columns else 0)
sort_ascending = st.sidebar.checkbox("Sort ascending", value=False)
if sort_by in filtered.columns:
    filtered = filtered.sort_values(sort_by, ascending=sort_ascending)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Charts", len(filtered))
c2.metric("Total P&L", fmt_money(filtered["pnl"].sum()) if not filtered.empty else "$0")
c3.metric("Avg R", fmt_num(filtered["r_multiple"].mean()) if "r_multiple" in filtered.columns and not filtered.empty else "-")
c4.metric("Win Rate", f"{(filtered['pnl'] > 0).mean() * 100:.1f}%" if not filtered.empty else "-")

if filtered.empty:
    st.warning("No charts match the current filters.")
    st.stop()

view = st.radio("View", ["Grid", "Stack"], horizontal=True)
if view == "Grid":
    cols = st.slider("Columns", 2, 6, 4)
    render_grid(filtered, cols)
else:
    for _, trade in filtered.iterrows():
        render_chart_card(trade)
        st.divider()

with st.expander("Filtered trade rows"):
    visible_cols = [c for c in ["close_date", "entry_date", "asset", "side", "pnl", "r_multiple", "risk", "chart_link"] + selected_filter_cols if c in filtered.columns]
    st.dataframe(filtered[visible_cols], use_container_width=True, hide_index=True)
