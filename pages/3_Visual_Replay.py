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
        score = sum(any(token in c for c in cells) for token in ["asset", "side", "entry", "exit", "pnl", "pl", "risk", "grade", "trend", "mistake", "link"])
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


def candidate_filter_columns(raw: pd.DataFrame, mapped_cols: Iterable[str]) -> List[str]:
    excluded = {c for c in mapped_cols if c}
    return sorted([col for col in raw.columns if col not in excluded and is_categorical(raw[col])])


def attach_columns(trades: pd.DataFrame, raw: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = trades.copy()
    for col in cols:
        if col in raw.columns and col not in out.columns:
            out[col] = raw.loc[out.index, col].astype(str).replace({"nan": ""}).str.strip()
    return out


def tradingview_snapshot_to_image(url: object) -> Optional[str]:
    if not isinstance(url, str):
        return None
    match = re.search(r"tradingview\.com/x/([A-Za-z0-9]+)/?", url.strip())
    if not match:
        return None
    code = match.group(1)
    return f"https://s3.tradingview.com/snapshots/{code[0].lower()}/{code}.png"


def find_link_column(raw: pd.DataFrame, mapped_link: Optional[str]) -> Optional[str]:
    if mapped_link in raw.columns:
        return mapped_link
    for col in raw.columns:
        if normalize(col) in {"link", "chartlink", "tradingviewlink", "url"}:
            return col
    for col in raw.columns:
        sample = raw[col].dropna().astype(str).head(25)
        if sample.str.contains(r"tradingview\.com/x/", case=False, regex=True).any():
            return col
    return None


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
    for field in ["asset", "side"]:
        if field in trade and str(trade.get(field, "")).strip() not in {"", "nan"}:
            meta.append(str(trade.get(field)).strip())
    if "pnl" in trade:
        meta.append(fmt_money(trade.get("pnl")))
    if "r_multiple" in trade and pd.notna(trade.get("r_multiple")):
        meta.append(f"{trade.get('r_multiple'):.2f}R")
    st.caption(" | ".join(meta))
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


st.markdown("""
<style>
.visual-hero { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 20px; background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(235,247,239,.88)); padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10); }
.visual-hero h1 { margin: 0; font-size: 2rem; }
.visual-hero p { margin: .35rem 0 0 0; color: #395443; font-weight: 500; }
</style>
<div class="visual-hero"><h1>Visual Replay</h1><p>Filter your trade log and review TradingView snapshots as a grid or large vertical replay.</p></div>
""", unsafe_allow_html=True)

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
link_col_guess = find_link_column(raw, matches.get("chart_link"))
if link_col_guess:
    matches["chart_link"] = link_col_guess

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

link_col = find_link_column(raw, mapping.get("chart_link"))
if link_col:
    trades["chart_link"] = raw.loc[trades.index, link_col].astype(str).replace({"nan": "", "None": ""}).str.strip()
else:
    trades["chart_link"] = ""
trades["_chart_img"] = trades["chart_link"].apply(tradingview_snapshot_to_image)

filter_cols = candidate_filter_columns(raw, mapping.values())
default_filters = [c for c in filter_cols if normalize(c) in {"grade", "trend", "freshness", "coverage", "cot", "valuation", "seasonality", "mistake"}]
selected_filter_cols = st.sidebar.multiselect("Available filter columns (optional)", filter_cols, default=[])
trades = attach_columns(trades, raw, selected_filter_cols)

filtered = trades.copy()
st.sidebar.markdown("---")
outcome = st.sidebar.radio("Outcome", ["All", "Winners", "Losers", "Breakeven"])
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
        values = sorted([v for v in filtered[col].fillna("__BLANK__").astype(str).str.strip().replace({"": "__BLANK__"}).unique()])
        label_map = {"__BLANK__": "(blank)"}
        chosen_labels = st.sidebar.multiselect(
            f"Filter: {col}",
            [label_map.get(v, v) for v in values],
            default=[label_map.get(v, v) for v in values],
        )
        chosen_raw = ["__BLANK__" if v == "(blank)" else v for v in chosen_labels]
        if set(chosen_raw) != set(values):
            comparable = filtered[col].fillna("__BLANK__").astype(str).str.strip().replace({"": "__BLANK__"})
            filtered = filtered[comparable.isin(chosen_raw)]

only_with_images = st.sidebar.checkbox("Only rows with images", value=True)
if only_with_images:
    filtered = filtered[filtered["_chart_img"].notna()]

sort_options = [c for c in ["close_date", "pnl", "r_multiple", "risk", "asset"] if c in filtered.columns]
sort_by = st.sidebar.selectbox("Sort by", sort_options, index=0) if sort_options else None
sort_ascending = st.sidebar.checkbox("Sort ascending", value=False)
if sort_by:
    filtered = filtered.sort_values(sort_by, ascending=sort_ascending)

raw_link_count = int(raw[link_col].astype(str).str.contains(r"tradingview\.com/x/", case=False, regex=True, na=False).sum()) if link_col else 0
converted_count = int(trades["_chart_img"].notna().sum())
filtered_with_images = int(filtered["_chart_img"].notna().sum()) if "_chart_img" in filtered.columns else 0

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Loaded trades", len(trades))
c2.metric("Raw TV links", raw_link_count)
c3.metric("Converted images", converted_count)
c4.metric("Showing", len(filtered))
c5.metric("Showing w/images", filtered_with_images)

with st.expander("Diagnostics", expanded=False):
    st.write("Mapped chart_link column:", link_col or "None")
    st.write("Rows in raw file:", len(raw))
    st.write("Rows prepared as trades:", len(trades))
    st.write("Rows after filters:", len(filtered))
    st.write("Available suggested filters (not applied by default):", default_filters)
    if link_col:
        failed = trades[(trades["chart_link"].astype(str).str.contains("tradingview", case=False, na=False)) & (trades["_chart_img"].isna())]
        st.write("TradingView links that failed conversion:", len(failed))
        st.dataframe(failed[[c for c in ["asset", "pnl", "chart_link"] if c in failed.columns]].head(25), use_container_width=True, hide_index=True)

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
    visible_cols = [c for c in ["close_date", "entry_date", "asset", "side", "pnl", "r_multiple", "risk", "chart_link", "_chart_img"] + selected_filter_cols if c in filtered.columns]
    st.dataframe(filtered[visible_cols], use_container_width=True, hide_index=True)
