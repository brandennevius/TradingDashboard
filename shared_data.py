from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from kpi_engine import auto_match_columns, prepare_trades

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
def _list_sheets_from_bytes(payload: bytes, name: str) -> List[str]:
    if not name.lower().endswith((".xlsx", ".xls")):
        return []
    return list(pd.ExcelFile(io.BytesIO(payload)).sheet_names)


@st.cache_data(show_spinner=False)
def _preview_from_bytes(payload: bytes, name: str, sheet_name: Optional[str]) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(payload), header=None, nrows=40)
    return pd.read_excel(io.BytesIO(payload), sheet_name=sheet_name, header=None, nrows=40)


@st.cache_data(show_spinner=False)
def _load_from_bytes(payload: bytes, name: str, sheet_name: Optional[str], header_row: int) -> pd.DataFrame:
    if name.lower().endswith(".csv"):
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


def load_shared_trade_data(page_key: str = "global") -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Optional[str]], List[str]]:
    uploaded = st.sidebar.file_uploader(
        "Upload trade log",
        type=["csv", "xlsx", "xls"],
        key=f"shared_upload_{page_key}",
    )

    if uploaded is not None:
        st.session_state["shared_trade_file_name"] = uploaded.name
        st.session_state["shared_trade_file_bytes"] = uploaded.getvalue()

    if "shared_trade_file_bytes" not in st.session_state:
        st.info("Upload your trade log once. It will stay loaded while you switch pages in this Streamlit session.")
        st.stop()

    if st.sidebar.button("Clear loaded trade log", key=f"clear_shared_trade_log_{page_key}"):
        for key in ["shared_trade_file_name", "shared_trade_file_bytes", "shared_sheet_name", "shared_header_row"]:
            st.session_state.pop(key, None)
        st.rerun()

    payload = st.session_state["shared_trade_file_bytes"]
    name = st.session_state.get("shared_trade_file_name", "uploaded_trade_log")
    st.sidebar.caption(f"Using: {name}")

    sheets = _list_sheets_from_bytes(payload, name)
    if sheets:
        current_sheet = st.session_state.get("shared_sheet_name")
        default_index = sheets.index(current_sheet) if current_sheet in sheets else (sheets.index("Trade Log") if "Trade Log" in sheets else 0)
        sheet_name = st.sidebar.selectbox("Sheet", sheets, index=default_index, key=f"shared_sheet_{page_key}")
        st.session_state["shared_sheet_name"] = sheet_name
    else:
        sheet_name = None

    preview = _preview_from_bytes(payload, name, sheet_name)
    detected = detect_header_row(preview)
    default_header = int(st.session_state.get("shared_header_row", detected))
    header_row = st.sidebar.number_input("Header row (0-indexed)", min_value=0, max_value=50, value=default_header, step=1, key=f"shared_header_{page_key}")
    st.session_state["shared_header_row"] = int(header_row)

    raw = _load_from_bytes(payload, name, sheet_name, int(header_row))
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.loc[:, [not str(c).startswith("Unnamed") for c in raw.columns]]

    matches = auto_match_columns(raw.columns)
    mapping: Dict[str, Optional[str]] = {}
    with st.sidebar.expander("Column mapping", expanded=False):
        options = [None] + list(raw.columns)
        for field in STANDARD_FIELDS:
            default = matches.get(field)
            index = options.index(default) if default in options else 0
            mapping[field] = st.selectbox(field, options, index=index, format_func=lambda x: "-" if x is None else str(x), key=f"map_{page_key}_{field}")

    trades = fix_excel_serial_dates(prepare_trades(raw, mapping))
    if trades.empty:
        st.error("Could not prepare trades. Map Exit and P&L.")
        st.write("Detected columns:", list(raw.columns))
        st.stop()

    return raw, trades, mapping, sheets
