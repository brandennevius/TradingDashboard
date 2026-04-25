from __future__ import annotations

import io
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from kpi_engine import auto_match_columns, compute_kpis, prepare_trades

st.set_page_config(page_title="Edge Intelligence", layout="wide")

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

DISPLAY_METRICS = [
    "Trades",
    "Net P&L",
    "Expectancy (R)",
    "Avg R",
    "Win Rate",
    "Profit Factor",
    "Payoff Ratio R",
    "Max Drawdown R",
    "System Quality Number",
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
        for token in ["asset", "side", "entry", "exit", "pnl", "pl", "risk", "grade", "trend", "mistake"]:
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
    return f"{value * 100:,.1f}%"


def display_metric_value(metric: str, value: float) -> str:
    if metric in {"Net P&L", "Best Trade", "Worst Trade", "Avg Risk", "Total Risk", "Largest Risk", "Max Drawdown"}:
        return fmt_money(value)
    if "Rate" in metric or metric.endswith("%"):
        return fmt_pct(value)
    return fmt_num(value)


def style_figure(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.72)",
        margin=dict(l=18, r=18, t=58, b=28),
        font=dict(family="IBM Plex Sans, sans-serif", color="#233426", size=13),
        title=dict(font=dict(family="Space Grotesk, sans-serif", size=20, color="#1f2f24")),
        legend=dict(bgcolor="rgba(255,255,255,0.72)", bordercolor="rgba(44,88,62,0.12)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor="rgba(44, 88, 62, 0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(44, 88, 62, 0.12)", zeroline=False)


def is_categorical(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str).str.strip()
    non_null = non_null[non_null != ""]
    if non_null.empty:
        return False
    unique_count = non_null.nunique()
    return 2 <= unique_count <= min(40, max(8, int(len(non_null) * 0.45)))


def candidate_characteristics(raw: pd.DataFrame, mapped_cols: Iterable[str]) -> List[str]:
    excluded = {c for c in mapped_cols if c}
    candidates: List[str] = []
    for col in raw.columns:
        if col in excluded:
            continue
        if is_categorical(raw[col]):
            candidates.append(col)
    return sorted(candidates)


def attach_characteristics(trades: pd.DataFrame, raw: pd.DataFrame, characteristic_cols: List[str]) -> pd.DataFrame:
    out = trades.copy()
    for col in characteristic_cols:
        if col in raw.columns and col not in out.columns:
            out[col] = raw.loc[out.index, col].astype(str).replace({"nan": ""}).str.strip()
    return out


def segment_table(df: pd.DataFrame, group_cols: List[str], min_trades: int) -> pd.DataFrame:
    if df.empty or not group_cols:
        return pd.DataFrame()
    rows = []
    for keys, chunk in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(chunk) < min_trades:
            continue
        kpis = compute_kpis(chunk)
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update({metric: kpis.get(metric, np.nan) for metric in DISPLAY_METRICS})
        row["Best Trade"] = kpis.get("Best Trade", np.nan)
        row["Worst Trade"] = kpis.get("Worst Trade", np.nan)
        row["Best R"] = kpis.get("Best R", np.nan)
        row["Worst R"] = kpis.get("Worst R", np.nan)
        row["Avg Risk"] = kpis.get("Avg Risk", np.nan)
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    table = pd.DataFrame(rows)
    sort_col = "Expectancy (R)" if table["Expectancy (R)"].notna().any() else "Net P&L"
    return table.sort_values(sort_col, ascending=False).reset_index(drop=True)


def mistake_impact_table(df: pd.DataFrame, mistake_col: str, min_trades: int) -> pd.DataFrame:
    if df.empty or mistake_col not in df.columns:
        return pd.DataFrame()
    rows = []
    total_kpis = compute_kpis(df)
    actual_net = total_kpis.get("Net P&L", np.nan)
    actual_r = total_kpis.get("Expectancy (R)", np.nan)
    for mistake, chunk in df.groupby(mistake_col, dropna=False):
        label = str(mistake).strip() or "Unlabeled"
        if len(chunk) < min_trades:
            continue
        without = df.drop(chunk.index)
        mistake_kpis = compute_kpis(chunk)
        clean_kpis = compute_kpis(without)
        rows.append(
            {
                "Mistake Type": label,
                "Trades": mistake_kpis.get("Trades", 0),
                "Net P&L": mistake_kpis.get("Net P&L", np.nan),
                "Expectancy (R)": mistake_kpis.get("Expectancy (R)", np.nan),
                "Win Rate": mistake_kpis.get("Win Rate", np.nan),
                "Avg Loss": mistake_kpis.get("Avg Loss", np.nan),
                "Worst R": mistake_kpis.get("Worst R", np.nan),
                "P&L Without This": clean_kpis.get("Net P&L", np.nan),
                "P&L Improvement If Removed": clean_kpis.get("Net P&L", np.nan) - actual_net,
                "R Expectancy Without This": clean_kpis.get("Expectancy (R)", np.nan),
                "R Expectancy Change": clean_kpis.get("Expectancy (R)", np.nan) - actual_r,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("P&L Improvement If Removed", ascending=False).reset_index(drop=True)


def characteristic_extremes(df: pd.DataFrame, characteristics: List[str], min_trades: int) -> pd.DataFrame:
    rows = []
    for col in characteristics:
        table = segment_table(df, [col], min_trades=min_trades)
        if table.empty:
            continue
        best = table.iloc[0]
        sort_col = "Expectancy (R)" if table["Expectancy (R)"].notna().any() else "Net P&L"
        worst = table.sort_values(sort_col).iloc[0]
        rows.append(
            {
                "Characteristic": col,
                "Best Value": best[col],
                "Best Trades": best["Trades"],
                "Best Exp R": best.get("Expectancy (R)", np.nan),
                "Best Net P&L": best.get("Net P&L", np.nan),
                "Worst Value": worst[col],
                "Worst Trades": worst["Trades"],
                "Worst Exp R": worst.get("Expectancy (R)", np.nan),
                "Worst Net P&L": worst.get("Net P&L", np.nan),
            }
        )
    return pd.DataFrame(rows)


def overlap_exposure(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "entry_date" not in df.columns or "close_date" not in df.columns:
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
        rows.append(
            {
                "Date": day,
                "Open Trades": len(open_trades),
                "Open Risk": open_trades["risk"].sum() if "risk" in open_trades.columns else np.nan,
                "Assets": ", ".join(sorted(open_trades.get("asset", pd.Series(dtype=str)).dropna().astype(str).unique())[:8]),
            }
        )
    return pd.DataFrame(rows)


def asset_pair_overlap(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "asset" not in df.columns or "entry_date" not in df.columns:
        return pd.DataFrame()
    working = df.dropna(subset=["entry_date", "close_date"]).copy()
    working["asset"] = working["asset"].astype(str).str.strip()
    working = working[working["asset"] != ""]
    assets = sorted(working["asset"].unique())
    if len(assets) < 2:
        return pd.DataFrame()
    rows = []
    for i, asset_a in enumerate(assets):
        a = working[working["asset"] == asset_a]
        for asset_b in assets[i + 1 :]:
            b = working[working["asset"] == asset_b]
            overlap_count = 0
            overlap_risk = 0.0
            for _, trade_a in a.iterrows():
                overlaps = b[(b["entry_date"] <= trade_a["close_date"]) & (b["close_date"] >= trade_a["entry_date"])]
                overlap_count += len(overlaps)
                if "risk" in working.columns and not overlaps.empty:
                    overlap_risk += float(overlaps["risk"].fillna(0).sum()) + float(trade_a.get("risk", 0) or 0)
            if overlap_count:
                rows.append({"Pair": f"{asset_a} / {asset_b}", "Overlaps": overlap_count, "Approx Overlap Risk": overlap_risk})
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["Overlaps", "Approx Overlap Risk"], ascending=False).reset_index(drop=True)


def format_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    money_cols = [c for c in df.columns if "P&L" in c or c in {"Avg Loss", "Avg Risk", "Best Trade", "Worst Trade", "Open Risk", "Approx Overlap Risk"}]
    pct_cols = [c for c in df.columns if "Rate" in c or c.endswith("%")]
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in money_cols + pct_cols]
    return df.style.format({**{c: "${:,.0f}" for c in money_cols}, **{c: "{:.1%}" for c in pct_cols}, **{c: "{:,.2f}" for c in num_cols}})


st.markdown(
    """
<style>
.edge-hero { border: 1px solid rgba(44, 88, 62, 0.15); border-radius: 20px; background: linear-gradient(135deg, rgba(255,255,255,.92), rgba(235,247,239,.88)); padding: 1rem 1.2rem; margin-bottom: .8rem; box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10); }
.edge-hero h1 { margin: 0; font-size: 2rem; }
.edge-hero p { margin: .35rem 0 0 0; color: #395443; font-weight: 500; }
</style>
<div class="edge-hero"><h1>Edge Intelligence</h1><p>Find what works, what does not, which mistakes cost the most, and where exposure pressure shows up.</p></div>
""",
    unsafe_allow_html=True,
)

uploaded = st.sidebar.file_uploader("Upload trade log", type=["csv", "xlsx", "xls"])
if not uploaded:
    st.info("Upload your trade log to start. Close date and P&L are required. R, risk, asset, side, grade, trend, setup, and mistake columns make the analysis better.")
    st.stop()

sheets = list_sheets(uploaded)
sheet_name = st.sidebar.selectbox("Sheet", sheets, index=sheets.index("Trade Log") if "Trade Log" in sheets else 0) if sheets else None
preview = preview_workbook(uploaded, sheet_name)
detected_header = detect_header_row(preview)
header_row = st.sidebar.number_input("Header row (0-indexed)", min_value=0, max_value=50, value=detected_header, step=1)
st.sidebar.caption(f"Auto-detected header row: {detected_header}. For this Trade Log workbook, it should be 13.")

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
    st.error("Could not prepare trades. Use sheet 'Trade Log', header row 13, then map Exit to exit_date and P&L to pnl.")
    st.write("Detected columns:", list(raw.columns))
    st.stop()

characteristics = candidate_characteristics(raw, mapping.values())
default_chars = [c for c in characteristics if normalize(c) in {"grade", "trend", "freshness", "coverage", "cot", "valuation", "seasonality", "mistake"}]
manual_chars = st.sidebar.multiselect("Characteristics to analyze", characteristics, default=default_chars[:8])
trades = attach_characteristics(trades, raw, manual_chars)

st.sidebar.markdown("---")
min_date = pd.Timestamp(trades["close_date"].min()).date()
max_date = pd.Timestamp(trades["close_date"].max()).date()
date_range = st.sidebar.date_input("Close date range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
filtered = trades.copy()
if isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = [pd.Timestamp(d) for d in date_range]
    filtered = filtered[(filtered["close_date"] >= start) & (filtered["close_date"] <= end)]

for col in ["asset", "side"] + manual_chars:
    if col in filtered.columns and is_categorical(filtered[col]):
        values = sorted([v for v in filtered[col].dropna().astype(str).str.strip().unique() if v])
        selected = st.sidebar.multiselect(f"Filter: {col}", values, default=values)
        if selected:
            filtered = filtered[filtered[col].astype(str).str.strip().isin(selected)]

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

st.markdown("### What is working vs not working")
extremes = characteristic_extremes(filtered, manual_chars, min_trades)
if extremes.empty:
    st.info("No characteristic segments met the minimum trade count. Lower the minimum or select more characteristics.")
else:
    st.dataframe(format_dataframe(extremes), use_container_width=True, hide_index=True)

best_setup_cols = [c for c in manual_chars if c in filtered.columns]
if best_setup_cols:
    st.markdown("### Best and worst setup combinations")
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
        style_figure(fig)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Mistake impact")
mistake_options = [c for c in ["mistake_type", "notes"] + manual_chars if c in filtered.columns]
if mistake_options:
    mistake_col = st.selectbox("Mistake / issue column", mistake_options, index=0)
    mistake_table = mistake_impact_table(filtered, mistake_col, min_trades=min_trades)
    if mistake_table.empty:
        st.info("No mistake categories met the minimum trade count.")
    else:
        a, b = st.columns([1.15, 0.85])
        with a:
            st.dataframe(format_dataframe(mistake_table), use_container_width=True, hide_index=True)
        with b:
            fig = px.bar(mistake_table.sort_values("P&L Improvement If Removed"), x="P&L Improvement If Removed", y="Mistake Type", orientation="h", title="Estimated P&L lift if removed", hover_data=["Trades", "Expectancy (R)", "Worst R"])
            style_figure(fig)
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Map or select a mistake column to unlock mistake impact.")

st.markdown("### Trade characteristics: winners vs losers")
if manual_chars:
    long = []
    for char in manual_chars:
        table = segment_table(filtered, [char], min_trades=min_trades)
        if table.empty:
            continue
        top = table.head(5).assign(Characteristic=char, Rank="Best")
        bottom = table.tail(5).assign(Characteristic=char, Rank="Worst")
        long.append(pd.concat([top, bottom], ignore_index=True))
    if long:
        ranked = pd.concat(long, ignore_index=True)
        ranked["Value"] = ranked.apply(lambda r: r.get(r["Characteristic"], ""), axis=1)
        fig = px.scatter(ranked, x="Win Rate", y="Expectancy (R)", size="Trades", color="Rank", facet_col="Characteristic", facet_col_wrap=2, hover_name="Value", hover_data=["Net P&L", "Profit Factor", "Payoff Ratio R", "Max Drawdown R"], title="Which characteristics produce quality trades?")
        style_figure(fig)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("### Exposure and correlation pressure")
if "entry_date" not in filtered.columns:
    st.info("Map Entry to unlock overlap exposure analysis.")
else:
    exposure = overlap_exposure(filtered)
    pairs = asset_pair_overlap(filtered)
    if exposure.empty:
        st.info("No overlapping trades found in the filtered sample.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Simultaneous Trades", int(exposure["Open Trades"].max()))
        c2.metric("Avg Simultaneous Trades", fmt_num(exposure["Open Trades"].mean()))
        c3.metric("Max Open Risk", fmt_money(exposure["Open Risk"].max()) if "Open Risk" in exposure.columns and exposure["Open Risk"].notna().any() else "Map risk")
        fig = px.area(exposure, x="Date", y="Open Trades", title="Open trade count over time", hover_data=["Assets"])
        style_figure(fig)
        st.plotly_chart(fig, use_container_width=True)
        if not pairs.empty:
            left, right = st.columns([0.95, 1.05])
            with left:
                st.markdown("**Most frequent overlapping asset pairs**")
                st.dataframe(format_dataframe(pairs.head(20)), use_container_width=True, hide_index=True)
            with right:
                fig = px.bar(pairs.head(15).sort_values("Overlaps"), x="Overlaps", y="Pair", orientation="h", title="Overlap count by asset pair")
                style_figure(fig)
                st.plotly_chart(fig, use_container_width=True)

st.markdown("### Raw filtered trades")
visible_cols = [c for c in ["close_date", "entry_date", "asset", "side", "pnl", "r_multiple", "risk", "mistake_type"] + manual_chars if c in filtered.columns]
st.dataframe(filtered[visible_cols].sort_values("close_date", ascending=False), use_container_width=True, hide_index=True)
