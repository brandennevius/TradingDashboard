from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="CANSLIM Model Book", layout="wide")

APP_DIR = Path(__file__).resolve().parents[1]
REFERENCE_DIR = APP_DIR / "data" / "canslim_reference"
INDEX_FILE = REFERENCE_DIR / "chart_setup_index.csv"


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


@st.cache_data(show_spinner=False)
def load_model_book() -> pd.DataFrame:
    if not INDEX_FILE.exists():
        return pd.DataFrame()
    df = pd.read_csv(INDEX_FILE, dtype=str).fillna("")
    for col in df.columns:
        df[col] = df[col].map(clean_text)
    return df


def image_path(row: pd.Series) -> Path:
    return APP_DIR / row["source_image_path"]


def option_label(row: pd.Series) -> str:
    page = f"p. {row['source_page']} - " if row.get("source_page") else ""
    ticker = row.get("ticker") or row.get("image_id")
    setup = row.get("setup_type") or "unlabeled"
    return f"{row['image_id']} | {page}{ticker} | {setup}"


def render_note(label: str, value: str) -> None:
    if value:
        st.markdown(f"**{label}:** {value}")


st.markdown(
    """
<style>
.canslim-hero {
    border: 1px solid rgba(44, 88, 62, 0.15);
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(255,255,255,.94), rgba(235,247,239,.88));
    padding: 1rem 1.2rem;
    margin-bottom: .9rem;
    box-shadow: 0 16px 40px rgba(42, 83, 58, 0.10);
}
.canslim-hero h1 { margin: 0; font-size: 2rem; }
.canslim-hero p { margin: .35rem 0 0 0; color: #395443; font-weight: 500; }
.lesson-panel {
    border: 1px solid rgba(44, 88, 62, 0.14);
    border-radius: 14px;
    background: rgba(255,255,255,.78);
    padding: .85rem 1rem;
}
</style>
<div class="canslim-hero">
  <h1>CANSLIM Model Book</h1>
  <p>Browse your private model-chart library and study the setup lesson next to the source chart.</p>
</div>
""",
    unsafe_allow_html=True,
)

df = load_model_book()
if df.empty:
    st.warning("No CANSLIM reference index found. Run `python3 scripts/build_canslim_chart_index.py` first.")
    st.stop()

status_counts = df["review_status"].value_counts()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Screenshots", len(df))
c2.metric("Model examples", int(status_counts.get("labeled_seed", 0)))
c3.metric("Duplicates", int(status_counts.get("duplicate", 0)))
c4.metric("Excluded", int(status_counts.get("exclude", 0)))

show_duplicates = st.sidebar.checkbox("Include duplicates", value=False)
show_excluded = st.sidebar.checkbox("Include excluded rows", value=False)

filtered = df.copy()
if not show_duplicates:
    filtered = filtered[filtered["review_status"] != "duplicate"]
if not show_excluded:
    filtered = filtered[filtered["review_status"] != "exclude"]

statuses = sorted(filtered["review_status"].dropna().unique())
selected_statuses = st.sidebar.multiselect("Status", statuses, default=statuses)
if selected_statuses:
    filtered = filtered[filtered["review_status"].isin(selected_statuses)]

setup_options = sorted([value for value in filtered["setup_type"].unique() if value])
selected_setups = st.sidebar.multiselect("Setup type", setup_options, default=[])
if selected_setups:
    filtered = filtered[filtered["setup_type"].isin(selected_setups)]

tickers = sorted([value for value in filtered["ticker"].unique() if value])
selected_tickers = st.sidebar.multiselect("Ticker / company", tickers, default=[])
if selected_tickers:
    filtered = filtered[filtered["ticker"].isin(selected_tickers)]

query = st.sidebar.text_input("Search")
if query:
    haystack_cols = [
        "image_id",
        "ticker",
        "setup_type",
        "base_quality",
        "volume_notes",
        "relative_strength_notes",
        "buy_point_notes",
        "failure_warnings",
        "model_lesson",
        "outcome_note",
    ]
    mask = pd.Series(False, index=filtered.index)
    for col in haystack_cols:
        if col in filtered.columns:
            mask = mask | filtered[col].str.contains(query, case=False, regex=False, na=False)
    filtered = filtered[mask]

filtered = filtered.sort_values(["source_page", "image_id"], key=lambda col: col.map(lambda value: int(value) if str(value).isdigit() else 10_000))

st.sidebar.markdown("---")
st.sidebar.caption(f"Showing {len(filtered)} rows")

if filtered.empty:
    st.warning("No model examples match the current filters.")
    st.stop()

labels = {option_label(row): idx for idx, row in filtered.iterrows()}
selected_label = st.selectbox("Model example", list(labels.keys()))
selected = filtered.loc[labels[selected_label]]

left, right = st.columns([1.35, 0.9], gap="large")

with left:
    path = image_path(selected)
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.error(f"Image file not found: {path}")

with right:
    st.markdown(
        f"""
<div class="lesson-panel">
  <h3>{selected.get('ticker') or selected.get('image_id')}</h3>
  <p><strong>{selected.get('setup_type') or 'Setup not labeled'}</strong></p>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("")
    render_note("Model lesson", selected.get("model_lesson", ""))
    render_note("Base quality", selected.get("base_quality", ""))
    render_note("Volume", selected.get("volume_notes", ""))
    render_note("Relative strength", selected.get("relative_strength_notes", ""))
    render_note("Buy point", selected.get("buy_point_notes", ""))
    render_note("Failure warning", selected.get("failure_warnings", ""))
    render_note("Outcome", selected.get("outcome_note", ""))
    st.caption(
        " | ".join(
            part
            for part in [
                f"Image: {selected.get('image_id')}",
                f"Page: {selected.get('source_page')}" if selected.get("source_page") else "",
                f"Confidence: {selected.get('confidence')}" if selected.get("confidence") else "",
                f"Status: {selected.get('review_status')}",
            ]
            if part
        )
    )

st.markdown("## Library Table")
visible_cols = [
    "image_id",
    "source_page",
    "ticker",
    "setup_type",
    "model_lesson",
    "failure_warnings",
    "outcome_note",
    "confidence",
    "review_status",
]
st.dataframe(filtered[[col for col in visible_cols if col in filtered.columns]], use_container_width=True, hide_index=True)

with st.expander("Raw selected row"):
    st.json(selected.to_dict())
