from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from kpi_engine import (
    auto_match_columns,
    build_deviation_table,
    calendar_matrix,
    compute_equity_curve,
    compute_kpis,
    horizon_snapshot,
    period_kpi_table,
    prepare_trades,
)

APP_DIR = Path(__file__).resolve().parent
ANNOTATION_FILE = APP_DIR / "trade_annotations.csv"
ANNOTATION_COLUMNS = [
    "trade_key",
    "user_setup",
    "user_note",
    "user_mistake_type",
    "user_chart_link",
    "user_chart_image",
    "updated_at",
]
WEEKLY_CHECKIN_FILE = APP_DIR / "weekly_checkins.csv"
WEEKLY_CHECKIN_COLUMNS = [
    "week_key",
    "week_start",
    "week_end",
    "week_label",
    "followed_rules_process",
    "weekly_reflection",
    "updated_at",
]
WEEKLY_QUESTION_FILE = APP_DIR / "weekly_checkin_questions.csv"
WEEKLY_QUESTION_COLUMNS = [
    "question_id",
    "section",
    "prompt",
    "display_order",
    "is_active",
    "created_at",
    "updated_at",
]
WEEKLY_RESPONSE_FILE = APP_DIR / "weekly_checkin_responses.csv"
WEEKLY_RESPONSE_COLUMNS = [
    "week_key",
    "question_id",
    "answer_bool",
    "updated_at",
]
DEFAULT_WEEKLY_QUESTIONS = [
    ("rule_predefined_grades", "A. Rule Adherence", "Did I trade predefined grades ?"),
    ("rule_max_risk_by_grade", "A. Rule Adherence", "Did I respect max risk per trade based on grade ?"),
    ("rule_weekly_max_loss", "A. Rule Adherence", "Did I respect weekly max loss ?"),
    ("rule_no_impulse_revenge", "A. Rule Adherence", "Did I avoid impulse / revenge trades ?"),
    ("exec_entries_as_planned", "B. Execution Quality", "Were all entries taken as planned ?"),
    ("exec_stops_placed_per_plan", "B. Execution Quality", "Were stops placed exactly as per plan ?"),
    ("exec_no_moving_stops", "B. Execution Quality", "Did I avoid moving stops ?"),
    ("exec_winners_to_1r_before_manage", "B. Execution Quality", "Did I let winners reach at least 1R before management ?"),
    ("journal_all_trades_logged_tradezella", "C. Journaling Discipline", "Were 100% of the trades logged in Tradezella ? Ideally same day when they closed ?"),
    ("journal_execution_graded_separately", "C. Journaling Discipline", "Was execution graded separately from outcome ?"),
]
LEGACY_QUESTION_MAP = {qid: qid for qid, _, _ in DEFAULT_WEEKLY_QUESTIONS}
PERIODIC_CHECKIN_COLUMNS = [
    "period_key",
    "period_start",
    "period_end",
    "period_label",
    "followed_rules_process",
    "reflection",
    "updated_at",
]
PERIODIC_QUESTION_COLUMNS = [
    "question_id",
    "section",
    "prompt",
    "response_type",
    "display_order",
    "is_active",
    "created_at",
    "updated_at",
]
PERIODIC_RESPONSE_COLUMNS = [
    "period_key",
    "question_id",
    "answer_bool",
    "answer_text",
    "updated_at",
]
MONTHLY_CHECKIN_FILE = APP_DIR / "monthly_checkins.csv"
MONTHLY_QUESTION_FILE = APP_DIR / "monthly_checkin_questions.csv"
MONTHLY_RESPONSE_FILE = APP_DIR / "monthly_checkin_responses.csv"
DEFAULT_MONTHLY_QUESTIONS = [
    ("monthly_pct_grade_a", "A. Process Consistency", "Percentage of trades that were A", "number"),
    ("monthly_pct_grade_b", "A. Process Consistency", "Percentage of trades that were B", "number"),
    ("monthly_pct_grade_c", "A. Process Consistency", "Percentage of trades that were C", "number"),
    ("monthly_rule_violations", "A. Process Consistency", "Number of rule violations", "number"),
    ("monthly_pct_weeks_passed", "A. Process Consistency", "Percentage of weeks passing weekly rules", "number"),
    ("monthly_max_drawdown_pct", "B. Risk and Drawdown", "Monthly max drawdown (percentage)", "number"),
    ("monthly_largest_losing_streak", "B. Risk and Drawdown", "Largest losing streak", "number"),
    ("monthly_risk_reduce_after_dd", "B. Risk and Drawdown", "Did risk reduce after drawdown as per rules?", "yesno"),
    ("monthly_avg_r_per_trade", "C. Execution Statistics (Tradezella)", "Average R per trade", "number"),
    ("monthly_expectancy", "C. Execution Statistics (Tradezella)", "Expectancy", "number"),
    ("monthly_pct_reached_2r", "C. Execution Statistics (Tradezella)", "Percentage of trades reaching at least 2R", "number"),
    ("monthly_confidence_execution", "D. Psychological Stability (Rate 1-5)", "Confidence in execution", "rating_1_5"),
    ("monthly_emotional_neutrality", "D. Psychological Stability (Rate 1-5)", "Emotional neutrality", "rating_1_5"),
    ("monthly_patience_selectivity", "D. Psychological Stability (Rate 1-5)", "Patience and selectivity", "rating_1_5"),
    ("monthly_trust_process", "D. Psychological Stability (Rate 1-5)", "Trust in process", "rating_1_5"),
]
QUARTERLY_CHECKIN_FILE = APP_DIR / "quarterly_checkins.csv"
QUARTERLY_QUESTION_FILE = APP_DIR / "quarterly_checkin_questions.csv"
QUARTERLY_RESPONSE_FILE = APP_DIR / "quarterly_checkin_responses.csv"
DEFAULT_QUARTERLY_QUESTIONS = [
    ("quarterly_expectancy_by_grade", "A. Edge Strength", "Expectancy by trade grade", "text"),
    ("quarterly_expectancy_by_asset", "A. Edge Strength", "Expectancy by asset", "text"),
    ("quarterly_expectancy_by_market_type", "A. Edge Strength", "Expectancy by market type", "text"),
    ("quarterly_sample_size_ok", "A. Edge Strength", "Sample size met (at least 30 trades per segment)", "yesno"),
    ("quarterly_trades_by_market_regime", "B. Trade Selection and Focus", "Trades per quarter (downtrend / uptrend / counter-trend)", "text"),
    ("quarterly_results_by_market_regime", "B. Trade Selection and Focus", "Results by market regime", "text"),
    ("quarterly_top3_assets_expectancy", "B. Trade Selection and Focus", "Top 3 assets by expectancy", "text"),
    ("quarterly_emotional_triggers", "C. Behavioral Patterns", "Recurring emotional triggers", "text"),
    ("quarterly_rule_pressure_conditions", "C. Behavioral Patterns", "Markets/conditions causing rule pressure", "text"),
    ("quarterly_prep_confidence_correlation", "C. Behavioral Patterns", "Correlation between preparation and confidence", "text"),
]
LEGACY_PERIODIC_QUESTION_IDS = {
    "monthly_process_audit",
    "monthly_data_review",
    "monthly_corrections",
    "quarterly_plan_alignment",
    "quarterly_strategy_mix",
    "quarterly_risk_update",
}

st.set_page_config(page_title="Trading KPI Dashboard", layout="wide")
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

:root {
    --bg-top: #f9fbf7;
    --bg-mid: #edf4ee;
    --bg-bottom: #eef3f8;
    --ink-strong: #1f2f24;
    --ink-soft: #4a6253;
    --card: rgba(255, 255, 255, 0.78);
    --card-border: rgba(44, 88, 62, 0.15);
    --accent: #2f7d56;
    --accent-soft: #dcefe3;
}

.stApp {
    background:
        radial-gradient(1200px 420px at 8% -6%, rgba(145, 203, 169, 0.28), transparent 58%),
        radial-gradient(900px 360px at 92% -10%, rgba(131, 178, 210, 0.24), transparent 64%),
        linear-gradient(180deg, var(--bg-top) 0%, var(--bg-mid) 45%, var(--bg-bottom) 100%);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, rgba(244, 251, 246, 0.96), rgba(232, 244, 237, 0.92));
    border-right: 1px solid rgba(44, 88, 62, 0.14);
}

h1, h2, h3, h4, .stMarkdown [data-testid="stMarkdownContainer"] strong {
    font-family: "Space Grotesk", "Avenir Next", sans-serif;
    color: var(--ink-strong);
    letter-spacing: -0.01em;
}

[data-testid="stMetricLabel"], label, .stCaption {
    font-family: "IBM Plex Sans", "Trebuchet MS", sans-serif;
    color: var(--ink-soft);
}

[data-testid="stMetric"] {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    box-shadow: 0 14px 34px rgba(42, 83, 58, 0.08);
    padding: 0.65rem 0.8rem;
    animation: fade-up 380ms ease-out both;
}

[data-testid="stMetricValue"] {
    font-family: "Space Grotesk", "Avenir Next", sans-serif;
    color: var(--ink-strong);
}

[data-testid="stDataFrame"], [data-testid="stAlert"], .stExpander {
    border-radius: 14px;
    border: 1px solid var(--card-border);
    background: rgba(255, 255, 255, 0.68);
}

[data-baseweb="tab-list"] {
    gap: 0.45rem;
}

[data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(44, 88, 62, 0.15);
    border-radius: 12px 12px 0 0;
    font-family: "Space Grotesk", "Avenir Next", sans-serif;
}

.hero {
    border: 1px solid var(--card-border);
    border-radius: 18px;
    background:
        linear-gradient(140deg, rgba(255, 255, 255, 0.88), rgba(242, 250, 245, 0.88)),
        radial-gradient(350px 130px at 82% 22%, rgba(145, 203, 169, 0.24), transparent 70%);
    padding: 1.1rem 1.2rem 0.95rem 1.2rem;
    margin: 0.25rem 0 0.8rem 0;
    box-shadow: 0 16px 42px rgba(42, 83, 58, 0.1);
    animation: fade-up 440ms ease-out both;
}

.hero-kicker {
    display: inline-block;
    padding: 0.24rem 0.52rem;
    border-radius: 999px;
    background: var(--accent-soft);
    color: #255c41;
    font: 600 0.74rem/1.2 "IBM Plex Sans", sans-serif;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

.hero h1 {
    margin: 0.45rem 0 0.1rem 0;
    font-size: clamp(1.35rem, 2vw, 2rem);
}

.hero p {
    margin: 0;
    color: #395443;
    font: 500 0.95rem/1.45 "IBM Plex Sans", sans-serif;
}

@keyframes fade-up {
    from { transform: translateY(8px); opacity: 0; }
    to { transform: translateY(0); opacity: 1; }
}
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
<div class="hero">
  <span class="hero-kicker">Performance Command Center</span>
  <h1>Trading KPI Dashboard</h1>
  <p>Import trade logs, compare KPIs to goals, filter by setups, and annotate specific trades.</p>
</div>
""",
    unsafe_allow_html=True,
)


def style_figure(fig) -> None:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.68)",
        margin=dict(l=20, r=20, t=58, b=24),
        font=dict(family="IBM Plex Sans, Trebuchet MS, sans-serif", color="#233426", size=13),
        title=dict(font=dict(family="Space Grotesk, Avenir Next, sans-serif", size=20, color="#1f2f24")),
        colorway=["#2f7d56", "#4f9a6f", "#7fba89", "#246b89", "#3f8ca8", "#6da7bd"],
        legend=dict(
            bgcolor="rgba(255,255,255,0.72)",
            bordercolor="rgba(44,88,62,0.12)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(gridcolor="rgba(44, 88, 62, 0.12)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(44, 88, 62, 0.12)", zeroline=False)


def apply_view_mode_styles(density_mode: str, presentation_mode: bool) -> None:
    if density_mode == "Compact":
        container_top = "0.8rem"
        container_bottom = "1.1rem"
        block_gap = "0.55rem"
        metric_padding = "0.45rem 0.62rem"
    else:
        container_top = "1.35rem"
        container_bottom = "1.9rem"
        block_gap = "0.95rem"
        metric_padding = "0.8rem 0.95rem"

    hero_display = "none" if presentation_mode else "block"
    frame_shadow = "0 10px 24px rgba(42, 83, 58, 0.08)" if presentation_mode else "0 14px 34px rgba(42, 83, 58, 0.08)"

    st.markdown(
        f"""
<style>
.hero {{
    display: {hero_display};
}}
[data-testid="block-container"] {{
    padding-top: {container_top};
    padding-bottom: {container_bottom};
}}
[data-testid="stVerticalBlock"] {{
    gap: {block_gap};
}}
[data-testid="stMetric"] {{
    padding: {metric_padding};
}}
[data-testid="stDataFrame"], [data-testid="stAlert"], .stExpander {{
    box-shadow: {frame_shadow};
}}
</style>
""",
        unsafe_allow_html=True,
    )


def fmt_money(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"${value:,.2f}"


def fmt_pct(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value * 100:,.2f}%"


def fmt_num(value: float, digits: int = 2) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:,.{digits}f}"


def as_text(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def detect_setup_columns(raw: pd.DataFrame, excluded: set[str]) -> List[str]:
    setup_cols: List[str] = []
    for col in raw.columns:
        if col in excluded:
            continue

        series = raw[col]
        non_null = series.dropna()
        if non_null.empty:
            continue

        unique_count = non_null.astype(str).str.strip().replace("", np.nan).dropna().nunique()
        if unique_count < 2 or unique_count > 25:
            continue

        if pd.api.types.is_numeric_dtype(series) and unique_count > 10:
            continue

        setup_cols.append(col)

    return sorted(setup_cols)


def make_trade_key(df: pd.DataFrame) -> pd.Series:
    close = df["close_date"].dt.strftime("%Y-%m-%d").fillna("")
    pnl = df["pnl"].round(4).astype(str)

    asset = df["asset"].fillna("").astype(str).str.strip() if "asset" in df.columns else ""
    side = df["side"].fillna("").astype(str).str.strip() if "side" in df.columns else ""
    entry = (
        df["entry_date"].dt.strftime("%Y-%m-%d").fillna("")
        if "entry_date" in df.columns
        else ""
    )

    return close + "|" + asset + "|" + side + "|" + pnl + "|" + entry


def load_annotations() -> pd.DataFrame:
    if not ANNOTATION_FILE.exists():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    ann = pd.read_csv(ANNOTATION_FILE)
    for col in ANNOTATION_COLUMNS:
        if col not in ann.columns:
            ann[col] = ""

    return ann[ANNOTATION_COLUMNS].copy()


def upsert_annotation(record: Dict[str, str]) -> None:
    ann = load_annotations()
    ann = ann[ann["trade_key"] != record["trade_key"]]
    new_row = pd.DataFrame([record], columns=ANNOTATION_COLUMNS)
    ann = pd.concat([ann, new_row], ignore_index=True)
    ann.to_csv(ANNOTATION_FILE, index=False)


def parse_bool_cell(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_weekly_checkins() -> pd.DataFrame:
    if not WEEKLY_CHECKIN_FILE.exists():
        return pd.DataFrame(columns=WEEKLY_CHECKIN_COLUMNS)

    checkins = pd.read_csv(WEEKLY_CHECKIN_FILE)
    for col in WEEKLY_CHECKIN_COLUMNS:
        if col not in checkins.columns:
            checkins[col] = ""

    checkins["followed_rules_process"] = checkins["followed_rules_process"].map(parse_bool_cell)

    return checkins[WEEKLY_CHECKIN_COLUMNS].copy()


def upsert_weekly_checkin(record: Dict[str, object]) -> None:
    checkins = load_weekly_checkins()
    checkins = checkins[checkins["week_key"] != str(record["week_key"])]

    normalized = {col: record.get(col, "") for col in WEEKLY_CHECKIN_COLUMNS}
    normalized["followed_rules_process"] = bool(normalized.get("followed_rules_process", False))

    new_row = pd.DataFrame([normalized], columns=WEEKLY_CHECKIN_COLUMNS)
    checkins = pd.concat([checkins, new_row], ignore_index=True)
    checkins.to_csv(WEEKLY_CHECKIN_FILE, index=False)


def slugify_text(value: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return text[:64]


def load_weekly_questions() -> pd.DataFrame:
    if not WEEKLY_QUESTION_FILE.exists():
        now = datetime.utcnow().isoformat(timespec="seconds")
        seeded = pd.DataFrame(
            [
                {
                    "question_id": qid,
                    "section": section,
                    "prompt": prompt,
                    "display_order": idx + 1,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
                for idx, (qid, section, prompt) in enumerate(DEFAULT_WEEKLY_QUESTIONS)
            ],
            columns=WEEKLY_QUESTION_COLUMNS,
        )
        seeded.to_csv(WEEKLY_QUESTION_FILE, index=False)
        return seeded

    questions = pd.read_csv(WEEKLY_QUESTION_FILE)
    for col in WEEKLY_QUESTION_COLUMNS:
        if col not in questions.columns:
            questions[col] = ""

    questions["question_id"] = questions["question_id"].fillna("").astype(str).str.strip()
    questions["section"] = questions["section"].fillna("").astype(str).str.strip()
    questions["prompt"] = questions["prompt"].fillna("").astype(str).str.strip()
    questions["display_order"] = pd.to_numeric(questions["display_order"], errors="coerce").fillna(0).astype(int)
    questions["is_active"] = questions["is_active"].map(parse_bool_cell)

    questions = questions[questions["question_id"] != ""].copy()
    questions = questions.sort_values(["section", "display_order", "prompt"], kind="stable").reset_index(drop=True)
    return questions[WEEKLY_QUESTION_COLUMNS].copy()


def save_weekly_questions(questions: pd.DataFrame) -> None:
    out = questions[WEEKLY_QUESTION_COLUMNS].copy()
    out.to_csv(WEEKLY_QUESTION_FILE, index=False)


def add_weekly_question(section: str, prompt: str) -> None:
    questions = load_weekly_questions()
    now = datetime.utcnow().isoformat(timespec="seconds")

    base_id = slugify_text(prompt) or f"q_{int(datetime.utcnow().timestamp())}"
    question_id = base_id
    suffix = 2
    existing_ids = set(questions["question_id"].astype(str).tolist())
    while question_id in existing_ids:
        question_id = f"{base_id}_{suffix}"
        suffix += 1

    section_rows = questions[questions["section"] == section]
    next_order = int(section_rows["display_order"].max()) + 1 if not section_rows.empty else 1

    new_row = pd.DataFrame(
        [
            {
                "question_id": question_id,
                "section": section.strip(),
                "prompt": prompt.strip(),
                "display_order": next_order,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
        ],
        columns=WEEKLY_QUESTION_COLUMNS,
    )
    questions = pd.concat([questions, new_row], ignore_index=True)
    save_weekly_questions(questions)


def set_weekly_questions_active(question_ids: List[str], active: bool) -> None:
    if not question_ids:
        return
    questions = load_weekly_questions()
    mask = questions["question_id"].isin(question_ids)
    if not mask.any():
        return

    questions.loc[mask, "is_active"] = bool(active)
    questions.loc[mask, "updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    save_weekly_questions(questions)


def load_weekly_responses() -> pd.DataFrame:
    if not WEEKLY_RESPONSE_FILE.exists():
        return pd.DataFrame(columns=WEEKLY_RESPONSE_COLUMNS)

    responses = pd.read_csv(WEEKLY_RESPONSE_FILE)
    for col in WEEKLY_RESPONSE_COLUMNS:
        if col not in responses.columns:
            responses[col] = ""

    responses["week_key"] = responses["week_key"].fillna("").astype(str).str.strip()
    responses["question_id"] = responses["question_id"].fillna("").astype(str).str.strip()
    responses["answer_bool"] = responses["answer_bool"].map(parse_bool_cell)

    responses = responses[(responses["week_key"] != "") & (responses["question_id"] != "")].copy()
    return responses[WEEKLY_RESPONSE_COLUMNS].copy()


def save_weekly_responses(responses: pd.DataFrame) -> None:
    out = responses[WEEKLY_RESPONSE_COLUMNS].copy()
    out.to_csv(WEEKLY_RESPONSE_FILE, index=False)


def upsert_weekly_responses(week_key: str, answers: Dict[str, bool]) -> None:
    if not answers:
        return

    responses = load_weekly_responses()
    question_ids = list(answers.keys())
    drop_mask = (responses["week_key"] == week_key) & responses["question_id"].isin(question_ids)
    responses = responses[~drop_mask].copy()

    now = datetime.utcnow().isoformat(timespec="seconds")
    new_rows = pd.DataFrame(
        [
            {
                "week_key": week_key,
                "question_id": qid,
                "answer_bool": bool(answer),
                "updated_at": now,
            }
            for qid, answer in answers.items()
        ],
        columns=WEEKLY_RESPONSE_COLUMNS,
    )
    responses = pd.concat([responses, new_rows], ignore_index=True)
    save_weekly_responses(responses)


def bootstrap_legacy_weekly_responses(checkins: pd.DataFrame) -> None:
    if WEEKLY_RESPONSE_FILE.exists() or not WEEKLY_CHECKIN_FILE.exists():
        return

    raw = pd.read_csv(WEEKLY_CHECKIN_FILE)
    legacy_cols = [col for col in LEGACY_QUESTION_MAP if col in raw.columns]
    if raw.empty or not legacy_cols:
        return

    migrated_rows = []
    now = datetime.utcnow().isoformat(timespec="seconds")
    for _, row in raw.iterrows():
        week_key = as_text(row.get("week_key", "")).strip()
        if not week_key:
            continue
        for col in legacy_cols:
            migrated_rows.append(
                {
                    "week_key": week_key,
                    "question_id": LEGACY_QUESTION_MAP[col],
                    "answer_bool": parse_bool_cell(row.get(col, False)),
                    "updated_at": now,
                }
            )

    if migrated_rows:
        pd.DataFrame(migrated_rows, columns=WEEKLY_RESPONSE_COLUMNS).to_csv(WEEKLY_RESPONSE_FILE, index=False)


def monday_of_week(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value).normalize()
    return ts - pd.Timedelta(days=ts.weekday())


def build_week_ranges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close_date" not in df.columns:
        return pd.DataFrame(columns=["week_key", "week_start", "week_end", "week_label"])

    min_day = pd.Timestamp(df["close_date"].min()).normalize()
    max_day = max(pd.Timestamp(df["close_date"].max()).normalize(), pd.Timestamp.today().normalize())
    first_week = monday_of_week(min_day)
    last_week = monday_of_week(max_day)
    week_starts = pd.date_range(first_week, last_week, freq="W-MON")
    if week_starts.empty:
        week_starts = pd.DatetimeIndex([monday_of_week(pd.Timestamp.today())])

    weeks = pd.DataFrame({"week_start": week_starts})
    weeks["week_end"] = weeks["week_start"] + pd.Timedelta(days=6)
    weeks["week_key"] = weeks["week_start"].dt.strftime("%Y-%m-%d")
    weeks["week_label"] = weeks["week_start"].dt.strftime("%Y-%m-%d") + " to " + weeks["week_end"].dt.strftime("%Y-%m-%d")
    return weeks[["week_key", "week_start", "week_end", "week_label"]].sort_values("week_key", ascending=False).reset_index(drop=True)


def month_start(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value).normalize()
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)


def quarter_start(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value).normalize()
    q_month = ((ts.month - 1) // 3) * 3 + 1
    return pd.Timestamp(year=ts.year, month=q_month, day=1)


def build_month_ranges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close_date" not in df.columns:
        return pd.DataFrame(columns=["period_key", "period_start", "period_end", "period_label"])

    min_day = pd.Timestamp(df["close_date"].min()).normalize()
    max_day = max(pd.Timestamp(df["close_date"].max()).normalize(), pd.Timestamp.today().normalize())
    first_month = month_start(min_day)
    last_month = month_start(max_day)
    month_starts = pd.date_range(first_month, last_month, freq="MS")
    if month_starts.empty:
        month_starts = pd.DatetimeIndex([month_start(pd.Timestamp.today())])

    months = pd.DataFrame({"period_start": month_starts})
    months["period_end"] = months["period_start"] + pd.offsets.MonthEnd(1)
    months["period_key"] = months["period_start"].dt.strftime("%Y-%m")
    months["period_label"] = months["period_start"].dt.strftime("%Y-%m")
    return months[["period_key", "period_start", "period_end", "period_label"]].sort_values("period_key", ascending=False).reset_index(drop=True)


def build_quarter_ranges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "close_date" not in df.columns:
        return pd.DataFrame(columns=["period_key", "period_start", "period_end", "period_label"])

    min_day = pd.Timestamp(df["close_date"].min()).normalize()
    max_day = max(pd.Timestamp(df["close_date"].max()).normalize(), pd.Timestamp.today().normalize())
    first_quarter = quarter_start(min_day)
    last_quarter = quarter_start(max_day)
    quarter_starts = pd.date_range(first_quarter, last_quarter, freq="QS")
    if quarter_starts.empty:
        quarter_starts = pd.DatetimeIndex([quarter_start(pd.Timestamp.today())])

    quarters = pd.DataFrame({"period_start": quarter_starts})
    quarters["period_end"] = quarters["period_start"] + pd.offsets.QuarterEnd(0)
    q_num = ((quarters["period_start"].dt.month - 1) // 3 + 1).astype(str)
    quarters["period_key"] = quarters["period_start"].dt.year.astype(str) + "-Q" + q_num
    quarters["period_label"] = quarters["period_key"]
    return quarters[["period_key", "period_start", "period_end", "period_label"]].sort_values("period_key", ascending=False).reset_index(drop=True)


def current_month_key() -> str:
    return month_start(pd.Timestamp.today()).strftime("%Y-%m")


def current_quarter_key() -> str:
    ts = pd.Timestamp.today().normalize()
    q_num = (ts.month - 1) // 3 + 1
    return f"{ts.year}-Q{q_num}"


def load_periodic_checkins(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=PERIODIC_CHECKIN_COLUMNS)

    checkins = pd.read_csv(file_path)
    for col in PERIODIC_CHECKIN_COLUMNS:
        if col not in checkins.columns:
            checkins[col] = ""
    checkins["followed_rules_process"] = checkins["followed_rules_process"].map(parse_bool_cell)
    return checkins[PERIODIC_CHECKIN_COLUMNS].copy()


def upsert_periodic_checkin(file_path: Path, record: Dict[str, object]) -> None:
    checkins = load_periodic_checkins(file_path)
    checkins = checkins[checkins["period_key"] != str(record["period_key"])]

    normalized = {col: record.get(col, "") for col in PERIODIC_CHECKIN_COLUMNS}
    normalized["followed_rules_process"] = bool(normalized.get("followed_rules_process", False))

    new_row = pd.DataFrame([normalized], columns=PERIODIC_CHECKIN_COLUMNS)
    checkins = pd.concat([checkins, new_row], ignore_index=True)
    checkins.to_csv(file_path, index=False)


def unpack_periodic_default_question(item: tuple) -> tuple[str, str, str, str]:
    if len(item) == 4:
        qid, section, prompt, response_type = item
        return str(qid), str(section), str(prompt), str(response_type)
    if len(item) == 3:
        qid, section, prompt = item
        return str(qid), str(section), str(prompt), "yesno"
    raise ValueError("Periodic question defaults must be (id, section, prompt[, response_type]).")


def normalize_response_type(value: object) -> str:
    token = str(value).strip().lower().replace("-", "_")
    aliases = {
        "bool": "yesno",
        "boolean": "yesno",
        "checkbox": "yesno",
        "yes_no": "yesno",
        "rating": "rating_1_5",
        "rating1_5": "rating_1_5",
        "numeric": "number",
    }
    mapped = aliases.get(token, token)
    return mapped if mapped in {"yesno", "number", "rating_1_5", "text"} else "yesno"


def load_periodic_questions(file_path: Path, defaults: List[tuple]) -> pd.DataFrame:
    default_rows = [unpack_periodic_default_question(item) for item in defaults]
    default_map = {
        qid: {
            "section": section,
            "prompt": prompt,
            "response_type": normalize_response_type(response_type),
            "display_order": idx + 1,
        }
        for idx, (qid, section, prompt, response_type) in enumerate(default_rows)
    }
    default_ids = set(default_map.keys())

    if not file_path.exists():
        now = datetime.utcnow().isoformat(timespec="seconds")
        seeded = pd.DataFrame(
            [
                {
                    "question_id": qid,
                    "section": section,
                    "prompt": prompt,
                    "response_type": normalize_response_type(response_type),
                    "display_order": idx + 1,
                    "is_active": True,
                    "created_at": now,
                    "updated_at": now,
                }
                for idx, (qid, section, prompt, response_type) in enumerate(default_rows)
            ],
            columns=PERIODIC_QUESTION_COLUMNS,
        )
        seeded.to_csv(file_path, index=False)
        return seeded

    questions = pd.read_csv(file_path)
    for col in PERIODIC_QUESTION_COLUMNS:
        if col not in questions.columns:
            questions[col] = ""

    questions["question_id"] = questions["question_id"].fillna("").astype(str).str.strip()
    questions["section"] = questions["section"].fillna("").astype(str).str.strip()
    questions["prompt"] = questions["prompt"].fillna("").astype(str).str.strip()
    questions["response_type"] = questions["response_type"].map(normalize_response_type)
    questions["display_order"] = pd.to_numeric(questions["display_order"], errors="coerce").fillna(0).astype(int)
    questions["is_active"] = questions["is_active"].map(parse_bool_cell)
    questions = questions[questions["question_id"] != ""].copy()

    now = datetime.utcnow().isoformat(timespec="seconds")
    changed = False

    for qid, meta in default_map.items():
        mask = questions["question_id"] == qid
        if mask.any():
            idx = questions.index[mask][0]
            for col in ["section", "prompt", "response_type"]:
                if as_text(questions.at[idx, col]).strip() != as_text(meta[col]).strip():
                    questions.at[idx, col] = meta[col]
                    changed = True
            if int(questions.at[idx, "display_order"]) <= 0:
                questions.at[idx, "display_order"] = int(meta["display_order"])
                changed = True
        else:
            append_row = pd.DataFrame(
                [
                    {
                        "question_id": qid,
                        "section": meta["section"],
                        "prompt": meta["prompt"],
                        "response_type": meta["response_type"],
                        "display_order": int(meta["display_order"]),
                        "is_active": True,
                        "created_at": now,
                        "updated_at": now,
                    }
                ],
                columns=PERIODIC_QUESTION_COLUMNS,
            )
            questions = pd.concat([questions, append_row], ignore_index=True)
            changed = True

    legacy_mask = questions["question_id"].isin(LEGACY_PERIODIC_QUESTION_IDS - default_ids) & questions["is_active"]
    if legacy_mask.any():
        questions.loc[legacy_mask, "is_active"] = False
        questions.loc[legacy_mask, "updated_at"] = now
        changed = True

    questions = questions.sort_values(["section", "display_order", "prompt"], kind="stable").reset_index(drop=True)
    if changed:
        save_periodic_questions(file_path, questions)
    return questions[PERIODIC_QUESTION_COLUMNS].copy()


def save_periodic_questions(file_path: Path, questions: pd.DataFrame) -> None:
    out = questions[PERIODIC_QUESTION_COLUMNS].copy()
    out.to_csv(file_path, index=False)


def add_periodic_question(file_path: Path, defaults: List[tuple], section: str, prompt: str, response_type: str) -> None:
    questions = load_periodic_questions(file_path, defaults)
    now = datetime.utcnow().isoformat(timespec="seconds")

    base_id = slugify_text(prompt) or f"q_{int(datetime.utcnow().timestamp())}"
    question_id = base_id
    suffix = 2
    existing_ids = set(questions["question_id"].astype(str).tolist())
    while question_id in existing_ids:
        question_id = f"{base_id}_{suffix}"
        suffix += 1

    section_rows = questions[questions["section"] == section]
    next_order = int(section_rows["display_order"].max()) + 1 if not section_rows.empty else 1

    new_row = pd.DataFrame(
        [
            {
                "question_id": question_id,
                "section": section.strip(),
                "prompt": prompt.strip(),
                "response_type": normalize_response_type(response_type),
                "display_order": next_order,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
        ],
        columns=PERIODIC_QUESTION_COLUMNS,
    )
    questions = pd.concat([questions, new_row], ignore_index=True)
    save_periodic_questions(file_path, questions)


def set_periodic_questions_active(
    file_path: Path,
    defaults: List[tuple],
    question_ids: List[str],
    active: bool,
) -> None:
    if not question_ids:
        return

    questions = load_periodic_questions(file_path, defaults)
    mask = questions["question_id"].isin(question_ids)
    if not mask.any():
        return

    questions.loc[mask, "is_active"] = bool(active)
    questions.loc[mask, "updated_at"] = datetime.utcnow().isoformat(timespec="seconds")
    save_periodic_questions(file_path, questions)


def load_periodic_responses(file_path: Path) -> pd.DataFrame:
    if not file_path.exists():
        return pd.DataFrame(columns=PERIODIC_RESPONSE_COLUMNS)

    responses = pd.read_csv(file_path)
    for col in PERIODIC_RESPONSE_COLUMNS:
        if col not in responses.columns:
            responses[col] = ""

    responses["period_key"] = responses["period_key"].fillna("").astype(str).str.strip()
    responses["question_id"] = responses["question_id"].fillna("").astype(str).str.strip()
    responses["answer_bool"] = responses["answer_bool"].map(parse_bool_cell)
    responses["answer_text"] = responses["answer_text"].fillna("").astype(str)
    responses = responses[(responses["period_key"] != "") & (responses["question_id"] != "")].copy()
    return responses[PERIODIC_RESPONSE_COLUMNS].copy()


def save_periodic_responses(file_path: Path, responses: pd.DataFrame) -> None:
    out = responses[PERIODIC_RESPONSE_COLUMNS].copy()
    out.to_csv(file_path, index=False)


def upsert_periodic_responses(file_path: Path, period_key: str, answers: Dict[str, Dict[str, object]]) -> None:
    if not answers:
        return

    responses = load_periodic_responses(file_path)
    question_ids = list(answers.keys())
    drop_mask = (responses["period_key"] == period_key) & responses["question_id"].isin(question_ids)
    responses = responses[~drop_mask].copy()

    now = datetime.utcnow().isoformat(timespec="seconds")
    new_rows = pd.DataFrame(
        [
            {
                "period_key": period_key,
                "question_id": qid,
                "answer_bool": parse_bool_cell(payload.get("answer_bool", False)),
                "answer_text": as_text(payload.get("answer_text", "")).strip(),
                "updated_at": now,
            }
            for qid, payload in answers.items()
        ],
        columns=PERIODIC_RESPONSE_COLUMNS,
    )

    for idx, row in new_rows.iterrows():
        answer_type = normalize_response_type(answers[row["question_id"]].get("response_type", "yesno"))
        text_value = as_text(row["answer_text"]).strip()
        bool_value = bool(row["answer_bool"])

        if answer_type == "yesno":
            if not text_value:
                text_value = "Yes" if bool_value else "No"
            bool_value = text_value.strip().lower() == "yes"
        elif answer_type == "rating_1_5":
            try:
                rating = int(float(text_value))
            except ValueError:
                rating = 0
            text_value = str(min(5, max(1, rating))) if rating else ""
            bool_value = False
        elif answer_type in {"number", "text"}:
            bool_value = False

        new_rows.at[idx, "answer_bool"] = bool_value
        new_rows.at[idx, "answer_text"] = text_value

    responses = pd.concat([responses, new_rows], ignore_index=True)
    save_periodic_responses(file_path, responses)


def reminder_status(
    cadence_name: str,
    ranges: pd.DataFrame,
    checkins: pd.DataFrame,
    current_key: str,
    due_soon_days: int,
) -> tuple[str, str]:
    today = pd.Timestamp.today().normalize()
    if ranges.empty:
        return ("info", f"{cadence_name}: no periods available yet.")

    submitted = set(checkins["period_key"].astype(str).tolist()) if not checkins.empty else set()
    overdue = ranges[(ranges["period_end"] < today) & (~ranges["period_key"].isin(submitted))].copy()
    if not overdue.empty:
        latest = overdue.sort_values("period_end", ascending=False).iloc[0]
        return (
            "error",
            f"{cadence_name}: overdue ({latest['period_label']} missing).",
        )

    current_row = ranges[ranges["period_key"] == current_key]
    if current_row.empty:
        return ("info", f"{cadence_name}: no current period found.")

    cur = current_row.iloc[0]
    if current_key in submitted:
        return ("success", f"{cadence_name}: current period completed ({cur['period_label']}).")

    days_left = int((pd.Timestamp(cur["period_end"]).normalize() - today).days)
    if days_left <= due_soon_days:
        return ("warning", f"{cadence_name}: due in {max(days_left, 0)} day(s) ({cur['period_label']}).")
    return ("info", f"{cadence_name}: not due yet ({days_left} day(s) left).")


def render_periodic_checkin_section(
    title: str,
    key_prefix: str,
    ranges: pd.DataFrame,
    current_key: str,
    checkin_file: Path,
    question_file: Path,
    response_file: Path,
    defaults: List[tuple],
    foundation_prompt: str,
) -> None:
    st.subheader(title)
    checkins = load_periodic_checkins(checkin_file)
    questions = load_periodic_questions(question_file, defaults)
    responses = load_periodic_responses(response_file)

    with st.expander(f"Customize {title} Questions", expanded=False):
        section_options = sorted(v for v in questions["section"].dropna().astype(str).str.strip().unique() if v)
        if not section_options:
            section_options = ["A. Process", "B. Execution", "C. Review"]

        with st.form(f"{key_prefix}_add_question_form"):
            st.markdown("**Add new question**")
            section_choice = st.selectbox(
                "Section",
                options=section_options + ["Custom"],
                key=f"{key_prefix}_q_section_choice",
            )
            custom_section = st.text_input("Custom section name", value="", key=f"{key_prefix}_q_section_custom")
            prompt_input = st.text_input("Question prompt", value="", key=f"{key_prefix}_q_prompt")
            response_type_input = st.selectbox(
                "Answer type",
                options=["number", "yesno", "rating_1_5", "text"],
                format_func=lambda t: {
                    "number": "Number / %",
                    "yesno": "Yes / No",
                    "rating_1_5": "Rating 1-5",
                    "text": "Text",
                }.get(t, t),
                key=f"{key_prefix}_q_response_type",
            )
            add_question_submit = st.form_submit_button("Add Question")

        if add_question_submit:
            section_name = custom_section.strip() if section_choice == "Custom" else section_choice
            if not section_name:
                st.warning("Enter a section name for custom questions.")
            elif not prompt_input.strip():
                st.warning("Question prompt cannot be empty.")
            else:
                add_periodic_question(
                    file_path=question_file,
                    defaults=defaults,
                    section=section_name,
                    prompt=prompt_input.strip(),
                    response_type=response_type_input,
                )
                st.success("Question added.")
                st.rerun()

        question_label_map = {
            row.question_id: f"{row.section} | {row.prompt} ({row.response_type})"
            for row in questions.itertuples()
        }

        active_questions_all = questions[questions["is_active"]].copy()
        inactive_questions_all = questions[~questions["is_active"]].copy()

        if not active_questions_all.empty:
            remove_ids = st.multiselect(
                "Remove options from form",
                options=active_questions_all["question_id"].tolist(),
                format_func=lambda qid: question_label_map.get(qid, qid),
                key=f"{key_prefix}_q_remove_ids",
            )
            if st.button("Remove Selected", key=f"{key_prefix}_q_remove_btn"):
                if remove_ids:
                    set_periodic_questions_active(question_file, defaults, remove_ids, active=False)
                    st.success("Selected questions removed from active form.")
                    st.rerun()
                else:
                    st.caption("Select at least one question to remove.")

        if not inactive_questions_all.empty:
            restore_ids = st.multiselect(
                "Restore removed options",
                options=inactive_questions_all["question_id"].tolist(),
                format_func=lambda qid: question_label_map.get(qid, qid),
                key=f"{key_prefix}_q_restore_ids",
            )
            if st.button("Restore Selected", key=f"{key_prefix}_q_restore_btn"):
                if restore_ids:
                    set_periodic_questions_active(question_file, defaults, restore_ids, active=True)
                    st.success("Selected questions restored.")
                    st.rerun()
                else:
                    st.caption("Select at least one question to restore.")

        catalog = questions.copy()
        catalog["status"] = np.where(catalog["is_active"], "Active", "Removed")
        catalog["answer_type"] = catalog["response_type"].map(
            {
                "number": "Number / %",
                "yesno": "Yes / No",
                "rating_1_5": "Rating 1-5",
                "text": "Text",
            }
        )
        st.dataframe(
            catalog[["section", "prompt", "answer_type", "status", "display_order"]],
            use_container_width=True,
            hide_index=True,
        )

    label_map = dict(zip(ranges["period_key"], ranges["period_label"]))
    if not checkins.empty:
        for _, saved in checkins.iterrows():
            key = as_text(saved.get("period_key", "")).strip()
            label = as_text(saved.get("period_label", "")).strip()
            if key and key not in label_map:
                label_map[key] = label or key

    options = sorted(label_map.keys(), reverse=True)
    if not options:
        options = [current_key]
        label_map[current_key] = current_key

    default_idx = options.index(current_key) if current_key in options else 0
    selected_key = st.selectbox(
        "Select period",
        options=options,
        index=default_idx,
        format_func=lambda k: label_map.get(k, k),
        key=f"{key_prefix}_period_select",
    )

    selected_range = ranges[ranges["period_key"] == selected_key]
    if not selected_range.empty:
        selected_start = pd.Timestamp(selected_range.iloc[0]["period_start"])
        selected_end = pd.Timestamp(selected_range.iloc[0]["period_end"])
        selected_label = as_text(selected_range.iloc[0]["period_label"])
    else:
        existing = checkins[checkins["period_key"] == selected_key]
        if not existing.empty:
            row = existing.iloc[0]
            selected_start = pd.to_datetime(row.get("period_start", selected_key), errors="coerce")
            selected_end = pd.to_datetime(row.get("period_end", selected_key), errors="coerce")
            selected_start = selected_start if pd.notna(selected_start) else pd.Timestamp.today().normalize()
            selected_end = selected_end if pd.notna(selected_end) else selected_start
            selected_label = as_text(row.get("period_label", selected_key)) or selected_key
        else:
            selected_start = pd.Timestamp.today().normalize()
            selected_end = selected_start
            selected_label = selected_key

    existing_row = checkins[checkins["period_key"] == selected_key]
    existing_record: Dict[str, object] = existing_row.iloc[0].to_dict() if not existing_row.empty else {}

    response_map = {
        row["question_id"]: {
            "answer_bool": parse_bool_cell(row["answer_bool"]),
            "answer_text": as_text(row.get("answer_text", "")).strip(),
        }
        for _, row in responses[responses["period_key"] == selected_key].iterrows()
    }
    active_questions = questions[questions["is_active"]].copy()
    active_questions["section"] = active_questions["section"].replace("", "Uncategorized")
    active_questions = active_questions.sort_values(["section", "display_order", "prompt"], kind="stable")

    with st.form(f"{key_prefix}_checkin_form"):
        st.markdown("**Foundation Question**")
        followed_rules = st.radio(
            foundation_prompt,
            options=["Yes", "No"],
            index=0 if parse_bool_cell(existing_record.get("followed_rules_process", False)) else 1,
            horizontal=True,
            key=f"{key_prefix}_foundation",
        )

        answers: Dict[str, Dict[str, object]] = {}
        if active_questions.empty:
            st.caption("No active checklist questions. Add options above.")
        else:
            for section_name in active_questions["section"].dropna().unique():
                st.markdown(f"**{section_name}**")
                rows = active_questions[active_questions["section"] == section_name]
                for row in rows.itertuples():
                    response_type = normalize_response_type(row.response_type)
                    saved = response_map.get(row.question_id, {"answer_bool": False, "answer_text": ""})

                    if response_type == "yesno":
                        current_text = as_text(saved.get("answer_text", "")).strip().lower()
                        default_yes = (
                            current_text == "yes"
                            or (not current_text and parse_bool_cell(saved.get("answer_bool", False)))
                        )
                        value = st.radio(
                            row.prompt,
                            options=["Yes", "No"],
                            index=0 if default_yes else 1,
                            horizontal=True,
                            key=f"{key_prefix}_answer_{selected_key}_{row.question_id}",
                        )
                        answers[row.question_id] = {
                            "response_type": response_type,
                            "answer_bool": value == "Yes",
                            "answer_text": value,
                        }
                    elif response_type == "rating_1_5":
                        saved_text = as_text(saved.get("answer_text", "")).strip()
                        try:
                            default_rating = int(float(saved_text))
                        except ValueError:
                            default_rating = 3
                        default_rating = min(5, max(1, default_rating))
                        rating = st.slider(
                            row.prompt,
                            min_value=1,
                            max_value=5,
                            value=default_rating,
                            key=f"{key_prefix}_answer_{selected_key}_{row.question_id}",
                        )
                        answers[row.question_id] = {
                            "response_type": response_type,
                            "answer_bool": False,
                            "answer_text": str(rating),
                        }
                    elif response_type == "number":
                        number_value = st.text_input(
                            row.prompt,
                            value=as_text(saved.get("answer_text", "")).strip(),
                            placeholder="Enter number or percentage",
                            key=f"{key_prefix}_answer_{selected_key}_{row.question_id}",
                        )
                        answers[row.question_id] = {
                            "response_type": response_type,
                            "answer_bool": False,
                            "answer_text": number_value.strip(),
                        }
                    else:
                        text_value = st.text_area(
                            row.prompt,
                            value=as_text(saved.get("answer_text", "")).strip(),
                            height=90,
                            key=f"{key_prefix}_answer_{selected_key}_{row.question_id}",
                        )
                        answers[row.question_id] = {
                            "response_type": response_type,
                            "answer_bool": False,
                            "answer_text": text_value.strip(),
                        }

        reflection = st.text_area(
            "Period reflection (optional)",
            value=as_text(existing_record.get("reflection", "")),
            height=120,
            key=f"{key_prefix}_reflection",
        )
        save_submit = st.form_submit_button("Save Check-In")

    if save_submit:
        upsert_periodic_checkin(
            checkin_file,
            {
                "period_key": selected_key,
                "period_start": selected_start.strftime("%Y-%m-%d"),
                "period_end": selected_end.strftime("%Y-%m-%d"),
                "period_label": selected_label,
                "followed_rules_process": followed_rules == "Yes",
                "reflection": reflection.strip(),
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
            },
        )
        upsert_periodic_responses(response_file, selected_key, answers)
        st.success("Check-in saved.")
        st.rerun()

    if checkins.empty:
        st.caption("No check-ins saved yet.")
        return

    question_meta = questions[["question_id", "section", "prompt", "response_type"]].copy()
    review_rows = []
    for _, rec in checkins.sort_values("period_key", ascending=False).iterrows():
        period_key = as_text(rec.get("period_key", "")).strip()
        period_label = as_text(rec.get("period_label", "")) or label_map.get(period_key, period_key)
        rec_resp = responses[responses["period_key"] == period_key].copy()
        rec_resp = rec_resp.merge(question_meta, on="question_id", how="left")
        rec_resp["section"] = rec_resp["section"].fillna("Uncategorized")
        rec_resp["response_type"] = rec_resp["response_type"].map(normalize_response_type)
        rec_resp["answer_text"] = rec_resp["answer_text"].fillna("").astype(str).str.strip()
        rec_resp["answer_display"] = np.where(
            rec_resp["response_type"] == "yesno",
            np.where(
                rec_resp["answer_text"] != "",
                rec_resp["answer_text"],
                np.where(rec_resp["answer_bool"], "Yes", "No"),
            ),
            rec_resp["answer_text"],
        )
        rec_resp["is_answered"] = np.where(
            rec_resp["response_type"] == "yesno",
            True,
            rec_resp["answer_display"].astype(str).str.strip() != "",
        )

        parts: List[str] = []
        if not rec_resp.empty:
            for section_name, group in rec_resp.groupby("section", sort=False):
                completed_count = int(group["is_answered"].sum())
                parts.append(f"{section_name}: {completed_count}/{len(group)}")
            total_completed = int(rec_resp["is_answered"].sum())
            total_score = f"{total_completed}/{len(rec_resp)}"
            answered = int(total_completed)
        else:
            total_score = "0/0"
            answered = 0

        review_rows.append(
            {
                "period_key": period_key,
                "period_label": period_label,
                "Foundation": "Yes" if parse_bool_cell(rec.get("followed_rules_process", False)) else "No",
                "Section Scores": " | ".join(parts),
                "Total Score": total_score,
                "Questions Answered": answered,
                "Last Updated": as_text(rec.get("updated_at", "")),
            }
        )

    review = pd.DataFrame(review_rows).sort_values("period_key", ascending=False)
    st.markdown("**Saved Reviews**")
    st.dataframe(
        review[
            [
                "period_label",
                "Foundation",
                "Section Scores",
                "Total Score",
                "Questions Answered",
                "Last Updated",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    detail_key = st.selectbox(
        "Review saved period",
        options=review["period_key"].tolist(),
        format_func=lambda k: label_map.get(k, k),
        key=f"{key_prefix}_review_select",
    )
    detail = review[review["period_key"] == detail_key].iloc[0]
    detail_checkin = checkins[checkins["period_key"] == detail_key].iloc[0]
    detail_responses = responses[responses["period_key"] == detail_key].copy()
    detail_responses = detail_responses.merge(question_meta, on="question_id", how="left")
    detail_responses["section"] = detail_responses["section"].fillna("Uncategorized")
    detail_responses["response_type"] = detail_responses["response_type"].map(normalize_response_type)
    detail_responses["prompt"] = detail_responses["prompt"].fillna(detail_responses["question_id"])
    detail_responses["answer_text"] = detail_responses["answer_text"].fillna("").astype(str).str.strip()
    detail_responses["Answer"] = np.where(
        detail_responses["response_type"] == "yesno",
        np.where(
            detail_responses["answer_text"] != "",
            detail_responses["answer_text"],
            np.where(detail_responses["answer_bool"], "Yes", "No"),
        ),
        detail_responses["answer_text"],
    )
    detail_responses["Answer"] = detail_responses["Answer"].replace("", "-")
    detail_responses = detail_responses.sort_values(["section", "prompt"], kind="stable")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Foundation", detail["Foundation"])
    d2.metric("Total Score", detail["Total Score"])
    d3.metric("Questions Answered", str(detail["Questions Answered"]))
    d4.metric("Last Updated", as_text(detail["Last Updated"])[:10] or "-")
    if as_text(detail_checkin.get("reflection", "")).strip():
        st.caption(f"Reflection: {as_text(detail_checkin['reflection'])}")
    if not detail_responses.empty:
        st.dataframe(
            detail_responses[["section", "prompt", "Answer"]],
            use_container_width=True,
            hide_index=True,
        )


def format_deviation_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    for idx, row in out.iterrows():
        metric = row["Metric"]
        if metric == "Win Rate":
            out.at[idx, "Current"] = fmt_pct(row["Current"])
            out.at[idx, "Baseline (3 prior avg)"] = fmt_pct(row["Baseline (3 prior avg)"])
        elif metric in {"Net P&L", "Expectancy ($)"}:
            out.at[idx, "Current"] = fmt_money(row["Current"])
            out.at[idx, "Baseline (3 prior avg)"] = fmt_money(row["Baseline (3 prior avg)"])
        else:
            out.at[idx, "Current"] = fmt_num(row["Current"])
            out.at[idx, "Baseline (3 prior avg)"] = fmt_num(row["Baseline (3 prior avg)"])

    out["Deviation %"] = out["Deviation %"].map(fmt_pct)
    return out


def flatten_columns(columns: pd.Index) -> List[str]:
    names: List[str] = []
    seen: Dict[str, int] = {}

    for idx, col in enumerate(columns):
        parts: List[str] = []
        tokens = list(col) if isinstance(col, tuple) else [col]
        for token in tokens:
            label = str(token).strip()
            if not label or label.lower().startswith("unnamed"):
                continue
            parts.append(label)

        base = " | ".join(dict.fromkeys(parts)) if parts else f"Column_{idx + 1}"
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 1
            name = base
        names.append(name)

    return names


def auto_detect_header_row(preview: pd.DataFrame) -> int:
    keywords = [
        "entry",
        "exit",
        "p&l",
        "pnl",
        "r multiple",
        "risk",
        "asset",
        "side",
        "notes",
        "mistake",
    ]
    best_row = 0
    best_score = -1.0

    for idx, row in preview.iterrows():
        values = [str(v).strip() for v in row.tolist()]
        non_empty = [v for v in values if v and v.lower() not in {"nan", "none"}]
        if len(non_empty) < 3:
            continue

        text = " | ".join(non_empty).lower()
        keyword_hits = sum(1 for k in keywords if k in text)
        numeric_ratio = sum(1 for v in non_empty if v.replace(".", "", 1).replace("-", "", 1).isdigit()) / max(len(non_empty), 1)
        score = (keyword_hits * 5.0) + (len(set(non_empty)) * 0.2) - (numeric_ratio * 3.0)

        if score > best_score:
            best_score = score
            best_row = int(idx)

    return best_row


@st.cache_data(show_spinner=False)
def get_excel_sheet_names(file_bytes: bytes) -> List[str]:
    xl = pd.ExcelFile(io.BytesIO(file_bytes))
    return list(xl.sheet_names)


@st.cache_data(show_spinner=False)
def read_preview_rows(file_name: str, file_bytes: bytes, sheet_name: Optional[str], nrows: int = 60) -> pd.DataFrame:
    lower = file_name.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), header=None, nrows=nrows)

    if lower.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        selected = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected, header=None, nrows=nrows)

    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def read_uploaded_file(
    file_name: str,
    file_bytes: bytes,
    sheet_name: Optional[str],
    header_row_idx: int,
    header_depth: int,
) -> pd.DataFrame:
    lower = file_name.lower()
    header_value: int | List[int] = header_row_idx if header_depth == 1 else [header_row_idx, header_row_idx + 1]

    if lower.endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), header=header_value)

    if lower.endswith((".xlsx", ".xls")):
        xl = pd.ExcelFile(io.BytesIO(file_bytes))
        selected = sheet_name if sheet_name in xl.sheet_names else xl.sheet_names[0]
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected, header=header_value)

    return pd.DataFrame()


def select_col(label: str, map_key: str, columns: List[str], defaults: Dict[str, Optional[str]], ui_key: str) -> Optional[str]:
    default = defaults.get(map_key)
    index = columns.index(default) if default in columns else 0
    selected = st.selectbox(label, columns, index=index, key=ui_key)
    return None if selected == "<None>" else selected


top_import_col, top_checkin_col = st.columns([1.45, 1.0], gap="large")

with top_import_col:
    st.markdown("### Trade Log Import")
    uploaded_files = st.file_uploader(
        "Upload trade-log file(s)",
        type=["xlsx", "xls", "csv"],
        accept_multiple_files=True,
        help="Upload one or many files; data will be combined.",
    )

    density_mode = st.radio("Density", options=["Compact", "Expanded"], horizontal=True, index=1, key="top_density_mode")
    presentation_mode = st.toggle(
        "Presentation mode",
        value=False,
        help="Hides editing-heavy sections for cleaner screenshots.",
        key="top_presentation_mode",
    )

with top_checkin_col:
    st.markdown("### Check-In Hub")
    quick_weekly = load_weekly_checkins()
    quick_monthly = load_periodic_checkins(MONTHLY_CHECKIN_FILE)
    quick_quarterly = load_periodic_checkins(QUARTERLY_CHECKIN_FILE)

    current_week = monday_of_week(pd.Timestamp.today()).strftime("%Y-%m-%d")
    current_month = current_month_key()
    current_quarter = current_quarter_key()

    quick_week_done = current_week in set(quick_weekly["week_key"].astype(str).tolist()) if not quick_weekly.empty else False
    quick_month_done = current_month in set(quick_monthly["period_key"].astype(str).tolist()) if not quick_monthly.empty else False
    quick_quarter_done = current_quarter in set(quick_quarterly["period_key"].astype(str).tolist()) if not quick_quarterly.empty else False

    qc1, qc2, qc3 = st.columns(3)
    qc1.metric("Weekly", "Done" if quick_week_done else "Due")
    qc2.metric("Monthly", "Done" if quick_month_done else "Due")
    qc3.metric("Quarterly", "Done" if quick_quarter_done else "Due")

    qs1, qs2, qs3 = st.columns(3)
    qs1.metric("Saved Weeks", f"{len(quick_weekly)}")
    qs2.metric("Saved Months", f"{len(quick_monthly)}")
    qs3.metric("Saved Quarters", f"{len(quick_quarterly)}")
    st.caption("Full check-in forms and analytics are available below.")

if not uploaded_files:
    apply_view_mode_styles(density_mode=density_mode, presentation_mode=presentation_mode)
    st.info("Upload at least one trade-log file to start.")
    st.stop()

sheet_options: List[str] = []
for file in uploaded_files:
    if file.name.lower().endswith((".xlsx", ".xls")):
        for sheet in get_excel_sheet_names(file.getvalue()):
            if sheet not in sheet_options:
                sheet_options.append(sheet)

sheet_choice = None
with top_import_col:
    st.markdown("#### Parsing")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        header_mode = st.radio("Header row mode", options=["Auto-detect", "Manual"], index=0, key="top_header_mode")
    with p2:
        manual_header_row = st.number_input("Header row (1-based)", min_value=1, value=1, step=1, key="top_manual_header_row")
    with p3:
        header_depth = st.selectbox(
            "Header depth",
            options=[1, 2],
            index=0,
            help="Use 2 when your table has grouped headers (top row categories + second row field names).",
            key="top_header_depth",
        )
    with p4:
        if sheet_options:
            sheet_choice = st.selectbox("Sheet to import (Excel)", options=sheet_options, index=0, key="top_sheet_choice")
        else:
            st.caption("No Excel sheets detected.")

    preview_file_name = st.selectbox(
        "Preview file",
        options=[f.name for f in uploaded_files],
        index=0,
        key="top_preview_file_name",
    )

apply_view_mode_styles(density_mode=density_mode, presentation_mode=presentation_mode)

preview_file = next((f for f in uploaded_files if f.name == preview_file_name), uploaded_files[0])
preview_rows = read_preview_rows(preview_file.name, preview_file.getvalue(), sheet_choice, nrows=60)
detected_preview_header = auto_detect_header_row(preview_rows) + 1 if not preview_rows.empty else 1

if not presentation_mode:
    with st.expander("Header Row Preview (for fixing 'Unnamed' columns)", expanded=False):
        st.caption(f"Auto-detected header row: {detected_preview_header}")
        preview_display = preview_rows.fillna("").copy()
        preview_display.insert(0, "Row #", np.arange(1, len(preview_display) + 1))
        st.dataframe(preview_display, use_container_width=True, hide_index=True)

header_rows: Dict[str, int] = {}
for file in uploaded_files:
    if header_mode == "Auto-detect":
        preview = read_preview_rows(file.name, file.getvalue(), sheet_choice, nrows=60)
        header_rows[file.name] = auto_detect_header_row(preview)
    else:
        header_rows[file.name] = int(manual_header_row) - 1

frames = []
for file in uploaded_files:
    frame = read_uploaded_file(
        file.name,
        file.getvalue(),
        sheet_choice,
        header_row_idx=header_rows[file.name],
        header_depth=int(header_depth),
    )
    if not frame.empty:
        frame.columns = flatten_columns(frame.columns)
        frame["__source_file"] = file.name
        frames.append(frame)

if not frames:
    st.error("No rows were loaded from the uploaded files.")
    st.stop()

raw_df = pd.concat(frames, ignore_index=True)
raw_df.columns = flatten_columns(raw_df.columns)

auto_map = auto_match_columns(raw_df.columns)
all_columns = ["<None>"] + raw_df.columns.tolist()

if presentation_mode:
    mapping_parent = st.expander("Column Mapping (collapsed in presentation mode)", expanded=False)
else:
    st.subheader("Column Mapping")
    mapping_parent = st.container()

with mapping_parent:
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        exit_col = select_col("Exit / Close Date", "exit_date", all_columns, auto_map, "col_exit")
        entry_col = select_col("Entry Date", "entry_date", all_columns, auto_map, "col_entry")
    with mc2:
        pnl_col = select_col("P&L", "pnl", all_columns, auto_map, "col_pnl")
        r_col = select_col("R Multiple", "r_multiple", all_columns, auto_map, "col_r")
    with mc3:
        risk_col = select_col("Risk $", "risk", all_columns, auto_map, "col_risk")
        asset_col = select_col("Asset / Symbol", "asset", all_columns, auto_map, "col_asset")
        side_col = select_col("Side", "side", all_columns, auto_map, "col_side")
    with mc4:
        notes_col = select_col("Notes", "notes", all_columns, auto_map, "col_notes")
        mistake_col = select_col("Mistake Type", "mistake_type", all_columns, auto_map, "col_mistake")
        link_col = select_col("Chart Link", "chart_link", all_columns, auto_map, "col_link")
        image_col = select_col("Chart Image URL", "chart_image", all_columns, auto_map, "col_image")

mapping: Dict[str, Optional[str]] = {
    "exit_date": exit_col,
    "entry_date": entry_col,
    "pnl": pnl_col,
    "r_multiple": r_col,
    "risk": risk_col,
    "asset": asset_col,
    "side": side_col,
    "notes": notes_col,
    "mistake_type": mistake_col,
    "chart_link": link_col,
    "chart_image": image_col,
}

if mapping["pnl"] is None:
    st.error("P&L mapping is required. If only 'Unnamed' columns appear, adjust Header Parsing (row/depth) and re-map.")
    st.stop()

trades_all = prepare_trades(raw_df, mapping)
if trades_all.empty:
    st.error("No valid trades after parsing. Check date and P&L mapping.")
    st.stop()

trades_all["trade_key"] = make_trade_key(trades_all)

mapped_cols = {col for col in mapping.values() if col}
setup_candidates = detect_setup_columns(raw_df, excluded=mapped_cols | {"__source_file"})

if setup_candidates:
    extra_cols = [col for col in setup_candidates if col not in trades_all.columns]
    if extra_cols:
        trades_all = trades_all.join(raw_df[extra_cols], how="left")

annotations = load_annotations()
if not annotations.empty:
    trades_all = trades_all.merge(annotations, on="trade_key", how="left")
else:
    for col in ANNOTATION_COLUMNS:
        if col != "trade_key":
            trades_all[col] = ""

if "mistake_type" not in trades_all.columns:
    trades_all["mistake_type"] = ""
if "notes" not in trades_all.columns:
    trades_all["notes"] = ""
if "chart_link" not in trades_all.columns:
    trades_all["chart_link"] = ""
if "chart_image" not in trades_all.columns:
    trades_all["chart_image"] = ""

trades_all["mistake_type_final"] = np.where(
    trades_all["user_mistake_type"].fillna("").str.strip() != "",
    trades_all["user_mistake_type"].fillna(""),
    trades_all["mistake_type"].fillna(""),
)
trades_all["chart_link_final"] = np.where(
    trades_all["user_chart_link"].fillna("").str.strip() != "",
    trades_all["user_chart_link"].fillna(""),
    trades_all["chart_link"].fillna(""),
)
trades_all["chart_image_final"] = np.where(
    trades_all["user_chart_image"].fillna("").str.strip() != "",
    trades_all["user_chart_image"].fillna(""),
    trades_all["chart_image"].fillna(""),
)

min_date = trades_all["close_date"].min().date()
max_date = trades_all["close_date"].max().date()

with st.sidebar:
    st.header("Filters")
    date_range = st.date_input(
        "Close date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) or isinstance(date_range, list):
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    start_equity = st.number_input("Starting Equity", value=0.0, step=1000.0)

    selected_assets: List[str] = []
    if "asset" in trades_all.columns:
        asset_options = sorted(v for v in trades_all["asset"].dropna().astype(str).str.strip().unique() if v)
        selected_assets = st.multiselect("Asset filter", options=asset_options)

    selected_sides: List[str] = []
    if "side" in trades_all.columns:
        side_options = sorted(v for v in trades_all["side"].dropna().astype(str).str.strip().unique() if v)
        selected_sides = st.multiselect("Side filter", options=side_options)

    mistake_options = sorted(v for v in trades_all["mistake_type_final"].dropna().astype(str).str.strip().unique() if v)
    selected_mistakes = st.multiselect("Mistake type filter", options=mistake_options)

    st.subheader("Setup Filters")
    setup_filter_columns = st.multiselect(
        "Setup columns",
        options=setup_candidates,
        default=[col for col in setup_candidates if col.lower() in {"trend", "grade", "freshness", "coverage", "valuation", "seasonality", "earnings (new)", "cot"}],
    )

    setup_values_map: Dict[str, List[str]] = {}
    for col in setup_filter_columns:
        options = sorted(v for v in trades_all[col].dropna().astype(str).str.strip().unique() if v)
        setup_values_map[col] = st.multiselect(f"{col} values", options=options, key=f"setup_{col}")

    note_search = st.text_input("Notes text contains")

    st.header("KPI Goals")
    target_month = st.number_input("Goal Net P&L (MTD)", value=0.0, step=100.0)
    target_quarter = st.number_input("Goal Net P&L (QTD)", value=0.0, step=100.0)
    target_6m = st.number_input("Goal Net P&L (6M)", value=0.0, step=100.0)
    target_year = st.number_input("Goal Net P&L (YTD)", value=0.0, step=100.0)
    target_win_rate = st.number_input("Goal Win Rate %", value=40.0, min_value=0.0, max_value=100.0, step=1.0) / 100
    target_expectancy_r = st.number_input("Goal Expectancy (R)", value=0.10, step=0.01)
    compare_horizon = st.selectbox("Goal comparison horizon", options=["MTD", "QTD", "6M", "YTD"], index=0)

start_ts = pd.Timestamp(start_date)
end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


def apply_non_date_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if selected_assets and "asset" in out.columns:
        out = out[out["asset"].astype(str).isin(selected_assets)]

    if selected_sides and "side" in out.columns:
        out = out[out["side"].astype(str).isin(selected_sides)]

    if selected_mistakes:
        out = out[out["mistake_type_final"].astype(str).isin(selected_mistakes)]

    for col, selected_values in setup_values_map.items():
        if selected_values and col in out.columns:
            out = out[out[col].astype(str).isin(selected_values)]

    if note_search:
        combined_notes = (
            out["notes"].fillna("").astype(str)
            + " "
            + out["user_note"].fillna("").astype(str)
        )
        out = out[combined_notes.str.contains(note_search, case=False, na=False)]

    return out


trades_scope = apply_non_date_filters(trades_all)
trades = trades_scope[(trades_scope["close_date"] >= start_ts) & (trades_scope["close_date"] <= end_ts)].copy()

if trades.empty:
    st.warning("No trades in the selected date range/filters.")
    st.stop()

kpi = compute_kpis(trades)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Trades", f"{int(kpi['Trades'])}")
m2.metric("Net P&L", fmt_money(kpi["Net P&L"]))
m3.metric("Win Rate", fmt_pct(kpi["Win Rate"]))
m4.metric("Profit Factor", fmt_num(kpi["Profit Factor"]))
m5.metric("Expectancy ($)", fmt_money(kpi["Expectancy ($)"]))
m6.metric("Expectancy (R)", fmt_num(kpi["Expectancy (R)"]))

m7, m8, m9, m10 = st.columns(4)
m7.metric("Avg R", fmt_num(kpi["Avg R"]))
m8.metric("Longest Win Streak", f"{int(kpi['Longest Win Streak'])}")
m9.metric("Longest Loss Streak", f"{int(kpi['Longest Loss Streak'])}")
m10.metric("Max Drawdown", fmt_money(kpi["Max Drawdown"]))

curve = compute_equity_curve(trades, starting_equity=start_equity)
if not curve.empty:
    eq_fig = px.line(curve, x="close_date", y="equity", title="Equity Curve")
    eq_fig.update_layout(xaxis_title="Close Date", yaxis_title="Equity")
    style_figure(eq_fig)
    st.plotly_chart(eq_fig, use_container_width=True)

st.subheader("KPI Base vs Goal")
snapshot = horizon_snapshot(trades_scope[trades_scope["close_date"] <= end_ts], end_ts)

if not snapshot.empty:
    goals = {
        "MTD": target_month,
        "QTD": target_quarter,
        "6M": target_6m,
        "YTD": target_year,
    }
    snapshot["Goal Net P&L"] = snapshot["Horizon"].map(goals)
    snapshot["Net P&L Gap"] = snapshot["Net P&L"] - snapshot["Goal Net P&L"]
    snapshot["Goal Progress %"] = np.where(
        snapshot["Goal Net P&L"].abs() > 0,
        snapshot["Net P&L"] / snapshot["Goal Net P&L"].abs(),
        np.nan,
    )
    snapshot["Goal Win Rate"] = target_win_rate
    snapshot["Win Rate Gap"] = snapshot["Win Rate"] - snapshot["Goal Win Rate"]
    snapshot["Goal Expectancy (R)"] = target_expectancy_r
    snapshot["Expectancy (R) Gap"] = snapshot["Expectancy (R)"] - snapshot["Goal Expectancy (R)"]

    selected_cmp = snapshot[snapshot["Horizon"] == compare_horizon]
    if not selected_cmp.empty:
        row = selected_cmp.iloc[0]
        g1, g2, g3 = st.columns(3)
        g1.metric("Net P&L vs Goal", fmt_money(row["Net P&L"]), delta=fmt_money(row["Net P&L Gap"]))
        g2.metric("Win Rate vs Goal", fmt_pct(row["Win Rate"]), delta=fmt_pct(row["Win Rate Gap"]))
        g3.metric("Expectancy (R) vs Goal", fmt_num(row["Expectancy (R)"]), delta=fmt_num(row["Expectancy (R) Gap"]))

    show = snapshot.copy()
    show["Win Rate"] = show["Win Rate"].map(fmt_pct)
    show["Goal Win Rate"] = show["Goal Win Rate"].map(fmt_pct)
    show["Win Rate Gap"] = show["Win Rate Gap"].map(fmt_pct)
    show["Net P&L"] = show["Net P&L"].map(fmt_money)
    show["Goal Net P&L"] = show["Goal Net P&L"].map(fmt_money)
    show["Net P&L Gap"] = show["Net P&L Gap"].map(fmt_money)
    show["Goal Progress %"] = show["Goal Progress %"].map(fmt_pct)
    show["Expectancy ($)"] = show["Expectancy ($)"].map(fmt_money)
    show["Expectancy (R)"] = show["Expectancy (R)"].map(fmt_num)
    show["Goal Expectancy (R)"] = show["Goal Expectancy (R)"].map(fmt_num)
    show["Expectancy (R) Gap"] = show["Expectancy (R) Gap"].map(fmt_num)
    st.dataframe(show, use_container_width=True, hide_index=True)

period_tables: Dict[str, pd.DataFrame] = {
    freq: period_kpi_table(trades, frequency=freq) for freq in ["M", "Q", "H", "Y"]
}

if not presentation_mode:
    period_tabs = st.tabs(["Monthly", "Quarterly", "6-Month", "Yearly"])
    for tab, freq in zip(period_tabs, ["M", "Q", "H", "Y"]):
        period = period_tables[freq]

        with tab:
            if period.empty:
                st.info("No rows for this frequency.")
                continue

            display = period.copy()
            display["Win Rate"] = display["Win Rate"].map(fmt_pct)
            for col in ["Net P&L", "Expectancy ($)", "Max Drawdown"]:
                display[col] = display[col].map(fmt_money)
            for col in ["Profit Factor", "Expectancy (R)", "Avg R"]:
                display[col] = display[col].map(fmt_num)

            st.dataframe(display, use_container_width=True, hide_index=True)
            st.markdown("Deviation vs trailing baseline")
            dev = build_deviation_table(period, lookback=3)
            if dev.empty:
                st.caption("Need at least two periods.")
            else:
                st.dataframe(format_deviation_table(dev), use_container_width=True, hide_index=True)

viz_tab1, viz_tab2, viz_tab3 = st.tabs(["P&L Calendar", "P&L By Period", "Distributions & Mistakes"])

with viz_tab1:
    years = sorted(trades["close_date"].dt.year.unique())
    selected_year = st.selectbox("Calendar year", years, index=len(years) - 1)
    matrix = calendar_matrix(trades, selected_year)

    if matrix.empty:
        st.info("No calendar data for selected year.")
    else:
        cal_fig = px.imshow(
            matrix.values,
            x=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            y=list(range(1, 32)),
            color_continuous_scale=["#b2182b", "#f7f7f7", "#1a9641"],
            labels={"x": "Month", "y": "Day", "color": "P&L"},
            aspect="auto",
            title=f"Daily Realized P&L Heatmap ({selected_year})",
        )
        cal_fig.update_yaxes(autorange="reversed")
        style_figure(cal_fig)
        st.plotly_chart(cal_fig, use_container_width=True)

with viz_tab2:
    freq_choice = st.selectbox(
        "Bar frequency",
        options=[("Monthly", "M"), ("Quarterly", "Q"), ("6-Month", "H"), ("Yearly", "Y")],
        format_func=lambda x: x[0],
    )
    choice = period_tables[freq_choice[1]]

    if choice.empty:
        st.info("No rows for selected frequency.")
    else:
        bar = px.bar(
            choice,
            x="Period",
            y="Net P&L",
            color="Net P&L",
            color_continuous_scale="RdYlGn",
            title=f"Net P&L by {freq_choice[0]}",
        )
        style_figure(bar)
        st.plotly_chart(bar, use_container_width=True)

with viz_tab3:
    d1, d2, d3 = st.columns(3)
    with d1:
        pnl_hist = px.histogram(trades, x="pnl", nbins=30, title="Trade P&L Distribution")
        style_figure(pnl_hist)
        st.plotly_chart(pnl_hist, use_container_width=True)

    with d2:
        if "r_multiple" in trades.columns and trades["r_multiple"].notna().any():
            r_hist = px.histogram(trades.dropna(subset=["r_multiple"]), x="r_multiple", nbins=20, title="R Multiple Histogram")
            style_figure(r_hist)
            st.plotly_chart(r_hist, use_container_width=True)
        else:
            st.caption("Map R Multiple column to enable histogram.")

    with d3:
        losses = trades[trades["pnl"] < 0].copy()
        losses["mistake_type_final"] = losses["mistake_type_final"].fillna("").replace("", "Unclassified")
        if not losses.empty:
            grouped = losses.groupby("mistake_type_final", as_index=False)["pnl"].sum()
            grouped["loss_abs"] = grouped["pnl"].abs()
            pie = px.pie(grouped, names="mistake_type_final", values="loss_abs", title="Losses by Mistake Type")
            style_figure(pie)
            st.plotly_chart(pie, use_container_width=True)
        else:
            st.caption("No losing trades in current filter.")

st.subheader("Check-In Reminders")
monthly_checkins_rem = load_periodic_checkins(MONTHLY_CHECKIN_FILE)
quarterly_checkins_rem = load_periodic_checkins(QUARTERLY_CHECKIN_FILE)
month_ranges_rem = build_month_ranges(trades_all)
quarter_ranges_rem = build_quarter_ranges(trades_all)

weekly_checkins_for_rem = load_weekly_checkins().rename(
    columns={
        "week_key": "period_key",
        "week_start": "period_start",
        "week_end": "period_end",
        "week_label": "period_label",
    }
)
week_ranges_rem = build_week_ranges(trades_all).rename(
    columns={
        "week_key": "period_key",
        "week_start": "period_start",
        "week_end": "period_end",
        "week_label": "period_label",
    }
)

r1, r2, r3 = st.columns(3)
weekly_level, weekly_msg = reminder_status(
    "Weekly",
    week_ranges_rem,
    weekly_checkins_for_rem,
    monday_of_week(pd.Timestamp.today()).strftime("%Y-%m-%d"),
    due_soon_days=1,
)
monthly_level, monthly_msg = reminder_status(
    "Monthly",
    month_ranges_rem,
    monthly_checkins_rem,
    current_month_key(),
    due_soon_days=3,
)
quarterly_level, quarterly_msg = reminder_status(
    "Quarterly",
    quarter_ranges_rem,
    quarterly_checkins_rem,
    current_quarter_key(),
    due_soon_days=7,
)

for col, level, msg in [
    (r1, weekly_level, weekly_msg),
    (r2, monthly_level, monthly_msg),
    (r3, quarterly_level, quarterly_msg),
]:
    with col:
        if level == "error":
            st.error(msg)
        elif level == "warning":
            st.warning(msg)
        elif level == "success":
            st.success(msg)
        else:
            st.info(msg)

st.subheader("Weekly Check-In")
weekly_checkins = load_weekly_checkins()
weekly_questions = load_weekly_questions()
bootstrap_legacy_weekly_responses(weekly_checkins)
weekly_responses = load_weekly_responses()
week_ranges = build_week_ranges(trades_all)

with st.expander("Customize Weekly Form Questions", expanded=False):
    section_options = sorted(v for v in weekly_questions["section"].dropna().astype(str).str.strip().unique() if v)
    if not section_options:
        section_options = ["A. Rule Adherence", "B. Execution Quality", "C. Journaling Discipline"]

    with st.form("weekly_add_question_form"):
        st.markdown("**Add new question**")
        section_choice = st.selectbox("Section", options=section_options + ["Custom"], key="weekly_q_section_choice")
        custom_section = st.text_input("Custom section name", value="", key="weekly_q_section_custom")
        prompt_input = st.text_input("Question prompt", value="", key="weekly_q_prompt")
        add_question_submit = st.form_submit_button("Add Question")

    if add_question_submit:
        section_name = custom_section.strip() if section_choice == "Custom" else section_choice
        if not section_name:
            st.warning("Enter a section name for custom questions.")
        elif not prompt_input.strip():
            st.warning("Question prompt cannot be empty.")
        else:
            add_weekly_question(section=section_name, prompt=prompt_input.strip())
            st.success("Question added.")
            st.rerun()

    question_label_map = {
        row.question_id: f"{row.section} | {row.prompt}"
        for row in weekly_questions.itertuples()
    }

    active_questions_all = weekly_questions[weekly_questions["is_active"]].copy()
    inactive_questions_all = weekly_questions[~weekly_questions["is_active"]].copy()

    if not active_questions_all.empty:
        remove_ids = st.multiselect(
            "Remove options from form",
            options=active_questions_all["question_id"].tolist(),
            format_func=lambda qid: question_label_map.get(qid, qid),
            key="weekly_q_remove_ids",
        )
        if st.button("Remove Selected", key="weekly_q_remove_btn"):
            if remove_ids:
                set_weekly_questions_active(remove_ids, active=False)
                st.success("Selected questions removed from active form.")
                st.rerun()
            else:
                st.caption("Select at least one question to remove.")

    if not inactive_questions_all.empty:
        restore_ids = st.multiselect(
            "Restore removed options",
            options=inactive_questions_all["question_id"].tolist(),
            format_func=lambda qid: question_label_map.get(qid, qid),
            key="weekly_q_restore_ids",
        )
        if st.button("Restore Selected", key="weekly_q_restore_btn"):
            if restore_ids:
                set_weekly_questions_active(restore_ids, active=True)
                st.success("Selected questions restored.")
                st.rerun()
            else:
                st.caption("Select at least one question to restore.")

    catalog = weekly_questions.copy()
    catalog["status"] = np.where(catalog["is_active"], "Active", "Removed")
    st.dataframe(
        catalog[["section", "prompt", "status", "display_order"]],
        use_container_width=True,
        hide_index=True,
    )

week_label_map = dict(zip(week_ranges["week_key"], week_ranges["week_label"]))
if not weekly_checkins.empty:
    for _, saved in weekly_checkins.iterrows():
        key = as_text(saved.get("week_key", "")).strip()
        label = as_text(saved.get("week_label", "")).strip()
        if key and key not in week_label_map:
            week_label_map[key] = label or key

week_options = sorted(week_label_map.keys(), reverse=True)
if not week_options:
    current_monday = monday_of_week(pd.Timestamp.today())
    week_options = [current_monday.strftime("%Y-%m-%d")]
    week_label_map[week_options[0]] = f"{current_monday:%Y-%m-%d} to {(current_monday + pd.Timedelta(days=6)):%Y-%m-%d}"

current_week_key = monday_of_week(pd.Timestamp.today()).strftime("%Y-%m-%d")
default_week_idx = week_options.index(current_week_key) if current_week_key in week_options else 0
selected_week_key = st.selectbox(
    "Select week",
    options=week_options,
    index=default_week_idx,
    format_func=lambda wk: week_label_map.get(wk, wk),
    key="weekly_checkin_week",
)

selected_week_start = pd.Timestamp(selected_week_key)
selected_week_end = selected_week_start + pd.Timedelta(days=6)
selected_week_label = week_label_map.get(selected_week_key, f"{selected_week_start:%Y-%m-%d} to {selected_week_end:%Y-%m-%d}")

existing_week = weekly_checkins[weekly_checkins["week_key"] == selected_week_key]
existing_record: Dict[str, object] = existing_week.iloc[0].to_dict() if not existing_week.empty else {}
week_response_map = {
    row["question_id"]: parse_bool_cell(row["answer_bool"])
    for _, row in weekly_responses[weekly_responses["week_key"] == selected_week_key].iterrows()
}
active_questions = weekly_questions[weekly_questions["is_active"]].copy()
active_questions["section"] = active_questions["section"].replace("", "Uncategorized")
active_questions = active_questions.sort_values(["section", "display_order", "prompt"], kind="stable")

with st.form("weekly_checkin_form"):
    st.markdown("**Foundation Question**")
    followed_rules = st.radio(
        "Did I follow my rules/process this week, regardless of outcome?",
        options=["Yes", "No"],
        index=0 if parse_bool_cell(existing_record.get("followed_rules_process", False)) else 1,
        horizontal=True,
    )

    question_answers: Dict[str, bool] = {}
    if active_questions.empty:
        st.caption("No active checklist questions. Add options in 'Customize Weekly Form Questions'.")
    else:
        for section_name in active_questions["section"].dropna().unique():
            st.markdown(f"**{section_name}**")
            section_rows = active_questions[active_questions["section"] == section_name]
            for row in section_rows.itertuples():
                question_answers[row.question_id] = st.checkbox(
                    row.prompt,
                    value=parse_bool_cell(week_response_map.get(row.question_id, False)),
                    key=f"weekly_answer_{selected_week_key}_{row.question_id}",
                )

    weekly_reflection = st.text_area(
        "Weekly reflection (optional)",
        value=as_text(existing_record.get("weekly_reflection", "")),
        height=120,
    )
    save_weekly_checkin = st.form_submit_button("Save Weekly Check-In")

if save_weekly_checkin:
    upsert_weekly_checkin(
        {
            "week_key": selected_week_key,
            "week_start": selected_week_start.strftime("%Y-%m-%d"),
            "week_end": selected_week_end.strftime("%Y-%m-%d"),
            "week_label": selected_week_label,
            "followed_rules_process": followed_rules == "Yes",
            "weekly_reflection": weekly_reflection.strip(),
            "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
        }
    )
    upsert_weekly_responses(selected_week_key, question_answers)
    st.success("Weekly check-in saved.")
    st.rerun()

if weekly_checkins.empty:
    st.caption("No weekly check-ins saved yet.")
else:
    review_rows = []
    question_meta = weekly_questions[["question_id", "section", "prompt"]].copy()

    for _, wk in weekly_checkins.sort_values("week_key", ascending=False).iterrows():
        wk_key = as_text(wk.get("week_key", "")).strip()
        wk_label = as_text(wk.get("week_label", "")).strip() or week_label_map.get(wk_key, wk_key)
        wk_resp = weekly_responses[weekly_responses["week_key"] == wk_key].copy()
        wk_resp = wk_resp.merge(question_meta, on="question_id", how="left")
        wk_resp["section"] = wk_resp["section"].fillna("Uncategorized")
        wk_resp["prompt"] = wk_resp["prompt"].fillna(wk_resp["question_id"])

        section_parts: List[str] = []
        if not wk_resp.empty:
            for section_name, group in wk_resp.groupby("section", sort=False):
                yes_count = int(group["answer_bool"].sum())
                section_parts.append(f"{section_name}: {yes_count}/{len(group)}")
            total_yes = int(wk_resp["answer_bool"].sum())
            total_score = f"{total_yes}/{len(wk_resp)}"
            answered_count = int(len(wk_resp))
        else:
            total_score = "0/0"
            answered_count = 0

        review_rows.append(
            {
                "week_key": wk_key,
                "week_label": wk_label,
                "Foundation": "Yes" if parse_bool_cell(wk.get("followed_rules_process", False)) else "No",
                "Section Scores": " | ".join(section_parts),
                "Total Score": total_score,
                "Questions Answered": answered_count,
                "Last Updated": as_text(wk.get("updated_at", "")),
            }
        )

    review = pd.DataFrame(review_rows).sort_values("week_key", ascending=False)

    st.markdown("**Saved Weekly Reviews**")
    st.dataframe(
        review[
            [
                "week_label",
                "Foundation",
                "Section Scores",
                "Total Score",
                "Questions Answered",
                "Last Updated",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )

    detail_week = st.selectbox(
        "Review saved week",
        options=review["week_key"].tolist(),
        format_func=lambda wk: week_label_map.get(wk, wk),
        key="weekly_checkin_review_week",
    )
    detail = review[review["week_key"] == detail_week].iloc[0]
    detail_checkin = weekly_checkins[weekly_checkins["week_key"] == detail_week].iloc[0]
    detail_responses = weekly_responses[weekly_responses["week_key"] == detail_week].copy()
    detail_responses = detail_responses.merge(
        weekly_questions[["question_id", "section", "prompt"]],
        on="question_id",
        how="left",
    )
    detail_responses["section"] = detail_responses["section"].fillna("Uncategorized")
    detail_responses["prompt"] = detail_responses["prompt"].fillna(detail_responses["question_id"])
    detail_responses["Answer"] = np.where(detail_responses["answer_bool"], "Yes", "No")
    detail_responses = detail_responses.sort_values(["section", "prompt"], kind="stable")

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Foundation", detail["Foundation"])
    d2.metric("Total Score", detail["Total Score"])
    d3.metric("Questions Answered", str(detail["Questions Answered"]))
    d4.metric("Last Updated", as_text(detail["Last Updated"])[:10] or "-")
    if as_text(detail_checkin.get("weekly_reflection", "")).strip():
        st.caption(f"Reflection: {as_text(detail_checkin['weekly_reflection'])}")
    if detail_responses.empty:
        st.caption("No checklist responses saved for this week.")
    else:
        st.dataframe(
            detail_responses[["section", "prompt", "Answer"]],
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Weekly Trend & Summary**")
    if weekly_responses.empty:
        st.caption("Need at least one saved weekly checklist response to show trend analytics.")
    else:
        weekly_analytics = weekly_responses.merge(
            weekly_questions[["question_id", "section", "prompt"]],
            on="question_id",
            how="left",
        )
        weekly_analytics["section"] = weekly_analytics["section"].fillna("Uncategorized")
        weekly_analytics["prompt"] = weekly_analytics["prompt"].fillna(weekly_analytics["question_id"])

        week_stats = (
            weekly_analytics.groupby("week_key", as_index=False)
            .agg(
                yes_count=("answer_bool", "sum"),
                total_questions=("question_id", "count"),
            )
            .merge(
                weekly_checkins[["week_key", "week_label", "followed_rules_process"]],
                on="week_key",
                how="left",
            )
            .sort_values("week_key")
        )
        week_stats["week_label"] = week_stats["week_label"].fillna(week_stats["week_key"])
        week_stats["score_pct"] = np.where(
            week_stats["total_questions"] > 0,
            week_stats["yes_count"] / week_stats["total_questions"] * 100.0,
            np.nan,
        )
        week_stats["Foundation"] = np.where(week_stats["followed_rules_process"], "Yes", "No")

        weeks_logged = int(len(week_stats))
        avg_score = float(week_stats["score_pct"].mean()) if weeks_logged else np.nan
        foundation_rate = float(week_stats["followed_rules_process"].mean() * 100.0) if weeks_logged else np.nan
        avg_answered = float(week_stats["total_questions"].mean()) if weeks_logged else np.nan

        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Weeks Logged", f"{weeks_logged}")
        w2.metric("Avg Weekly Score", f"{avg_score:.1f}%" if not np.isnan(avg_score) else "-")
        w3.metric("Foundation Yes Rate", f"{foundation_rate:.1f}%" if not np.isnan(foundation_rate) else "-")
        w4.metric("Avg Questions / Week", f"{avg_answered:.1f}" if not np.isnan(avg_answered) else "-")

        if weeks_logged > 0:
            best_week = week_stats.loc[week_stats["score_pct"].idxmax()]
            worst_week = week_stats.loc[week_stats["score_pct"].idxmin()]
            q_stats = weekly_analytics.groupby("prompt", as_index=False).agg(
                yes_count=("answer_bool", "sum"),
                total=("answer_bool", "count"),
            )
            q_stats["yes_rate_pct"] = np.where(q_stats["total"] > 0, q_stats["yes_count"] / q_stats["total"] * 100.0, np.nan)
            weakest = q_stats.sort_values(["yes_rate_pct", "total"], ascending=[True, False]).iloc[0] if not q_stats.empty else None

            summary_text = (
                f"Average weekly checklist score is {avg_score:.1f}% across {weeks_logged} week(s). "
                f"Best week: {as_text(best_week['week_label'])} ({best_week['score_pct']:.1f}%). "
                f"Lowest week: {as_text(worst_week['week_label'])} ({worst_week['score_pct']:.1f}%)."
            )
            if weakest is not None:
                summary_text += f" Most missed check: \"{as_text(weakest['prompt'])}\" ({weakest['yes_rate_pct']:.1f}% yes rate)."
            st.caption(summary_text)

        score_fig = px.line(
            week_stats,
            x="week_label",
            y="score_pct",
            markers=True,
            title="Weekly Checklist Score (%)",
        )
        score_fig.update_layout(xaxis_title="Week", yaxis_title="Score %")
        score_fig.update_yaxes(range=[0, 100])
        score_fig.update_xaxes(tickangle=-30)
        style_figure(score_fig)
        st.plotly_chart(score_fig, use_container_width=True)

        section_stats = weekly_analytics.groupby(["week_key", "section"], as_index=False).agg(
            yes_count=("answer_bool", "sum"),
            total_questions=("question_id", "count"),
        )
        section_stats["score_pct"] = np.where(
            section_stats["total_questions"] > 0,
            section_stats["yes_count"] / section_stats["total_questions"] * 100.0,
            np.nan,
        )
        section_pivot = section_stats.pivot(index="week_key", columns="section", values="score_pct")
        section_pivot = section_pivot.reindex(week_stats["week_key"])
        section_pivot.index = week_stats["week_label"].tolist()
        if not section_pivot.empty and section_pivot.shape[1] > 0:
            section_heat = px.imshow(
                section_pivot.values,
                x=section_pivot.columns.tolist(),
                y=section_pivot.index.tolist(),
                color_continuous_scale=["#b2182b", "#f7f7f7", "#1a9641"],
                labels={"x": "Section", "y": "Week", "color": "Score %"},
                title="Section Score by Week (%)",
                aspect="auto",
            )
            section_heat.update_coloraxes(cmin=0, cmax=100)
            style_figure(section_heat)
            st.plotly_chart(section_heat, use_container_width=True)

        q_stats = weekly_analytics.groupby("prompt", as_index=False).agg(
            yes_count=("answer_bool", "sum"),
            total=("answer_bool", "count"),
        )
        q_stats["yes_rate_pct"] = np.where(q_stats["total"] > 0, q_stats["yes_count"] / q_stats["total"] * 100.0, np.nan)
        q_stats["miss_rate_pct"] = 100.0 - q_stats["yes_rate_pct"]
        missed = q_stats.sort_values(["miss_rate_pct", "total"], ascending=[False, False]).head(8)
        if not missed.empty:
            miss_fig = px.bar(
                missed.sort_values("miss_rate_pct", ascending=True),
                x="miss_rate_pct",
                y="prompt",
                orientation="h",
                title="Most Missed Weekly Checks (%)",
            )
            miss_fig.update_layout(xaxis_title="Miss Rate %", yaxis_title="Question")
            miss_fig.update_xaxes(range=[0, 100])
            style_figure(miss_fig)
            st.plotly_chart(miss_fig, use_container_width=True)

        weekly_table = week_stats[["week_label", "Foundation", "yes_count", "total_questions", "score_pct"]].copy()
        weekly_table["Checklist Score"] = weekly_table["score_pct"].map(lambda v: f"{v:.1f}%")
        weekly_table["Checks Completed"] = (
            weekly_table["yes_count"].astype(int).astype(str)
            + "/"
            + weekly_table["total_questions"].astype(int).astype(str)
        )
        st.dataframe(
            weekly_table[["week_label", "Foundation", "Checklist Score", "Checks Completed"]],
            use_container_width=True,
            hide_index=True,
        )

render_periodic_checkin_section(
    title="Monthly Check-In",
    key_prefix="monthly",
    ranges=build_month_ranges(trades_all),
    current_key=current_month_key(),
    checkin_file=MONTHLY_CHECKIN_FILE,
    question_file=MONTHLY_QUESTION_FILE,
    response_file=MONTHLY_RESPONSE_FILE,
    defaults=DEFAULT_MONTHLY_QUESTIONS,
    foundation_prompt="Am I executing my system consistently and controlling the downside?",
)

render_periodic_checkin_section(
    title="Quarterly Check-In",
    key_prefix="quarterly",
    ranges=build_quarter_ranges(trades_all),
    current_key=current_quarter_key(),
    checkin_file=QUARTERLY_CHECKIN_FILE,
    question_file=QUARTERLY_QUESTION_FILE,
    response_file=QUARTERLY_RESPONSE_FILE,
    defaults=DEFAULT_QUARTERLY_QUESTIONS,
    foundation_prompt="Is my edge real, robust, and improving over time?",
)

if not presentation_mode:
    st.subheader("Specific Trade Logging")
    trade_view = trades.sort_values("close_date", ascending=False).copy()
    trade_view["display"] = trade_view.apply(
        lambda r: f"{r['close_date'].date()} | {as_text(r.get('asset', 'N/A'))} | {fmt_money(r['pnl'])}",
        axis=1,
    )

    selected_display = st.selectbox("Select trade", options=trade_view["display"].tolist())
    selected_trade = trade_view[trade_view["display"] == selected_display].iloc[0]

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Close Date", str(selected_trade["close_date"].date()))
    s2.metric("Asset", as_text(selected_trade.get("asset", "-")))
    s3.metric("P&L", fmt_money(float(selected_trade["pnl"])))
    s4.metric("R Multiple", fmt_num(float(selected_trade["r_multiple"])) if "r_multiple" in selected_trade and pd.notna(selected_trade["r_multiple"]) else "-")

    if as_text(selected_trade.get("chart_link_final", "")).strip():
        st.markdown(f"Chart link: [{as_text(selected_trade['chart_link_final'])}]({as_text(selected_trade['chart_link_final'])})")

    if as_text(selected_trade.get("chart_image_final", "")).strip():
        st.image(as_text(selected_trade["chart_image_final"]), caption="Trade chart image", use_container_width=True)

    with st.form("trade_annotation_form"):
        setup_input = st.text_input("Setup Tag / Label", value=as_text(selected_trade.get("user_setup", "")))
        mistake_input = st.text_input(
            "Mistake Type",
            value=as_text(selected_trade.get("user_mistake_type", "")) or as_text(selected_trade.get("mistake_type", "")),
        )
        trade_note = st.text_area(
            "Trade Review Note",
            value=as_text(selected_trade.get("user_note", "")) or as_text(selected_trade.get("notes", "")),
        )
        link_input = st.text_input(
            "Chart Link URL",
            value=as_text(selected_trade.get("chart_link_final", "")),
        )
        image_input = st.text_input(
            "Chart Image URL or local image path",
            value=as_text(selected_trade.get("chart_image_final", "")),
        )

        save_trade = st.form_submit_button("Save Trade Annotation")

    if save_trade:
        upsert_annotation(
            {
                "trade_key": as_text(selected_trade["trade_key"]),
                "user_setup": setup_input.strip(),
                "user_note": trade_note.strip(),
                "user_mistake_type": mistake_input.strip(),
                "user_chart_link": link_input.strip(),
                "user_chart_image": image_input.strip(),
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
            }
        )
        st.success("Trade annotation saved.")
        st.rerun()

    with st.expander("Raw Parsed Trades"):
        preview = trades.copy()
        preview["close_date"] = preview["close_date"].dt.strftime("%Y-%m-%d")
        if "entry_date" in preview.columns:
            preview["entry_date"] = preview["entry_date"].dt.strftime("%Y-%m-%d")

        st.dataframe(
            preview,
            use_container_width=True,
            hide_index=True,
            column_config={
                "chart_link_final": st.column_config.LinkColumn("Chart Link"),
            },
        )
        csv_bytes = trades.to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered data as CSV", data=csv_bytes, file_name="filtered_trades.csv", mime="text/csv")
