from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


def normalize_name(value: str) -> str:
    """Normalize column names for lightweight fuzzy matching."""
    return "".join(ch for ch in str(value).lower().strip() if ch.isalnum())


def parse_numeric(series: pd.Series) -> pd.Series:
    """Parse numeric strings including $, commas, and (negatives)."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")

    as_str = series.astype(str).str.strip()
    cleaned = (
        as_str.str.replace(r"\(([^)]+)\)", r"-\1", regex=True)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    cleaned = cleaned.replace({"": np.nan, "nan": np.nan, "None": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def auto_match_columns(columns: Iterable[str]) -> Dict[str, Optional[str]]:
    candidates = {
        "exit_date": ["exit", "closedate", "exitdate", "close", "dateclosed", "closeout"],
        "entry_date": ["entry", "entrydate", "opendate", "dateopened", "open"],
        "pnl": ["pnl", "profitloss", "profit", "pl", "netpnl", "p&l", "return$", "totpayout"],
        "r_multiple": ["rmultiple", "r", "rmult", "rmultiple"],
        "risk": ["risk", "risk$", "riskusd", "riskvalue"],
        "asset": ["asset", "symbol", "ticker", "instrument"],
        "side": ["side", "direction", "longshort", "position"],
        "notes": ["note", "notes", "comment", "mistake"],
        "mistake_type": ["mistake", "errortype", "error", "tag", "reason"],
        "chart_link": ["link", "chartlink", "tradingviewlink", "url"],
        "chart_image": ["image", "chartimage", "screenshot", "imgurl", "imageurl"],
    }

    lookup = {normalize_name(col): col for col in columns}
    output: Dict[str, Optional[str]] = {k: None for k in candidates}

    for key, tokens in candidates.items():
        for token in tokens:
            if token in lookup:
                output[key] = lookup[token]
                break
        if output[key] is not None:
            continue

        for normalized, original in lookup.items():
            if any(token in normalized for token in tokens):
                output[key] = original
                break

    return output


def prepare_trades(raw: pd.DataFrame, mapping: Dict[str, Optional[str]]) -> pd.DataFrame:
    df = raw.copy()

    close_col = mapping.get("exit_date") or mapping.get("entry_date")
    if not close_col:
        return pd.DataFrame()

    prepared = pd.DataFrame(index=df.index)
    prepared["close_date"] = pd.to_datetime(df[close_col], errors="coerce")

    entry_col = mapping.get("entry_date")
    if entry_col:
        prepared["entry_date"] = pd.to_datetime(df[entry_col], errors="coerce")

    pnl_col = mapping.get("pnl")
    if pnl_col:
        prepared["pnl"] = parse_numeric(df[pnl_col])
    else:
        prepared["pnl"] = np.nan

    r_col = mapping.get("r_multiple")
    if r_col:
        prepared["r_multiple"] = parse_numeric(df[r_col])

    risk_col = mapping.get("risk")
    if risk_col:
        prepared["risk"] = parse_numeric(df[risk_col])

    for field in ("asset", "side", "notes", "mistake_type", "chart_link", "chart_image"):
        col = mapping.get(field)
        if col:
            prepared[field] = df[col].astype(str)

    prepared = prepared.dropna(subset=["close_date", "pnl"]).copy()
    prepared = prepared.sort_values("close_date")
    prepared["trade_id"] = np.arange(1, len(prepared) + 1)
    return prepared


def _longest_streaks(pnl: pd.Series) -> tuple[int, int]:
    longest_win = 0
    longest_loss = 0
    curr_win = 0
    curr_loss = 0

    for value in pnl.fillna(0):
        if value > 0:
            curr_win += 1
            curr_loss = 0
        elif value < 0:
            curr_loss += 1
            curr_win = 0
        else:
            curr_win = 0
            curr_loss = 0
        longest_win = max(longest_win, curr_win)
        longest_loss = max(longest_loss, curr_loss)

    return longest_win, longest_loss


def compute_kpis(trades: pd.DataFrame) -> Dict[str, float]:
    if trades.empty:
        return {
            "Trades": 0,
            "Wins": 0,
            "Losses": 0,
            "Win Rate": np.nan,
            "Net P&L": 0.0,
            "Avg Trade": np.nan,
            "Avg Win": np.nan,
            "Avg Loss": np.nan,
            "Profit Factor": np.nan,
            "Expectancy ($)": np.nan,
            "Avg R": np.nan,
            "Expectancy (R)": np.nan,
            "Avg Win R": np.nan,
            "Avg Loss R": np.nan,
            "Longest Win Streak": 0,
            "Longest Loss Streak": 0,
            "Sharpe (approx)": np.nan,
            "Max Drawdown": 0.0,
        }

    pnl = trades["pnl"].astype(float).fillna(0)
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    total = len(pnl)
    gross_win = wins.sum()
    gross_loss = losses.sum()
    net = pnl.sum()

    profit_factor = gross_win / abs(gross_loss) if gross_loss < 0 else np.nan

    longest_win, longest_loss = _longest_streaks(pnl)

    rolling_equity = pnl.cumsum()
    drawdown = rolling_equity - rolling_equity.cummax()
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0

    sharpe = np.nan
    if "risk" in trades.columns:
        returns = (trades["pnl"] / trades["risk"].replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).dropna()
    else:
        returns = rolling_equity.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if len(returns) > 1 and returns.std(ddof=1) > 0:
        sharpe = float(np.sqrt(252) * returns.mean() / returns.std(ddof=1))

    avg_r = np.nan
    expectancy_r = np.nan
    avg_win_r = np.nan
    avg_loss_r = np.nan
    if "r_multiple" in trades.columns:
        valid_r = trades["r_multiple"].dropna()
        if len(valid_r):
            avg_r = float(valid_r.mean())
            expectancy_r = float(valid_r.mean())
            positive_r = valid_r[valid_r > 0]
            negative_r = valid_r[valid_r < 0]
            avg_win_r = float(positive_r.mean()) if len(positive_r) else np.nan
            avg_loss_r = float(negative_r.mean()) if len(negative_r) else np.nan

    return {
        "Trades": total,
        "Wins": int((pnl > 0).sum()),
        "Losses": int((pnl < 0).sum()),
        "Win Rate": float((pnl > 0).mean()) if total else np.nan,
        "Net P&L": float(net),
        "Avg Trade": float(pnl.mean()) if total else np.nan,
        "Avg Win": float(wins.mean()) if len(wins) else np.nan,
        "Avg Loss": float(losses.mean()) if len(losses) else np.nan,
        "Profit Factor": float(profit_factor) if not np.isnan(profit_factor) else np.nan,
        "Expectancy ($)": float(pnl.mean()) if total else np.nan,
        "Avg R": avg_r,
        "Expectancy (R)": expectancy_r,
        "Avg Win R": avg_win_r,
        "Avg Loss R": avg_loss_r,
        "Longest Win Streak": longest_win,
        "Longest Loss Streak": longest_loss,
        "Sharpe (approx)": sharpe,
        "Max Drawdown": max_drawdown,
    }


def compute_equity_curve(trades: pd.DataFrame, starting_equity: float = 0.0) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["close_date", "equity"])

    curve = trades[["close_date", "pnl"]].copy()
    curve = curve.groupby("close_date", as_index=False)["pnl"].sum().sort_values("close_date")
    curve["equity"] = starting_equity + curve["pnl"].cumsum()
    return curve


def _period_labels(series: pd.Series, frequency: str) -> pd.Series:
    if frequency == "H":
        return series.dt.year.astype(str) + "-H" + np.where(series.dt.month <= 6, "1", "2")

    return series.dt.to_period(frequency).astype(str)


def period_kpi_table(trades: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    working = trades.copy()
    working["Period"] = _period_labels(working["close_date"], frequency)

    rows = []
    for period_name, chunk in working.groupby("Period", sort=True):
        metrics = compute_kpis(chunk)
        rows.append(
            {
                "Period": period_name,
                "Trades": metrics["Trades"],
                "Net P&L": metrics["Net P&L"],
                "Win Rate": metrics["Win Rate"],
                "Profit Factor": metrics["Profit Factor"],
                "Expectancy ($)": metrics["Expectancy ($)"],
                "Expectancy (R)": metrics["Expectancy (R)"],
                "Avg R": metrics["Avg R"],
                "Max Drawdown": metrics["Max Drawdown"],
            }
        )

    return pd.DataFrame(rows)


def build_deviation_table(period_table: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
    if period_table.empty or len(period_table) < 2:
        return pd.DataFrame()

    numeric_cols = [
        "Net P&L",
        "Win Rate",
        "Profit Factor",
        "Expectancy ($)",
        "Expectancy (R)",
        "Avg R",
    ]

    current = period_table.iloc[-1]
    baseline = period_table.iloc[max(0, len(period_table) - 1 - lookback) : -1][numeric_cols].mean(numeric_only=True)

    rows = []
    for metric in numeric_cols:
        current_val = current.get(metric, np.nan)
        base_val = baseline.get(metric, np.nan)
        if pd.isna(base_val) or base_val == 0:
            pct_change = np.nan
        else:
            pct_change = (current_val - base_val) / abs(base_val)

        rows.append(
            {
                "Metric": metric,
                "Current": current_val,
                f"Baseline ({lookback} prior avg)": base_val,
                "Deviation %": pct_change,
            }
        )

    return pd.DataFrame(rows)


def horizon_snapshot(trades: pd.DataFrame, end_date: pd.Timestamp) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    end_date = pd.Timestamp(end_date).normalize()

    month_start = end_date.replace(day=1)
    quarter_start_month = ((end_date.month - 1) // 3) * 3 + 1
    quarter_start = pd.Timestamp(year=end_date.year, month=quarter_start_month, day=1)
    halfyear_start = end_date - pd.DateOffset(months=6) + pd.Timedelta(days=1)
    year_start = pd.Timestamp(year=end_date.year, month=1, day=1)

    windows = {
        "MTD": month_start,
        "QTD": quarter_start,
        "6M": halfyear_start,
        "YTD": year_start,
    }

    rows = []
    for label, start in windows.items():
        chunk = trades[(trades["close_date"] >= start) & (trades["close_date"] <= end_date)]
        k = compute_kpis(chunk)
        rows.append(
            {
                "Horizon": label,
                "Start": start.date(),
                "End": end_date.date(),
                "Trades": k["Trades"],
                "Net P&L": k["Net P&L"],
                "Win Rate": k["Win Rate"],
                "Expectancy ($)": k["Expectancy ($)"],
                "Expectancy (R)": k["Expectancy (R)"],
            }
        )

    return pd.DataFrame(rows)


def calendar_matrix(trades: pd.DataFrame, year: int) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()

    cal = trades.copy()
    cal = cal[cal["close_date"].dt.year == year]
    if cal.empty:
        return pd.DataFrame()

    grouped = (
        cal.assign(month=cal["close_date"].dt.month, day=cal["close_date"].dt.day)
        .groupby(["day", "month"], as_index=False)["pnl"]
        .sum()
    )

    matrix = grouped.pivot(index="day", columns="month", values="pnl")
    matrix = matrix.reindex(index=range(1, 32), columns=range(1, 13))
    return matrix
