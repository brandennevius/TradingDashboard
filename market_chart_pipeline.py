"""Market data and CANSLIM chart pipeline.

Designed for the post-close MarketSurge workflow. The module deliberately fails
closed: a symbol is never considered chart-verified unless adjusted daily bars
are available through the requested session and the minimum history checks pass.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
import requests

ALPACA_DATA_URL = "https://data.alpaca.markets/v2/stocks/bars"


class MarketDataError(RuntimeError):
    pass


@dataclass
class ChartMetrics:
    ticker: str
    session_date: str
    current_price: float
    sma21: float
    sma50: float
    sma200: float
    atr14: float
    atr_pct: float
    avg_volume_50: float
    relative_volume: float
    avg_dollar_volume_50: float
    high_52w: float
    pct_from_52w_high: float
    pct_from_sma50: float
    pct_from_sma200: float
    tightness_5d_pct: float
    tightness_10d_pct: float
    tightness_15d_pct: float
    up_down_volume_ratio_50: Optional[float]
    accumulation_days_25: int
    distribution_days_25: int
    data_source: str = "ALPACA"

    def to_dict(self) -> dict:
        return asdict(self)


def _credentials() -> tuple[str, str]:
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_API_SECRET") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        raise MarketDataError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_API_SECRET "
            "(or APCA_API_KEY_ID/APCA_API_SECRET_KEY)."
        )
    return key, secret


def fetch_daily_bars(
    symbols: Iterable[str],
    session_date: str | date,
    lookback_days: int = 550,
    feed: str = "iex",
) -> dict[str, pd.DataFrame]:
    """Fetch split/dividend-adjusted daily OHLCV for current manifest symbols."""
    symbols = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    if not symbols:
        return {}
    key, secret = _credentials()
    end = pd.Timestamp(session_date).normalize()
    start = end - pd.Timedelta(days=lookback_days)
    params = {
        "symbols": ",".join(symbols),
        "timeframe": "1Day",
        "start": start.strftime("%Y-%m-%dT00:00:00Z"),
        "end": (end + pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"),
        "adjustment": "all",
        "feed": feed,
        "limit": 10000,
        "sort": "asc",
    }
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    raw: dict[str, list[dict]] = {s: [] for s in symbols}
    page_token = None
    while True:
        if page_token:
            params["page_token"] = page_token
        r = requests.get(ALPACA_DATA_URL, params=params, headers=headers, timeout=30)
        if r.status_code != 200:
            raise MarketDataError(f"Alpaca bars request failed ({r.status_code}): {r.text[:300]}")
        payload = r.json()
        for symbol, bars in (payload.get("bars") or {}).items():
            raw.setdefault(symbol.upper(), []).extend(bars or [])
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    result: dict[str, pd.DataFrame] = {}
    for symbol in symbols:
        rows = raw.get(symbol, [])
        if not rows:
            continue
        df = pd.DataFrame(rows).rename(columns={
            "t": "Date", "o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"
        })
        df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None).dt.normalize()
        df = df.set_index("Date")[["Open", "High", "Low", "Close", "Volume"]].sort_index()
        df = df[~df.index.duplicated(keep="last")]
        result[symbol] = df
    return result


def _tightness(df: pd.DataFrame, days: int) -> float:
    w = df.tail(days)
    if len(w) < days or w["Close"].iloc[-1] <= 0:
        return float("nan")
    return float((w["High"].max() - w["Low"].min()) / w["Close"].iloc[-1] * 100)


def calculate_metrics(symbol: str, df: pd.DataFrame, session_date: str | date) -> ChartMetrics:
    end = pd.Timestamp(session_date).normalize()
    df = df.loc[df.index <= end].copy()
    if len(df) < 200:
        raise MarketDataError(f"{symbol}: only {len(df)} daily bars; 200 required")
    if df.index[-1] != end:
        raise MarketDataError(f"{symbol}: latest bar is {df.index[-1].date()}, expected {end.date()}")

    close = df["Close"]
    volume = df["Volume"]
    sma21 = close.rolling(21).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    prev_close = close.shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr14 = tr.rolling(14).mean()
    avg_vol50 = volume.rolling(50).mean()
    dollar_vol50 = (close * volume).rolling(50).mean()
    one_year = df.tail(252)
    high52 = float(one_year["High"].max())
    price = float(close.iloc[-1])

    daily_change = close.pct_change()
    up_vol = volume.where(daily_change > 0, 0).tail(50).sum()
    down_vol = volume.where(daily_change < 0, 0).tail(50).sum()
    ud_ratio = None if down_vol <= 0 else float(up_vol / down_vol)
    recent = df.tail(26).copy()
    recent_change = recent["Close"].pct_change()
    recent_prev_vol = recent["Volume"].shift(1)
    accumulation = int(((recent_change >= 0.002) & (recent["Volume"] > recent_prev_vol)).sum())
    distribution = int(((recent_change <= -0.002) & (recent["Volume"] > recent_prev_vol)).sum())

    return ChartMetrics(
        ticker=symbol,
        session_date=end.date().isoformat(),
        current_price=price,
        sma21=float(sma21.iloc[-1]),
        sma50=float(sma50.iloc[-1]),
        sma200=float(sma200.iloc[-1]),
        atr14=float(atr14.iloc[-1]),
        atr_pct=float(atr14.iloc[-1] / price * 100),
        avg_volume_50=float(avg_vol50.iloc[-1]),
        relative_volume=float(volume.iloc[-1] / avg_vol50.iloc[-1]),
        avg_dollar_volume_50=float(dollar_vol50.iloc[-1]),
        high_52w=high52,
        pct_from_52w_high=float((price / high52 - 1) * 100),
        pct_from_sma50=float((price / sma50.iloc[-1] - 1) * 100),
        pct_from_sma200=float((price / sma200.iloc[-1] - 1) * 100),
        tightness_5d_pct=_tightness(df, 5),
        tightness_10d_pct=_tightness(df, 10),
        tightness_15d_pct=_tightness(df, 15),
        up_down_volume_ratio_50=ud_ratio,
        accumulation_days_25=accumulation,
        distribution_days_25=distribution,
    )


def quantitative_gate(m: ChartMetrics) -> tuple[str, list[str]]:
    """Conservative pre-chart gate. This ranks/rejects; it never declares ACTIONABLE."""
    reasons: list[str] = []
    if m.avg_dollar_volume_50 < 20_000_000:
        reasons.append("average dollar volume below $20M")
    if m.current_price < m.sma200:
        reasons.append("price below 200-day moving average")
    if m.current_price < m.sma50:
        reasons.append("price below 50-day moving average")
    if m.pct_from_52w_high < -25:
        reasons.append("more than 25% below 52-week high")
    if m.pct_from_sma50 > 20:
        reasons.append("more than 20% above 50-day moving average")
    if reasons:
        return "DEPRIORITIZE", reasons
    score = 0
    score += int(m.current_price >= m.sma21 >= m.sma50 >= m.sma200)
    score += int(m.pct_from_52w_high >= -10)
    score += int(m.relative_volume >= 1.0)
    score += int(m.tightness_10d_pct <= 12)
    score += int(m.accumulation_days_25 >= m.distribution_days_25)
    return ("CHART_REVIEW_PRIORITY" if score >= 3 else "CHART_REVIEW"), reasons


def render_chart(
    symbol: str,
    df: pd.DataFrame,
    session_date: str | date,
    output_path: str | Path,
    pivot: Optional[float] = None,
    stop: Optional[float] = None,
    sessions: int = 180,
) -> Path:
    """Render a standardized daily CANSLIM review chart from verified OHLCV."""
    end = pd.Timestamp(session_date).normalize()
    data = df.loc[df.index <= end].copy()
    if len(data) < 200 or data.index[-1] != end:
        raise MarketDataError(f"{symbol}: chart data failed freshness/history gate")
    data["MA21"] = data["Close"].rolling(21).mean()
    data["MA50"] = data["Close"].rolling(50).mean()
    data["MA200"] = data["Close"].rolling(200).mean()
    view = data.tail(sessions).copy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hlines = []
    if pivot is not None:
        hlines.append(float(pivot))
    if stop is not None:
        hlines.append(float(stop))
    kwargs = {}
    if hlines:
        kwargs["hlines"] = dict(hlines=hlines, linestyle="--", linewidths=1.0)
    mpf.plot(
        view,
        type="candle",
        volume=True,
        mav=(21, 50, 200),
        title=f"{symbol} — Daily through {end.date().isoformat()}",
        ylabel="Price",
        ylabel_lower="Volume",
        style="yahoo",
        figsize=(14, 8),
        tight_layout=True,
        savefig=dict(fname=str(output_path), dpi=160, bbox_inches="tight"),
        **kwargs,
    )
    plt.close("all")
    return output_path


def build_chart_package(
    symbols: Iterable[str],
    session_date: str | date,
    output_dir: str | Path,
    feed: str = "iex",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Fetch all manifest symbols, compute metrics, and chart every valid symbol.

    Returns a metrics dataframe plus per-symbol validation errors. A downstream
    recommendation engine must require a successfully rendered chart before it can
    classify a symbol ACTIONABLE.
    """
    symbols = sorted({str(s).strip().upper() for s in symbols if str(s).strip()})
    output_dir = Path(output_dir)
    chart_dir = output_dir / "charts"
    bars = fetch_daily_bars(symbols, session_date, feed=feed)
    rows: list[dict] = []
    errors: dict[str, str] = {}
    for symbol in symbols:
        df = bars.get(symbol)
        if df is None:
            errors[symbol] = "no daily bars returned"
            continue
        try:
            m = calculate_metrics(symbol, df, session_date)
            gate, reasons = quantitative_gate(m)
            chart = render_chart(symbol, df, session_date, chart_dir / f"{symbol}.png")
            row = m.to_dict()
            row.update({"quantitative_gate": gate, "gate_reasons": "; ".join(reasons), "chart_path": str(chart)})
            rows.append(row)
        except Exception as exc:
            errors[symbol] = str(exc)
    metrics = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_json(output_dir / "chart_metrics.json", orient="records", indent=2)
    pd.DataFrame([{"ticker": k, "error": v} for k, v in errors.items()]).to_json(
        output_dir / "chart_errors.json", orient="records", indent=2
    )
    return metrics, errors
