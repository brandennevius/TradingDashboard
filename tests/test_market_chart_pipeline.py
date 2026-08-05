import pandas as pd
import pytest

from market_chart_pipeline import MarketDataError, calculate_metrics, quantitative_gate


def make_bars(periods=260, end="2026-08-04"):
    idx = pd.bdate_range(end=end, periods=periods)
    close = pd.Series([50 + i * 0.2 for i in range(periods)], index=idx)
    return pd.DataFrame({
        "Open": close - 0.1,
        "High": close + 0.5,
        "Low": close - 0.5,
        "Close": close,
        "Volume": 1_000_000,
    }, index=idx)


def test_metrics_require_session_freshness():
    bars = make_bars(end="2026-08-03")
    with pytest.raises(MarketDataError):
        calculate_metrics("TEST", bars, "2026-08-04")


def test_metrics_require_200_bars():
    bars = make_bars(periods=199)
    with pytest.raises(MarketDataError):
        calculate_metrics("TEST", bars, "2026-08-04")


def test_constructive_trend_reaches_chart_review():
    m = calculate_metrics("TEST", make_bars(), "2026-08-04")
    gate, reasons = quantitative_gate(m)
    assert gate in {"CHART_REVIEW", "CHART_REVIEW_PRIORITY"}
    assert not reasons


def test_below_200_day_is_deprioritized():
    bars = make_bars()
    bars.loc[bars.index[-1], ["Open", "High", "Low", "Close"]] = [20, 21, 19, 20]
    m = calculate_metrics("TEST", bars, "2026-08-04")
    gate, reasons = quantitative_gate(m)
    assert gate == "DEPRIORITIZE"
    assert any("200-day" in reason for reason in reasons)
