#!/usr/bin/env python3
"""Trade-by-trade shakeout and re-entry study using FMP daily OHLCV data."""

from __future__ import annotations

import csv
import json
import math
import os
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


AS_OF = "2026-08-11"
START = "2025-08-01"
TRADE_START = "2026-06-01"
SOURCE_CSV = Path("/Users/brandennevius/.codex/attachments/c17d8ace-9254-4254-b0b0-440f20000ba3/branden-trade-log-2026-08-11.csv")
OUTPUT_DIR = Path("/Users/brandennevius/Desktop/TradingDashboard/analysis/shakeout_reentry_2026-08-11")
CACHE_DIR = OUTPUT_DIR / "fmp_daily"

# Exact exits shown in the journal table; the CSV export omits this column.
# Keyed by (symbol, open_date, entry rounded to 4 decimals).
EXACT_EXITS = {
    ("LGIH", "2026-08-05", 61.9700): 54.44, ("RELY", "2026-07-17", 24.8400): 23.32,
    ("S", "2026-07-16", 19.9500): 19.07, ("LUV", "2026-07-15", 49.1799): 48.10,
    ("SEDG", "2026-07-15", 58.0000): 51.76, ("AFRM", "2026-07-14", 84.4400): 79.43,
    ("ANET", "2026-07-14", 181.3000): 171.90, ("DELL", "2026-07-13", 440.0230): 450.01,
    ("NTAP", "2026-07-13", 166.2500): 166.47, ("DXCM", "2026-07-10", 75.6406): 74.85,
    ("NTAP", "2026-07-09", 168.7212): 169.46, ("W", "2026-07-09", 89.0207): 89.37,
    ("HOOD", "2026-07-08", 113.6216): 109.28, ("GLW", "2026-07-02", 217.3700): 211.97,
    ("NTAP", "2026-07-02", 161.5187): 160.94, ("INTC", "2026-07-01", 132.6846): 123.59,
    ("SNOW", "2026-07-01", 259.7800): 260.04, ("UPST", "2026-06-29", 35.0192): 33.66,
    ("HOOD", "2026-06-26", 104.6212): 111.41, ("MCHP", "2026-06-25", 95.7190): 88.42,
    ("HIMS", "2026-06-24", 32.6650): 30.89, ("W", "2026-06-24", 94.1636): 88.10,
    ("SEDG", "2026-06-22", 59.8625): 51.01, ("INTC", "2026-06-18", 132.1030): 132.88,
    ("SNOW", "2026-06-18", 232.7507): 229.43, ("TWLO", "2026-06-18", 187.2487): 187.50,
    ("HIMS", "2026-06-17", 33.0841): 33.01, ("LABU", "2026-06-17", 215.4567): 274.61,
    ("APH", "2026-06-16", 167.2409): 164.77, ("SNOW", "2026-06-16", 245.4100): 222.28,
    ("HOOD", "2026-06-12", 97.6640): 98.59, ("APH", "2026-06-10", 158.8000): 151.61,
    ("ARM", "2026-06-10", 356.5387): 393.14, ("CRDO", "2026-06-10", 253.8800): 265.42,
    ("CRDO", "2026-06-09", 236.8600): 235.05, ("CRDO", "2026-06-09", 235.7911): 210.77,
    ("LLY", "2026-06-04", 1164.8077): 1141.86, ("TWLO", "2026-06-01", 210.0321): 221.76,
    ("MDT", "2026-07-10", 83.3303): 81.45, ("UPRO", "2026-07-10", 144.3614): 138.70,
    ("MDT", "2026-07-09", 82.6387): 82.69, ("MDT", "2026-07-02", 82.3948): 82.45,
    ("BABA", "2026-06-16", 109.5391): 102.26, ("NOC", "2026-06-15", 545.3225): 531.43,
    ("BSX", "2026-06-12", 46.9352): 45.63, ("ORCL", "2026-06-11", 183.2954): 168.19,
    ("ORCL", "2026-06-11", 179.8100): 177.80, ("UPS", "2026-06-11", 105.6697): 105.66,
    ("EGO", "2026-06-10", 29.0877): 33.19, ("COST", "2026-06-08", 973.5025): 968.00,
    ("LOW", "2026-06-08", 214.0386): 220.00, ("NVO", "2026-06-08", 47.1686): 49.12,
    ("NOC", "2026-06-05", 545.2648): 539.87, ("T", "2026-06-03", 23.7822): 23.45,
    ("COST", "2026-06-02", 955.2204): 971.45,
}


def fnum(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value):
    return "-" if value is None else f"{value:.1%}"


def median(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return statistics.median(values) if values else None


def fetch_symbol(symbol, api_key):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe = symbol.replace("/", "_").replace(".", "_")
    cache = CACHE_DIR / f"{safe}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    params = urllib.parse.urlencode({"symbol": symbol, "from": START, "to": AS_OF, "apikey": api_key})
    url = f"https://financialmodelingprep.com/stable/historical-price-eod/full?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "trade-shakeout-study/1.0"})
    with urllib.request.urlopen(req, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected FMP response for {symbol}: {payload}")
    cache.write_text(json.dumps(payload, indent=2))
    time.sleep(0.06)
    return payload


def indicators(raw):
    bars = sorted(raw, key=lambda x: x["date"])
    closes = []
    trs = []
    prev_close = None
    for bar in bars:
        close = fnum(bar.get("close"))
        high = fnum(bar.get("high"))
        low = fnum(bar.get("low"))
        if None in (close, high, low):
            continue
        closes.append(close)
        tr = high - low if prev_close is None else max(high - low, abs(high - prev_close), abs(low - prev_close))
        trs.append(tr)
        bar["close"] = close
        bar["high"] = high
        bar["low"] = low
        bar["open"] = fnum(bar.get("open"))
        bar["volume"] = fnum(bar.get("volume"))
        for n in (10, 20, 50, 200):
            bar[f"sma{n}"] = sum(closes[-n:]) / n if len(closes) >= n else None
        bar["atr14"] = sum(trs[-14:]) / 14 if len(trs) >= 14 else None
        prev_close = close
    return bars


def first_index_on_or_after(bars, day):
    for i, bar in enumerate(bars):
        if bar["date"] >= day:
            return i
    return None


def first_trigger(bars, start_i, end_i, kind, original_entry, exit_high, exit_low):
    for i in range(start_i, min(end_i + 1, len(bars))):
        b = bars[i]
        p = bars[i - 1] if i > 0 else None
        if p is None:
            continue
        if kind == "entry_reclaim":
            hit = b["close"] > original_entry and p["close"] <= original_entry
        elif kind == "entry_reclaim_strength":
            hit = b["close"] > original_entry and p["close"] <= original_entry and b["close"] > p["high"]
        elif kind == "sma10_reclaim_strength":
            hit = (b.get("sma10") is not None and p.get("sma10") is not None and
                   p["close"] <= p["sma10"] and b["close"] > b["sma10"] and b["close"] > p["high"])
        elif kind == "sma20_reclaim_strength":
            hit = (b.get("sma20") is not None and p.get("sma20") is not None and
                   p["close"] <= p["sma20"] and b["close"] > b["sma20"] and b["close"] > p["high"])
        elif kind == "sma20_reclaim":
            hit = (b.get("sma20") is not None and p.get("sma20") is not None and
                   p["close"] <= p["sma20"] and b["close"] > b["sma20"])
        elif kind == "exit_low_undercut_reclaim":
            hit = b["low"] < exit_low and b["close"] > exit_low and b["close"] > b["open"]
        elif kind == "exit_high_break":
            hit = b["close"] > exit_high
        elif kind == "fresh_5d_high":
            prior = bars[max(0, i - 5):i]
            hit = len(prior) == 5 and b.get("sma20") is not None and b["close"] > max(x["high"] for x in prior) and b["close"] > b["sma20"]
        else:
            hit = False
        if hit:
            return i
    return None


def outcome_after_trigger(bars, trigger_i, atr, horizon=10):
    if trigger_i is None:
        return None
    end = min(len(bars), trigger_i + horizon + 1)
    future = bars[trigger_i + 1:end]
    if not future:
        return {"gain": None, "drawdown": None, "success_2atr_before_1atr": None, "sessions": 0}
    price = bars[trigger_i]["close"]
    max_high = max(b["high"] for b in future)
    min_low = min(b["low"] for b in future)
    target = price + 2 * atr
    stop = price - atr
    success = None
    for b in future:
        if b["low"] <= stop and b["high"] >= target:
            success = False
            break
        if b["low"] <= stop:
            success = False
            break
        if b["high"] >= target:
            success = True
            break
    return {
        "gain": max_high / price - 1,
        "drawdown": min_low / price - 1,
        "success_2atr_before_1atr": success,
        "sessions": len(future),
    }


def stop_before_target(path, stop_fn, target):
    for b in path:
        stop = stop_fn(b)
        stop_hit = stop is not None and b["low"] <= stop
        target_hit = b["high"] >= target
        if stop_hit and target_hit:
            return True
        if stop_hit:
            return True
        if target_hit:
            return False
    return None


def close_stop_before_target(path, stop_fn, target):
    for b in path:
        stop = stop_fn(b)
        if stop is not None and b["close"] <= stop:
            return True
        if b["high"] >= target:
            return False
    return None


def simulate_20d_stop(path, entry, atr, stop, mode):
    """Fixed-dollar-risk result: size is scaled to the distance from entry to stop."""
    if stop >= entry or not path:
        return None
    exit_price = path[-1]["close"]
    for b in path:
        if mode == "hard" and b["low"] <= stop:
            exit_price = stop  # stop-price fill; gaps/slippage are not modeled
            break
        if mode == "close" and b["close"] <= stop:
            exit_price = b["close"]
            break
    return (exit_price - entry) / (entry - stop)


def analyze_trade(trade, bars):
    entry_i = first_index_on_or_after(bars, trade["open_date"])
    exit_i = first_index_on_or_after(bars, trade["close_date"])
    if entry_i is None or exit_i is None or bars[entry_i]["date"] != trade["open_date"] or bars[exit_i]["date"] != trade["close_date"]:
        return None, "missing aligned market date"
    entry_bar = bars[entry_i]
    exit_bar = bars[exit_i]
    atr = entry_bar.get("atr14")
    entry = fnum(trade["entry"])
    if not atr or not entry:
        return None, "missing ATR or entry"
    market_match = entry_bar["low"] * 0.97 <= entry <= entry_bar["high"] * 1.03
    available_after_exit = max(0, len(bars) - exit_i - 1)
    post = bars[exit_i + 1:min(len(bars), exit_i + 21)]
    post_to_asof = bars[exit_i + 1:]
    max_post_high = max((b["high"] for b in post), default=None)
    min_post_low = min((b["low"] for b in post), default=None)
    max_asof_high = max((b["high"] for b in post_to_asof), default=None)
    asof_bar = bars[-1]
    target = entry + 2 * atr
    path20 = bars[entry_i + 1:min(len(bars), entry_i + 21)]
    prior5 = bars[max(0, entry_i - 5):entry_i + 1]
    structure_stop = min(b["low"] for b in prior5) - 0.25 * atr
    exact_exit = EXACT_EXITS.get((trade["symbol"], trade["open_date"], round(entry, 4)))
    pnl = fnum(trade["net_return"])
    risk_dollars = fnum(trade["risk"])
    shares = None
    planned_stop_distance = None
    if exact_exit is not None and pnl is not None and risk_dollars and abs(exact_exit - entry) > 0.005:
        shares = abs(pnl / (exact_exit - entry))
        if shares > 0:
            planned_stop_distance = risk_dollars / shares
    result = {
        "symbol": trade["symbol"], "setup": trade["setup"], "status": trade["status"],
        "open_date": trade["open_date"], "close_date": trade["close_date"],
        "entry": entry, "exit_day_close": exit_bar["close"], "realized_r": fnum(trade["r"]),
        "exact_exit": exact_exit, "inferred_shares": shares,
        "inferred_planned_stop_price": None if planned_stop_distance is None else entry - planned_stop_distance,
        "inferred_planned_stop_atr": None if planned_stop_distance is None else planned_stop_distance / atr,
        "inferred_planned_stop_pct": None if planned_stop_distance is None else -planned_stop_distance / entry,
        "grade": trade["grade"], "atr14_entry": atr, "atr_pct": atr / entry,
        "entry_price_matches_fmp": market_match,
        "available_sessions_after_exit": available_after_exit,
        "available_sessions_from_entry": max(0, len(bars) - entry_i - 1),
        "post_20d_max_gain_from_entry": None if max_post_high is None else max_post_high / entry - 1,
        "post_20d_max_gain_from_exit_close": None if max_post_high is None else max_post_high / exit_bar["close"] - 1,
        "post_20d_max_drawdown_from_exit_close": None if min_post_low is None else min_post_low / exit_bar["close"] - 1,
        "asof_market_date": asof_bar["date"], "asof_close": asof_bar["close"],
        "asof_gain_from_entry": asof_bar["close"] / entry - 1,
        "asof_gain_from_exact_exit": None if exact_exit is None else asof_bar["close"] / exact_exit - 1,
        "max_gain_from_entry_through_asof": None if max_asof_high is None else max_asof_high / entry - 1,
        "reached_2atr_after_exit": None if max_post_high is None else max_post_high >= target,
        "reached_5pct_after_exit": None if max_post_high is None else max_post_high >= entry * 1.05,
        "reached_10pct_after_exit": None if max_post_high is None else max_post_high >= entry * 1.10,
        "structure_stop_pct": structure_stop / entry - 1,
    }
    for k in (1.0, 1.5, 2.0, 2.5, 3.0):
        hard = entry - k * atr
        result[f"hard_{k:g}atr_stopped_before_2atr"] = stop_before_target(path20, lambda b, s=hard: s, target)
        result[f"close_{k:g}atr_stopped_before_2atr"] = close_stop_before_target(path20, lambda b, s=hard: s, target)
        result[f"hard_{k:g}atr_20d_r"] = simulate_20d_stop(path20, entry, atr, hard, "hard")
        result[f"close_{k:g}atr_20d_r"] = simulate_20d_stop(path20, entry, atr, hard, "close")
    result["structure_stopped_before_2atr"] = stop_before_target(path20, lambda b: structure_stop, target)
    below20 = 0
    two_close_result = None
    for b in path20:
        if b.get("sma20") is not None and b["close"] < b["sma20"]:
            below20 += 1
        else:
            below20 = 0
        if below20 >= 2:
            two_close_result = True
            break
        if b["high"] >= target:
            two_close_result = False
            break
    result["two_closes_below_sma20_before_2atr"] = two_close_result
    trigger_start = exit_i + 1
    trigger_end = min(len(bars) - 1, exit_i + 20)
    for rule in ("entry_reclaim", "entry_reclaim_strength", "sma10_reclaim_strength", "sma20_reclaim", "sma20_reclaim_strength", "exit_low_undercut_reclaim", "exit_high_break", "fresh_5d_high"):
        ti = first_trigger(bars, trigger_start, trigger_end, rule, entry, exit_bar["high"], exit_bar["low"])
        outcome = outcome_after_trigger(bars, ti, atr)
        result[f"{rule}_date"] = bars[ti]["date"] if ti is not None else None
        result[f"{rule}_price"] = bars[ti]["close"] if ti is not None else None
        result[f"{rule}_gain_10d"] = None if outcome is None else outcome["gain"]
        result[f"{rule}_drawdown_10d"] = None if outcome is None else outcome["drawdown"]
        result[f"{rule}_success"] = None if outcome is None else outcome["success_2atr_before_1atr"]
        result[f"{rule}_forward_sessions"] = 0 if outcome is None else outcome["sessions"]
    return result, None


def rule_summary(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return {"eligible": len(rows), "observations": len(vals), "unresolved": len(rows) - len(vals),
            "stopped_before_target": sum(v is True for v in vals),
            "survived_to_target": sum(v is False for v in vals),
            "survival_rate": (sum(v is False for v in vals) / len(vals)) if vals else None}


def return_summary(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return {"observations": len(vals), "mean_r": sum(vals) / len(vals) if vals else None,
            "median_r": median(vals), "win_rate": sum(v > 0 for v in vals) / len(vals) if vals else None,
            "worst_r": min(vals) if vals else None, "best_r": max(vals) if vals else None}


def reentry_summary(rows, rule):
    triggered = [r for r in rows if r.get(f"{rule}_date")]
    mature = [r for r in triggered if r.get(f"{rule}_forward_sessions", 0) >= 5]
    outcomes = [r[f"{rule}_success"] for r in mature if r.get(f"{rule}_success") is not None]
    return {
        "eligible": len(rows), "triggered": len(triggered), "mature_triggers": len(mature),
        "successful": sum(v is True for v in outcomes), "failed": sum(v is False for v in outcomes),
        "success_rate": sum(v is True for v in outcomes) / len(outcomes) if outcomes else None,
        "median_10d_gain": median(r.get(f"{rule}_gain_10d") for r in mature),
        "median_10d_drawdown": median(r.get(f"{rule}_drawdown_10d") for r in mature),
    }


def main():
    api_key = os.environ.get("FMP_API_KEY")
    if not api_key:
        raise SystemExit("FMP_API_KEY is required")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with SOURCE_CSV.open(newline="") as f:
        trades = list(csv.DictReader(f))
    eligible = [t for t in trades if t["side"] == "LONG" and t["close_date"] and t["open_date"] >= TRADE_START and "/" not in t["symbol"] and not t["symbol"].startswith(".")]
    symbols = sorted({t["symbol"] for t in eligible})
    market = {}
    errors = {}
    for symbol in symbols:
        try:
            market[symbol] = indicators(fetch_symbol(symbol, api_key))
        except Exception as exc:
            errors[symbol] = str(exc)
    rows = []
    skipped = []
    for trade in eligible:
        if trade["symbol"] not in market:
            skipped.append({"symbol": trade["symbol"], "reason": errors.get(trade["symbol"], "no data")})
            continue
        result, error = analyze_trade(trade, market[trade["symbol"]])
        if result:
            rows.append(result)
        else:
            skipped.append({"symbol": trade["symbol"], "reason": error})
    valid = [r for r in rows if r["entry_price_matches_fmp"]]
    mature = [r for r in valid if r["available_sessions_after_exit"] >= 10]
    canslim_mature = [r for r in mature if r["setup"] == "CANSLIM"]
    canslim_20d = [r for r in valid if r["setup"] == "CANSLIM" and r["available_sessions_from_entry"] >= 20]
    losers_be = [r for r in canslim_mature if r["status"] in ("LOSS", "BREAKEVEN")]
    stop_rules = {}
    for key in ("hard_1atr_stopped_before_2atr", "hard_1.5atr_stopped_before_2atr", "hard_2atr_stopped_before_2atr",
                "hard_2.5atr_stopped_before_2atr", "hard_3atr_stopped_before_2atr", "close_1.5atr_stopped_before_2atr",
                "close_2atr_stopped_before_2atr", "close_2.5atr_stopped_before_2atr", "structure_stopped_before_2atr",
                "two_closes_below_sma20_before_2atr"):
        stop_rules[key] = rule_summary(canslim_20d, key)
    stop_returns = {}
    for key in ("hard_1atr_20d_r", "hard_1.5atr_20d_r", "hard_2atr_20d_r", "hard_2.5atr_20d_r", "hard_3atr_20d_r",
                "close_1atr_20d_r", "close_1.5atr_20d_r", "close_2atr_20d_r", "close_2.5atr_20d_r", "close_3atr_20d_r"):
        stop_returns[key] = return_summary(canslim_20d, key)
    re_rules = {rule: reentry_summary(losers_be, rule) for rule in
                ("entry_reclaim", "entry_reclaim_strength", "sma10_reclaim_strength", "sma20_reclaim", "sma20_reclaim_strength", "exit_low_undercut_reclaim", "exit_high_break", "fresh_5d_high")}
    summary = {
        "as_of": AS_OF, "source_rows": len(trades), "eligible_closed_equity_longs_since_june": len(eligible),
        "analyzed": len(rows), "market_matched": len(valid), "mature_10_sessions": len(mature),
        "mature_canslim": len(canslim_mature), "mature_canslim_loss_or_breakeven": len(losers_be),
        "canslim_with_20_sessions_from_entry": len(canslim_20d),
        "unmatched_entry_prices": [r["symbol"] + " " + r["open_date"] for r in rows if not r["entry_price_matches_fmp"]],
        "skipped": skipped, "fetch_errors": errors,
        "post_exit": {
            "2atr_worked_count": sum(r["reached_2atr_after_exit"] is True for r in losers_be),
            "5pct_count": sum(r["reached_5pct_after_exit"] is True for r in losers_be),
            "10pct_count": sum(r["reached_10pct_after_exit"] is True for r in losers_be),
            "median_max_gain_from_entry": median(r["post_20d_max_gain_from_entry"] for r in losers_be),
            "median_max_gain_from_exit_close": median(r["post_20d_max_gain_from_exit_close"] for r in losers_be),
            "unique_symbols_reaching_2atr": sorted({r["symbol"] for r in losers_be if r["reached_2atr_after_exit"] is True}),
            "median_inferred_planned_stop_atr": median(r["inferred_planned_stop_atr"] for r in losers_be if r.get("inferred_planned_stop_atr") is not None and r["inferred_planned_stop_atr"] < 10),
        },
        "actual_canslim_realized_r": return_summary(canslim_20d, "realized_r"),
        "stop_rules": stop_rules, "stop_20d_returns": stop_returns, "reentry_rules": re_rules,
    }
    with (OUTPUT_DIR / "trade_level_results.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
