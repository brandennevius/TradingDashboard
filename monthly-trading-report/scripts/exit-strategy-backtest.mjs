import fs from "node:fs";
import path from "node:path";

const tradesPath = process.argv[2] || "/private/tmp/branden-exit-lab-trades.json";
const candlesDirectory = process.argv[3] || "/private/tmp/exit-candles";
const outputPath = process.argv[4] || "/private/tmp/branden-exit-strategy-results.json";

const rawTrades = JSON.parse(fs.readFileSync(tradesPath, "utf8"));
const cleanSymbol = (value) => String(value || "").trim().replace(/^#/, "").toUpperCase();
const numberValue = (value) => (Number.isFinite(Number(value)) ? Number(value) : 0);
const timestamp = (execution) => `${execution.date || ""}T${execution.time || "00:00:00"}`;

function weightedAverage(items) {
  const shares = items.reduce((sum, item) => sum + numberValue(item.shares), 0);
  return shares
    ? items.reduce((sum, item) => sum + numberValue(item.price) * numberValue(item.shares), 0) / shares
    : 0;
}

function normalizedEntry(trade) {
  const executions = [...(trade.executions || [])].sort((a, b) => timestamp(a).localeCompare(timestamp(b)));
  const lots = [];
  let openShares = 0;
  let maxShares = 0;
  let maxEntryPrice = 0;
  let maxEntryDate = trade.entry_date;

  for (const execution of executions) {
    let shares = Math.max(0, numberValue(execution.shares));
    if (execution.type === "ENTRY") {
      lots.push({ shares, price: numberValue(execution.price) });
      openShares += shares;
    } else {
      openShares -= shares;
      while (shares > 0 && lots.length) {
        const lot = lots[0];
        const matched = Math.min(shares, lot.shares);
        lot.shares -= matched;
        shares -= matched;
        if (lot.shares <= 1e-8) lots.shift();
      }
    }

    if (openShares > maxShares + 1e-8) {
      maxShares = openShares;
      maxEntryPrice = weightedAverage(lots);
      maxEntryDate = execution.date || maxEntryDate;
    }
  }

  const entryShares = executions
    .filter((execution) => execution.type === "ENTRY")
    .reduce((sum, execution) => sum + numberValue(execution.shares), 0);
  const exitShares = executions
    .filter((execution) => execution.type === "EXIT")
    .reduce((sum, execution) => sum + numberValue(execution.shares), 0);
  const finalExitDate = executions
    .filter((execution) => execution.type === "EXIT")
    .map((execution) => execution.date)
    .sort()
    .at(-1);

  return {
    entryShares,
    exitShares,
    finalExitDate,
    maxShares,
    entryPrice: maxEntryPrice || numberValue(trade.avg_entry),
    entryDate: maxEntryDate || trade.entry_date
  };
}

function loadCandles(symbol) {
  const payload = JSON.parse(fs.readFileSync(path.join(candlesDirectory, `${symbol}.json`), "utf8"));
  const candles = payload.candles || [];
  let ema8 = null;
  let ema21 = null;
  const ema8Multiplier = 2 / 9;
  const ema21Multiplier = 2 / 22;
  return candles.map((candle) => {
    ema8 = ema8 === null ? numberValue(candle.close) : numberValue(candle.close) * ema8Multiplier + ema8 * (1 - ema8Multiplier);
    ema21 = ema21 === null ? numberValue(candle.close) : numberValue(candle.close) * ema21Multiplier + ema21 * (1 - ema21Multiplier);
    return { ...candle, ema8, ema21 };
  });
}

function resultR(trade, exitPrice, entryPrice, riskPerShare) {
  const direction = trade.side === "SHORT" ? -1 : 1;
  return (direction * (exitPrice - entryPrice)) / riskPerShare - numberValue(trade.commission) / numberValue(trade.risk);
}

function simulateTarget(trade, candles, entry, targetPrice, label) {
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  const stopPrice = entry.entryPrice - direction * riskPerShare;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    const targetHit = direction === 1 ? candle.high >= targetPrice : candle.low <= targetPrice;
    if (stopHit) {
      return { r: resultR(trade, stopPrice, entry.entryPrice, riskPerShare), exitDate: candle.time, reason: "stop" };
    }
    if (targetHit) {
      return { r: resultR(trade, targetPrice, entry.entryPrice, riskPerShare), exitDate: candle.time, reason: label };
    }
  }

  const last = path.at(-1) || candles.filter((candle) => candle.time >= entry.entryDate).at(-1);
  return last
    ? { r: resultR(trade, last.close, entry.entryPrice, riskPerShare), exitDate: last.time, reason: "marked" }
    : { r: 0, exitDate: entry.entryDate, reason: "no-data" };
}

function simulateEma8(trade, candles, entry) {
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  const stopPrice = entry.entryPrice - direction * riskPerShare;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      return { r: resultR(trade, stopPrice, entry.entryPrice, riskPerShare), exitDate: candle.time, reason: "stop" };
    }
    const emaExit = direction === 1 ? candle.close < candle.ema8 : candle.close > candle.ema8;
    if (emaExit) {
      return { r: resultR(trade, candle.close, entry.entryPrice, riskPerShare), exitDate: candle.time, reason: "ema8-close" };
    }
  }

  const last = path.at(-1) || candles.filter((candle) => candle.time >= entry.entryDate).at(-1);
  return last
    ? { r: resultR(trade, last.close, entry.entryPrice, riskPerShare), exitDate: last.time, reason: "marked" }
    : { r: 0, exitDate: entry.entryDate, reason: "no-data" };
}

function simulateActivatedEma8(trade, candles, entry, activationR) {
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  let stopPrice = entry.entryPrice - direction * riskPerShare;
  const activationPrice = entry.entryPrice + direction * activationR * riskPerShare;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);
  let activated = false;

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      return {
        r: resultR(trade, stopPrice, entry.entryPrice, riskPerShare),
        exitDate: candle.time,
        reason: activated ? `${activationR}r-then-breakeven` : "stop"
      };
    }

    if (!activated) {
      const activationHit = direction === 1 ? candle.high >= activationPrice : candle.low <= activationPrice;
      if (activationHit) {
        activated = true;
        stopPrice = entry.entryPrice;
      }
    }

    if (activated) {
      const emaExit = direction === 1 ? candle.close < candle.ema8 : candle.close > candle.ema8;
      if (emaExit) {
        return {
          r: resultR(trade, candle.close, entry.entryPrice, riskPerShare),
          exitDate: candle.time,
          reason: `${activationR}r-then-ema8`
        };
      }
    }
  }

  const last = path.at(-1) || candles.filter((candle) => candle.time >= entry.entryDate).at(-1);
  return last
    ? { r: resultR(trade, last.close, entry.entryPrice, riskPerShare), exitDate: last.time, reason: "marked" }
    : { r: 0, exitDate: entry.entryDate, reason: "no-data" };
}

function simulateTwoRThenEma8(trade, candles, entry) {
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  let stopPrice = entry.entryPrice - direction * riskPerShare;
  const targetPrice = entry.entryPrice + direction * 2 * riskPerShare;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);
  let firstHalfR = null;

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      const stopR = resultR(trade, stopPrice, entry.entryPrice, riskPerShare);
      return {
        r: firstHalfR === null ? stopR : firstHalfR * 0.5 + stopR * 0.5,
        exitDate: candle.time,
        reason: firstHalfR === null ? "stop" : "2r-then-stop"
      };
    }

    if (firstHalfR === null) {
      const targetHit = direction === 1 ? candle.high >= targetPrice : candle.low <= targetPrice;
      if (targetHit) {
        firstHalfR = resultR(trade, targetPrice, entry.entryPrice, riskPerShare);
        stopPrice = entry.entryPrice;
      }
      continue;
    }

    const emaExit = direction === 1 ? candle.close < candle.ema8 : candle.close > candle.ema8;
    if (emaExit) {
      const trailingR = resultR(trade, candle.close, entry.entryPrice, riskPerShare);
      return { r: firstHalfR * 0.5 + trailingR * 0.5, exitDate: candle.time, reason: "2r-then-ema8" };
    }
  }

  const last = path.at(-1) || candles.filter((candle) => candle.time >= entry.entryDate).at(-1);
  const markedR = last ? resultR(trade, last.close, entry.entryPrice, riskPerShare) : 0;
  return {
    r: firstHalfR === null ? markedR : firstHalfR * 0.5 + markedR * 0.5,
    exitDate: last?.time || entry.entryDate,
    reason: "marked"
  };
}

function excursionThroughActualExit(trade, candles, entry) {
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  const path = candles.filter((candle) => candle.time > entry.entryDate && candle.time <= entry.finalExitDate);
  let mfe = 0;
  let mae = 0;
  for (const candle of path) {
    const favorablePrice = direction === 1 ? candle.high : candle.low;
    const adversePrice = direction === 1 ? candle.low : candle.high;
    mfe = Math.max(mfe, direction * (favorablePrice - entry.entryPrice) / riskPerShare);
    mae = Math.min(mae, direction * (adversePrice - entry.entryPrice) / riskPerShare);
  }
  return { mfe, mae };
}

const normalized = rawTrades
  .map((trade) => ({ trade, entry: normalizedEntry(trade), symbol: cleanSymbol(trade.symbol) }))
  .filter(({ trade, entry, symbol }) =>
    /^[A-Z]+$/.test(symbol) &&
    numberValue(trade.risk) > 0 &&
    entry.maxShares > 0 &&
    entry.exitShares >= entry.entryShares - 1e-6 &&
    entry.finalExitDate
  )
  .sort((a, b) => b.entry.finalExitDate.localeCompare(a.entry.finalExitDate))
  .slice(0, 50);

const strategyNames = [
  "actual",
  "1.5R",
  "2R",
  "2.5R",
  "3R",
  "4R",
  "5%",
  "7%",
  "8%",
  "10%",
  "12%",
  "15%",
  "20%",
  "8EMA",
  "1R activate 8EMA",
  "2R activate 8EMA",
  "3R activate 8EMA",
  "2R + 8EMA"
];
const rows = normalized.map(({ trade, entry, symbol }) => {
  const candles = loadCandles(symbol);
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  const strategy = {
    actual: { r: numberValue(trade.pnl) / numberValue(trade.risk), exitDate: entry.finalExitDate, reason: "actual" },
    "1.5R": simulateTarget(trade, candles, entry, entry.entryPrice + direction * 1.5 * riskPerShare, "1.5r-target"),
    "2R": simulateTarget(trade, candles, entry, entry.entryPrice + direction * 2 * riskPerShare, "2r-target"),
    "2.5R": simulateTarget(trade, candles, entry, entry.entryPrice + direction * 2.5 * riskPerShare, "2.5r-target"),
    "3R": simulateTarget(trade, candles, entry, entry.entryPrice + direction * 3 * riskPerShare, "3r-target"),
    "4R": simulateTarget(trade, candles, entry, entry.entryPrice + direction * 4 * riskPerShare, "4r-target"),
    "5%": simulateTarget(trade, candles, entry, entry.entryPrice * (1 + direction * 0.05), "5%-target"),
    "7%": simulateTarget(trade, candles, entry, entry.entryPrice * (1 + direction * 0.07), "7%-target"),
    "8%": simulateTarget(trade, candles, entry, entry.entryPrice * (1 + direction * 0.08), "8%-target"),
    "10%": simulateTarget(trade, candles, entry, entry.entryPrice * (1 + direction * 0.1), "10%-target"),
    "12%": simulateTarget(trade, candles, entry, entry.entryPrice * (1 + direction * 0.12), "12%-target"),
    "15%": simulateTarget(trade, candles, entry, entry.entryPrice * (1 + direction * 0.15), "15%-target"),
    "20%": simulateTarget(trade, candles, entry, entry.entryPrice * (1 + direction * 0.2), "20%-target"),
    "8EMA": simulateEma8(trade, candles, entry),
    "1R activate 8EMA": simulateActivatedEma8(trade, candles, entry, 1),
    "2R activate 8EMA": simulateActivatedEma8(trade, candles, entry, 2),
    "3R activate 8EMA": simulateActivatedEma8(trade, candles, entry, 3),
    "2R + 8EMA": simulateTwoRThenEma8(trade, candles, entry)
  };
  const excursion = excursionThroughActualExit(trade, candles, entry);
  return {
    id: trade.id,
    symbol,
    side: trade.side,
    setup: trade.setup_tags?.[0] || "Unassigned",
    normalizedEntryDate: entry.entryDate,
    actualExitDate: entry.finalExitDate,
    normalizedEntryPrice: entry.entryPrice,
    maxShares: entry.maxShares,
    risk: numberValue(trade.risk),
    mfeThroughActualExitR: excursion.mfe,
    maeThroughActualExitR: excursion.mae,
    strategies: strategy
  };
});

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function summarize(name) {
  const values = rows.map((row) => row.strategies[name].r);
  const ordered = rows
    .map((row) => ({ date: row.strategies[name].exitDate, r: row.strategies[name].r }))
    .sort((a, b) => a.date.localeCompare(b.date));
  let equity = 0;
  let peak = 0;
  let maxDrawdown = 0;
  for (const item of ordered) {
    equity += item.r;
    peak = Math.max(peak, equity);
    maxDrawdown = Math.min(maxDrawdown, equity - peak);
  }
  const reasons = rows.reduce((totals, row) => {
    const reason = row.strategies[name].reason;
    totals[reason] = (totals[reason] || 0) + 1;
    return totals;
  }, {});
  return {
    strategy: name,
    trades: values.length,
    totalR: values.reduce((sum, value) => sum + value, 0),
    averageR: values.reduce((sum, value) => sum + value, 0) / values.length,
    medianR: median(values),
    winRate: values.filter((value) => value > 0.05).length / values.length,
    maxDrawdownR: maxDrawdown,
    markedOpen: reasons.marked || 0,
    reasons
  };
}

const summaries = strategyNames.map(summarize).sort((a, b) => b.totalR - a.totalR);
const actualTotalR = summaries.find((summary) => summary.strategy === "actual").totalR;
for (const summary of summaries) summary.deltaVsActualR = summary.totalR - actualTotalR;

const diagnostics = {
  reached2RBeforeActualExit: rows.filter((row) => row.mfeThroughActualExitR >= 2).length,
  reached3RBeforeActualExit: rows.filter((row) => row.mfeThroughActualExitR >= 3).length,
  reached2RButExitedBelow1R: rows.filter(
    (row) => row.mfeThroughActualExitR >= 2 && row.strategies.actual.r < 1
  ).length,
  reached3RButExitedBelow2R: rows.filter(
    (row) => row.mfeThroughActualExitR >= 3 && row.strategies.actual.r < 2
  ).length,
  stoppedAtMinus1ROrWorse: rows.filter((row) => row.strategies.actual.r <= -0.95).length,
  averageMfeR: rows.reduce((sum, row) => sum + row.mfeThroughActualExitR, 0) / rows.length,
  averageMaeR: rows.reduce((sum, row) => sum + row.maeThroughActualExitR, 0) / rows.length
};

fs.writeFileSync(
  outputPath,
  JSON.stringify(
    {
      generatedAt: new Date().toISOString(),
      methodology: {
        universe: "Latest fully closed U.S. stock/ETF trades with executions and saved initial risk",
        entryNormalization: "Maximum open share count, average cost at that point",
        priceFrequency: "Daily OHLC",
        signalTiming: "Begins on the first full daily bar after normalized entry",
        stop: "Implied 1R stop from saved dollar risk divided by maximum shares",
        sameDayStopAndTarget: "Stop assumed first",
        maximumHoldingPeriod: "60 trading sessions; unresolved positions marked at latest available close",
        commissions: "Saved total trade commission deducted from simulated R"
      },
      tradeCount: rows.length,
      dateRange: {
        firstEntry: rows.map((row) => row.normalizedEntryDate).sort()[0],
        lastActualExit: rows.map((row) => row.actualExitDate).sort().at(-1)
      },
      summaries,
      diagnostics,
      rows
    },
    null,
    2
  )
);

console.log(JSON.stringify({ tradeCount: rows.length, summaries, diagnostics }, null, 2));
