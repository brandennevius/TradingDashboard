import type { TradeExecution, TradeLogEntry } from "./types";

export type ExitAnalysisCandle = {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type ExitStrategyResult = {
  name: string;
  resultR: number;
  exitDate: string;
  exitPrice: number;
  outcome: "actual" | "target" | "stop" | "signal" | "marked";
};

export type TradeExitAnalysis = {
  actualR: number;
  mfeR: number | null;
  maeR: number | null;
  captureRate: number | null;
  givebackR: number | null;
  normalizedEntryDate: string;
  normalizedEntryPrice: number;
  impliedStopPrice: number;
  finalExitDate: string;
  latestMarketDate: string;
  recommendation: string;
  strategies: ExitStrategyResult[];
};

type NormalizedEntry = {
  entryDate: string;
  entryPrice: number;
  maxShares: number;
  entryShares: number;
  exitShares: number;
  finalExitDate: string;
};

export function tradeExitAnalysisEligibility(trade: TradeLogEntry) {
  if (!numberValue(trade.risk)) {
    return { eligible: false, reason: "missing-risk" as const };
  }

  const entry = normalizedEntry(trade);
  if (!entry) {
    return { eligible: false, reason: "incomplete-executions" as const };
  }

  return { eligible: true, reason: "eligible" as const };
}

function numberValue(value: unknown) {
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function executionTimestamp(execution: TradeExecution) {
  return `${execution.date || ""}T${execution.time || "00:00:00"}`;
}

function openLotAverage(lots: { shares: number; price: number }[]) {
  const shares = lots.reduce((sum, lot) => sum + lot.shares, 0);
  return shares ? lots.reduce((sum, lot) => sum + lot.price * lot.shares, 0) / shares : 0;
}

function normalizedEntry(trade: TradeLogEntry): NormalizedEntry | null {
  const executions = [...(trade.executions || [])].sort((a, b) => executionTimestamp(a).localeCompare(executionTimestamp(b)));

  if (!executions.length) {
    if (trade.status === "OPEN" || !trade.exitDate || !numberValue(trade.avgEntry) || !numberValue(trade.shares)) {
      return null;
    }

    return {
      entryDate: trade.entryDate,
      entryPrice: numberValue(trade.avgEntry),
      maxShares: numberValue(trade.shares),
      entryShares: numberValue(trade.shares),
      exitShares: numberValue(trade.shares),
      finalExitDate: trade.exitDate
    };
  }

  const lots: { shares: number; price: number }[] = [];
  let openShares = 0;
  let maxShares = 0;
  let entryPrice = 0;
  let entryDate = trade.entryDate;

  for (const execution of executions) {
    let shares = Math.max(0, numberValue(execution.shares));

    if (execution.type === "ENTRY") {
      lots.push({ shares, price: numberValue(execution.price) });
      openShares += shares;
    } else {
      openShares = Math.max(0, openShares - shares);
      while (shares > 0 && lots.length) {
        const lot = lots[0];
        const matchedShares = Math.min(shares, lot.shares);
        lot.shares -= matchedShares;
        shares -= matchedShares;
        if (lot.shares <= 1e-8) {
          lots.shift();
        }
      }
    }

    if (openShares > maxShares + 1e-8) {
      maxShares = openShares;
      entryPrice = openLotAverage(lots);
      entryDate = execution.date || entryDate;
    }
  }

  const entryShares = executions
    .filter((execution) => execution.type === "ENTRY")
    .reduce((sum, execution) => sum + numberValue(execution.shares), 0);
  const exitShares = executions
    .filter((execution) => execution.type === "EXIT")
    .reduce((sum, execution) => sum + numberValue(execution.shares), 0);
  const finalExitDate = executions
    .filter((execution) => execution.type === "EXIT" && execution.date)
    .map((execution) => execution.date)
    .sort()
    .at(-1) || trade.exitDate;

  if (!maxShares || !entryPrice || !finalExitDate || exitShares < entryShares - 1e-6) {
    return null;
  }

  return { entryDate, entryPrice, maxShares, entryShares, exitShares, finalExitDate };
}

function addIndicators(candles: ExitAnalysisCandle[]) {
  let ema8: number | null = null;
  let ema21: number | null = null;
  let atr14: number | null = null;
  let previousClose: number | null = null;

  return candles.map((candle, index) => {
    ema8 = ema8 === null ? candle.close : candle.close * (2 / 9) + ema8 * (7 / 9);
    ema21 = ema21 === null ? candle.close : candle.close * (2 / 22) + ema21 * (20 / 22);
    const trueRange = previousClose === null
      ? candle.high - candle.low
      : Math.max(candle.high - candle.low, Math.abs(candle.high - previousClose), Math.abs(candle.low - previousClose));
    atr14 = atr14 === null ? trueRange : index < 14
      ? (atr14 * index + trueRange) / (index + 1)
      : (atr14 * 13 + trueRange) / 14;
    previousClose = candle.close;
    return { ...candle, ema8, ema21, atr14 };
  });
}

type IndicatorCandle = ReturnType<typeof addIndicators>[number];

function simulatedR(trade: TradeLogEntry, exitPrice: number, entry: NormalizedEntry, riskPerShare: number) {
  const direction = trade.side === "SHORT" ? -1 : 1;
  const commissionR = numberValue(trade.commission) / numberValue(trade.risk);
  return direction * (exitPrice - entry.entryPrice) / riskPerShare - commissionR;
}

function simulateTarget(
  trade: TradeLogEntry,
  candles: IndicatorCandle[],
  entry: NormalizedEntry,
  targetPrice: number,
  name: ExitStrategyResult["name"]
): ExitStrategyResult {
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  const stopPrice = entry.entryPrice - direction * riskPerShare;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    const targetHit = direction === 1 ? candle.high >= targetPrice : candle.low <= targetPrice;

    if (stopHit) {
      return {
        name,
        resultR: simulatedR(trade, stopPrice, entry, riskPerShare),
        exitDate: candle.time,
        exitPrice: stopPrice,
        outcome: "stop"
      };
    }

    if (targetHit) {
      return {
        name,
        resultR: simulatedR(trade, targetPrice, entry, riskPerShare),
        exitDate: candle.time,
        exitPrice: targetPrice,
        outcome: "target"
      };
    }
  }

  const last = path.at(-1) || candles.filter((candle) => candle.time >= entry.entryDate).at(-1);
  return {
    name,
    resultR: last ? simulatedR(trade, last.close, entry, riskPerShare) : 0,
    exitDate: last?.time || entry.entryDate,
    exitPrice: last?.close || entry.entryPrice,
    outcome: "marked"
  };
}

function markedResult(
  trade: TradeLogEntry,
  candles: IndicatorCandle[],
  entry: NormalizedEntry,
  riskPerShare: number,
  name: string
): ExitStrategyResult {
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);
  const last = path.at(-1) || candles.filter((candle) => candle.time >= entry.entryDate).at(-1);
  return {
    name,
    resultR: last ? simulatedR(trade, last.close, entry, riskPerShare) : 0,
    exitDate: last?.time || entry.entryDate,
    exitPrice: last?.close || entry.entryPrice,
    outcome: "marked"
  };
}

function simulateMovingAverage(
  trade: TradeLogEntry,
  candles: IndicatorCandle[],
  entry: NormalizedEntry,
  period: 8 | 21
): ExitStrategyResult {
  const name = `${period}EMA close`;
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  const stopPrice = entry.entryPrice - direction * riskPerShare;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      return {
        name,
        resultR: simulatedR(trade, stopPrice, entry, riskPerShare),
        exitDate: candle.time,
        exitPrice: stopPrice,
        outcome: "stop"
      };
    }

    const average = period === 8 ? candle.ema8 : candle.ema21;
    const emaExit = direction === 1 ? candle.close < average : candle.close > average;
    if (emaExit) {
      return {
        name,
        resultR: simulatedR(trade, candle.close, entry, riskPerShare),
        exitDate: candle.time,
        exitPrice: candle.close,
        outcome: "signal"
      };
    }
  }

  return markedResult(trade, candles, entry, riskPerShare, name);
}

function simulateTimeExit(
  trade: TradeLogEntry,
  candles: IndicatorCandle[],
  entry: NormalizedEntry,
  sessions: number
): ExitStrategyResult {
  const name = `${sessions}-session exit`;
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  const stopPrice = entry.entryPrice - direction * riskPerShare;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (let index = 0; index < path.length; index += 1) {
    const candle = path[index];
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      return { name, resultR: simulatedR(trade, stopPrice, entry, riskPerShare), exitDate: candle.time, exitPrice: stopPrice, outcome: "stop" };
    }
    if (index + 1 === sessions) {
      return { name, resultR: simulatedR(trade, candle.close, entry, riskPerShare), exitDate: candle.time, exitPrice: candle.close, outcome: "signal" };
    }
  }

  return markedResult(trade, candles, entry, riskPerShare, name);
}

function simulatePercentTrail(
  trade: TradeLogEntry,
  candles: IndicatorCandle[],
  entry: NormalizedEntry,
  percentage: number
): ExitStrategyResult {
  const name = `${Math.round(percentage * 100)}% trailing stop`;
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  let stopPrice = entry.entryPrice - direction * riskPerShare;
  let bestPrice = entry.entryPrice;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      return { name, resultR: simulatedR(trade, stopPrice, entry, riskPerShare), exitDate: candle.time, exitPrice: stopPrice, outcome: "stop" };
    }

    bestPrice = direction === 1 ? Math.max(bestPrice, candle.high) : Math.min(bestPrice, candle.low);
    const nextStop = direction === 1 ? bestPrice * (1 - percentage) : bestPrice * (1 + percentage);
    stopPrice = direction === 1 ? Math.max(stopPrice, nextStop) : Math.min(stopPrice, nextStop);
  }

  return markedResult(trade, candles, entry, riskPerShare, name);
}

function simulateAtrTrail(
  trade: TradeLogEntry,
  candles: IndicatorCandle[],
  entry: NormalizedEntry,
  multiple: number
): ExitStrategyResult {
  const name = `${multiple}x ATR trailing stop`;
  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = numberValue(trade.risk) / entry.maxShares;
  let stopPrice = entry.entryPrice - direction * riskPerShare;
  let bestPrice = entry.entryPrice;
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      return { name, resultR: simulatedR(trade, stopPrice, entry, riskPerShare), exitDate: candle.time, exitPrice: stopPrice, outcome: "stop" };
    }

    bestPrice = direction === 1 ? Math.max(bestPrice, candle.high) : Math.min(bestPrice, candle.low);
    const nextStop = direction === 1 ? bestPrice - multiple * candle.atr14 : bestPrice + multiple * candle.atr14;
    stopPrice = direction === 1 ? Math.max(stopPrice, nextStop) : Math.min(stopPrice, nextStop);
  }

  return markedResult(trade, candles, entry, riskPerShare, name);
}

function simulatePartialTargetThenEma(
  trade: TradeLogEntry,
  candles: IndicatorCandle[],
  entry: NormalizedEntry,
  targetR: number,
  period: 8 | 21
): ExitStrategyResult {
  const name = `50% at ${targetR}R + ${period}EMA`;
  const direction = trade.side === "SHORT" ? -1 : 1;
  const risk = numberValue(trade.risk);
  const riskPerShare = risk / entry.maxShares;
  const targetPrice = entry.entryPrice + direction * targetR * riskPerShare;
  let stopPrice = entry.entryPrice - direction * riskPerShare;
  let targetDate = "";
  const path = candles.filter((candle) => candle.time > entry.entryDate).slice(0, 60);

  for (const candle of path) {
    const stopHit = direction === 1 ? candle.low <= stopPrice : candle.high >= stopPrice;
    if (stopHit) {
      const remainderR = direction * (stopPrice - entry.entryPrice) / riskPerShare;
      const resultR = targetDate
        ? 0.5 * targetR + 0.5 * remainderR - numberValue(trade.commission) / risk
        : simulatedR(trade, stopPrice, entry, riskPerShare);
      return { name, resultR, exitDate: candle.time, exitPrice: stopPrice, outcome: "stop" };
    }

    if (!targetDate) {
      const targetHit = direction === 1 ? candle.high >= targetPrice : candle.low <= targetPrice;
      if (targetHit) {
        targetDate = candle.time;
        stopPrice = entry.entryPrice;
      }
      continue;
    }

    const average = period === 8 ? candle.ema8 : candle.ema21;
    const emaExit = direction === 1 ? candle.close < average : candle.close > average;
    if (emaExit) {
      const remainderR = direction * (candle.close - entry.entryPrice) / riskPerShare;
      return {
        name,
        resultR: 0.5 * targetR + 0.5 * remainderR - numberValue(trade.commission) / risk,
        exitDate: candle.time,
        exitPrice: candle.close,
        outcome: "signal"
      };
    }
  }

  const last = path.at(-1) || candles.filter((candle) => candle.time >= entry.entryDate).at(-1);
  const remainderR = last ? direction * (last.close - entry.entryPrice) / riskPerShare : 0;
  return {
    name,
    resultR: targetDate
      ? 0.5 * targetR + 0.5 * remainderR - numberValue(trade.commission) / risk
      : last ? simulatedR(trade, last.close, entry, riskPerShare) : 0,
    exitDate: last?.time || entry.entryDate,
    exitPrice: last?.close || entry.entryPrice,
    outcome: "marked"
  };
}

function analysisRecommendation(actualR: number, mfeR: number | null) {
  if (mfeR === null) {
    return "This trade does not have a complete daily bar between the normalized entry and final exit, so intraday MFE and MAE cannot be measured reliably.";
  }

  if (mfeR >= 3) {
    const halfAtThreeR = 0.5 * 3 + 0.5 * actualR;
    if (actualR < 2) {
      return `High giveback: 3R was available but only ${actualR.toFixed(2)}R was realized. A 50% trim at 3R with the remainder exited normally would have produced about ${halfAtThreeR.toFixed(2)}R.`;
    }

    return halfAtThreeR > actualR
      ? `This trade reached 3R. A 50% trim at 3R with the remainder exited normally would have improved the result to about ${halfAtThreeR.toFixed(2)}R.`
      : `The actual ${actualR.toFixed(2)}R exit outperformed a 50% trim at 3R, which would have produced about ${halfAtThreeR.toFixed(2)}R.`;
  }

  if (mfeR >= 2 && actualR < 1) {
    return `High giveback: the trade reached ${mfeR.toFixed(2)}R but finished at ${actualR.toFixed(2)}R. Review profit protection once 2R is reached.`;
  }

  if (mfeR < 1) {
    return "The trade never reached 1R on a full daily bar. Review entry quality and the failure exit rather than adding a wider profit target.";
  }

  const captureRate = mfeR > 0 ? actualR / mfeR : 0;
  return captureRate >= 0.7
    ? "The exit captured most of the available move. There is no clear evidence that a wider target would have improved this trade."
    : `The trade captured ${(captureRate * 100).toFixed(0)}% of its available favorable excursion. Review whether a profit floor could have reduced the giveback.`;
}

export function analyzeTradeExit(trade: TradeLogEntry, rawCandles: ExitAnalysisCandle[]): TradeExitAnalysis | null {
  const entry = normalizedEntry(trade);
  const risk = numberValue(trade.risk);
  if (!entry || !risk || !rawCandles.length) {
    return null;
  }

  const direction = trade.side === "SHORT" ? -1 : 1;
  const riskPerShare = risk / entry.maxShares;
  const impliedStopPrice = entry.entryPrice - direction * riskPerShare;
  const candles = addIndicators(rawCandles);
  const actualPath = candles.filter((candle) => candle.time > entry.entryDate && candle.time <= entry.finalExitDate);
  let mfeR: number | null = actualPath.length ? 0 : null;
  let maeR: number | null = actualPath.length ? 0 : null;

  for (const candle of actualPath) {
    const favorablePrice = direction === 1 ? candle.high : candle.low;
    const adversePrice = direction === 1 ? candle.low : candle.high;
    mfeR = Math.max(mfeR || 0, direction * (favorablePrice - entry.entryPrice) / riskPerShare);
    maeR = Math.min(maeR || 0, direction * (adversePrice - entry.entryPrice) / riskPerShare);
  }

  const actualR = numberValue(trade.pnl) / risk;
  const captureRate = mfeR !== null && mfeR > 0 ? actualR / mfeR : null;
  const givebackR = mfeR !== null ? Math.max(0, mfeR - actualR) : null;
  const actualExitPrice = numberValue(trade.exitPrice) || entry.entryPrice + direction * actualR * riskPerShare;
  const strategies: ExitStrategyResult[] = [
    {
      name: "Actual",
      resultR: actualR,
      exitDate: entry.finalExitDate,
      exitPrice: actualExitPrice,
      outcome: "actual"
    },
    ...[1.5, 2, 2.5, 3, 4].map((targetR) =>
      simulateTarget(trade, candles, entry, entry.entryPrice + direction * targetR * riskPerShare, `${targetR}R target`)
    ),
    ...[0.05, 0.08, 0.1, 0.12, 0.15, 0.2].map((percentage) =>
      simulateTarget(
        trade,
        candles,
        entry,
        entry.entryPrice * (1 + direction * percentage),
        `${Math.round(percentage * 100)}% target`
      )
    ),
    simulateMovingAverage(trade, candles, entry, 8),
    simulateMovingAverage(trade, candles, entry, 21),
    ...[5, 10, 15].map((sessions) => simulateTimeExit(trade, candles, entry, sessions)),
    ...[0.05, 0.08, 0.1].map((percentage) => simulatePercentTrail(trade, candles, entry, percentage)),
    simulateAtrTrail(trade, candles, entry, 2),
    simulateAtrTrail(trade, candles, entry, 3),
    simulatePartialTargetThenEma(trade, candles, entry, 2, 8),
    simulatePartialTargetThenEma(trade, candles, entry, 3, 8),
    simulatePartialTargetThenEma(trade, candles, entry, 3, 21)
  ];

  return {
    actualR,
    mfeR,
    maeR,
    captureRate,
    givebackR,
    normalizedEntryDate: entry.entryDate,
    normalizedEntryPrice: entry.entryPrice,
    impliedStopPrice,
    finalExitDate: entry.finalExitDate,
    latestMarketDate: rawCandles.at(-1)?.time || "",
    recommendation: analysisRecommendation(actualR, mfeR),
    strategies
  };
}
