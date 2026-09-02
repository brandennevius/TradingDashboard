import type { TradeExecution, TradeLogInput, TradeSide, TradeStatus } from "./types";

type ParsedImportTrade = TradeLogInput;
export type ParsedOpenPositionRow = {
  entryDate: string;
  timeValue: string;
  side: TradeSide;
  shares: number;
  symbol: string;
  entryPrice: number;
  currentPrice: number;
  usedMargin: number;
  stopPrice: number;
  takeProfitPrice: number;
  floatingPnl: number;
  commission: number;
};
type RawTransactionRow = {
  transactionId: string;
  tradeDate: string;
  timeValue: string;
  direction: "Buy" | "Sell";
  shares: number;
  symbol: string;
  price: number;
  settledPnlRaw: string;
  settledPnl: number;
  commission: number;
};
export type ParsedWorkingOrderRow = {
  orderId: string;
  orderDate: string;
  timeValue: string;
  direction: "Buy" | "Sell";
  shares: number;
  symbol: string;
  orderType: "LIMIT" | "STOP" | string;
  orderPrice: number;
};
type OpenLot = {
  id: string;
  transactionId: string;
  cycleId: string;
  symbol: string;
  originalShares: number;
  sharesRemaining: number;
  entryDate: string;
  openTime: string;
  entryPrice: number;
  side: TradeSide;
  commissionPerShare: number;
  realizedPnl: number;
  exitValue: number;
  exitedShares: number;
  latestExitDate: string;
  latestExitTime: string;
  closeCommission: number;
  executions: NonNullable<TradeLogInput["executions"]>;
  stopPrice: number;
  takeProfitPrice: number;
  usedMargin: number;
};
type ParsedCfStatement = {
  trades: ParsedImportTrade[];
  transactions: RawTransactionRow[];
  openPositions: ParsedOpenPositionRow[];
  workingOrders: ParsedWorkingOrderRow[];
  balance: number;
  currentEquity: number;
  statementEquity: number;
  floatingPnl: number;
  equityStatementStartDate: string;
  equityStatementDate: string;
};

type TradeCycle = {
  id: string;
  symbol: string;
  side: TradeSide;
  entryDate: string;
  openTime: string;
  firstTransactionId: string;
  totalEntryShares: number;
  totalEntryValue: number;
  totalEntryCommission: number;
  remainingShares: number;
  realizedPnl: number;
  exitValue: number;
  exitedShares: number;
  latestExitDate: string;
  latestExitTime: string;
  closeCommission: number;
  executions: NonNullable<TradeLogInput["executions"]>;
  stopPrice: number;
  takeProfitPrice: number;
  usedMargin: number;
  displayEntryPrice: number;
};

const SHARE_EPSILON = 0.000001;

function normalizeMinus(value: string) {
  return value.replace(/[‑−–]/g, "-").replace(/,/g, "").trim();
}

function parseNumber(value: string) {
  const normalized = normalizeMinus(String(value || ""));

  if (!normalized || normalized === "-" || normalized === "—") {
    return 0;
  }

  const number = Number(normalized);
  return Number.isFinite(number) ? number : 0;
}

function parseLabeledMoney(line: string, labels: string[]) {
  const normalized = line.replace(/\s+/g, " ").trim();
  const escapedLabels = labels.map((label) => label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const immediateMatch = normalized.match(new RegExp(`\\b(?:${escapedLabels.join("|")})\\b\\s+(?:\\$|USD\\s*)?\\(?(-?[0-9][\\d,]*\\.\\d{2})\\)?`, "i"));

  if (!labels.some((label) => new RegExp(`\\b${label}\\b`, "i").test(normalized))) {
    return 0;
  }

  const moneyMatches = Array.from(normalized.matchAll(/(?:\$|USD\s*)?\(?(-?[0-9][\d,]*\.\d{2})\)?/gi));
  const value = immediateMatch ? parseNumber(immediateMatch[1]) : moneyMatches.length ? parseNumber(moneyMatches[0][1]) : 0;

  return Number.isFinite(value) ? value : 0;
}

const statementMonthNumbers: Record<string, string> = {
  jan: "01",
  feb: "02",
  mar: "03",
  apr: "04",
  may: "05",
  jun: "06",
  jul: "07",
  aug: "08",
  sep: "09",
  oct: "10",
  nov: "11",
  dec: "12"
};

function namedStatementDate(day: string, month: string, year: string) {
  const monthNumber = statementMonthNumbers[month.toLowerCase()];
  if (!monthNumber) return "";
  return `${year}-${monthNumber}-${day.padStart(2, "0")}`;
}

function parseStatementPeriod(lines: string[]) {
  for (const line of lines) {
    const period = line.match(
      /\b(\d{1,2})\s+([A-Z]{3})\s+(\d{4})\s+\d{1,2}:\d{2}\s*[—–-]\s*(\d{1,2})\s+([A-Z]{3})\s+(\d{4})\s+\d{1,2}:\d{2}\b/i
    );
    if (period) {
      return {
        startDate: namedStatementDate(period[1], period[2], period[3]),
        endDate: namedStatementDate(period[4], period[5], period[6])
      };
    }
  }

  for (const line of lines) {
    const created = line.match(/^Created\s+(\d{2}\/\d{2}\/\d{4})\b/i);
    if (created) return { startDate: "", endDate: isoDate(created[1]) };
  }

  return { startDate: "", endDate: "" };
}

function parseSummaryMetrics(lines: string[]) {
  const balanceLabels = ["balance"];
  const equityLabels = ["current equity", "net liquidation", "net liquidation value", "account equity", "equity"];
  const floatingLabels = ["floating pnl", "floating p&l", "unrealized p&l", "open p&l", "p&l"];
  let balance = 0;
  let equity = 0;
  let floatingPnl = 0;

  for (const line of lines.slice(0, 80)) {
    balance ||= parseLabeledMoney(line, balanceLabels);
    equity ||= parseLabeledMoney(line, equityLabels);
    floatingPnl ||= parseLabeledMoney(line, floatingLabels);
  }

  return { balance, equity, floatingPnl };
}

function isoDate(value: string) {
  const match = String(value || "").trim().match(/^(\d{2})\/(\d{2})\/(\d{4})$/);
  if (!match) {
    return "";
  }

  const [, day, month, year] = match;
  return `${year}-${month}-${day}`;
}

function cfStatus(pnl: number, isOpen: boolean): TradeStatus {
  if (isOpen) {
    return "OPEN";
  }
  if (pnl > 0) {
    return "WIN";
  }
  if (pnl < 0) {
    return "LOSS";
  }
  return "BREAKEVEN";
}

function openPositionSide(direction: string): TradeSide {
  return String(direction).toUpperCase() === "SELL" ? "SHORT" : "LONG";
}

function splitLines(text: string) {
  return text
    .split(/\r?\n/)
    .map((line) => line.replace(/\s+/g, " ").trim())
    .filter(Boolean);
}

const openPositionPattern =
  /^(\d{2}\/\d{2}\/\d{4}) (\d{2}:\d{2}) (Buy|Sell) ([\d,]+\.\d+) ([A-Z0-9/._-]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+)$/i;
const openPositionPrefixPattern =
  /^(\d{2}\/\d{2}\/\d{4}) (\d{2}:\d{2}) (Buy|Sell) ([\d,]+(?:\.\d+)?) ([A-Z0-9/._-]+) (.+)$/i;
const workingOrderPattern =
  /^(\d{2}\/\d{2}\/\d{4}) (\d{2}:\d{2}) (\d+) (Buy|Sell) ([\d,]+\.\d+) ([A-Z0-9/._-]+) ([A-Z]+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+) [A-Z-]+(?: [\d,\-.‑−–—]+){0,3}$/i;

const transactionPattern =
  /^([0-9:]+) (\d{2}\/\d{2}\/\d{4}) (\d{2}:\d{2}:\d{2}(?:\.\d+)?) (Buy|Sell) ([\d,]+\.\d+) ([A-Z0-9/._-]+) ([\d,\-.‑−–]+) (\d+) ([\d,\-.‑−–—]+) ([\d,\-.‑−–—]+)(?: .*)?$/i;

function toTimestamp(dateValue: string, timeValue: string) {
  return new Date(`${dateValue}T${timeValue.replace(/\.\d+$/, "")}Z`).getTime();
}

function roundShareQuantity(value: number) {
  if (Math.abs(value) <= SHARE_EPSILON) return 0;
  return Number(value.toFixed(6));
}

function isSettledTransaction(value: string) {
  const normalized = normalizeMinus(value);
  return Boolean(normalized && normalized !== "-" && normalized !== "—");
}

function daysBetween(entryDate: string, exitDate: string) {
  if (!entryDate || !exitDate) {
    return 0;
  }

  const start = new Date(`${entryDate}T00:00:00Z`).getTime();
  const end = new Date(`${exitDate}T00:00:00Z`).getTime();

  if (!Number.isFinite(start) || !Number.isFinite(end)) {
    return 0;
  }

  return Math.max(0, Math.round((end - start) / 86400000));
}

function pushOpenLot(lotsBySymbol: Record<string, OpenLot[]>, symbol: string, lot: OpenLot) {
  (lotsBySymbol[symbol] ||= []).push(lot);
}

function openingSide(direction: "Buy" | "Sell"): TradeSide {
  return direction === "Buy" ? "LONG" : "SHORT";
}

function closingSide(direction: "Buy" | "Sell"): TradeSide {
  return direction === "Sell" ? "LONG" : "SHORT";
}

function pushLotForSide(longLotsBySymbol: Record<string, OpenLot[]>, shortLotsBySymbol: Record<string, OpenLot[]>, lot: OpenLot) {
  if (lot.side === "LONG") {
    pushOpenLot(longLotsBySymbol, lot.symbol, lot);
  } else {
    pushOpenLot(shortLotsBySymbol, lot.symbol, lot);
  }
}

function seedCarryoverLots(
  rows: RawTransactionRow[],
  openPositions: ParsedOpenPositionRow[],
  longLotsBySymbol: Record<string, OpenLot[]>,
  shortLotsBySymbol: Record<string, OpenLot[]>,
  cyclesById: Map<string, TradeCycle>,
  cycles: TradeCycle[]
) {
  const transactionNetBySymbol = new Map<string, number>();
  const openNetBySymbol = new Map<string, number>();
  const weightedOpenEntryBySymbol = new Map<string, { longValue: number; longShares: number; shortValue: number; shortShares: number }>();
  const firstTransactionBySymbol = new Map<string, RawTransactionRow>();

  for (const row of rows) {
    transactionNetBySymbol.set(
      row.symbol,
      (transactionNetBySymbol.get(row.symbol) || 0) + (row.direction === "Buy" ? row.shares : -row.shares)
    );

    if (!firstTransactionBySymbol.has(row.symbol)) {
      firstTransactionBySymbol.set(row.symbol, row);
    }
  }

  for (const position of openPositions) {
    openNetBySymbol.set(
      position.symbol,
      (openNetBySymbol.get(position.symbol) || 0) + (position.side === "LONG" ? position.shares : -position.shares)
    );

    const current = weightedOpenEntryBySymbol.get(position.symbol) || { longValue: 0, longShares: 0, shortValue: 0, shortShares: 0 };
    if (position.side === "LONG") {
      current.longValue += position.entryPrice * position.shares;
      current.longShares += position.shares;
    } else {
      current.shortValue += position.entryPrice * position.shares;
      current.shortShares += position.shares;
    }
    weightedOpenEntryBySymbol.set(position.symbol, current);
  }

  const symbols = new Set([...transactionNetBySymbol.keys(), ...openNetBySymbol.keys()]);

  for (const symbol of symbols) {
    const currentOpenNet = openNetBySymbol.get(symbol) || 0;

    if (Math.abs(currentOpenNet) <= SHARE_EPSILON) {
      continue;
    }

    const baselineNet = currentOpenNet - (transactionNetBySymbol.get(symbol) || 0);

    if (Math.abs(baselineNet) <= SHARE_EPSILON) {
      continue;
    }

    const side: TradeSide = baselineNet > 0 ? "LONG" : "SHORT";
    const shares = roundShareQuantity(Math.abs(baselineNet));
    const firstRow = firstTransactionBySymbol.get(symbol);
    const openEntry = weightedOpenEntryBySymbol.get(symbol);
    const displayEntryPrice =
      side === "LONG"
        ? openEntry?.longShares
          ? openEntry.longValue / openEntry.longShares
          : firstRow?.price || 0
        : openEntry?.shortShares
          ? openEntry.shortValue / openEntry.shortShares
          : firstRow?.price || 0;
    const entryDate = firstRow?.tradeDate || new Date().toISOString().slice(0, 10);
    const openTime = firstRow?.timeValue || "00:00:00";
    const cycleId = `cf-baseline:${symbol}:${side}:${entryDate}`;

    const cycle: TradeCycle = {
      id: cycleId,
      symbol,
      side,
      entryDate,
      openTime,
      firstTransactionId: "baseline",
      totalEntryShares: shares,
      totalEntryValue: displayEntryPrice * shares,
      totalEntryCommission: 0,
      remainingShares: shares,
      realizedPnl: 0,
      exitValue: 0,
      exitedShares: 0,
      latestExitDate: "",
      latestExitTime: "",
      closeCommission: 0,
      executions: [],
      stopPrice: 0,
      takeProfitPrice: 0,
      usedMargin: 0,
      displayEntryPrice
    };

    const lot: OpenLot = {
      id: `cf-carryover:${symbol}:${side}:${entryDate}`,
      transactionId: "baseline",
      cycleId,
      symbol,
      originalShares: shares,
      sharesRemaining: shares,
      entryDate,
      openTime,
      entryPrice: displayEntryPrice,
      side,
      commissionPerShare: 0,
      realizedPnl: 0,
      exitValue: 0,
      exitedShares: 0,
      latestExitDate: "",
      latestExitTime: "",
      closeCommission: 0,
      executions: [],
      stopPrice: 0,
      takeProfitPrice: 0,
      usedMargin: 0
    };

    cyclesById.set(cycleId, cycle);
    cycles.push(cycle);
    pushLotForSide(longLotsBySymbol, shortLotsBySymbol, lot);
  }
}

function parseTransactions(lines: string[]) {
  const rows: RawTransactionRow[] = [];

  for (const line of lines) {
    const transactionMatch = line.match(transactionPattern);

    if (!transactionMatch) {
      continue;
    }

    const [, transactionId, dateValue, timeValue, directionValue, sizeValue, symbol, priceValue, _orderId, settledPnlValue, commissionValue] =
      transactionMatch;

    rows.push({
      transactionId,
      tradeDate: isoDate(dateValue),
      timeValue,
      direction: directionValue as "Buy" | "Sell",
      shares: parseNumber(sizeValue),
      symbol,
      price: parseNumber(priceValue),
      settledPnlRaw: settledPnlValue,
      settledPnl: parseNumber(settledPnlValue),
      commission: Math.abs(parseNumber(commissionValue))
    });
  }

  rows.sort((a, b) => toTimestamp(a.tradeDate, a.timeValue) - toTimestamp(b.tradeDate, b.timeValue));
  return rows;
}

function splitPossiblyGluedCfNumberToken(value: string): string[] {
  const normalized = value.replace(/[‑−–]/g, "-").trim();
  if (!normalized || !/^-?\d/.test(normalized)) return [];

  const gluedMatch = normalized.match(/^(-?\d[\d,]*\.\d{2})(\d[\d,]*\.\d{2})$/);
  if (gluedMatch) {
    return [gluedMatch[1], ...splitPossiblyGluedCfNumberToken(gluedMatch[2])];
  }

  return [normalized];
}

function tokenizeOpenPositionNumbers(value: string) {
  return value
    .split(/\s+/)
    .flatMap(splitPossiblyGluedCfNumberToken)
    .filter((token) => Number.isFinite(parseNumber(token)));
}

function parseOpenPositions(lines: string[]) {
  const positions: ParsedOpenPositionRow[] = [];

  for (const line of lines) {
    const openMatch = line.match(openPositionPattern);

    if (openMatch) {
      const [, dateValue, timeValue, direction, sizeValue, symbol, entryPriceValue, currentPriceValue, usedMarginValue, _notionalVolume, stopLossValue, takeProfitValue, floatingPnlValue, commissionValue] = openMatch;

      positions.push({
        entryDate: isoDate(dateValue),
        timeValue,
        side: openPositionSide(direction),
        shares: parseNumber(sizeValue),
        symbol,
        entryPrice: parseNumber(entryPriceValue),
        currentPrice: parseNumber(currentPriceValue),
        usedMargin: parseNumber(usedMarginValue),
        stopPrice: parseNumber(stopLossValue),
        takeProfitPrice: parseNumber(takeProfitValue),
        floatingPnl: parseNumber(floatingPnlValue),
        commission: Math.abs(parseNumber(commissionValue))
      });
      continue;
    }

    const prefixMatch = line.match(openPositionPrefixPattern);
    if (!prefixMatch) {
      continue;
    }

    const [, dateValue, timeValue, direction, sizeValue, symbol, numericRemainder] = prefixMatch;
    const values = tokenizeOpenPositionNumbers(numericRemainder);

    if (values.length < 8) {
      continue;
    }

    positions.push({
      entryDate: isoDate(dateValue),
      timeValue,
      side: openPositionSide(direction),
      shares: parseNumber(sizeValue),
      symbol,
      entryPrice: parseNumber(values[0]),
      currentPrice: parseNumber(values[1]),
      usedMargin: parseNumber(values[2]),
      stopPrice: parseNumber(values[4]),
      takeProfitPrice: parseNumber(values[5]),
      floatingPnl: parseNumber(values[6]),
      commission: Math.abs(parseNumber(values[7]))
    });
  }

  return positions;
}

function parseWorkingOrders(lines: string[]) {
  const orders: ParsedWorkingOrderRow[] = [];

  for (const line of lines) {
    const match = line.match(workingOrderPattern);

    if (!match) {
      continue;
    }

    const [, dateValue, timeValue, orderId, directionValue, sizeValue, symbol, orderTypeValue, orderPriceValue] = match;
    const normalizedType = String(orderTypeValue).toUpperCase();

    if (normalizedType !== "LIMIT" && normalizedType !== "STOP") {
      continue;
    }

    orders.push({
      orderId,
      orderDate: isoDate(dateValue),
      timeValue,
      direction: directionValue as "Buy" | "Sell",
      shares: parseNumber(sizeValue),
      symbol,
      orderType: normalizedType,
      orderPrice: parseNumber(orderPriceValue)
    });
  }

  return orders;
}

function transactionDirectionFromExecution(execution: TradeExecution): "Buy" | "Sell" {
  if (execution.type === "ENTRY") {
    return execution.side === "LONG" ? "Buy" : "Sell";
  }

  return execution.side === "LONG" ? "Sell" : "Buy";
}

function transactionIdFromExecution(execution: TradeExecution) {
  return String(execution.sourceKey || execution.id || "").replace(/^cf-transaction:/, "");
}

function transactionsFromExecutions(executions: TradeExecution[]) {
  const grouped = new Map<string, RawTransactionRow>();
  const seenExecutions = new Set<string>();

  for (const execution of executions) {
    const transactionId = transactionIdFromExecution(execution);

    if (!transactionId) {
      continue;
    }

    const direction = transactionDirectionFromExecution(execution);
    const key = `${transactionId}|${direction}`;
    const current = grouped.get(key);
    const shares = Math.abs(parseNumber(String(execution.shares)));
    const price = parseNumber(String(execution.price));
    const pnl = parseNumber(String(execution.pnl));
    const commission = Math.abs(parseNumber(String(execution.commission)));
    const executionFingerprint = [transactionId, direction, execution.type, execution.date, execution.time, shares, price, pnl, commission].join("|");
    if (seenExecutions.has(executionFingerprint)) continue;
    seenExecutions.add(executionFingerprint);

    if (!current) {
      grouped.set(key, {
        transactionId,
        tradeDate: execution.date,
        timeValue: execution.time || "00:00:00",
        direction,
        shares,
        symbol: execution.source || "",
        price,
        settledPnlRaw: execution.type === "EXIT" ? String(pnl) : "-",
        settledPnl: execution.type === "EXIT" ? pnl : 0,
        commission
      });
      continue;
    }

    const nextShares = current.shares + shares;
    current.price = nextShares ? (current.price * current.shares + price * shares) / nextShares : current.price;
    current.shares = nextShares;
    current.settledPnl += execution.type === "EXIT" ? pnl : 0;
    current.settledPnlRaw = current.settledPnl ? String(current.settledPnl) : current.settledPnlRaw;
    current.commission += commission;
    if (`${execution.date}T${execution.time}` < `${current.tradeDate}T${current.timeValue}`) {
      current.tradeDate = execution.date;
      current.timeValue = execution.time || current.timeValue;
    }
  }

  return Array.from(grouped.values())
    .filter((row) => row.shares > 0 && row.symbol && row.tradeDate)
    .sort((a, b) => toTimestamp(a.tradeDate, a.timeValue) - toTimestamp(b.tradeDate, b.timeValue));
}

export function buildCfTradesFromExecutionHistory(
  executions: TradeExecution[],
  latestOpenPositions: ParsedOpenPositionRow[],
  latestWorkingOrders: ParsedWorkingOrderRow[],
  userId: string,
  portfolioTag: string
) {
  return buildPositionTrades(transactionsFromExecutions(executions), latestOpenPositions, latestWorkingOrders, userId, portfolioTag);
}

function buildPositionTrades(
  rows: RawTransactionRow[],
  openPositions: ParsedOpenPositionRow[],
  workingOrders: ParsedWorkingOrderRow[],
  userId: string,
  portfolioTag: string
) {
  const longLotsBySymbol: Record<string, OpenLot[]> = {};
  const shortLotsBySymbol: Record<string, OpenLot[]> = {};
  const cyclesById = new Map<string, TradeCycle>();
  const cycles: TradeCycle[] = [];
  const activeCycleByKey = new Map<string, string>();

  seedCarryoverLots(rows, openPositions, longLotsBySymbol, shortLotsBySymbol, cyclesById, cycles);

  const currentOpenSideKeys = new Set(openPositions.map((position) => `${position.symbol}|${position.side}`));

  function cycleKey(symbol: string, side: TradeSide) {
    return `${symbol}|${side}`;
  }

  function startCycle(symbol: string, side: TradeSide, entryDate: string, openTime: string, firstTransactionId: string, displayEntryPrice: number) {
    const id = `cf-cycle:${symbol}:${side}:${entryDate}:${firstTransactionId}`;
    const cycle: TradeCycle = {
      id,
      symbol,
      side,
      entryDate,
      openTime,
      firstTransactionId,
      totalEntryShares: 0,
      totalEntryValue: 0,
      totalEntryCommission: 0,
      remainingShares: 0,
      realizedPnl: 0,
      exitValue: 0,
      exitedShares: 0,
      latestExitDate: "",
      latestExitTime: "",
      closeCommission: 0,
      executions: [],
      stopPrice: 0,
      takeProfitPrice: 0,
      usedMargin: 0,
      displayEntryPrice
    };

    cyclesById.set(id, cycle);
    cycles.push(cycle);
    activeCycleByKey.set(cycleKey(symbol, side), id);
    return cycle;
  }

  function getOrStartCycle(symbol: string, side: TradeSide, row: RawTransactionRow) {
    const key = cycleKey(symbol, side);
    const existing = activeCycleByKey.get(key);

    if (existing) {
      return cyclesById.get(existing)!;
    }

    return startCycle(symbol, side, row.tradeDate, row.timeValue, row.transactionId, row.price);
  }

  function hasFutureOppositeTransaction(rowIndex: number, row: RawTransactionRow) {
    return rows.slice(rowIndex + 1).some((future) => future.symbol === row.symbol && future.direction !== row.direction);
  }

  function shouldTreatUnmatchedSharesAsOpening(rowIndex: number, row: RawTransactionRow) {
    const side = openingSide(row.direction);
    return currentOpenSideKeys.has(cycleKey(row.symbol, side)) || hasFutureOppositeTransaction(rowIndex, row);
  }

  function recordUnmatchedClosingRow(row: RawTransactionRow, shares: number) {
    const side = closingSide(row.direction);
    const matchedSize = roundShareQuantity(shares);
    const pnlShare = row.shares ? row.settledPnl * (matchedSize / row.shares) : row.settledPnl;
    const closeCommissionShare = row.shares ? row.commission * (matchedSize / row.shares) : row.commission;
    const id = `cf-unmatched-exit:${row.symbol}:${side}:${row.tradeDate}:${row.transactionId}`;
    cycles.push({
      id,
      symbol: row.symbol,
      side,
      entryDate: row.tradeDate,
      openTime: row.timeValue,
      firstTransactionId: row.transactionId,
      totalEntryShares: matchedSize,
      totalEntryValue: row.price * matchedSize,
      totalEntryCommission: 0,
      remainingShares: 0,
      realizedPnl: pnlShare,
      exitValue: row.price * matchedSize,
      exitedShares: matchedSize,
      latestExitDate: row.tradeDate,
      latestExitTime: row.timeValue,
      closeCommission: closeCommissionShare,
      executions: [{
        id: `unmatched-exit-${row.transactionId}`,
        type: "EXIT",
        date: row.tradeDate,
        time: row.timeValue,
        side,
        shares: matchedSize,
        price: row.price,
        pnl: pnlShare,
        commission: closeCommissionShare,
        source: "CF statement",
        sourceKey: `cf-transaction:${row.transactionId}`
      }],
      stopPrice: 0,
      takeProfitPrice: 0,
      usedMargin: 0,
      displayEntryPrice: row.price
    });
  }

  for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
    const row = rows[rowIndex];
    const matchedLots = row.direction === "Sell" ? longLotsBySymbol[row.symbol] || [] : shortLotsBySymbol[row.symbol] || [];
    const shouldCloseExistingLot = matchedLots.some((lot) => lot.sharesRemaining > 0.000001);

    if (!isSettledTransaction(row.settledPnlRaw) && !shouldCloseExistingLot && shouldTreatUnmatchedSharesAsOpening(rowIndex, row)) {
      const side = openingSide(row.direction);
      const cycle = getOrStartCycle(row.symbol, side, row);
      const lot: OpenLot = {
        id: `cf-lot:${row.symbol}:${side}:${row.tradeDate}:${row.timeValue}:${row.shares}:${row.price}:${row.transactionId}`,
        transactionId: row.transactionId,
        cycleId: cycle.id,
        symbol: row.symbol,
        originalShares: roundShareQuantity(row.shares),
        sharesRemaining: roundShareQuantity(row.shares),
        entryDate: row.tradeDate,
        openTime: row.timeValue,
        entryPrice: row.price,
        side,
        commissionPerShare: row.shares ? row.commission / row.shares : 0,
        realizedPnl: 0,
        exitValue: 0,
        exitedShares: 0,
        latestExitDate: "",
        latestExitTime: "",
        closeCommission: 0,
        executions: [
          {
            id: `entry-${row.transactionId}`,
            type: "ENTRY",
            date: row.tradeDate,
            time: row.timeValue,
            side,
            shares: roundShareQuantity(row.shares),
            price: row.price,
            pnl: 0,
            commission: row.commission,
            source: "CF statement",
            sourceKey: `cf-transaction:${row.transactionId}`
          }
        ],
        stopPrice: 0,
        takeProfitPrice: 0,
        usedMargin: 0
      };

      cycle.totalEntryShares = roundShareQuantity(cycle.totalEntryShares + row.shares);
      cycle.totalEntryValue += row.price * row.shares;
      cycle.totalEntryCommission += row.commission;
      cycle.remainingShares = roundShareQuantity(cycle.remainingShares + row.shares);
      cycle.displayEntryPrice = cycle.totalEntryShares ? cycle.totalEntryValue / cycle.totalEntryShares : row.price;
      cycle.executions.push(lot.executions[0]);
      pushLotForSide(longLotsBySymbol, shortLotsBySymbol, lot);
      continue;
    }

    if (!shouldCloseExistingLot && !shouldTreatUnmatchedSharesAsOpening(rowIndex, row)) {
      recordUnmatchedClosingRow(row, row.shares);
      continue;
    }

    let remainingShares = row.shares;
    const sharesByCycleId = new Map<string, number>();

    while (remainingShares > 0 && matchedLots.length) {
      const lot = matchedLots[0];
      const matchedSize = roundShareQuantity(Math.min(remainingShares, lot.sharesRemaining));
      sharesByCycleId.set(lot.cycleId, roundShareQuantity((sharesByCycleId.get(lot.cycleId) || 0) + matchedSize));

      lot.sharesRemaining = roundShareQuantity(lot.sharesRemaining - matchedSize);
      remainingShares = roundShareQuantity(remainingShares - matchedSize);

      if (lot.sharesRemaining <= SHARE_EPSILON) {
        matchedLots.shift();
      }
    }

    const closedShares = Array.from(sharesByCycleId.values()).reduce((sum, size) => sum + size, 0);
    const openShares = Math.max(0, remainingShares);

    for (const [id, matchedSize] of sharesByCycleId.entries()) {
      const cycle = cyclesById.get(id);

      if (!cycle || matchedSize <= 0) {
        continue;
      }

      const pnlBaseShares = closedShares || row.shares;
      const pnlShare = pnlBaseShares ? row.settledPnl * (matchedSize / pnlBaseShares) : row.settledPnl;
      const closeCommissionBaseShares = closedShares + openShares || row.shares;
      const closeCommissionShare = closeCommissionBaseShares
        ? row.commission * (matchedSize / closeCommissionBaseShares)
        : row.commission;
      cycle.realizedPnl += pnlShare;
      cycle.exitValue += row.price * matchedSize;
      cycle.exitedShares = roundShareQuantity(cycle.exitedShares + matchedSize);
      cycle.remainingShares = roundShareQuantity(Math.max(0, cycle.remainingShares - matchedSize));
      cycle.latestExitDate = row.tradeDate;
      cycle.latestExitTime = row.timeValue;
      cycle.closeCommission += closeCommissionShare;
      cycle.executions.push({
        id: `exit-${row.transactionId}-${id}`,
        type: "EXIT",
        date: row.tradeDate,
        time: row.timeValue,
        side: cycle.side,
        shares: roundShareQuantity(matchedSize),
        price: row.price,
        pnl: pnlShare,
        commission: closeCommissionShare,
        source: "CF statement",
        sourceKey: `cf-transaction:${row.transactionId}`
      });

      if (cycle.remainingShares <= SHARE_EPSILON && activeCycleByKey.get(cycleKey(cycle.symbol, cycle.side)) === cycle.id) {
        activeCycleByKey.delete(cycleKey(cycle.symbol, cycle.side));
      }
    }

    if (remainingShares > SHARE_EPSILON && !shouldTreatUnmatchedSharesAsOpening(rowIndex, row)) {
      recordUnmatchedClosingRow(row, remainingShares);
      continue;
    }

    if (remainingShares > SHARE_EPSILON) {
      const side = openingSide(row.direction);
      const cycle = getOrStartCycle(row.symbol, side, row);
      const openCommissionShare = row.shares ? row.commission * (remainingShares / row.shares) : row.commission;
      const lot: OpenLot = {
        id: `cf-lot:${row.symbol}:${side}:${row.tradeDate}:${row.timeValue}:${remainingShares}:${row.price}:${row.transactionId}`,
        transactionId: row.transactionId,
        cycleId: cycle.id,
        symbol: row.symbol,
        originalShares: roundShareQuantity(remainingShares),
        sharesRemaining: roundShareQuantity(remainingShares),
        entryDate: row.tradeDate,
        openTime: row.timeValue,
        entryPrice: row.price,
        side,
        commissionPerShare: remainingShares ? openCommissionShare / remainingShares : 0,
        realizedPnl: 0,
        exitValue: 0,
        exitedShares: 0,
        latestExitDate: "",
        latestExitTime: "",
        closeCommission: 0,
        executions: [
          {
            id: `entry-${row.transactionId}`,
            type: "ENTRY",
            date: row.tradeDate,
            time: row.timeValue,
            side,
            shares: roundShareQuantity(remainingShares),
            price: row.price,
            pnl: 0,
            commission: openCommissionShare,
            source: "CF statement",
            sourceKey: `cf-transaction:${row.transactionId}`
          }
        ],
        stopPrice: 0,
        takeProfitPrice: 0,
        usedMargin: 0
      };

      cycle.totalEntryShares = roundShareQuantity(cycle.totalEntryShares + remainingShares);
      cycle.totalEntryValue += row.price * remainingShares;
      cycle.totalEntryCommission += openCommissionShare;
      cycle.remainingShares = roundShareQuantity(cycle.remainingShares + remainingShares);
      cycle.displayEntryPrice = cycle.totalEntryShares ? cycle.totalEntryValue / cycle.totalEntryShares : row.price;
      cycle.executions.push(lot.executions[0]);
      pushLotForSide(longLotsBySymbol, shortLotsBySymbol, lot);
    }
  }

  const openPositionsByKey = new Map<string, ParsedOpenPositionRow[]>();
  for (const position of openPositions) {
    const key = cycleKey(position.symbol, position.side);
    openPositionsByKey.set(key, [...(openPositionsByKey.get(key) || []), position]);
  }

  for (const [key, positions] of openPositionsByKey.entries()) {
    const activeCycleId = activeCycleByKey.get(key);
    const positionShares = positions.reduce((sum, position) => sum + position.shares, 0);
    const weightedStop = positions.reduce((sum, position) => sum + position.stopPrice * position.shares, 0);
    const weightedTakeProfit = positions.reduce((sum, position) => sum + position.takeProfitPrice * position.shares, 0);
    const usedMargin = positions.reduce((sum, position) => sum + position.usedMargin, 0);

    if (activeCycleId) {
      const cycle = cyclesById.get(activeCycleId);

      if (cycle) {
        cycle.stopPrice = positionShares ? weightedStop / positionShares : 0;
        cycle.takeProfitPrice = positionShares ? weightedTakeProfit / positionShares : 0;
        cycle.usedMargin = usedMargin;
      }
    }
  }

  const workingOrdersByKey = new Map<string, ParsedWorkingOrderRow[]>();
  for (const order of workingOrders) {
    const impliedPositionSide: TradeSide = order.direction === "Sell" ? "LONG" : "SHORT";
    const key = cycleKey(order.symbol, impliedPositionSide);
    workingOrdersByKey.set(key, [...(workingOrdersByKey.get(key) || []), order]);
  }

  for (const [key, orders] of workingOrdersByKey.entries()) {
    const activeCycleId = activeCycleByKey.get(key);
    if (!activeCycleId) {
      continue;
    }

    const cycle = cyclesById.get(activeCycleId);
    if (!cycle) {
      continue;
    }

    const latestStop = orders
      .filter((order) => order.orderType === "STOP")
      .sort((a, b) => toTimestamp(b.orderDate, b.timeValue) - toTimestamp(a.orderDate, a.timeValue))[0];
    const latestLimit = orders
      .filter((order) => order.orderType === "LIMIT")
      .sort((a, b) => toTimestamp(b.orderDate, b.timeValue) - toTimestamp(a.orderDate, a.timeValue))[0];

    if (latestStop?.orderPrice) {
      cycle.stopPrice = latestStop.orderPrice;
    }
    if (latestLimit?.orderPrice) {
      cycle.takeProfitPrice = latestLimit.orderPrice;
    }
  }

  return cycles.map((cycle) => {
    const isOpen = cycle.remainingShares > SHARE_EPSILON;
    const tags = ["CF Statement", isOpen ? "Open Position" : "Closed Transaction"];

    const exitExecutions = cycle.executions.filter((execution) => execution.type === "EXIT");
    if (exitExecutions.length > 1 || (isOpen && exitExecutions.length > 0)) {
      tags.push("Partial exits");
    }

    if (!cycle.executions.some((execution) => execution.type === "ENTRY")) {
      tags.push("Needs review");
    }

    const exitPrice = cycle.exitedShares ? cycle.exitValue / cycle.exitedShares : 0;
    const commission = cycle.totalEntryCommission + cycle.closeCommission;
    const costBasis = cycle.totalEntryValue;

    return {
      userId,
      importSource: "cf-statement-pdf",
      importRowKey: cycle.id,
      symbol: cycle.symbol,
      side: cycle.side,
      status: cfStatus(cycle.realizedPnl, isOpen),
      entryDate: cycle.entryDate,
      exitDate: isOpen ? "" : cycle.latestExitDate,
      openTime: cycle.openTime,
      closeTime: isOpen ? "" : cycle.latestExitTime,
      avgEntry: cycle.displayEntryPrice,
      exitPrice,
      stopPrice: cycle.stopPrice,
      takeProfitPrice: cycle.takeProfitPrice,
      shares: roundShareQuantity(isOpen ? cycle.remainingShares : cycle.totalEntryShares),
      commission,
      usedMargin: cycle.usedMargin,
      risk: 0,
      pnl: cycle.realizedPnl,
      rMultiple: 0,
      returnPercent: costBasis ? (cycle.realizedPnl / costBasis) * 100 : 0,
      daysInTrade: daysBetween(cycle.entryDate, isOpen ? new Date().toISOString().slice(0, 10) : cycle.latestExitDate),
      setupTags: [],
      mistakeTags: [],
      customTags: tags,
      manualGrade: "",
      portfolioTag,
      emotion: "",
      tradeQuality: "",
      checklistItems: [],
      notes: "",
      screenshots: [],
      chartLinks: [],
      executions: cycle.executions,
      groupId: "",
      groupRole: "none" as const
    };
  });
}

export function parseCfStatementText(text: string, userId: string, portfolioTag: string): ParsedCfStatement {
  const lines = splitLines(text);
  const transactionRows = parseTransactions(lines);
  const openPositions = parseOpenPositions(lines);
  const workingOrders = parseWorkingOrders(lines);
  const summary = parseSummaryMetrics(lines);
  const statementPeriod = parseStatementPeriod(lines);

  return {
    trades: buildPositionTrades(transactionRows, openPositions, workingOrders, userId, portfolioTag),
    transactions: transactionRows,
    openPositions,
    workingOrders,
    balance: summary.balance,
    currentEquity: summary.balance || summary.equity,
    statementEquity: summary.equity,
    floatingPnl: summary.floatingPnl,
    equityStatementStartDate: statementPeriod.startDate,
    equityStatementDate: statementPeriod.endDate
  };
}
