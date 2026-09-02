import type { TradeExecution, TradeLogEntry, TradeLogInput } from "./types";
import { normalizeTradeReviewSections } from "./trade-review";

export type CfWorkingOrderMetadata = {
  orderId: string;
  orderDate: string;
  timeValue: string;
  direction: "Buy" | "Sell";
  shares: number;
  symbol: string;
  orderType: string;
  orderPrice: number;
};

function tradeFingerprint(trade: TradeLogEntry | TradeLogInput) {
  return JSON.stringify({
    importRowKey: trade.importRowKey || "", symbol: trade.symbol, side: trade.side, status: trade.status,
    entryDate: trade.entryDate, exitDate: trade.exitDate, openTime: trade.openTime || "", closeTime: trade.closeTime || "",
    avgEntry: trade.avgEntry, exitPrice: trade.exitPrice, stopPrice: trade.stopPrice, takeProfitPrice: trade.takeProfitPrice,
    shares: trade.shares, commission: trade.commission, usedMargin: trade.usedMargin, risk: trade.risk, pnl: trade.pnl,
    rMultiple: trade.rMultiple, returnPercent: trade.returnPercent, daysInTrade: trade.daysInTrade,
    setupTags: trade.setupTags, mistakeTags: trade.mistakeTags, customTags: trade.customTags, manualGrade: trade.manualGrade,
    portfolioTag: trade.portfolioTag, emotion: trade.emotion, tradeQuality: trade.tradeQuality,
    checklistItems: trade.checklistItems, notes: trade.notes, reviewSections: normalizeTradeReviewSections(trade.reviewSections),
    screenshots: trade.screenshots, chartLinks: trade.chartLinks,
    executions: trade.executions, hidden: Boolean(trade.hidden)
  });
}

export function cfImportTradesEquivalent(existing: TradeLogEntry[], incoming: TradeLogInput[]) {
  if (existing.length !== incoming.length) return false;
  const existingFingerprints = existing.map(tradeFingerprint).sort();
  const incomingFingerprints = incoming.map(tradeFingerprint).sort();
  return existingFingerprints.every((fingerprint, index) => fingerprint === incomingFingerprints[index]);
}

function brokerTransactionKey(execution: TradeExecution) {
  const transactionId = String(execution.sourceKey || execution.id || "").replace(/^cf-transaction:/, "");
  const direction = execution.type === "ENTRY"
    ? execution.side === "LONG" ? "Buy" : "Sell"
    : execution.side === "LONG" ? "Sell" : "Buy";
  return transactionId ? `${transactionId}|${direction}` : "";
}

function dateDaysBetween(startDate: string, endDate: string) {
  const start = new Date(`${startDate}T00:00:00Z`).getTime();
  const end = new Date(`${endDate}T00:00:00Z`).getTime();
  if (!Number.isFinite(start) || !Number.isFinite(end)) return 0;
  return Math.max(0, Math.round((end - start) / 86400000));
}

export function mergeCfExecutionHistory(
  existing: TradeExecution[],
  currentStatement: TradeExecution[],
  options: {
    statementStartDate?: string;
    statementEndDate?: string;
    currentStatementSymbols?: string[];
    currentOpenSymbols?: string[];
  } = {}
) {
  const currentKeys = new Set(currentStatement.map(brokerTransactionKey).filter(Boolean));
  const currentSymbols = new Set((options.currentStatementSymbols || []).map((symbol) => symbol.trim().toUpperCase()).filter(Boolean));
  const isBroadStatementReplay = Boolean(
    options.statementStartDate
      && options.statementEndDate
      && dateDaysBetween(options.statementStartDate, options.statementEndDate) >= 7
  );

  return [
    ...existing.filter((execution) => {
      const key = brokerTransactionKey(execution);
      const symbol = String(execution.source || "").trim().toUpperCase();
      if (isBroadStatementReplay && currentSymbols.has(symbol)) {
        return false;
      }
      return !key || !currentKeys.has(key);
    }),
    ...currentStatement
  ];
}

export function replaceActiveWorkingOrders(orders: CfWorkingOrderMetadata[]) {
  const byOrderId = new Map<string, CfWorkingOrderMetadata>();
  for (const order of orders) {
    const key = order.orderId || [order.symbol, order.direction, order.orderType, order.orderPrice, order.shares].join("|");
    byOrderId.set(key, { ...order, symbol: order.symbol.trim().toUpperCase(), orderType: order.orderType.trim().toUpperCase() });
  }
  return Array.from(byOrderId.values()).sort((a, b) =>
    a.orderId.localeCompare(b.orderId)
    || a.symbol.localeCompare(b.symbol)
    || a.orderType.localeCompare(b.orderType)
    || a.orderPrice - b.orderPrice
  );
}

export type CfImportTransaction = {
  begin(): Promise<void>;
  commit(): Promise<void>;
  rollback(): Promise<void>;
};

export async function runAtomicCfImport<T>(transaction: CfImportTransaction, operation: () => Promise<T>) {
  await transaction.begin();
  try {
    const result = await operation();
    await transaction.commit();
    return result;
  } catch (error) {
    await transaction.rollback();
    throw error;
  }
}
