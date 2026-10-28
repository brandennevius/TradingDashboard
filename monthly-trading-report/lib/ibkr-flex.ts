import { XMLParser, XMLValidator } from "fast-xml-parser";
import type { TradeLogEntry, TradeLogInput } from "./types";
import type { ParsedOpenPositionRow } from "./cf-statement";

export const IBKR_SOURCE = "ibkr-flex-xml";
type Row = Record<string, string>;
export type IbkrFlexData = {
  accountId: string; currency: string; fromDate: string; toDate: string;
  executions: Row[]; cashTransactions: Row[]; equities: Row[]; positions: Row[];
  securities: Row[]; conversionRates: Row[]; nav: Row;
};
export class IbkrImportError extends Error {}
const fail = (message: string): never => { throw new IbkrImportError(message); };
const rows = (value: unknown): Row[] => !value ? [] : (Array.isArray(value) ? value : [value]) as Row[];
export function ibkrDate(value: string) {
  const d = String(value || "").split(";")[0];
  if (!/^\d{8}$/.test(d)) return fail(`Invalid IBKR date: ${d}`);
  const formatted = `${d.slice(0,4)}-${d.slice(4,6)}-${d.slice(6,8)}`;
  if (new Date(formatted).toISOString().slice(0,10) !== formatted) return fail("Invalid IBKR calendar date.");
  return formatted;
}
export function ibkrNumber(row: Row, field: string) {
  const value = row[field];
  if (value == null || value === "" || !Number.isFinite(Number(value))) return fail(`Missing or invalid IBKR ${field}.`);
  return Number(value);
}
export function parseIbkrFlex(xml: string): IbkrFlexData {
  if (/<!DOCTYPE|<!ENTITY/i.test(xml)) fail("XML entity declarations are not supported.");
  if (XMLValidator.validate(xml) !== true) fail("The uploaded file is not valid XML.");
  const parsed = new XMLParser({ ignoreAttributes: false, attributeNamePrefix: "", parseAttributeValue: false, processEntities: false }).parse(xml);
  const response = parsed.FlexQueryResponse;
  const statements = rows(response?.FlexStatements?.FlexStatement);
  if (response?.type !== "AF" || statements.length !== 1) fail("Upload an Activity Flex XML for exactly one IBKR account.");
  const s = statements[0] as unknown as Record<string, any>;
  for (const section of ["AccountInformation", "ChangeInNAV", "EquitySummaryInBase", "OpenPositions", "Trades", "CashTransactions", "SecuritiesInfo"]) {
    if (!(section in s)) fail(`The Flex Query must include ${section}.`);
  }
  const accountId = String(s.accountId || "");
  if (!accountId || s.AccountInformation.accountId !== accountId) fail("IBKR account identity is missing or inconsistent.");
  const currency = String(s.AccountInformation.currency);
  if (currency !== "USD") fail("This importer currently supports USD-base personal portfolios.");
  const fromDate = ibkrDate(s.fromDate), toDate = ibkrDate(s.toDate);
  if (fromDate > toDate) fail("Invalid statement date range.");
  const executions = rows(s.Trades?.Trade).filter(r => r.levelOfDetail === "EXECUTION");
  if (rows(s.Trades?.Trade).some(r => r.levelOfDetail !== "EXECUTION")) fail("Select Execution detail only in the Trades section.");
  const positions = rows(s.OpenPositions?.OpenPosition);
  for (const r of [...executions, ...positions]) {
    if (r.accountId !== accountId) fail("The report contains another account.");
    if (r.currency !== "USD" || r.assetCategory !== "STK") fail("This importer currently supports USD stock and ETF trades/positions. Other assets require a separate importer.");
  }
  for (const r of executions) {
    if (!r.ibExecID || !r.transactionID || !r.conid) fail("Trades must include IB Execution ID, Transaction ID, and Conid.");
    if (!["BUY", "SELL"].includes(r.buySell) || !["O", "C"].includes(r.openCloseIndicator)) fail("Unsupported trade direction or open/close indicator.");
    if (ibkrDate(r.tradeDate) < fromDate || ibkrDate(r.tradeDate) > toDate) fail("Execution outside statement coverage.");
    for (const field of ["quantity", "tradePrice", "ibCommission", "fifoPnlRealized"]) ibkrNumber(r, field);
  }
  const equities = rows(s.EquitySummaryInBase?.EquitySummaryByReportDateInBase);
  const latest = equities.find(r => ibkrDate(r.reportDate) === toDate);
  if (!latest) fail("Missing ending equity snapshot.");
  const nav = s.ChangeInNAV as Row;
  if (Math.abs(ibkrNumber(latest!, "total") - ibkrNumber(nav, "endingValue")) > .01) fail("Ending NAV does not reconcile.");
  const positionValue = positions.reduce((sum, r) => sum + ibkrNumber(r, "positionValue"), 0);
  if (Math.abs(positionValue - ibkrNumber(latest!, "stock")) > .02) fail("Stock holdings do not reconcile to account equity.");
  const cashTransactions = rows(s.CashTransactions?.CashTransaction).filter(r => r.levelOfDetail === "DETAIL");
  for (const r of cashTransactions) {
    if (r.accountId !== accountId || !r.transactionID) fail("Cash transactions require matching account and Transaction ID.");
    ibkrNumber(r, "amount"); ibkrDate(r.reportDate); ibkrDate(r.dateTime);
  }
  if (rows(s.CorporateActions?.CorporateAction).length || rows(s.Transfers?.Transfer).length) fail("This file contains corporate actions or asset transfers. Import support is required before applying it safely.");
  return { accountId, currency, fromDate, toDate, executions, cashTransactions, equities, positions,
    securities: rows(s.SecuritiesInfo?.SecurityInfo), conversionRates: rows(s.ConversionRates?.ConversionRate), nav };
}

export function mergeIbkrData(previous: IbkrFlexData | undefined, incoming: IbkrFlexData): IbkrFlexData {
  if (!previous) return incoming;
  if (previous.accountId !== incoming.accountId) fail("This portfolio is already linked to a different IBKR account.");
  const merge = (old: Row[], next: Row[], key: (r: Row) => string) => [...new Map([...old, ...next].map(r => [key(r), r])).values()];
  // The incoming report is authoritative for its coverage, including corrections/removals.
  const outside = (r: Row, field: string) => ibkrDate(r[field]) < incoming.fromDate || ibkrDate(r[field]) > incoming.toDate;
  const latest = incoming.toDate >= previous.toDate ? incoming : previous;
  return { ...latest, fromDate: previous.fromDate < incoming.fromDate ? previous.fromDate : incoming.fromDate,
    executions: merge(previous.executions.filter(r => outside(r,"tradeDate")), incoming.executions, r => r.ibExecID),
    cashTransactions: merge(previous.cashTransactions.filter(r => outside(r,"reportDate")), incoming.cashTransactions, r => r.transactionID),
    equities: merge(previous.equities, incoming.equities, r => r.reportDate).sort((a,b) => a.reportDate.localeCompare(b.reportDate)),
    securities: merge(previous.securities, incoming.securities, r => r.conid),
    conversionRates: merge(previous.conversionRates, incoming.conversionRates, r => `${r.reportDate}:${r.fromCurrency}:${r.toCurrency}`)
  };
}

export function buildIbkrTrades(data: IbkrFlexData, userId: string, portfolioTag: string, existing: TradeLogEntry[]) {
  type Cycle = { trade: TradeLogInput; conid: string; quantity: number; entered: number; entryValue: number; exited: number; exitValue: number };
  const active = new Map<string, Cycle>();
  const cycles: Cycle[] = [];
  for (const r of [...data.executions].sort((a,b) => a.dateTime.localeCompare(b.dateTime) || a.ibExecID.localeCompare(b.ibExecID))) {
    const qty = Math.abs(ibkrNumber(r,"quantity")), price = ibkrNumber(r,"tradePrice");
    if (!qty || price <= 0) fail("Execution quantity and price must be positive.");
    const opening = r.openCloseIndicator === "O";
    const side = (opening ? r.buySell === "BUY" : r.buySell === "SELL") ? "LONG" : "SHORT";
    const key = `${r.conid}:${side}`;
    let cycle = active.get(key);
    if (!cycle) {
      if (!opening) fail("Opening executions are missing. Upload account-opening-to-date history before this statement.");
      const trade: TradeLogInput = {
        userId, portfolioTag, importSource: IBKR_SOURCE, importRowKey: `${data.accountId}:${r.conid}:${r.ibExecID}`,
        symbol: r.symbol, side, status: "OPEN", entryDate: ibkrDate(r.tradeDate), exitDate: "", openTime: time(r.dateTime), closeTime: "",
        avgEntry: 0, exitPrice: 0, stopPrice: 0, takeProfitPrice: 0, shares: 0, commission: 0, usedMargin: 0,
        risk: 0, pnl: 0, rMultiple: 0, returnPercent: 0, daysInTrade: 0, setupTags: [], mistakeTags: [], customTags: ["IBKR Flex"],
        manualGrade: "", emotion: "", tradeQuality: "", checklistItems: [], notes: "", screenshots: [], chartLinks: [], executions: []
      };
      cycle = { trade, conid: r.conid, quantity: 0, entered: 0, entryValue: 0, exited: 0, exitValue: 0 };
      active.set(key, cycle); cycles.push(cycle);
    }
    const t = cycle.trade;
    if (opening) { cycle.quantity += qty; cycle.entered += qty; cycle.entryValue += qty * price; }
    else {
      if (qty > cycle.quantity + 1e-8) fail("Closing quantity exceeds imported entries. Upload complete trade history.");
      cycle.quantity -= qty; cycle.exited += qty; cycle.exitValue += qty * price;
      t.pnl += ibkrNumber(r,"fifoPnlRealized");
      t.exitDate = ibkrDate(r.tradeDate); t.closeTime = time(r.dateTime);
    }
    const commission = -ibkrNumber(r,"ibCommission");
    t.commission += commission;
    t.executions!.push({ id: r.ibExecID, sourceKey: `${data.accountId}:${r.ibExecID}`, source: IBKR_SOURCE,
      type: opening ? "ENTRY" : "EXIT", side, date: ibkrDate(r.tradeDate), time: time(r.dateTime), shares: qty, price,
      pnl: ibkrNumber(r,"fifoPnlRealized"), commission });
    if (cycle.quantity < 1e-8) { cycle.quantity = 0; active.delete(key); }
  }
  for (const c of cycles) {
    const t = c.trade;
    t.status = c.quantity ? "OPEN" : t.pnl > 0 ? "WIN" : t.pnl < 0 ? "LOSS" : "BREAKEVEN";
    t.shares = c.quantity || c.entered;
    t.avgEntry = c.entryValue / c.entered;
    t.exitPrice = c.exited ? c.exitValue / c.exited : 0;
    if (c.quantity) { t.exitDate = ""; t.closeTime = ""; }
    t.returnPercent = c.entryValue ? t.pnl / c.entryValue * 100 : 0;
    t.daysInTrade = Math.max(0, Math.round((Date.parse(t.exitDate || data.toDate) - Date.parse(t.entryDate))/86400000));
    const old = existing.find(e => e.importSource === IBKR_SOURCE && e.importRowKey === t.importRowKey);
    if (old) {
      for (const field of ["stopPrice", "takeProfitPrice", "risk", "setupTags", "mistakeTags", "customTags", "manualGrade", "emotion", "tradeQuality", "checklistItems", "notes", "reviewSections", "screenshots", "chartLinks", "hidden"] as const) {
        Object.assign(t, { [field]: old[field] });
      }
    }
  }
  const positionKeys = new Set<string>();
  for (const p of data.positions) {
    const qty = ibkrNumber(p,"position");
    const security = data.securities.filter(s => s.symbol === p.symbol && s.currency === p.currency);
    const conid = p.conid || (security.length === 1 ? security[0].conid : "");
    const key = `${conid}:${qty < 0 ? "SHORT" : "LONG"}`;
    positionKeys.add(key);
    const c = active.get(key);
    if (!c || Math.abs(c.quantity - Math.abs(qty)) > 1e-7) fail(`Missing execution history for ${p.symbol}. Upload account-opening-to-date history.`);
    c!.trade.avgEntry = ibkrNumber(p,"costBasisPrice");
  }
  if ([...active.keys()].some(k => !positionKeys.has(k))) fail("Imported executions do not match ending holdings. Upload complete history.");
  return cycles.map(c => c.trade);
}
function time(value: string) { const t = (value.split(";")[1] || ""); return t ? `${t.slice(0,2)}:${t.slice(2,4)}:${t.slice(4,6)}` : ""; }
export function ibkrSnapshotPositions(data: IbkrFlexData, trades: TradeLogInput[]): ParsedOpenPositionRow[] {
  return data.positions.map(p => {
    const side = Number(p.position) < 0 ? "SHORT" : "LONG";
    const t = trades.find(t => t.symbol === p.symbol && t.side === side && t.status === "OPEN");
    return { symbol: p.symbol, side, shares: Math.abs(Number(p.position)), entryDate: t?.entryDate || "", timeValue: t?.openTime || "",
      entryPrice: Number(p.costBasisPrice), currentPrice: Number(p.markPrice), usedMargin: 0, stopPrice: 0,
      takeProfitPrice: 0, floatingPnl: Number(p.fifoPnlUnrealized), commission: t?.commission || 0 };
  });
}
