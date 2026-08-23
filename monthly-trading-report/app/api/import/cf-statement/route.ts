import { NextResponse } from "next/server";
import crypto from "node:crypto";
import { getSessionUser } from "@/lib/auth";
import { buildCfTradesFromExecutionHistory, parseCfStatementText } from "@/lib/cf-statement";
import { cfImportTradesEquivalent, mergeCfExecutionHistory, replaceActiveWorkingOrders } from "@/lib/cf-import-idempotency";
import {
  listCfStatementTrades,
  replaceCfStatementImport,
} from "@/lib/store";
import type { TradeExecution, TradeLogEntry, TradeLogInput } from "@/lib/types";

const CF_SYSTEM_TAGS = new Set([
  "CF Statement",
  "Open Position",
  "Closed Transaction",
  "Partial exits",
  "Needs review",
  "Combined trade",
  "Auto recalculated"
]);

function tradeManualKey(trade: Pick<TradeLogEntry | TradeLogInput, "symbol" | "side" | "entryDate" | "openTime">) {
  return [trade.symbol, trade.side, trade.entryDate, trade.openTime || ""].join("|");
}

function executionHistoryFromTrade(trade: TradeLogEntry | TradeLogInput): TradeExecution[] {
  return (trade.executions || []).map((execution) => ({
    ...execution,
    source: trade.symbol
  }));
}

function applyManualFields(rebuiltTrade: TradeLogInput, existingTrades: TradeLogEntry[]) {
  const exact = existingTrades.find((trade) => trade.importRowKey === rebuiltTrade.importRowKey);
  const fallback = existingTrades.find((trade) => tradeManualKey(trade) === tradeManualKey(rebuiltTrade));
  const existing = exact || fallback;

  if (!existing) {
    return rebuiltTrade;
  }

  const manualTags = existing.customTags.filter((tag) => !CF_SYSTEM_TAGS.has(tag));

  return {
    ...rebuiltTrade,
    risk: existing.risk || rebuiltTrade.risk,
    setupTags: existing.setupTags.length ? existing.setupTags : rebuiltTrade.setupTags,
    mistakeTags: existing.mistakeTags.length ? existing.mistakeTags : rebuiltTrade.mistakeTags,
    customTags: Array.from(new Set([...rebuiltTrade.customTags, ...manualTags])),
    manualGrade: existing.manualGrade || rebuiltTrade.manualGrade,
    emotion: existing.emotion || rebuiltTrade.emotion,
    tradeQuality: existing.tradeQuality || rebuiltTrade.tradeQuality,
    checklistItems: existing.checklistItems.length ? existing.checklistItems : rebuiltTrade.checklistItems,
    notes: existing.notes || rebuiltTrade.notes,
    reviewSections: existing.reviewSections || rebuiltTrade.reviewSections,
    screenshots: existing.screenshots.length ? existing.screenshots : rebuiltTrade.screenshots,
    chartLinks: existing.chartLinks.length ? existing.chartLinks : rebuiltTrade.chartLinks,
    hidden: exact ? existing.hidden : rebuiltTrade.hidden
  };
}

export async function POST(request: Request) {
  const user = await getSessionUser();

  if (!user) {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  if (user.readOnly) {
    return NextResponse.json({ error: "This account is read-only." }, { status: 403 });
  }

  const formData = await request.formData();
  const file = formData.get("file");
  const portfolioTag = String(formData.get("portfolioTag") || "").trim();

  if (!portfolioTag) {
    return NextResponse.json({ error: "A portfolio is required for CF import." }, { status: 400 });
  }

  if (!(file instanceof File)) {
    return NextResponse.json({ error: "A PDF statement file is required." }, { status: 400 });
  }

  try {
    const PDFParser = (await import("pdf2json")).default;
    const buffer = Buffer.from(await file.arrayBuffer());
    const sourceHash = crypto.createHash("sha256").update(buffer).digest("hex");
    const parser = new PDFParser(undefined, true);

    const sourceText = await new Promise<string>((resolve, reject) => {
      parser.on("pdfParser_dataError", (error: Error | { parserError: Error }) =>
        reject(error instanceof Error ? error : error?.parserError || new Error("CF PDF parse failed."))
      );
      parser.on("pdfParser_dataReady", () => resolve(parser.getRawTextContent() || ""));
      parser.parseBuffer(buffer);
    });

    if (!sourceText.trim()) {
      return NextResponse.json({ error: "No importable text was extracted from that CF statement." }, { status: 400 });
    }

    const parsedStatement = parseCfStatementText(sourceText, user.id, portfolioTag);
    const statementTrades = parsedStatement.trades;

    if (!statementTrades.length) {
      return NextResponse.json({ error: "No importable CF trades were found in that statement." }, { status: 400 });
    }

    if (!parsedStatement.equityStatementDate) {
      return NextResponse.json(
        { error: "The statement coverage date could not be determined, so a dated broker snapshot was not stored." },
        { status: 400 }
      );
    }

    const existingCfTrades = await listCfStatementTrades(user.id, portfolioTag);
    const existingKeys = new Set(existingCfTrades.map((trade) => trade.importRowKey));
    const existingExecutionHistory = existingCfTrades.flatMap(executionHistoryFromTrade);
    const statementExecutionHistory = statementTrades.flatMap(executionHistoryFromTrade);
    const executionHistory = mergeCfExecutionHistory(existingExecutionHistory, statementExecutionHistory);
    const rebuiltTrades = buildCfTradesFromExecutionHistory(
      executionHistory,
      parsedStatement.openPositions,
      parsedStatement.workingOrders,
      user.id,
      portfolioTag
    ).map((trade) => applyManualFields(trade, existingCfTrades));

    const tradesChanged = !cfImportTradesEquivalent(existingCfTrades, rebuiltTrades);
    const activeWorkingOrders = replaceActiveWorkingOrders(parsedStatement.workingOrders);
    const saved = await replaceCfStatementImport(user.id, portfolioTag, rebuiltTrades, {
      currentEquity: parsedStatement.currentEquity,
      statementEquity: parsedStatement.statementEquity,
      floatingPnl: parsedStatement.floatingPnl,
      equitySource: "CF Import",
      equityStatementDate: parsedStatement.equityStatementDate,
      workingOrders: activeWorkingOrders
    }, {
      userId: user.id,
      portfolioTag,
      coverageDate: parsedStatement.equityStatementDate,
      sourceHash,
      sourceFilename: file.name,
      balance: parsedStatement.balance,
      currentEquity: parsedStatement.statementEquity,
      statementEquity: parsedStatement.statementEquity,
      floatingPnl: parsedStatement.floatingPnl,
      openPositions: parsedStatement.openPositions,
      workingOrders: activeWorkingOrders
    }, tradesChanged);
    const created = rebuiltTrades.filter((trade) => !existingKeys.has(trade.importRowKey)).length;
    const updated = tradesChanged ? rebuiltTrades.length - created : 0;
    const openTrades = rebuiltTrades.filter((trade) => trade.status === "OPEN").length;
    const closedTrades = rebuiltTrades.length - openTrades;
    const needsReview = rebuiltTrades.filter((trade) => trade.customTags.includes("Needs review")).length;

    return NextResponse.json({
      imported: rebuiltTrades.length,
      created,
      updated,
      openTrades,
      closedTrades,
      needsReview,
      replaced: saved.count,
      tradesChanged: saved.tradesReplaced,
      currentEquity: parsedStatement.currentEquity,
      statementEquity: parsedStatement.statementEquity,
      floatingPnl: parsedStatement.floatingPnl,
      equityStatementDate: parsedStatement.equityStatementDate
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? `CF import failed on the server: ${error.message}`
            : "CF import failed on the server."
      },
      { status: 500 }
    );
  }
}
