import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { buildCfTradesFromExecutionHistory, parseCfStatementText } from "@/lib/cf-statement";
import {
  listCfStatementTrades,
  replaceCfStatementTrades,
  saveBrandenPortfolioMeta,
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

    const existingCfTrades = await listCfStatementTrades(user.id, portfolioTag);
    const existingKeys = new Set(existingCfTrades.map((trade) => trade.importRowKey));
    const executionHistory = [...existingCfTrades, ...statementTrades].flatMap(executionHistoryFromTrade);
    const rebuiltTrades = buildCfTradesFromExecutionHistory(
      executionHistory,
      parsedStatement.openPositions,
      parsedStatement.workingOrders,
      user.id,
      portfolioTag
    ).map((trade) => applyManualFields(trade, existingCfTrades));

    const saved = await replaceCfStatementTrades(user.id, portfolioTag, rebuiltTrades);
    const created = rebuiltTrades.filter((trade) => !existingKeys.has(trade.importRowKey)).length;
    const updated = rebuiltTrades.length - created;
    const openTrades = rebuiltTrades.filter((trade) => trade.status === "OPEN").length;
    const closedTrades = rebuiltTrades.length - openTrades;
    const needsReview = rebuiltTrades.filter((trade) => trade.customTags.includes("Needs review")).length;

    if (parsedStatement.currentEquity) {
      await saveBrandenPortfolioMeta(portfolioTag, {
        currentEquity: parsedStatement.currentEquity,
        statementEquity: parsedStatement.statementEquity,
        floatingPnl: parsedStatement.floatingPnl,
        equitySource: "CF Import",
        equityStatementDate: parsedStatement.equityStatementDate
      });
    }

    return NextResponse.json({
      imported: rebuiltTrades.length,
      created,
      updated,
      openTrades,
      closedTrades,
      needsReview,
      replaced: saved.count,
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
