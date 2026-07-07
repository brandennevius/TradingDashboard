import {
  AlignmentType,
  BorderStyle,
  Document,
  HeadingLevel,
  ImageRun,
  Packer,
  Paragraph,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType
} from "docx";
import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getSetupChecklistTemplates, getTradeScreenshot, listBrandenVisibleTrades } from "@/lib/store";
import type { SetupChecklistTemplate, TradeChecklistItem, TradeLogEntry } from "@/lib/types";

const BREAKEVEN_R_THRESHOLD = 0.1;
const PAGE_WIDTH_DXA = 9360;
const ACCENT = "6F8F5F";
const LIGHT_GREEN = "E9F6E4";
const BORDER = "C8DDBD";

type ReviewPromptTrade = {
  reviewKey: string;
  symbol: string;
  side: string;
  status: string;
  setup: string;
  grade: string;
  pnl: number;
  rMultiple: number;
  risk: number;
  entryDate: string;
  exitDate: string;
  notes: string;
  metCriteria: string[];
  missedCriteria: string[];
  strategyKnowledge: {
    setupName: string;
    title: string;
    sourceType: string;
    url: string;
    content: string;
    relevanceScore: number;
  }[];
  modelExampleMatches: {
    setupName: string;
    symbol: string;
    setupType: string;
    quality: string;
    outcome: string;
    source: string;
    notes: string;
    screenshotCount: number;
    relevanceScore: number;
  }[];
  executions: {
    type: string;
    date: string;
    price: number;
    shares: number;
    pnl: number;
  }[];
  screenshotCount: number;
  chartLinks: string[];
};

type TradeReview = {
  mainLesson: string;
  primaryRead: string;
  reviewNotes: string;
  chartAnalysis: {
    visibleText: string[];
    patternRead: string;
    keyLevels: string[];
    relativeStrengthRead: string;
    volumeRead: string;
    setupComparison: string;
    confidence: "low" | "medium" | "high";
  };
  actionItems: string[];
};

type AiReview = {
  overallTakeaway: string;
  keyThemes: string[];
  improved: string[];
  needsWork: string[];
  upcomingFocus: string[];
  bottomLine: string;
  tradeReviews: Record<string, TradeReview>;
};

const tradeReviewSchema = {
  type: "object",
  additionalProperties: false,
  required: ["mainLesson", "primaryRead", "reviewNotes", "chartAnalysis", "actionItems"],
  properties: {
    mainLesson: { type: "string" },
    primaryRead: { type: "string" },
    reviewNotes: { type: "string" },
    chartAnalysis: {
      type: "object",
      additionalProperties: false,
      required: ["visibleText", "patternRead", "keyLevels", "relativeStrengthRead", "volumeRead", "setupComparison", "confidence"],
      properties: {
        visibleText: { type: "array", items: { type: "string" } },
        patternRead: { type: "string" },
        keyLevels: { type: "array", items: { type: "string" } },
        relativeStrengthRead: { type: "string" },
        volumeRead: { type: "string" },
        setupComparison: {
          type: "string",
          description: "Compare the trade chart against setup criteria, strategy knowledge, and modelExampleMatches when provided."
        },
        confidence: { type: "string", enum: ["low", "medium", "high"] }
      }
    },
    actionItems: {
      type: "array",
      minItems: 1,
      items: { type: "string" }
    }
  }
};

function aiReviewJsonSchema(promptTrades: ReviewPromptTrade[]) {
  const tradeReviewProperties = Object.fromEntries(promptTrades.map((trade) => [trade.reviewKey, tradeReviewSchema]));
  const requiredTradeKeys = promptTrades.map((trade) => trade.reviewKey);

  return {
    name: "trade_review_export",
    strict: true,
    schema: {
      type: "object",
      additionalProperties: false,
      required: ["overallTakeaway", "keyThemes", "improved", "needsWork", "upcomingFocus", "bottomLine", "tradeReviews"],
      properties: {
        overallTakeaway: { type: "string" },
        keyThemes: { type: "array", items: { type: "string" } },
        improved: { type: "array", items: { type: "string" } },
        needsWork: { type: "array", items: { type: "string" } },
        upcomingFocus: { type: "array", items: { type: "string" } },
        bottomLine: { type: "string" },
        tradeReviews: {
          type: "object",
          additionalProperties: false,
          required: requiredTradeKeys,
          properties: tradeReviewProperties
        }
      }
    }
  };
}

function money(value: number) {
  const sign = value < 0 ? "-" : "";
  return `${sign}$${Math.abs(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
}

function pct(value: number) {
  return `${(value || 0).toFixed(2)}%`;
}

function fmt(value: number, digits = 2) {
  return Number(value || 0).toFixed(digits);
}

function safeFilePart(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/(^-|-$)/g, "") || "trade-review";
}

function normalizedTradeStatus(trade: TradeLogEntry): TradeLogEntry["status"] {
  const pnl = Number(trade.pnl || 0);
  const rMultiple = Number(trade.rMultiple || 0);
  if (trade.status === "OPEN") return "OPEN";
  if (Math.abs(rMultiple) < BREAKEVEN_R_THRESHOLD) return "BREAKEVEN";
  if (pnl > 0) return "WIN";
  if (pnl < 0) return "LOSS";
  return "BREAKEVEN";
}

function dateInRange(value: string, startDate: string, endDate: string) {
  if (!value) return false;
  if (startDate && value < startDate) return false;
  if (endDate && value > endDate) return false;
  return true;
}

function rangedExitExecutions(trade: TradeLogEntry, startDate: string, endDate: string) {
  return trade.executions.filter(
    (execution) => execution.type === "EXIT" && dateInRange(execution.date, startDate, endDate)
  );
}

function tradeForRange(trade: TradeLogEntry, startDate: string, endDate: string): TradeLogEntry {
  const exits = rangedExitExecutions(trade, startDate, endDate);
  if (!exits.length) return trade;
  const pnl = exits.reduce((total, execution) => total + Number(execution.pnl || 0), 0);
  const rMultiple = trade.risk ? pnl / trade.risk : 0;
  return {
    ...trade,
    pnl,
    rMultiple,
    status: normalizedTradeStatus({ ...trade, pnl, rMultiple }),
    exitDate: exits[exits.length - 1]?.date || trade.exitDate
  };
}

function countsAsSettledTrade(trade: TradeLogEntry) {
  const hasPartialExits = trade.customTags.some((tag) => tag.trim().toLowerCase() === "partial exits");
  return trade.status !== "OPEN" || (hasPartialExits && Number(trade.pnl || 0) !== 0);
}

function primarySetup(trade: TradeLogEntry) {
  return trade.setupTags[0] || "No setup";
}

function setupTemplateFor(setupName: string, templates: SetupChecklistTemplate[]) {
  return templates.find((template) => template.setupName.toLowerCase() === setupName.toLowerCase());
}

function resolvedChecklistItems(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const template = setupTemplateFor(primarySetup(trade), templates);
  const tradeItems = trade.checklistItems || [];

  if (!template) return tradeItems;

  return template.criteria.map((criterion) => {
    const existing = tradeItems.find((item) => item.id === criterion.id || item.criteria === criterion.criteria);
    return {
      id: criterion.id,
      criteria: criterion.criteria,
      points: criterion.points,
      inputType: criterion.inputType,
      groupName:
        template.groups.find((group) => group.criteria.some((groupCriterion) => groupCriterion.id === criterion.id))
          ?.name || "",
      met: Boolean(existing?.met),
      score: typeof existing?.score === "number" ? existing.score : existing?.met ? criterion.points : 0
    } satisfies TradeChecklistItem;
  });
}

function checklistScore(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const template = setupTemplateFor(primarySetup(trade), templates);
  const items = resolvedChecklistItems(trade, templates);
  const earned = items.reduce((total, item) => total + (item.inputType === "points" ? Number(item.score || 0) : item.met ? item.points : 0), 0);
  const total = items.reduce((sum, item) => sum + Number(item.points || 0), 0);
  const pctScore = total ? (earned / total) * 100 : 0;
  const grade =
    trade.manualGrade ||
    template?.gradeBands.find((band) => pctScore >= band.minScore && (band.maxScore === null || pctScore <= band.maxScore))?.label ||
    (total ? `${fmt(pctScore, 0)}%` : "Ungraded");

  return { earned, total, pctScore, grade };
}

function sortedTradesByRequest(trades: TradeLogEntry[], requestedIds: string[]) {
  const byId = new Map(trades.map((trade) => [trade.id, trade]));
  return requestedIds.map((id) => byId.get(id)).filter((trade): trade is TradeLogEntry => Boolean(trade));
}

function tokenize(value: string) {
  const stopwords = new Set([
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "trade",
    "trades",
    "setup",
    "price",
    "stock",
    "entry",
    "exit"
  ]);
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((token) => token.length >= 3 && !stopwords.has(token));
}

function limitText(value: string, maxLength: number) {
  const text = String(value || "").trim();
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
}

function fallbackChunks(source: NonNullable<SetupChecklistTemplate["knowledgeSources"]>[number]) {
  if (source.chunks?.length) return source.chunks;
  if (!source.content.trim()) return [];

  const chunks: { id: string; title: string; content: string; order: number }[] = [];
  const paragraphs = source.content.split(/\n{2,}/).map((paragraph) => paragraph.trim()).filter(Boolean);
  let current = "";

  for (const paragraph of paragraphs) {
    if (current && `${current}\n\n${paragraph}`.length > 4500) {
      chunks.push({ id: `${source.id}-chunk-${chunks.length + 1}`, title: `${source.title} ${chunks.length + 1}`, content: current, order: chunks.length });
      current = paragraph;
    } else {
      current = current ? `${current}\n\n${paragraph}` : paragraph;
    }
  }

  if (current) {
    chunks.push({ id: `${source.id}-chunk-${chunks.length + 1}`, title: `${source.title} ${chunks.length + 1}`, content: current, order: chunks.length });
  }

  return chunks;
}

function relevantStrategyKnowledge(
  trade: TradeLogEntry,
  templates: SetupChecklistTemplate[],
  metCriteria: string[],
  missedCriteria: string[]
) {
  const setupNames = trade.setupTags.length ? trade.setupTags : [primarySetup(trade)];
  const query = [
    trade.symbol,
    trade.side,
    normalizedTradeStatus(trade),
    primarySetup(trade),
    trade.notes,
    metCriteria.join(" "),
    missedCriteria.join(" "),
    trade.executions.map((execution) => `${execution.type} ${execution.date} ${execution.pnl}`).join(" ")
  ].join(" ");
  const queryTokens = new Set(tokenize(query));

  return setupNames
    .flatMap((setupName) => {
      const template = setupTemplateFor(setupName, templates);
      return (template?.knowledgeSources || []).flatMap((source) =>
        source.active === false
          ? []
          : fallbackChunks(source)
          .filter((chunk) => chunk.content.trim())
          .map((chunk) => {
            const chunkTokens = tokenize(`${chunk.title} ${chunk.content}`);
            const overlap = chunkTokens.reduce((score, token) => score + (queryTokens.has(token) ? 1 : 0), 0);
            const setupBoost = chunkTokens.some((token) => queryTokens.has(token)) ? 2 : 0;
            return {
              setupName: template?.setupName || setupName,
              title: `${source.title}${chunk.title && chunk.title !== source.title ? ` - ${chunk.title}` : ""}`,
              sourceType: source.sourceType,
              url: source.url,
              content: limitText(chunk.content, 2200),
              relevanceScore: overlap + setupBoost
            };
          })
      );
    })
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, 5);
}

function relevantStrategyExamples(
  trade: TradeLogEntry,
  templates: SetupChecklistTemplate[],
  metCriteria: string[],
  missedCriteria: string[]
) {
  const setupNames = trade.setupTags.length ? trade.setupTags : [primarySetup(trade)];
  const query = [
    trade.symbol,
    trade.side,
    normalizedTradeStatus(trade),
    primarySetup(trade),
    trade.notes,
    metCriteria.join(" "),
    missedCriteria.join(" "),
    trade.executions.map((execution) => `${execution.type} ${execution.date} ${execution.pnl}`).join(" ")
  ].join(" ");
  const queryTokens = new Set(tokenize(query));

  return setupNames
    .flatMap((setupName) => {
      const template = setupTemplateFor(setupName, templates);
      return (template?.strategyExamples || []).flatMap((example) => {
        if (example.active === false) return [];
        const exampleText = [
          example.symbol,
          example.setupType,
          example.quality,
          example.outcome,
          example.source,
          example.notes
        ].join(" ");
        const exampleTokens = tokenize(exampleText);
        const overlap = exampleTokens.reduce((score, token) => score + (queryTokens.has(token) ? 1 : 0), 0);
        const setupBoost =
          example.setupType && primarySetup(trade).toLowerCase().includes(example.setupType.toLowerCase()) ? 4 : 0;
        const qualityBoost = example.quality === "ideal" || example.quality === "failed" ? 1 : 0;
        return {
          setupName: template?.setupName || setupName,
          symbol: example.symbol,
          setupType: example.setupType,
          quality: example.quality,
          outcome: example.outcome,
          source: example.source,
          notes: limitText(example.notes, 1400),
          screenshotCount: example.screenshots?.length || 0,
          relevanceScore: overlap + setupBoost + qualityBoost
        };
      });
    })
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, 4);
}

function buildPromptTrades(trades: TradeLogEntry[], templates: SetupChecklistTemplate[]): ReviewPromptTrade[] {
  return trades.map((trade) => {
    const items = resolvedChecklistItems(trade, templates);
    const grade = checklistScore(trade, templates).grade;
    const metCriteria = items.filter((item) => item.met || Number(item.score || 0) > 0).map((item) => item.criteria);
    const missedCriteria = items.filter((item) => !item.met && !Number(item.score || 0)).map((item) => item.criteria);
    const strategyKnowledge = relevantStrategyKnowledge(trade, templates, metCriteria, missedCriteria);
    const modelExampleMatches = relevantStrategyExamples(trade, templates, metCriteria, missedCriteria);

    return {
      reviewKey: trade.id,
      symbol: trade.symbol,
      side: trade.side,
      status: normalizedTradeStatus(trade),
      setup: primarySetup(trade),
      grade,
      pnl: trade.pnl,
      rMultiple: trade.rMultiple,
      risk: trade.risk,
      entryDate: trade.entryDate,
      exitDate: trade.exitDate,
      notes: trade.notes,
      metCriteria,
      missedCriteria,
      strategyKnowledge,
      modelExampleMatches,
      executions: trade.executions.map((execution) => ({
        type: execution.type,
        date: execution.date,
        price: execution.price,
        shares: execution.shares,
        pnl: execution.pnl
      })),
      screenshotCount: trade.screenshots.length,
      chartLinks: trade.chartLinks
    };
  });
}

function promptTradesForOpenAi(promptTrades: ReviewPromptTrade[]) {
  return promptTrades.map((trade) => ({
    ...trade,
    notes: limitText(trade.notes, 1200),
    metCriteria: trade.metCriteria.slice(0, 25),
    missedCriteria: trade.missedCriteria.slice(0, 25),
    strategyKnowledge: trade.strategyKnowledge.slice(0, 3).map((source) => ({
      ...source,
      content: limitText(source.content, 1600)
    })),
    modelExampleMatches: trade.modelExampleMatches.slice(0, 3).map((example) => ({
      ...example,
      notes: limitText(example.notes, 1200)
    })),
    executions: trade.executions.slice(0, 40)
  }));
}

function numericContext(trades: TradeLogEntry[], templates: SetupChecklistTemplate[]) {
  const settled = trades.filter(countsAsSettledTrade);
  const wins = settled.filter((trade) => normalizedTradeStatus(trade) === "WIN");
  const losses = settled.filter((trade) => normalizedTradeStatus(trade) === "LOSS");
  const netPnl = settled.reduce((sum, trade) => sum + Number(trade.pnl || 0), 0);
  const totalR = settled.reduce((sum, trade) => sum + Number(trade.rMultiple || 0), 0);
  const winRate = settled.length ? (wins.length / settled.length) * 100 : 0;
  const scored = trades.map((trade) => ({ trade, score: checklistScore(trade, templates) }));
  const best = [...scored].sort((a, b) => b.score.pctScore - a.score.pctScore)[0];
  const worst = [...scored].sort((a, b) => a.score.pctScore - b.score.pctScore)[0];
  const biggestWin = [...wins].sort((a, b) => Number(b.pnl || 0) - Number(a.pnl || 0))[0];
  const biggestLoss = [...losses].sort((a, b) => Number(a.pnl || 0) - Number(b.pnl || 0))[0];
  const openCount = trades.filter((trade) => normalizedTradeStatus(trade) === "OPEN").length;
  const missingNotes = trades.filter((trade) => !trade.notes.trim()).length;
  const missingScreenshots = trades.filter((trade) => !trade.screenshots.length && !trade.chartLinks.length).length;

  return {
    tradeCount: trades.length,
    closedCount: settled.length,
    openCount,
    winCount: wins.length,
    lossCount: losses.length,
    netPnl,
    totalR,
    winRate,
    bestSetupScore: best ? { symbol: best.trade.symbol, grade: best.score.grade, score: best.score.pctScore } : null,
    weakestSetupScore: worst ? { symbol: worst.trade.symbol, grade: worst.score.grade, score: worst.score.pctScore } : null,
    biggestWin: biggestWin ? { symbol: biggestWin.symbol, pnl: biggestWin.pnl, rMultiple: biggestWin.rMultiple } : null,
    biggestLoss: biggestLoss ? { symbol: biggestLoss.symbol, pnl: biggestLoss.pnl, rMultiple: biggestLoss.rMultiple } : null,
    missingNotes,
    missingScreenshots
  };
}

function compactReviewKey(value: string) {
  return String(value || "").toUpperCase().replace(/[^A-Z0-9]/g, "");
}

function reviewForTrade(
  tradeReviews: Partial<Record<string, TradeReview>>,
  trade: TradeLogEntry,
  promptTrade: ReviewPromptTrade | undefined
) {
  const possibleKeys = [
    trade.id,
    promptTrade?.reviewKey,
    trade.symbol,
    compactReviewKey(trade.symbol),
    trade.symbol.replace("/", ""),
    trade.symbol.replace("-", ""),
    trade.symbol.split("/")[0]
  ].filter(Boolean) as string[];

  const direct = possibleKeys.map((key) => tradeReviews[key]).find(Boolean);
  if (direct) return direct;

  const compactSymbol = compactReviewKey(trade.symbol);
  const looseMatch = Object.entries(tradeReviews).find(([key]) => compactReviewKey(key) === compactSymbol);
  return looseMatch?.[1];
}

function validateAiReview(value: unknown, trades: TradeLogEntry[], promptTrades: ReviewPromptTrade[]): AiReview {
  const parsed = value as Partial<AiReview>;
  const tradeReviews = parsed.tradeReviews || {};
  const promptTradeById = new Map(promptTrades.map((trade) => [trade.reviewKey, trade]));
  const normalizedTradeReviews: Record<string, TradeReview> = {};

  for (const trade of trades) {
    const review = reviewForTrade(tradeReviews, trade, promptTradeById.get(trade.id));
    if (!review?.primaryRead || !review.reviewNotes || !Array.isArray(review.actionItems) || !review.actionItems.length) {
      throw new Error(`OpenAI review was missing the required trade review for ${trade.symbol}.`);
    }
    normalizedTradeReviews[trade.id] = {
      mainLesson: String(review.mainLesson || ""),
      primaryRead: String(review.primaryRead || ""),
      reviewNotes: String(review.reviewNotes || ""),
      chartAnalysis: normalizeChartAnalysis(review.chartAnalysis),
      actionItems: Array.isArray(review.actionItems) ? review.actionItems.map(String) : []
    };
  }

  if (
    !parsed.overallTakeaway ||
    !Array.isArray(parsed.keyThemes) ||
    !Array.isArray(parsed.improved) ||
    !Array.isArray(parsed.needsWork) ||
    !Array.isArray(parsed.upcomingFocus) ||
    !parsed.bottomLine
  ) {
    throw new Error("OpenAI review response was missing required summary fields.");
  }

  return {
    overallTakeaway: String(parsed.overallTakeaway),
    keyThemes: parsed.keyThemes.map(String),
    improved: parsed.improved.map(String),
    needsWork: parsed.needsWork.map(String),
    upcomingFocus: parsed.upcomingFocus.map(String),
    bottomLine: String(parsed.bottomLine),
    tradeReviews: normalizedTradeReviews
  };
}

function normalizeChartAnalysis(value: unknown): TradeReview["chartAnalysis"] {
  if (value && typeof value === "object") {
    const raw = value as Record<string, unknown>;
    const confidence = String(raw.confidence || "low").toLowerCase();
    return {
      visibleText: Array.isArray(raw.visibleText) ? raw.visibleText.map(String).filter(Boolean) : [],
      patternRead: String(raw.patternRead || "No clear chart pattern identified."),
      keyLevels: Array.isArray(raw.keyLevels) ? raw.keyLevels.map(String).filter(Boolean) : [],
      relativeStrengthRead: String(raw.relativeStrengthRead || "Relative strength was not clear from the screenshot."),
      volumeRead: String(raw.volumeRead || "Volume was not clear from the screenshot."),
      setupComparison: String(raw.setupComparison || "Insufficient visual evidence from screenshots."),
      confidence: confidence === "high" || confidence === "medium" ? confidence : "low"
    };
  }

  return {
    visibleText: [],
    patternRead: String(value || "No clear chart pattern identified."),
    keyLevels: [],
    relativeStrengthRead: "Relative strength was not clear from the screenshot.",
    volumeRead: "Volume was not clear from the screenshot.",
    setupComparison: "Insufficient visual evidence from screenshots.",
    confidence: "low"
  };
}

async function generateAiReview(trades: TradeLogEntry[], templates: SetupChecklistTemplate[], promptTrades: ReviewPromptTrade[]) {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    throw new Error("OPENAI_API_KEY is not configured. Add it in Vercel environment variables and try again.");
  }
  if (!trades.length) {
    throw new Error("No trades were provided for review.");
  }

  const model = process.env.OPENAI_REVIEW_MODEL || "gpt-4.1-mini";
  const context = numericContext(trades, templates);
  const userContent = await buildOpenAiReviewContent(trades, promptTrades, context);
  const responseFormat = { type: "json_schema", json_schema: aiReviewJsonSchema(promptTrades) };

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      temperature: 0.35,
      response_format: responseFormat,
      messages: [
        {
          role: "system",
          content:
            "You are a trading performance coach writing a weekly review for a swing trader. Be specific, direct, risk-aware, chart-aware, and process-focused. Do not use generic motivational language. Use only the provided trade data, notes, setup criteria, setup strategy knowledge, executions, screenshots, and stats. Return valid JSON only."
        },
        {
          role: "user",
          content: userContent
        }
      ]
    })
  });

  if (!response.ok) {
    const details = await response.text().catch(() => "");
    throw new Error(`OpenAI review failed (${response.status}). ${details.slice(0, 300)}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content;
  if (!content) {
    throw new Error("OpenAI review returned an empty response.");
  }

  return validateAiReview(JSON.parse(content), trades, promptTrades);
}

async function buildOpenAiReviewContent(trades: TradeLogEntry[], promptTrades: ReviewPromptTrade[], context: ReturnType<typeof numericContext>) {
  const compactPromptTrades = promptTradesForOpenAi(promptTrades);
  const requiredReviewKeys = compactPromptTrades.map((trade) => `${trade.reviewKey} (${trade.symbol})`);
  const content: Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string; detail: "low" | "high" | "auto" } }> = [
    {
      type: "text",
      text: `Write a high-quality trade review document from the filtered trades.

Required JSON shape:
{
  "overallTakeaway": "specific paragraph",
  "keyThemes": ["specific bullet", "..."],
  "improved": ["specific bullet", "..."],
  "needsWork": ["specific bullet", "..."],
  "upcomingFocus": ["specific bullet", "..."],
  "bottomLine": "specific closing paragraph",
  "tradeReviews": {
    "REVIEW_KEY_FROM_TRADES": {
      "mainLesson": "one specific sentence",
      "primaryRead": "one concise paragraph",
      "reviewNotes": "one detailed paragraph using notes/criteria/executions",
      "chartAnalysis": {
        "visibleText": ["exact readable annotations or on-screen labels from screenshots"],
        "patternRead": "chart pattern / structure identified from the screenshot",
        "keyLevels": ["visible support, resistance, pivot, moving average, stop, or breakout levels"],
        "relativeStrengthRead": "RS/relative strength interpretation if visible, otherwise say unclear",
        "volumeRead": "volume interpretation if visible, otherwise say unclear",
        "setupComparison": "comparison against setup criteria and strategy knowledge",
        "confidence": "low|medium|high"
      },
      "actionItems": ["specific action", "specific action"]
    }
  }
}

Rules:
- Include one tradeReviews entry for every trade.
- Key each tradeReviews entry by the exact reviewKey field from the Trades JSON. Do not key by ticker, symbol, or display name.
- Required review keys: ${requiredReviewKeys.join(", ")}
- Do not say "good discipline" or "needs discipline" unless tied to a concrete execution/criteria fact.
- Call out if risk size did not match setup grade/criteria.
- Review each trade against the strategyKnowledge attached to its setup tag. If strategyKnowledge conflicts with the checklist, explain the conflict.
- If strategyKnowledge is missing for a setup, say the strategy context is incomplete instead of pretending to know the full strategy.
- Use modelExampleMatches when provided. Explicitly compare the current trade to the closest ideal/good/failed/cautionary examples, and say when no example is close enough.
- Mention entry/exit quality, stop/invalidation quality, adds/partials if executions imply them, and missing notes/screenshots when relevant.
- Use attached screenshots for chartAnalysis. Read visible annotations/text on the screenshot exactly when readable. Identify chart pattern/structure, key levels, moving averages, volume behavior, relative strength behavior, extension/pullback/base/breakout/failure, and whether the screenshot supports or contradicts the setup tag.
- If text is blurry or cropped, do not invent it. Use an empty visibleText array or describe uncertainty in the relevant field.
- If the attached image is not a readable chart or there is no image for a trade, use low confidence and explain insufficient visual evidence.
- Keep bullets concise but not vague.

Numeric context:
${JSON.stringify(context)}

Trades:
${JSON.stringify(compactPromptTrades)}`
    }
  ];

  const imageParts = await openAiScreenshotParts(trades);
  content.push(...imageParts);
  return content;
}

function text(text: string, options: { bold?: boolean; color?: string; size?: number; italics?: boolean } = {}) {
  return new TextRun({
    text,
    bold: options.bold,
    color: options.color || "263026",
    size: options.size || 22,
    italics: options.italics,
    font: "Arial"
  });
}

function paragraph(children: (TextRun | ImageRun)[], options: { spacingAfter?: number; heading?: typeof HeadingLevel[keyof typeof HeadingLevel]; alignment?: typeof AlignmentType[keyof typeof AlignmentType] } = {}) {
  return new Paragraph({
    children,
    heading: options.heading,
    alignment: options.alignment,
    spacing: { after: options.spacingAfter ?? 160 }
  });
}

function heading(label: string, level: typeof HeadingLevel[keyof typeof HeadingLevel] = HeadingLevel.HEADING_1) {
  return new Paragraph({
    text: label,
    heading: level,
    spacing: { before: level === HeadingLevel.HEADING_1 ? 280 : 180, after: 120 }
  });
}

function bullet(label: string) {
  return new Paragraph({
    children: [text(label)],
    bullet: { level: 0 },
    spacing: { after: 80 }
  });
}

function tableCell(label: string, options: { bold?: boolean; fill?: string; width?: number; color?: string } = {}) {
  return new TableCell({
    width: options.width ? { size: options.width, type: WidthType.DXA } : undefined,
    shading: options.fill ? { fill: options.fill } : undefined,
    margins: { top: 100, bottom: 100, left: 120, right: 120 },
    children: [new Paragraph({ children: [text(label, { bold: options.bold, color: options.color, size: 19 })], spacing: { after: 0 } })]
  });
}

function simpleTable(rows: string[][], widths: number[]) {
  return new Table({
    width: { size: PAGE_WIDTH_DXA, type: WidthType.DXA },
    columnWidths: widths,
    borders: {
      top: { style: BorderStyle.SINGLE, color: BORDER, size: 1 },
      bottom: { style: BorderStyle.SINGLE, color: BORDER, size: 1 },
      left: { style: BorderStyle.SINGLE, color: BORDER, size: 1 },
      right: { style: BorderStyle.SINGLE, color: BORDER, size: 1 },
      insideHorizontal: { style: BorderStyle.SINGLE, color: BORDER, size: 1 },
      insideVertical: { style: BorderStyle.SINGLE, color: BORDER, size: 1 }
    },
    rows: rows.map((row, index) =>
      new TableRow({
        children: row.map((cell, cellIndex) =>
          tableCell(cell, {
            bold: index === 0 || cellIndex === 0,
            fill: index === 0 ? LIGHT_GREEN : undefined,
            width: widths[cellIndex]
          })
        )
      })
    )
  });
}

function summaryStatsTable(trades: TradeLogEntry[]) {
  const settled = trades.filter(countsAsSettledTrade);
  const wins = settled.filter((trade) => normalizedTradeStatus(trade) === "WIN");
  const losses = settled.filter((trade) => normalizedTradeStatus(trade) === "LOSS");
  const net = settled.reduce((sum, trade) => sum + Number(trade.pnl || 0), 0);
  const totalR = settled.reduce((sum, trade) => sum + Number(trade.rMultiple || 0), 0);
  const winRate = settled.length ? (wins.length / settled.length) * 100 : 0;

  return simpleTable(
    [
      ["Visible Trades", "Settled", "Wins", "Losses", "Settled P&L", "Total R", "Win Rate"],
      [String(trades.length), String(settled.length), String(wins.length), String(losses.length), money(net), `${fmt(totalR)}R`, pct(winRate)]
    ],
    [1200, 1200, 1200, 1200, 1500, 1200, 1860]
  );
}

function scorecardTable(trades: TradeLogEntry[], templates: SetupChecklistTemplate[], review: AiReview) {
  return simpleTable(
    [
      ["Ticker", "Setup", "Grade", "Status", "Result / Risk", "Main Lesson"],
      ...trades.map((trade) => {
        const status = normalizedTradeStatus(trade);
        const score = checklistScore(trade, templates);
        const tradeReview = review.tradeReviews[trade.id];
        return [
          trade.symbol,
          primarySetup(trade),
          score.grade,
          status,
          `${money(trade.pnl)} / ${fmt(trade.rMultiple)}R, risk ${money(trade.risk)}`,
          tradeReview?.mainLesson || tradeReview?.primaryRead || "Review note unavailable."
        ];
      })
    ],
    [900, 1500, 900, 1000, 1700, 3360]
  );
}

function criteriaSummary(trade: TradeLogEntry, templates: SetupChecklistTemplate[]) {
  const items = resolvedChecklistItems(trade, templates);
  if (!items.length) return ["No setup criteria were found for this setup."];

  return items.slice(0, 14).map((item) => {
    const earned = item.inputType === "points" ? Number(item.score || 0) : item.met ? item.points : 0;
    const prefix = earned > 0 ? "Met" : "Missed";
    return `${prefix}: ${item.criteria} (${fmt(earned, 1)}/${fmt(item.points, 1)} pts)`;
  });
}

function chartAnalysisBullets(chartAnalysis: TradeReview["chartAnalysis"]) {
  return [
    `Pattern: ${chartAnalysis.patternRead}`,
    chartAnalysis.keyLevels.length ? `Key levels: ${chartAnalysis.keyLevels.join(", ")}` : "",
    chartAnalysis.visibleText.length ? `Visible chart text: ${chartAnalysis.visibleText.join(" | ")}` : "",
    `Relative strength: ${chartAnalysis.relativeStrengthRead}`,
    `Volume: ${chartAnalysis.volumeRead}`,
    `Setup comparison: ${chartAnalysis.setupComparison}`,
    `Confidence: ${chartAnalysis.confidence}`
  ].filter(Boolean);
}

function extractScreenshotId(url: string) {
  const match = url.match(/\/screenshots\/([^/?#]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

function extensionFromMime(mimeType: string): "jpg" | "png" | "gif" | "bmp" {
  if (mimeType.includes("jpeg") || mimeType.includes("jpg")) return "jpg";
  if (mimeType.includes("gif")) return "gif";
  if (mimeType.includes("bmp")) return "bmp";
  return "png";
}

function imageFromDataUrl(value: string) {
  const match = value.match(/^data:([^;]+);base64,(.+)$/);
  if (!match) return null;
  return {
    data: Buffer.from(match[2], "base64"),
    type: extensionFromMime(match[1])
  };
}

async function screenshotDataUrl(screenshot: string, trade: TradeLogEntry) {
  const inlineImage = imageFromDataUrl(screenshot);
  if (inlineImage) {
    return `data:image/${inlineImage.type};base64,${inlineImage.data.toString("base64")}`;
  }

  const screenshotId = extractScreenshotId(screenshot);
  if (!screenshotId) return "";

  const stored = await getTradeScreenshot(screenshotId);
  if (!stored || stored.tradeId !== trade.id || stored.userId !== "branden") return "";

  return `data:${stored.mimeType};base64,${stored.imageData.toString("base64")}`;
}

async function openAiScreenshotParts(trades: TradeLogEntry[]) {
  const parts: Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string; detail: "low" } }> = [];
  let totalImages = 0;
  const maxImages = 10;
  const maxImagesPerTrade = 2;
  const maxDataUrlChars = 4_500_000;

  for (const trade of trades) {
    if (totalImages >= maxImages) break;
    const screenshots = trade.screenshots.slice(0, maxImagesPerTrade);

    for (let index = 0; index < screenshots.length; index += 1) {
      if (totalImages >= maxImages) break;
      const url = await screenshotDataUrl(screenshots[index], trade);
      if (!url || url.length > maxDataUrlChars) continue;

      parts.push({ type: "text", text: `Screenshot for ${trade.symbol}, image ${index + 1}. Analyze this only for ${trade.symbol}.` });
      parts.push({ type: "image_url", image_url: { url, detail: "low" } });
      totalImages += 1;
    }
  }

  return parts;
}

async function imageRunsForTrade(trade: TradeLogEntry) {
  const runs: Paragraph[] = [];
  const screenshots = trade.screenshots.slice(0, 4);

  for (let index = 0; index < screenshots.length; index += 1) {
    const screenshot = screenshots[index];
    let image: { data: Buffer; type: "jpg" | "png" | "gif" | "bmp" } | null = imageFromDataUrl(screenshot);

    if (!image) {
      const screenshotId = extractScreenshotId(screenshot);
      if (screenshotId) {
        const stored = await getTradeScreenshot(screenshotId);
        if (stored && stored.tradeId === trade.id && stored.userId === "branden") {
          image = { data: stored.imageData, type: extensionFromMime(stored.mimeType) };
        }
      }
    }

    if (!image) continue;

    runs.push(paragraph([text(`${trade.symbol} chart screenshot ${index + 1}`, { bold: true, color: ACCENT })], { spacingAfter: 80 }));
    runs.push(
      new Paragraph({
        children: [
          new ImageRun({
            data: image.data,
            type: image.type,
            transformation: { width: 540, height: 310 },
            altText: {
              name: `${trade.symbol} screenshot ${index + 1}`,
              title: `${trade.symbol} screenshot ${index + 1}`,
              description: `Chart screenshot attached to ${trade.symbol}`
            }
          })
        ],
        spacing: { after: 180 }
      })
    );
  }

  if (!runs.length && trade.chartLinks.length) {
    runs.push(paragraph([text(`Chart links: ${trade.chartLinks.join(", ")}`)], { spacingAfter: 120 }));
  }

  return runs;
}

async function buildDocument(trades: TradeLogEntry[], templates: SetupChecklistTemplate[], startDate: string, endDate: string) {
  const promptTrades = buildPromptTrades(trades, templates);
  const review = await generateAiReview(trades, templates, promptTrades);
  const children: (Paragraph | Table)[] = [
    new Paragraph({
      children: [text("Branden Trade Review", { bold: true, size: 46, color: "263026" })],
      spacing: { after: 80 }
    }),
    paragraph([text(`Filtered review - ${startDate || "All dates"} to ${endDate || "today"}`, { color: ACCENT, bold: true, size: 24 })], {
      spacingAfter: 220
    }),
    heading("Overall Takeaway"),
    paragraph([text(review.overallTakeaway)], { spacingAfter: 180 }),
    summaryStatsTable(trades),
    heading("Key Themes", HeadingLevel.HEADING_2),
    ...review.keyThemes.map(bullet),
    heading("Trade Scorecard", HeadingLevel.HEADING_2),
    scorecardTable(trades, templates, review),
    heading("What Improved", HeadingLevel.HEADING_2),
    ...review.improved.map(bullet),
    heading("What Needs Work", HeadingLevel.HEADING_2),
    ...review.needsWork.map(bullet)
  ];

  for (const trade of trades) {
    const score = checklistScore(trade, templates);
    const status = normalizedTradeStatus(trade);
    const tradeReview = review.tradeReviews[trade.id];
    children.push(heading(`${trade.symbol} - ${primarySetup(trade)}`, HeadingLevel.HEADING_1));
    children.push(
      simpleTable(
        [
          ["Grade", score.grade, "Status", status],
          ["Result / Risk", `${money(trade.pnl)} / ${fmt(trade.rMultiple)}R`, "Risk", money(trade.risk)],
          ["Side", trade.side, "Shares", String(trade.shares || "-")],
          ["Entry", `${trade.entryDate} @ ${fmt(trade.avgEntry)}`, "Exit", trade.exitDate ? `${trade.exitDate} @ ${fmt(trade.exitPrice)}` : "Still open"],
          ["Stop", trade.stopPrice ? fmt(trade.stopPrice) : "-", "Target", trade.takeProfitPrice ? fmt(trade.takeProfitPrice) : "-"]
        ],
        [1400, 3280, 1400, 3280]
      )
    );
    children.push(heading("Primary Read", HeadingLevel.HEADING_2));
    children.push(paragraph([text(tradeReview.primaryRead)]));
    children.push(heading("Review Notes", HeadingLevel.HEADING_2));
    children.push(paragraph([text(tradeReview.reviewNotes)]));
    children.push(heading("Chart Analysis", HeadingLevel.HEADING_2));
    chartAnalysisBullets(tradeReview.chartAnalysis).forEach((item) => children.push(bullet(item)));
    children.push(heading("Action Items", HeadingLevel.HEADING_2));
    tradeReview.actionItems.forEach((item) => children.push(bullet(item)));
    children.push(heading("Setup Criteria Summary", HeadingLevel.HEADING_2));
    criteriaSummary(trade, templates).forEach((item) => children.push(bullet(item)));
  }

  children.push(heading("Bottom Line"));
  children.push(paragraph([text(review.bottomLine)]));
  children.push(heading("Upcoming Week Focus", HeadingLevel.HEADING_2));
  review.upcomingFocus.forEach((item) => children.push(bullet(item)));

  return new Document({
    creator: "Branden Journal",
    title: "Trade Review",
    description: "Generated from the filtered Branden trade log.",
    styles: {
      paragraphStyles: [
        {
          id: "Normal",
          name: "Normal",
          run: { font: "Arial", size: 22, color: "263026" },
          paragraph: { spacing: { after: 120, line: 276 } }
        },
        {
          id: "Heading1",
          name: "Heading 1",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: "Arial", size: 30, bold: true, color: "263026" },
          paragraph: { spacing: { before: 280, after: 120 } }
        },
        {
          id: "Heading2",
          name: "Heading 2",
          basedOn: "Normal",
          next: "Normal",
          quickFormat: true,
          run: { font: "Arial", size: 24, bold: true, color: ACCENT },
          paragraph: { spacing: { before: 180, after: 100 } }
        }
      ]
    },
    sections: [
      {
        properties: {
          page: {
            margin: { top: 720, right: 720, bottom: 720, left: 720 }
          }
        },
        children
      }
    ]
  });
}

export async function POST(request: Request) {
  const user = await getSessionUser();

  if (!user || (user.journalOwnerId || user.id) !== "branden") {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const body = await request.json().catch(() => ({}));
  const tradeIds = Array.isArray(body.tradeIds) ? body.tradeIds.map(String).filter(Boolean) : [];
  const startDate = String(body.startDate || "");
  const endDate = String(body.endDate || "");

  if (!tradeIds.length) {
    return NextResponse.json({ error: "No filtered trades were selected for export." }, { status: 400 });
  }

  const [allTrades, templates] = await Promise.all([listBrandenVisibleTrades(), getSetupChecklistTemplates()]);
  const selectedTrades = sortedTradesByRequest(allTrades, tradeIds).map((trade) => tradeForRange(trade, startDate, endDate));

  if (!selectedTrades.length) {
    return NextResponse.json({ error: "No matching trades were found for export." }, { status: 404 });
  }

  let buffer: Buffer;
  try {
    const document = await buildDocument(selectedTrades, templates, startDate, endDate);
    buffer = await Packer.toBuffer(document);
  } catch (error) {
    const message = error instanceof Error ? error.message : "OpenAI review generation failed.";
    return NextResponse.json(
      {
        error: `${message} Try again after OpenAI is fixed.`
      },
      { status: 502 }
    );
  }

  const filename = `branden-trade-review-${safeFilePart(startDate || "all")}-to-${safeFilePart(endDate || new Date().toISOString().slice(0, 10))}.docx`;

  return new NextResponse(new Uint8Array(buffer), {
    headers: {
      "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "Content-Disposition": `attachment; filename="${filename}"`,
      "Cache-Control": "no-store"
    }
  });
}
