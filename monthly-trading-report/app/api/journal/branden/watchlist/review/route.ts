import { NextResponse } from "next/server";
import { getSessionUser } from "@/lib/auth";
import { getCamJournalScreenshot, getSetupChecklistTemplates } from "@/lib/store";
import type { SetupChecklistTemplate, TradeChecklistItem, WatchlistItem } from "@/lib/types";

export const runtime = "nodejs";

type WatchlistReview = {
  verdict: "Actionable" | "Watch" | "Pass" | "Manage Existing";
  buyPlan: {
    recommendation: "Buy" | "Starter" | "Add" | "Wait" | "No Trade";
    primaryBuyLevel: string;
    starterBuyLevel: string;
    addOnBuyLevel: string;
    stopLevel: string;
    noTradeReason: string;
    canslimRule: string;
  };
  tradeDeskVerdict: string;
  technicalSetupGrade: string;
  canslimQualityGrade: string;
  entryQuality: string;
  riskQuality: string;
  actionPlan: string;
  addTrigger: string;
  invalidationSignal: string;
  positionSizeGuidance: string;
  whatWouldUpgradeThis: string[];
  whatWouldKillThisSetup: string[];
  decisionSummary: string;
  checklistGradeContext: string;
  independentCanslimAssessment: string;
  valueAddInsight: string;
  contradictionFlags: string[];
  positionSizingImplication: string;
  gradeRead: string;
  setupRead: string;
  entryRead: string;
  riskRead: string;
  chartAnalysis: {
    visibleText: string[];
    patternRead: string;
    keyLevels: string[];
    relativeStrengthRead: string;
    volumeRead: string;
    modelComparison: string;
    confidence: "low" | "medium" | "high";
  };
  modelExamplesUsed: string[];
  missingEvidence: string[];
  actionItems: string[];
};

const reviewSchema = {
  name: "watchlist_setup_review",
  strict: true,
  schema: {
    type: "object",
    additionalProperties: false,
    required: [
      "verdict",
      "buyPlan",
      "tradeDeskVerdict",
      "technicalSetupGrade",
      "canslimQualityGrade",
      "entryQuality",
      "riskQuality",
      "actionPlan",
      "addTrigger",
      "invalidationSignal",
      "positionSizeGuidance",
      "whatWouldUpgradeThis",
      "whatWouldKillThisSetup",
      "decisionSummary",
      "checklistGradeContext",
      "independentCanslimAssessment",
      "valueAddInsight",
      "contradictionFlags",
      "positionSizingImplication",
      "gradeRead",
      "setupRead",
      "entryRead",
      "riskRead",
      "chartAnalysis",
      "modelExamplesUsed",
      "missingEvidence",
      "actionItems"
    ],
    properties: {
      verdict: { type: "string", enum: ["Actionable", "Watch", "Pass", "Manage Existing"] },
      buyPlan: {
        type: "object",
        additionalProperties: false,
        required: [
          "recommendation",
          "primaryBuyLevel",
          "starterBuyLevel",
          "addOnBuyLevel",
          "stopLevel",
          "noTradeReason",
          "canslimRule"
        ],
        properties: {
          recommendation: { type: "string", enum: ["Buy", "Starter", "Add", "Wait", "No Trade"] },
          primaryBuyLevel: { type: "string", maxLength: 100 },
          starterBuyLevel: { type: "string", maxLength: 100 },
          addOnBuyLevel: { type: "string", maxLength: 100 },
          stopLevel: { type: "string", maxLength: 100 },
          noTradeReason: { type: "string", maxLength: 120 },
          canslimRule: { type: "string", maxLength: 120 }
        }
      },
      tradeDeskVerdict: { type: "string", maxLength: 80 },
      technicalSetupGrade: { type: "string", maxLength: 90 },
      canslimQualityGrade: { type: "string", maxLength: 90 },
      entryQuality: { type: "string", maxLength: 140 },
      riskQuality: { type: "string", maxLength: 140 },
      actionPlan: { type: "string", maxLength: 220 },
      addTrigger: { type: "string", maxLength: 140 },
      invalidationSignal: { type: "string", maxLength: 140 },
      positionSizeGuidance: { type: "string", maxLength: 140 },
      whatWouldUpgradeThis: { type: "array", maxItems: 3, items: { type: "string", maxLength: 120 } },
      whatWouldKillThisSetup: { type: "array", maxItems: 3, items: { type: "string", maxLength: 120 } },
      decisionSummary: { type: "string", maxLength: 280 },
      checklistGradeContext: { type: "string", maxLength: 220 },
      independentCanslimAssessment: { type: "string", maxLength: 240 },
      valueAddInsight: { type: "string", maxLength: 220 },
      contradictionFlags: { type: "array", maxItems: 3, items: { type: "string", maxLength: 140 } },
      positionSizingImplication: { type: "string", maxLength: 140 },
      gradeRead: { type: "string", maxLength: 180 },
      setupRead: { type: "string", maxLength: 220 },
      entryRead: { type: "string", maxLength: 220 },
      riskRead: { type: "string", maxLength: 220 },
      chartAnalysis: {
        type: "object",
        additionalProperties: false,
        required: [
          "visibleText",
          "patternRead",
          "keyLevels",
          "relativeStrengthRead",
          "volumeRead",
          "modelComparison",
          "confidence"
        ],
        properties: {
          visibleText: { type: "array", items: { type: "string" } },
          patternRead: { type: "string", maxLength: 220 },
          keyLevels: { type: "array", maxItems: 6, items: { type: "string", maxLength: 80 } },
          relativeStrengthRead: { type: "string", maxLength: 180 },
          volumeRead: { type: "string", maxLength: 180 },
          modelComparison: { type: "string", maxLength: 240 },
          confidence: { type: "string", enum: ["low", "medium", "high"] }
        }
      },
      modelExamplesUsed: { type: "array", maxItems: 5, items: { type: "string", maxLength: 80 } },
      missingEvidence: { type: "array", maxItems: 5, items: { type: "string", maxLength: 120 } },
      actionItems: { type: "array", minItems: 1, maxItems: 5, items: { type: "string", maxLength: 120 } }
    }
  }
};

function numberValue(value: unknown) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : 0;
}

function limitText(value: string, maxLength: number) {
  const text = String(value || "").trim();
  return text.length > maxLength ? `${text.slice(0, maxLength)}...` : text;
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

function setupTemplateFor(setupName: string, templates: SetupChecklistTemplate[]) {
  return templates.find((template) => template.setupName.trim().toLowerCase() === setupName.trim().toLowerCase());
}

function checklistSummary(items: TradeChecklistItem[]) {
  const metCriteria = items
    .filter((item) => item.met || numberValue(item.score) > 0)
    .map((item) => item.criteria)
    .slice(0, 30);
  const missedCriteria = items
    .filter((item) => !item.met && !numberValue(item.score))
    .map((item) => item.criteria)
    .slice(0, 30);
  const total = items.reduce((sum, item) => sum + numberValue(item.points), 0);
  const earned = items.reduce((sum, item) => {
    const max = numberValue(item.points);
    return sum + ((item.inputType || "boolean") === "points" ? Math.max(0, Math.min(max, numberValue(item.score))) : item.met ? max : 0);
  }, 0);
  return { earned, total, metCriteria, missedCriteria };
}

function fallbackChunks(source: NonNullable<SetupChecklistTemplate["knowledgeSources"]>[number]) {
  if (source.chunks?.length) return source.chunks;
  if (!source.content.trim()) return [];
  return [{ id: `${source.id}-content`, title: source.title, content: source.content, order: 0 }];
}

function relevantStrategyKnowledge(item: WatchlistItem, template: SetupChecklistTemplate | undefined) {
  if (!template) return [];
  const query = [
    item.symbol,
    item.side,
    item.setupTag,
    item.entryCriteria,
    item.entryNotes,
    item.invalidation,
    item.notes,
    item.chartLinks.join(" ")
  ].join(" ");
  const queryTokens = new Set(tokenize(query));

  return (template.knowledgeSources || [])
    .flatMap((source) =>
      source.active === false
        ? []
        : fallbackChunks(source).map((chunk) => {
            const chunkTokens = tokenize(`${chunk.title} ${chunk.content}`);
            const overlap = chunkTokens.reduce((score, token) => score + (queryTokens.has(token) ? 1 : 0), 0);
            return {
              title: `${source.title}${chunk.title && chunk.title !== source.title ? ` - ${chunk.title}` : ""}`,
              sourceType: source.sourceType,
              url: source.url,
              content: limitText(chunk.content, 1600),
              relevanceScore: overlap + 2
            };
          })
    )
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, 4);
}

function relevantModelExamples(item: WatchlistItem, template: SetupChecklistTemplate | undefined) {
  if (!template) return [];
  const query = [
    item.symbol,
    item.side,
    item.setupTag,
    item.entryCriteria,
    item.entryNotes,
    item.invalidation,
    item.notes,
    item.chartLinks.join(" ")
  ].join(" ");
  const queryTokens = new Set(tokenize(query));

  return (template.strategyExamples || [])
    .flatMap((example) => {
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
        example.setupType && item.setupTag.toLowerCase().includes(example.setupType.toLowerCase()) ? 4 : 0;
      const qualityBoost = example.quality === "ideal" || example.quality === "failed" ? 1 : 0;
      return {
        symbol: example.symbol,
        setupType: example.setupType,
        quality: example.quality,
        outcome: example.outcome,
        source: example.source,
        notes: limitText(example.notes, 1200),
        screenshotCount: example.screenshots?.length || 0,
        relevanceScore: overlap + setupBoost + qualityBoost
      };
    })
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, 5);
}

function normalizeWatchlistItem(value: unknown): WatchlistItem | null {
  if (!value || typeof value !== "object") return null;
  const raw = value as Partial<WatchlistItem>;
  return {
    id: String(raw.id || "watchlist-item"),
    symbol: String(raw.symbol || "").trim().toUpperCase(),
    side: raw.side === "SHORT" ? "SHORT" : "LONG",
    setupTag: String(raw.setupTag || "").trim(),
    setupGrade: String(raw.setupGrade || "").trim(),
    checklistItems: Array.isArray(raw.checklistItems) ? raw.checklistItems : [],
    plannedEntry: numberValue(raw.plannedEntry),
    stopPrice: numberValue(raw.stopPrice),
    takeProfitPrice: numberValue(raw.takeProfitPrice),
    entryCriteria: String(raw.entryCriteria || "").trim(),
    entryNotes: String(raw.entryNotes || "").trim(),
    invalidation: String(raw.invalidation || "").trim(),
    notes: String(raw.notes || "").trim(),
    screenshots: Array.isArray(raw.screenshots) ? raw.screenshots.map(String).filter(Boolean) : [],
    chartLinks: Array.isArray(raw.chartLinks) ? raw.chartLinks.map(String).filter(Boolean) : [],
    createdAt: String(raw.createdAt || new Date().toISOString()),
    updatedAt: String(raw.updatedAt || new Date().toISOString())
  };
}

function extractScreenshotId(url: string) {
  const match = url.match(/\/screenshots\/([^/?#]+)/);
  return match ? decodeURIComponent(match[1]) : "";
}

async function screenshotDataUrl(screenshot: string, item: WatchlistItem) {
  if (screenshot.startsWith("data:image/")) return screenshot;

  const screenshotId = extractScreenshotId(screenshot);
  if (!screenshotId) return "";

  const stored = await getCamJournalScreenshot(screenshotId);
  if (!stored || stored.entityType !== "watchlist-item" || stored.entityId !== item.id) return "";
  return `data:${stored.mimeType};base64,${stored.imageData.toString("base64")}`;
}

async function openAiImageParts(item: WatchlistItem) {
  const maxImages = 2;
  const maxDataUrlChars = 4_500_000;
  const parts: Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string; detail: "low" } }> = [];
  const screenshots = item.screenshots.slice(0, maxImages);

  for (let index = 0; index < screenshots.length; index += 1) {
    const url = await screenshotDataUrl(screenshots[index], item);
    if (!url || url.length > maxDataUrlChars) continue;
    parts.push({ type: "text", text: `Watchlist screenshot for ${item.symbol || "selected setup"}, image ${index + 1}.` });
    parts.push({ type: "image_url", image_url: { url, detail: "low" } });
  }

  return parts;
}

function validateReview(value: unknown): WatchlistReview {
  const raw = value as Partial<WatchlistReview>;
  const verdict = ["Actionable", "Watch", "Pass", "Manage Existing"].includes(String(raw.verdict))
    ? (raw.verdict as WatchlistReview["verdict"])
    : "Watch";
  const chart = raw.chartAnalysis || {};
  const chartRaw = chart as Partial<WatchlistReview["chartAnalysis"]>;
  const confidence = ["high", "medium"].includes(String(chartRaw.confidence)) ? chartRaw.confidence : "low";

  return {
    verdict,
    buyPlan: {
      recommendation: ["Buy", "Starter", "Add", "Wait", "No Trade"].includes(String(raw.buyPlan?.recommendation))
        ? (raw.buyPlan?.recommendation as WatchlistReview["buyPlan"]["recommendation"])
        : "Wait",
      primaryBuyLevel: String(raw.buyPlan?.primaryBuyLevel || "not defined"),
      starterBuyLevel: String(raw.buyPlan?.starterBuyLevel || "not defined"),
      addOnBuyLevel: String(raw.buyPlan?.addOnBuyLevel || "not defined"),
      stopLevel: String(raw.buyPlan?.stopLevel || "not defined"),
      noTradeReason: String(raw.buyPlan?.noTradeReason || ""),
      canslimRule: String(raw.buyPlan?.canslimRule || "")
    },
    tradeDeskVerdict: String(raw.tradeDeskVerdict || verdict),
    technicalSetupGrade: String(raw.technicalSetupGrade || "Unclear"),
    canslimQualityGrade: String(raw.canslimQualityGrade || "Unclear"),
    entryQuality: String(raw.entryQuality || ""),
    riskQuality: String(raw.riskQuality || ""),
    actionPlan: String(raw.actionPlan || ""),
    addTrigger: String(raw.addTrigger || ""),
    invalidationSignal: String(raw.invalidationSignal || ""),
    positionSizeGuidance: String(raw.positionSizeGuidance || raw.positionSizingImplication || ""),
    whatWouldUpgradeThis: Array.isArray(raw.whatWouldUpgradeThis) ? raw.whatWouldUpgradeThis.map(String).filter(Boolean) : [],
    whatWouldKillThisSetup: Array.isArray(raw.whatWouldKillThisSetup) ? raw.whatWouldKillThisSetup.map(String).filter(Boolean) : [],
    decisionSummary: String(raw.decisionSummary || ""),
    checklistGradeContext: String(raw.checklistGradeContext || ""),
    independentCanslimAssessment: String(raw.independentCanslimAssessment || ""),
    valueAddInsight: String(raw.valueAddInsight || ""),
    contradictionFlags: Array.isArray(raw.contradictionFlags) ? raw.contradictionFlags.map(String).filter(Boolean) : [],
    positionSizingImplication: String(raw.positionSizingImplication || ""),
    gradeRead: String(raw.gradeRead || ""),
    setupRead: String(raw.setupRead || ""),
    entryRead: String(raw.entryRead || ""),
    riskRead: String(raw.riskRead || ""),
    chartAnalysis: {
      visibleText: Array.isArray(chartRaw.visibleText) ? chartRaw.visibleText.map(String).filter(Boolean) : [],
      patternRead: String(chartRaw.patternRead || "No clear chart pattern identified."),
      keyLevels: Array.isArray(chartRaw.keyLevels) ? chartRaw.keyLevels.map(String).filter(Boolean) : [],
      relativeStrengthRead: String(chartRaw.relativeStrengthRead || "Relative strength was not clear from the screenshot."),
      volumeRead: String(chartRaw.volumeRead || "Volume was not clear from the screenshot."),
      modelComparison: String(chartRaw.modelComparison || "No close model-example comparison was available."),
      confidence: confidence as WatchlistReview["chartAnalysis"]["confidence"]
    },
    modelExamplesUsed: Array.isArray(raw.modelExamplesUsed) ? raw.modelExamplesUsed.map(String).filter(Boolean) : [],
    missingEvidence: Array.isArray(raw.missingEvidence) ? raw.missingEvidence.map(String).filter(Boolean) : [],
    actionItems: Array.isArray(raw.actionItems) && raw.actionItems.length ? raw.actionItems.map(String) : ["Wait for a cleaner setup or add missing evidence."]
  };
}

export async function POST(request: Request) {
  const user = await getSessionUser();
  if (!user || (user.journalOwnerId || user.id) !== "branden") {
    return NextResponse.json({ error: "Unauthorized." }, { status: 401 });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return NextResponse.json({ error: "OPENAI_API_KEY is not configured." }, { status: 500 });
  }

  const body = await request.json().catch(() => ({}));
  const item = normalizeWatchlistItem(body.item);
  if (!item) {
    return NextResponse.json({ error: "A watchlist item is required." }, { status: 400 });
  }
  if (!item.symbol && !item.screenshots.length && !item.notes) {
    return NextResponse.json({ error: "Add a symbol, notes, or screenshot before requesting review." }, { status: 400 });
  }

  const templates = await getSetupChecklistTemplates();
  const template = setupTemplateFor(item.setupTag, templates);
  const score = checklistSummary(item.checklistItems);
  const strategyKnowledge = relevantStrategyKnowledge(item, template);
  const modelExampleMatches = relevantModelExamples(item, template);
  const model = process.env.OPENAI_REVIEW_MODEL || "gpt-4.1-mini";
  const content: Array<{ type: "text"; text: string } | { type: "image_url"; image_url: { url: string; detail: "low" } }> = [
    {
      type: "text",
      text: `Review this watchlist setup as a CANSLIM-aware swing-trading coach.

Return JSON only. Do not invent evidence that is not visible or provided.

Rules:
- Use setup criteria, strategyKnowledge, modelExampleMatches, notes, levels, and screenshots.
- Be concise. Use short trade-desk phrases, not paragraphs. Respect max field lengths.
- Use exact price levels from plannedEntry, stopPrice, takeProfitPrice, notes, visible chart labels, or keyLevels whenever available. If a level is unavailable, say "not defined."
- The checklist score/setupGrade is deterministic context from the user's grader. Do not simply echo it as your own grade.
- Your main job is to add value beyond the checklist: CANSLIM rule interpretation, model-book comparison, chart/risk critique, contradictions, and the actual trading decision.
- Buy plan is mandatory. Use CANSLIM pivot/breakout/add-on logic from strategyKnowledge and modelExampleMatches.
- If the setup does not look buyable now, set buyPlan.recommendation to "Wait" or "No Trade"; do not force a buy level.
- If no valid pivot/entry is visible or provided, say primaryBuyLevel/starterBuyLevel/addOnBuyLevel "not defined" and explain noTradeReason.
- If a starter is reasonable before a full breakout, distinguish starterBuyLevel from addOnBuyLevel.
- If a trade should not be taken, be explicit: "No Trade" plus the exact defect such as extended, no pivot, weak RS, missing stop, failed base, or poor volume.
- buyPlan.noTradeReason and buyPlan.canslimRule must be one short complete sentence. Do not end mid-thought.
- stopLevel should use the provided stop if present; otherwise suggest a logical CANSLIM invalidation area only if visible/provided, otherwise "not defined".
- canslimRule should cite the relevant rule/pattern in plain language, such as "buy through pivot on volume", "avoid extended entries", "add only after secondary proper buy point", or "pass failed breakout".
- Fill the trade-desk scorecard first. It is the primary output the user will act on.
- tradeDeskVerdict should be a short decision phrase such as "Starter only", "Watch for breakout", "Actionable now", "Add only above ATH", "Too extended", or "Pass".
- technicalSetupGrade should grade only the chart/price-volume/entry structure, independent of fundamentals. Use A/B/C/D/F style with a brief reason.
- canslimQualityGrade should grade the full CANSLIM quality using fundamentals, leadership, RS, market, model examples, and chart. It may differ from technicalSetupGrade.
- entryQuality should state whether the proposed entry is actionable now, starter-only, add-only, late/extended, or not actionable.
- riskQuality should evaluate stop placement and distance from entry, not just repeat the stop.
- actionPlan must be specific and executable.
- addTrigger must name the exact confirmation level/condition for adding size, or "None until reset" if none exists.
- invalidationSignal must name the exact condition that kills the setup.
- positionSizeGuidance must translate setup quality and stop distance into size guidance.
- whatWouldUpgradeThis and whatWouldKillThisSetup must be concrete bullet items.
- checklistGradeContext should explain what the user grader says, including earned/total points, biggest misses, and why your AI grades differ if they differ.
- independentCanslimAssessment should state whether the setup resembles a high-quality CANSLIM model-book opportunity after considering chart, fundamentals checklist, RS, market, entry location, and risk.
- valueAddInsight should be the most important observation the checklist alone would not tell the user.
- contradictionFlags must list conflicts between checklist/notes/screenshots/model examples. Return [] only if there are no meaningful conflicts.
- decisionSummary must be concise and decision-oriented: one to three sentences.
- If model examples are provided, compare the candidate against the closest ideal/good/failed/cautionary examples.
- Verdict must be Actionable only when the setup, risk location, and evidence are sufficient. Prefer Watch or Pass when entry is extended, stop is too wide, or evidence is missing.
- If chart text is blurry or cropped, say so.
- This is decision support, not financial advice.

Watchlist item:
${JSON.stringify({
  symbol: item.symbol,
  side: item.side,
  setupTag: item.setupTag,
  setupGrade: item.setupGrade,
  plannedEntry: item.plannedEntry,
  stopPrice: item.stopPrice,
  takeProfitPrice: item.takeProfitPrice,
  entryCriteria: limitText(item.entryCriteria, 1200),
  entryNotes: limitText(item.entryNotes, 1200),
  invalidation: limitText(item.invalidation, 1200),
  notes: limitText(item.notes, 1200),
  chartLinks: item.chartLinks,
  checklistScore: score,
  strategyKnowledge,
  modelExampleMatches,
  screenshotCount: item.screenshots.length
})}`
    },
    ...(await openAiImageParts(item))
  ];

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${apiKey}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model,
      temperature: 0.25,
      response_format: { type: "json_schema", json_schema: reviewSchema },
      messages: [
        {
          role: "system",
          content:
            "You are a direct, risk-aware trading setup reviewer. Use only the supplied setup data, strategy context, model examples, and images. Return valid JSON only."
        },
        {
          role: "user",
          content
        }
      ]
    })
  });

  if (!response.ok) {
    const details = await response.text().catch(() => "");
    return NextResponse.json(
      { error: `OpenAI watchlist review failed (${response.status}). ${details.slice(0, 300)}` },
      { status: 502 }
    );
  }

  const data = await response.json();
  const rawContent = data.choices?.[0]?.message?.content;
  if (!rawContent) {
    return NextResponse.json({ error: "OpenAI returned an empty review." }, { status: 502 });
  }

  return NextResponse.json({
    review: validateReview(JSON.parse(rawContent)),
    context: {
      strategyKnowledgeCount: strategyKnowledge.length,
      modelExampleCount: modelExampleMatches.length,
      screenshotCount: item.screenshots.length
    }
  });
}
