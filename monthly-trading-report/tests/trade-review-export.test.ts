import assert from "node:assert/strict";
import test from "node:test";
import { Packer } from "docx";
import JSZip from "jszip";
import { createCanvas } from "@napi-rs/canvas";
import { writeFile } from "node:fs/promises";
import { buildDocument, buildPromptTrades, buildReviewRequest, completedAiReview, DEFAULT_TRADE_REVIEW_MODEL, MAX_TRADE_REVIEW_COST_USD, maximumTradeReviewCostUsd, pendingAiReviewId, retrieveAiReview, startAiReview, validateAiReview } from "../lib/trade-review-export";
import { tradeChecklistScore, tradeReviewMissingFields } from "../lib/trade-review";
import { loadReviewImage } from "../lib/trade-review-evidence";
import { trade, templates, evidence, review } from "./fixtures/trade-review-export";

function chart(label = "Chart layout test") {
  const canvas = createCanvas(1200, 600);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 1200, 600);
  ctx.fillStyle = "#263026"; ctx.font = "30px Arial"; ctx.fillText(label, 40, 55);
  ctx.strokeStyle = "#6f8f5f"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(40, 480); ctx.lineTo(340, 360); ctx.lineTo(560, 380); ctx.lineTo(880, 230); ctx.lineTo(1160, 280); ctx.stroke();
  return `data:image/png;base64,${canvas.toBuffer("image/png").toString("base64")}`;
}

test("preserves every structured note, full context, and executions beyond old cutoffs", () => {
  const item = trade({ notes: "x".repeat(1500), executions: Array.from({ length: 45 }, (_, i) => ({ ...trade().executions[0], id: `fill-${i}` })) });
  const sources = Array.from({ length: 5 }, (_, i) => ({ ...templates[0].knowledgeSources![0], id: `source-${i}`, content: "s".repeat(2000) }));
  const context = [{ ...templates[0], knowledgeSources: [...sources, { ...sources[0], active: false }] }];
  const payload = buildPromptTrades([item], context, evidence())[0];
  assert.deepEqual(payload.reviewSections, item.reviewSections);
  assert.equal(payload.notes.length, 1500);
  assert.equal(payload.strategyKnowledge.length, 5);
  assert.equal(payload.strategyKnowledge[0].content.length, 2000);
  assert.equal(payload.executions.length, 45);
  assert.deepEqual(payload.mistakeTags, item.mistakeTags);
  assert.equal(payload.excursion?.status, "UNAVAILABLE");
  assert.equal(payload.excursion?.mfeDollars, null);
});

test("export grade exactly matches log points grading and manual override", () => {
  const item = trade({ manualGrade: "", checklistItems: [{ id: "c1", criteria: "Entry", points: 10, met: false, inputType: "points", score: 6 }] });
  assert.equal(buildPromptTrades([item], templates, evidence())[0].grade, "C");
  assert.equal(buildPromptTrades([item], templates, evidence())[0].grade, tradeChecklistScore(item, templates).grade);
  assert.equal(buildPromptTrades([trade({ ...item, manualGrade: "B+" })], templates, evidence())[0].grade, "B+");
});

test("all charts including later trades use high detail with exact trade IDs", () => {
  const items = [trade(), trade({ id: "trade-2" })];
  const data = evidence();
  for (const item of items) data.images[item.id] = Array.from({ length: 6 }, (_, i) => ({ label: `chart ${i}`, dataUrl: chart(`${item.id} chart ${i}`) }));
  const request = buildReviewRequest(items, templates, data, "2026-09-01", "2026-09-07");
  const images = request.input[0].content.filter((part) => part.type === "input_image");
  assert.equal(images.length, 12); assert(images.every((part) => part.detail === "high"));
  assert(JSON.stringify(request).includes("Trade ID trade-2"));
  assert.equal(request.model, DEFAULT_TRADE_REVIEW_MODEL); assert.equal(request.reasoning.effort, "medium");
  assert.equal(request.store, false); assert(!("temperature" in request));
});

test("sends a shared setup-example chart once while retaining every trade association", () => {
  const items = [trade(), trade({ id: "trade-2", symbol: "TWO" })];
  const shared = chart("Shared model example");
  const data = evidence();
  for (const item of items) data.images[item.id] = [{ label: "Shared comparison example", dataUrl: shared, analysisDetail: "low" }];
  const request = buildReviewRequest(items, templates, data, "2026-09-01", "2026-09-07");
  const content = request.input[0].content;
  assert.equal(content.filter((part) => part.type === "input_image").length, 1);
  assert.equal(content.find((part) => part.type === "input_image")?.detail, "low");
  const labels = content.filter((part) => part.type === "input_text").map((part) => part.text).join("\n");
  assert(labels.includes("Trade ID trade-1"));
  assert(labels.includes("Trade ID trade-2"));
});

test("sends shared strategy text once while retaining references from every trade", () => {
  const items = [trade(), trade({ id: "trade-2", symbol: "TWO" })];
  const request = buildReviewRequest(items, templates, evidence(), "2026-09-01", "2026-09-07");
  const prompt = request.input[0].content.find((part) => part.type === "input_text")?.text || "";
  assert.equal(prompt.split("Enter near the pivot; avoid chasing extended breakouts.").length - 1, 1);
  assert.equal(prompt.split('"strategyReferenceIds":["strategy-1"]').length - 1, 2);
  assert(!prompt.includes('"strategyKnowledge"'));
});

test("supplies an identity hint for ambiguous ticker symbols", () => {
  const request = buildReviewRequest([trade({ symbol: "USB" })], templates, evidence(), "2026-09-01", "2026-09-07");
  const prompt = request.input[0].content.find((part) => part.type === "input_text")?.text || "";
  assert(prompt.includes("U.S. Bancorp"));
  assert(prompt.includes("Financials"));
});

test("oversized text fails explicitly instead of silently truncating", () => {
  assert.throws(() => buildReviewRequest([trade({ notes: "x".repeat(1_000_001) })], templates, evidence(), "", ""), /no evidence was omitted/);
});

test("completion gate lists exact missing fields and does not require general review", () => {
  assert.deepEqual(tradeReviewMissingFields(trade({ reviewSections: { ...trade().reviewSections!, general: "" } }), templates), []);
  assert.deepEqual(tradeReviewMissingFields(trade({ screenshots: [], reviewSections: { ...trade().reviewSections!, entry: "" } }), templates), ["Entry review", "screenshot"]);
});

test("rejects missing priorities, invented evidence IDs, empty actions, and ticker-based reviews", () => {
  const payload = buildPromptTrades([trade()], templates, evidence());
  assert.doesNotThrow(() => validateAiReview(review(), [trade()], payload));
  const invalid = review(); invalid.workOn.priorities[0].evidenceTradeIds = ["unknown"];
  assert.throws(() => validateAiReview(invalid, [trade()], payload), /valid trade evidence/);
  const empty = review(); empty.workOn.priorities[0].rule = " ";
  assert.throws(() => validateAiReview(empty, [trade()], payload), /actionable rules/);
  const recurring = review(); recurring.workOn.priorities[0].scope = "recurring";
  recurring.workOn.priorities[0].evidenceTradeIds = ["trade-1", "trade-1"];
  assert.throws(() => validateAiReview(recurring, [trade()], payload), /valid trade evidence/);
  const wrong = review(); wrong.tradeReviews = { FRO: wrong.tradeReviews["trade-1"] };
  assert.throws(() => validateAiReview(wrong, [trade()], payload), /missing the required trade review/);
});

test("image evidence decodes faithfully and rejects unreadable references", async () => {
  const image = await loadReviewImage(chart(), "Actual chart", trade());
  assert.match(image.dataUrl, /^data:image\/(?:png|jpeg);base64,/);
  assert.equal(image.analysisDetail, "high");
  await assert.rejects(loadReviewImage("https://example.com/chart", "Example chart", trade(), { kind: "strategy-example", exampleId: "ex1" }), /Re-upload/);
});

test("loads stored strategy-example charts from their journal screenshot owner", async () => {
  const dataUrl = chart();
  const bytes = Buffer.from(dataUrl.split(",")[1], "base64");
  const image = await loadReviewImage(
    "/api/cam-journal/screenshots/example-image-1",
    "Stored comparison chart",
    trade(),
    { kind: "strategy-example", exampleId: "ex1" },
    {
      trade: async () => null,
      journal: async () => ({
        id: "example-image-1",
        entityType: "setup-strategy-example",
        entityId: "ex1",
        fileName: "example.png",
        mimeType: "image/png",
        imageData: bytes
      })
    }
  );
  assert.match(image.dataUrl, /^data:image\/(?:png|jpeg);base64,/);
  assert.equal(image.analysisDetail, "low");
});

test("rejects a strategy-example chart owned by a different example", async () => {
  await assert.rejects(
    loadReviewImage(
      "/api/cam-journal/screenshots/example-image-1",
      "Stored comparison chart",
      trade(),
      { kind: "strategy-example", exampleId: "ex1" },
      {
        trade: async () => null,
        journal: async () => ({
          id: "example-image-1",
          entityType: "setup-strategy-example",
          entityId: "different-example",
          fileName: "example.png",
          mimeType: "image/png",
          imageData: Buffer.from("not-used")
        })
      }
    ),
    /Re-upload/
  );
});

test("background Responses flow submits, polls, completes, and rejects terminal failures", async () => {
  const oldFetch = globalThis.fetch; const oldKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-key";
  let mode = "in_progress";
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith("/resp_test")) return Response.json({ id: "resp_test", status: mode });
    if (String(url).endsWith("/responses/input_tokens")) {
      const body = JSON.parse(String(init?.body));
      assert.equal(body.model, "gpt-5.6-luna");
      return Response.json({ object: "response.input_tokens", input_tokens: 200_000 });
    }
    assert.equal(url, "https://api.openai.com/v1/responses");
    const body = JSON.parse(String(init?.body));
    assert.equal(body.text.format.type, "json_schema");
    assert.equal(body.background, true);
    if (mode === "failed") return new Response("Unavailable", { status: 429 });
    return Response.json({ id: "resp_test", status: mode });
  };
  try {
    const started = await startAiReview([trade()], templates, evidence(), "", "");
    assert.equal(pendingAiReviewId(started.response), "resp_test");
    assert(started.maximumCostUsd <= MAX_TRADE_REVIEW_COST_USD);
    assert.equal(pendingAiReviewId(await retrieveAiReview("resp_test")), "resp_test");
    const completed = { id: "resp_test", status: "completed", output: [{ type: "message", content: [{ type: "output_text", text: JSON.stringify(review()) }] }] };
    assert.equal(completedAiReview(completed, [trade()], templates, evidence()).workOn.priorities.length, 1);
    assert.throws(() => completedAiReview({ ...completed, status: "incomplete" }, [trade()], templates, evidence()), /did not finish/);
    assert.throws(() => completedAiReview({ ...completed, output: [{ type: "message", content: [{ type: "refusal" }] }] }, [trade()], templates, evidence()), /could not complete/);
    mode = "failed";
    await assert.rejects(startAiReview([trade()], templates, evidence(), "", ""), /HTTP 429/);
  } finally { globalThis.fetch = oldFetch; if (oldKey === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = oldKey; }
});

test("cost ceiling includes long-context and maximum output pricing", () => {
  assert(Math.abs(maximumTradeReviewCostUsd(200_000) - 0.046) < 1e-10);
  assert(maximumTradeReviewCostUsd(534_770) < MAX_TRADE_REVIEW_COST_USD);
  assert(maximumTradeReviewCostUsd(610_000) > MAX_TRADE_REVIEW_COST_USD);
});

test("price preflight blocks an over-budget generation request", async () => {
  const oldFetch = globalThis.fetch; const oldKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-key";
  let requests = 0;
  globalThis.fetch = async (url) => {
    requests += 1;
    assert(String(url).endsWith("/responses/input_tokens"));
    return Response.json({ object: "response.input_tokens", input_tokens: 610_000 });
  };
  try {
    await assert.rejects(startAiReview([trade()], templates, evidence(), "", ""), /safety ceiling.*No paid review was started/);
    assert.equal(requests, 1);
  } finally { globalThis.fetch = oldFetch; if (oldKey === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = oldKey; }
});

test("Word report is a concise period review and keeps strategy material private", async () => {
  const data = evidence(); data.images["trade-1"] = [{ label: "Actual trade chart", dataUrl: chart() }, { label: "Comparison example chart", dataUrl: chart() }];
  const document = await buildDocument([trade()], templates, "2026-09-01", "2026-09-07", data, review());
  const buffer = await Packer.toBuffer(document);
  const zip = await JSZip.loadAsync(buffer); const xml = await zip.file("word/document.xml")!.async("string");
  for (const text of ["Period Overview", "What Went Well", "Key Mistakes", "Exposure and Correlation", "Trade Snapshot", "What to Work On", "Rule going forward", "Track:", "Bottom Line"]) assert(xml.includes(text), text);
  for (const excluded of ["Strategy Context", "Charts and Model Examples", "Breakout rules", "Enter near the pivot", "General review", "MAE and MFE", "No bars available", "Legacy notes"]) assert(!xml.includes(excluded), excluded);
  assert.equal((xml.match(/<w:drawing>/g) || []).length, 0);
  assert(xml.includes('w:val="Title"'));
  if (process.env.REVIEW_QA_PATH) {
    const qaTrades = Array.from({ length: 20 }, (_, index) => trade({
      id: `trade-${index + 1}`,
      symbol: ["DELL", "USB", "AMZN", "LRCX", "NET", "SHOP"][index % 6],
      entryDate: `2026-09-${String((index % 12) + 1).padStart(2, "0")}`,
      pnl: index % 3 === 0 ? -125 : 90,
      rMultiple: index % 3 === 0 ? -0.68 : 0.49,
      status: index % 3 === 0 ? "LOSS" : "WIN"
    }));
    const qaReview = review();
    qaReview.overallTakeaway = "The period was positive overall, but repeated technology exposure and late additions made several trades behave like one larger position. Risk planning was generally consistent; entry selectivity and exposure limits are the clearest opportunities.";
    qaReview.keyThemes = ["Planned risk was usually documented before entry.", "Late additions and extended entries reduced the quality of otherwise valid setups.", "Several technology trades created overlapping exposure during the same window."];
    qaReview.improved = ["Position risk was defined consistently.", "The strongest trades followed the planned trigger and avoided unnecessary adjustments.", "Trade reviews identified specific behaviors instead of relying only on profit and loss."];
    qaReview.needsWork = ["Do not add after price has moved beyond the planned trigger.", "Treat related technology positions as one exposure bucket before sizing.", "Require clear exit criteria before entry."];
    qaReview.exposureAnalysis = { summary: "Technology and internet-related equities dominated the sample and produced mixed results. The concentration increased the chance that several positions would respond to the same market move.", groups: [{
      label: "Technology and internet (inferred)", type: "sector", symbols: ["DELL", "AMZN", "LRCX", "NET", "SHOP"], evidenceTradeIds: qaTrades.filter((item) => item.symbol !== "USB").map((item) => item.id),
      performance: "Mixed, with several losses offset by smaller wins.", correlation: "These positions likely shared sensitivity to growth and technology sentiment.", takeaway: "Set a combined risk cap for related positions before adding another name.", confidence: "medium"
    }, {
      label: "Repeated AMZN positions", type: "repeated_symbol", symbols: ["AMZN"], evidenceTradeIds: qaTrades.filter((item) => item.symbol === "AMZN").map((item) => item.id),
      performance: "Mixed across the period.", correlation: "Multiple entries in the same symbol are direct concentration.", takeaway: "Review the combined thesis and total symbol risk before re-entry.", confidence: "high"
    }] };
    qaReview.workOn.priorities[0].scope = "recurring";
    qaReview.workOn.priorities[0].evidenceTradeIds = ["trade-1", "trade-4", "trade-7"];
    qaReview.workOn.priorities[0].evidence = "Late additions appeared in three losing trades, making entry discipline a recurring issue in this period.";
    qaReview.tradeReviews = Object.fromEntries(qaTrades.map((item, index) => [item.id, { mainLesson: index % 3 === 0 ? "Avoid adding after the planned entry area." : "Keep the defined risk and wait for the intended trigger." }]));
    const qaDocument = await buildDocument(qaTrades, templates, "2026-09-01", "2026-09-13", data, qaReview);
    await writeFile(process.env.REVIEW_QA_PATH, await Packer.toBuffer(qaDocument));
  }
});
