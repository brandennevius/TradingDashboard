import assert from "node:assert/strict";
import test from "node:test";
import { Packer } from "docx";
import JSZip from "jszip";
import { createCanvas } from "@napi-rs/canvas";
import { writeFile } from "node:fs/promises";
import { buildDocument, buildPromptTrades, buildReviewRequest, DEFAULT_TRADE_REVIEW_MODEL, generateAiReview, validateAiReview } from "../lib/trade-review-export";
import { tradeChecklistScore, tradeReviewMissingFields } from "../lib/trade-review";
import { loadReviewImage } from "../lib/trade-review-evidence";
import { trade, templates, evidence, review } from "./fixtures/trade-review-export";

function chart() {
  const canvas = createCanvas(1200, 600);
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff"; ctx.fillRect(0, 0, 1200, 600);
  ctx.fillStyle = "#263026"; ctx.font = "30px Arial"; ctx.fillText("Chart layout test", 40, 55);
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
  for (const item of items) data.images[item.id] = Array.from({ length: 6 }, (_, i) => ({ label: `chart ${i}`, dataUrl: chart() }));
  const request = buildReviewRequest(items, templates, data, "2026-09-01", "2026-09-07");
  const images = request.input[0].content.filter((part) => part.type === "input_image");
  assert.equal(images.length, 12); assert(images.every((part) => part.detail === "high"));
  assert(JSON.stringify(request).includes("Trade ID trade-2"));
  assert.equal(request.model, DEFAULT_TRADE_REVIEW_MODEL); assert.equal(request.reasoning.effort, "medium");
  assert.equal(request.store, false); assert(!("temperature" in request));
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
  assert(image.dataUrl.startsWith("data:image/png;base64,"));
  await assert.rejects(loadReviewImage("https://example.com/chart", "Example chart", trade(), true), /Re-upload/);
});

test("Responses request handles completion, refusal, incomplete output, and API failure", async () => {
  const oldFetch = globalThis.fetch; const oldKey = process.env.OPENAI_API_KEY;
  process.env.OPENAI_API_KEY = "test-key";
  let mode = "completed";
  globalThis.fetch = async (url, init) => {
    assert.equal(url, "https://api.openai.com/v1/responses");
    const body = JSON.parse(String(init?.body)); assert.equal(body.text.format.type, "json_schema");
    if (mode === "failed") return new Response("Unavailable", { status: 429 });
    return Response.json({ status: mode === "incomplete" ? "incomplete" : "completed", output: [{ type: "message", content: mode === "refusal" ? [{ type: "refusal" }] : [{ type: "output_text", text: JSON.stringify(review()) }] }] });
  };
  try {
    assert.equal((await generateAiReview([trade()], templates, evidence(), "", "")).workOn.priorities.length, 1);
    for (const state of ["refusal", "incomplete", "failed"]) {
      mode = state; await assert.rejects(generateAiReview([trade()], templates, evidence(), "", ""));
    }
  } finally { globalThis.fetch = oldFetch; if (oldKey === undefined) delete process.env.OPENAI_API_KEY; else process.env.OPENAI_API_KEY = oldKey; }
});

test("Word report includes reflections, MAE/MFE, charts, source context and measurable work priorities", async () => {
  const data = evidence(); data.images["trade-1"] = [{ label: "Actual trade chart", dataUrl: chart() }, { label: "Comparison example chart", dataUrl: chart() }];
  const document = await buildDocument([trade()], templates, "2026-09-01", "2026-09-07", data, review());
  const buffer = await Packer.toBuffer(document);
  const zip = await JSZip.loadAsync(buffer); const xml = await zip.file("word/document.xml")!.async("string");
  for (const text of ["What to Work On", "Rule going forward", "Measure of improvement", "Added late", "General review", "MAE and MFE", "No bars available", "Charts and Model Examples", "Breakout rules"]) assert(xml.includes(text), text);
  assert(!xml.includes("Upcoming Week Focus"));
  assert.equal((xml.match(/<w:drawing>/g) || []).length, 2);
  assert(xml.includes('w:val="Title"'));
  if (process.env.REVIEW_QA_PATH) await writeFile(process.env.REVIEW_QA_PATH, buffer);
});
