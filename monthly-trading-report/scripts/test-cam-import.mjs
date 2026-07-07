import assert from "node:assert/strict";
import crypto from "node:crypto";
import fs from "node:fs";
import vm from "node:vm";

const source = fs.readFileSync(new URL("../public/cam-journal/app.js", import.meta.url), "utf8");

function functionSource(name) {
  const start = source.indexOf(`function ${name}(`);
  if (start < 0) throw new Error(`Missing function ${name}`);
  const brace = source.indexOf("{", start);
  let depth = 0;
  for (let index = brace; index < source.length; index++) {
    if (source[index] === "{") depth++;
    if (source[index] === "}") depth--;
    if (depth === 0) return source.slice(start, index + 1);
  }
  throw new Error(`Unclosed function ${name}`);
}

const context = vm.createContext({
  console,
  crypto,
  trades: [],
  money: value => `$${Number(value || 0).toFixed(2)}`,
  numberOrBlank: value => value,
  stableId: prefix => `${prefix}-test`
});

[
  "cleanBrokerImportNoteText",
  "hasDefinedR",
  "getRMultiple",
  "validRValue",
  "summarizeRoundTrip",
  "preserveManualFields",
  "syntheticExecutionsFromTrade",
  "dedupeExecutions",
  "netExecutionPosition",
  "reconcileWithExistingOpen",
  "upsertImportedTrades"
].forEach(name => vm.runInContext(functionSource(name), context));

function execution(key, action, qty, price, date = "2026-06-22", time = "10:00") {
  const signedQty = action === "BOT" ? qty : -qty;
  return {
    date,
    time,
    action,
    qty,
    signedQty,
    price,
    instrument: "XYZ",
    ticker: "XYZ",
    amount: action === "BOT" ? -(qty * price) : qty * price,
    miscFees: 0,
    commissions: 0,
    description: `${action} ${qty} XYZ @${price}`,
    executionKey: key
  };
}

const openExecution = execution("open-1", "BOT", 10, 100, "2026-06-20", "09:30");
const closeExecution = execution("close-1", "SOLD", 10, 105, "2026-06-22", "10:00");

context.trades = [{
  ...context.summarizeRoundTrip("XYZ", [openExecution], false),
  id: "existing-trade",
  setup: "BBC",
  notes: "Keep my note\n\nBroker import updated this trade with new execution data.",
  screenshots: [{ url: "/shot/1" }]
}];

let result = context.upsertImportedTrades([context.summarizeRoundTrip("XYZ", [closeExecution], false)]);
assert.equal(result.updated, 1);
assert.equal(result.closedOpen, 1);
assert.equal(context.trades.length, 1);
assert.equal(context.trades[0].status, "Closed");
assert.equal(context.trades[0].direction, "Long");
assert.equal(context.trades[0].notes, "Keep my note");
assert.equal(context.trades[0].screenshots.length, 1);

result = context.upsertImportedTrades([context.summarizeRoundTrip("XYZ", [closeExecution], false)]);
assert.equal(result.ignored, 1);
assert.equal(context.trades.length, 1);
assert.equal(context.trades[0].status, "Closed");

result = context.upsertImportedTrades([context.summarizeRoundTrip("XYZ", [openExecution, closeExecution], true)]);
assert.equal(result.ignored, 1);

assert.equal(context.validRValue({ pl: 500, risk: "" }), null);
assert.equal(context.validRValue({ pl: 500, risk: 250 }), 2);

console.log(JSON.stringify({
  sellOnlyClosedExistingLong: "passed",
  duplicateCloseIgnored: "passed",
  personalNotesPreserved: "passed",
  undefinedRExcludedFromAverages: "passed"
}, null, 2));
