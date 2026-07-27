import assert from "node:assert/strict";
import fs from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

type ParsedExecution = {
  ticker: string;
  qty: number;
  signedQty: number;
  description: string;
};

type SkippedTradeRow = {
  date: string;
  time: string;
  ref: string;
  description: string;
};

type CamBrokerCsvParser = {
  parseCsv(text: string): string[][];
  parseTradeDescription(description: string): ParsedExecution | null;
  statementExecutions(rows: string[][]): {
    executions: ParsedExecution[];
    skippedTradeRows: SkippedTradeRow[];
  };
  groupExecutionsIntoTrades(
    executions: ParsedExecution[],
    options?: { idFactory?: () => string }
  ): Array<Record<string, unknown>>;
  upsertImportedTrades(
    existingTrades: Array<Record<string, unknown>>,
    importedTrades: Array<Record<string, unknown>>,
    options?: { idFactory?: () => string }
  ): {
    trades: Array<Record<string, unknown>>;
    result: {
      added: number;
      updated: number;
      ignored: number;
      closedOpen: number;
      ambiguous: number;
    };
  };
};

const require = createRequire(import.meta.url);
const parser = require("../public/cam-journal/broker-csv-parser.js") as CamBrokerCsvParser;
const appSource = fs.readFileSync(new URL("../public/cam-journal/app.js", import.meta.url), "utf8");

const reportedStatementRows = `\uFEFFAccount Statement

Cash Balance
DATE,TIME,TYPE,REF #,DESCRIPTION,Misc Fees,Commissions & Fees,AMOUNT,BALANCE
7/21/26,09:52:12,TRD,="1007263172576","SOLD -1,111 HBAN @17.8876",-0.63,,"19,873.12","21,543.18"
7/21/26,10:24:57,TRD,="1007265259524",SOLD -500 HBAN @17.915,-0.28,,"8,957.50","30,500.40"
7/21/26,10:43:06,TRD,="1007266592912","BOT +1,611 HBAN @17.92",,,"-28,869.12","1,631.28"
7/22/26,15:00:26,TRD,="1007295404857","SOLD -3,333 QUBT @7.9036",-1.19,,"26,342.70","27,703.98"
7/22/26,15:16:24,TRD,="1007295405475","BOT +3,333 QUBT @7.815",,,"-26,047.40","1,656.58"
7/23/26,10:23:14,TRD,="1007302739670","SOLD -1,666 ONDS @8.025",-0.60,,"13,369.65","15,025.63"
7/23/26,10:37:25,TRD,="1007303727226","BOT +1,666 ONDS @8.0877",,,"-13,474.11","1,551.52"
7/23/26,10:56:23,TRD,="1007304731246","BOT +1,428 ONDS @8.005",,,"-11,431.14","-9,879.62"
7/23/26,10:58:12,TRD,="1007304731418","SOLD -1,428 ONDS @7.9601",-0.51,,"11,367.02","1,486.89"
7/24/26,10:56:54,TRD,="1007323317526","SOLD -1,000 FCX @61.825",-1.47,,"61,825.00","63,026.23"
7/24/26,10:58:02,TRD,="1007323317678","BOT +1,000 FCX @61.9399",,,"-61,939.90","1,086.33"

Futures Statements
`;

function idFactory(prefix: string) {
  let index = 0;
  return () => `${prefix}-${(index += 1)}`;
}

function importedTradesFrom(csv: string, prefix: string) {
  const { executions, skippedTradeRows } = parser.statementExecutions(parser.parseCsv(csv));
  assert.deepEqual(skippedTradeRows, []);
  return parser.groupExecutionsIntoTrades(executions, { idFactory: idFactory(prefix) });
}

function withoutQuantityCommas(csv: string) {
  return csv.replace(/([+-]\d),(?=\d{3}(?:\.\d+)?\s+[A-Z])/g, "$1");
}

test("CAM broker CSV parser accepts comma-formatted share quantities", () => {
  const rows = parser.parseCsv(reportedStatementRows);
  const { executions, skippedTradeRows } = parser.statementExecutions(rows);

  assert.equal(executions.length, 11);
  assert.deepEqual(skippedTradeRows, []);

  const quantities = executions.map((execution) => execution.qty);
  assert.deepEqual(quantities, [1111, 500, 1611, 3333, 3333, 1666, 1666, 1428, 1428, 1000, 1000]);
});

test("reported HBAN, QUBT, ONDS, and FCX executions do not create phantom positions", () => {
  const { executions } = parser.statementExecutions(parser.parseCsv(reportedStatementRows));
  const netQuantities = new Map<string, number>();

  for (const execution of executions) {
    netQuantities.set(execution.ticker, (netQuantities.get(execution.ticker) || 0) + execution.signedQty);
  }

  assert.deepEqual(Object.fromEntries(netQuantities), {
    HBAN: 0,
    QUBT: 0,
    ONDS: 0,
    FCX: 0
  });
});

test("unrecognized TRD rows are returned as diagnostics instead of silently dropped", () => {
  const csv = `DATE,TIME,TYPE,REF #,DESCRIPTION,Misc Fees,Commissions & Fees,AMOUNT,BALANCE
7/24/26,12:00:00,TRD,="unrecognized",UNSUPPORTED EXECUTION FORMAT,,,,`;
  const { executions, skippedTradeRows } = parser.statementExecutions(parser.parseCsv(csv));

  assert.deepEqual(executions, []);
  assert.deepEqual(skippedTradeRows, [
    {
      date: "2026-07-24",
      time: "12:00:00",
      ref: "=unrecognized",
      description: "UNSUPPORTED EXECUTION FORMAT"
    }
  ]);
});

test("fractional quantities remain supported", () => {
  const parsed = parser.parseTradeDescription("BOT +1,000.5 XYZ @12.34");

  assert.equal(parsed?.qty, 1000.5);
  assert.equal(parsed?.signedQty, 1000.5);
});

test("comma-formatted and comma-edited files generate the same execution identities", () => {
  const original = parser.statementExecutions(parser.parseCsv(reportedStatementRows)).executions;
  const edited = parser.statementExecutions(parser.parseCsv(withoutQuantityCommas(reportedStatementRows))).executions;

  assert.deepEqual(
    original.map((execution) => (execution as ParsedExecution & { executionKey: string }).executionKey),
    edited.map((execution) => (execution as ParsedExecution & { executionKey: string }).executionKey)
  );
});

test("importing the same statement twice is idempotent", () => {
  const firstImport = importedTradesFrom(reportedStatementRows, "first");
  const firstResult = parser.upsertImportedTrades([], firstImport, { idFactory: idFactory("upsert-first") });
  const fingerprint = JSON.stringify(firstResult.trades);
  const secondImport = importedTradesFrom(reportedStatementRows, "second");
  const secondResult = parser.upsertImportedTrades(firstResult.trades, secondImport, {
    idFactory: idFactory("upsert-second")
  });

  assert.equal(firstResult.result.added, 5);
  assert.equal(secondResult.result.added, 0);
  assert.equal(secondResult.result.updated, 0);
  assert.equal(secondResult.result.ignored, 5);
  assert.equal(JSON.stringify(secondResult.trades), fingerprint);
});

test("original and comma-edited statements are idempotent in either order", () => {
  const editedCsv = withoutQuantityCommas(reportedStatementRows);

  for (const [firstCsv, secondCsv] of [
    [reportedStatementRows, editedCsv],
    [editedCsv, reportedStatementRows]
  ]) {
    const first = parser.upsertImportedTrades([], importedTradesFrom(firstCsv, "first-variant"), {
      idFactory: idFactory("upsert-first-variant")
    });
    const fingerprint = JSON.stringify(first.trades);
    const second = parser.upsertImportedTrades(first.trades, importedTradesFrom(secondCsv, "second-variant"), {
      idFactory: idFactory("upsert-second-variant")
    });

    assert.equal(second.result.added, 0);
    assert.equal(second.result.updated, 0);
    assert.equal(second.result.ignored, 5);
    assert.equal(JSON.stringify(second.trades), fingerprint);
  }
});

test("a tagged legacy lifecycle is enriched rather than duplicated", () => {
  const importedQubt = importedTradesFrom(reportedStatementRows, "qubt").find((trade) => trade.ticker === "QUBT");
  assert.ok(importedQubt);

  const legacyTaggedTrade: Record<string, unknown> = {
    ...importedQubt,
    id: "legacy-qubt",
    setup: "Bollinger Band Capitulation",
    setupId: "setup-bbc",
    grade: "A",
    risk: 100,
    rMultiple: 2.94,
    secondaryTag: "High Conviction",
    mistakeTag: "No Mistake",
    notes: "Waited for confirmation.",
    screenshots: [{ id: "shot-1", url: "/api/cam-journal/screenshots/shot-1" }],
    setupScore: { earned: 9, max: 10 },
    playbookScreenshotIndex: 0
  };
  delete legacyTaggedTrade.rawExecutions;
  delete legacyTaggedTrade.executionKeys;
  delete legacyTaggedTrade.importTradeKey;
  delete legacyTaggedTrade.importOpenKey;

  const result = parser.upsertImportedTrades([legacyTaggedTrade], [importedQubt], {
    idFactory: idFactory("legacy")
  });
  const merged = result.trades[0];

  assert.equal(result.trades.length, 1);
  assert.equal(result.result.added, 0);
  assert.equal(result.result.updated, 1);
  assert.equal(merged.id, "legacy-qubt");
  assert.equal(merged.setup, "Bollinger Band Capitulation");
  assert.equal(merged.grade, "A");
  assert.equal(merged.notes, "Waited for confirmation.");
  assert.deepEqual(merged.screenshots, [{ id: "shot-1", url: "/api/cam-journal/screenshots/shot-1" }]);
  assert.deepEqual(merged.setupScore, { earned: 9, max: 10 });
  assert.ok(Array.isArray(merged.rawExecutions));
});

test("ambiguous legacy lifecycle matches are not duplicated or silently merged", () => {
  const importedQubt = importedTradesFrom(reportedStatementRows, "ambiguous").find((trade) => trade.ticker === "QUBT");
  assert.ok(importedQubt);

  const legacyOne: Record<string, unknown> = { ...importedQubt, id: "legacy-one" };
  const legacyTwo: Record<string, unknown> = { ...importedQubt, id: "legacy-two" };
  for (const trade of [legacyOne, legacyTwo]) {
    delete trade.rawExecutions;
    delete trade.executionKeys;
    delete trade.importTradeKey;
    delete trade.importOpenKey;
  }

  const result = parser.upsertImportedTrades([legacyOne, legacyTwo], [importedQubt]);

  assert.equal(result.result.ambiguous, 1);
  assert.equal(result.result.added, 0);
  assert.equal(result.trades.length, 2);
});

test("CAM import no longer resets report or trade-log filters", () => {
  assert.doesNotMatch(appSource, /clearReportAndTradeLogFiltersForImport/);
});
