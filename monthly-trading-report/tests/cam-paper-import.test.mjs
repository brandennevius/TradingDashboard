import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createRequire } from 'node:module';
import vm from 'node:vm';
import test from 'node:test';

const parser = createRequire(import.meta.url)('../public/cam-journal/broker-csv-parser.js');
const csv = fs.readFileSync(new URL('./fixtures/cam-papermoney.csv', import.meta.url), 'utf8');
const parse = (text = csv) => parser.statementExecutions(parser.parseCsv(text));
const group = (text = csv) => parser.groupExecutionsIntoTrades(parse(text).executions);
const cents = value => Math.round(value * 100);

// Independently reconciled from the attached statement's quantities and prices.
const expected = [
  ['TDOC', 'Short', 909, '10:02:13', '10:11:27', 19089],
  ['AVGO', 'Short', 131, '10:16:06', '10:16:51', -3930],
  ['AVGO', 'Short', 111, '10:23:10', '10:23:40', -3441],
  ['GOOG', 'Short', 222, '10:27:06', '10:30:34', 2220],
  ['GOOG', 'Short', 238, '10:32:51', '10:34:56', -476],
  ['ORCL', 'Short', 270, '10:47:09', '10:50:41', -6210],
  ['ORCL', 'Long', 370, '10:50:41', '11:02:40', 19350],
  ['NVDA', 'Short', 333, '11:35:58', '11:43:09', 3663],
  ['ORCL', 'Long', 500, '11:52:12', '11:55:09', 0],
  ['ORCL', 'Long', 333, '11:58:10', '12:01:10', 999]
];

test('attached paper statement reconciles every fill and all ten trade results', () => {
  const { executions, skippedTradeRows } = parse();
  assert.equal(executions.length, 20);
  assert.deepEqual(skippedTradeRows, []);
  const sourceRows = parser.parseCsv(csv).filter(row => /^10\/29\/26 /.test(row[2] || '') && row.length === 16 && !row.includes('CANCELED'));
  assert.equal(sourceRows.length, 20);
  for (let i = 0; i < sourceRows.length; i++) {
    const row = sourceRows[i];
    const fill = executions[i];
    assert.equal(fill.date, '2026-10-29');
    assert.equal(fill.time, row[2].split(' ')[1]);
    assert.equal(fill.ticker, row[7]);
    assert.equal(fill.signedQty, Number(row[5]));
    assert.equal(fill.price, Number(row[11]));
    assert.equal(cents(fill.amount), -Number(row[5]) * cents(Number(row[11])));
  }
  const trades = group();
  assert.deepEqual(trades.map(t => [t.ticker, t.direction, t.size, t.entryTime, t.exitTime, cents(t.pl)]), expected);
  assert.ok(trades.every(t => t.status === 'Closed' && t.date === '2026-10-29' && t.exitDate === '2026-10-29' && t.accountType === 'paper'));
  assert.equal(trades.reduce((sum, t) => sum + cents(t.pl), 0), 31264);
  // Splitting the ORCL reversal must conserve all shares and cash, by symbol.
  for (const ticker of new Set(executions.map(e => e.ticker))) {
    const original = executions.filter(e => e.ticker === ticker);
    const grouped = trades.filter(t => t.ticker === ticker).flatMap(t => t.rawExecutions);
    for (const field of ['qty', 'signedQty', 'amount']) {
      assert.ok(Math.abs(original.reduce((s, e) => s + e[field], 0) - grouped.reduce((s, e) => s + e[field], 0)) < 1e-7);
    }
  }
});

test('repeat paper imports add no duplicates and preserve journal annotations', () => {
  const first = parser.upsertImportedTrades([], group());
  first.trades[0].notes = 'Keep this note';
  first.trades[0].setup = 'My setup';
  const second = parser.upsertImportedTrades(first.trades, group());
  assert.deepEqual(second.result, { added: 0, updated: 0, ignored: 10, closedOpen: 0, ambiguous: 0 });
  assert.deepEqual(second.trades, first.trades);
});

test('sorting the fill table does not change trade execution identities', () => {
  const rows = parser.parseCsv(csv);
  const indexes = rows.flatMap((row, i) => /^10\/29\/26 /.test(row[2] || '') && row.length === 16 && !row.includes('CANCELED') ? [i] : []);
  const reversed = indexes.map(i => rows[i]).reverse();
  indexes.forEach((index, i) => { rows[index] = reversed[i]; });
  const imported = parser.groupExecutionsIntoTrades(parser.statementExecutions(rows).executions);
  assert.deepEqual(imported.map(t => t.executionKeys), group().map(t => t.executionKeys));
});

const header = ',,Exec Time,Spread,Side,Qty,Pos Effect,Symbol,Exp,Strike,Type,Price,Net Price,Amount,Price Improvement,Order Type';
const paper = rows => `paperMoney simulated trading\n\nFilled Orders\n${header}\n${rows.join('\n')}\n\nCanceled Orders\n${header}\n,,10/29/26 12:00:00,STOCK,SELL,-5,TO CLOSE,XYZ,,,STOCK,999,999,0,0,MKT`;
const fill = (time, side, qty, price, type = 'STOCK') => `,,10/29/26 ${time},${type},${side},${qty},TO OPEN,XYZ,,,${type},${price},${price},0,0,MKT`;

test('identical fills remain distinct, while repeat imports remain idempotent', () => {
  const text = paper([fill('10:00:00', 'BUY', 5, 100), fill('10:00:00', 'BUY', 5, 100), fill('10:01:00', 'SELL', -10, 101)]);
  assert.equal(parse(text).executions.length, 3);
  assert.equal(new Set(parse(text).executions.map(e => e.executionKey)).size, 3);
  const first = parser.upsertImportedTrades([], group(text));
  assert.equal(first.trades[0].size, 10);
  assert.equal(first.trades[0].pl, 10);
  assert.equal(parser.upsertImportedTrades(first.trades, group(text)).result.ignored, 1);
});

test('later paper fills close an imported open position without touching a live trade', () => {
  const open = paper([fill('10:00:00', 'BUY', 5, 100)]);
  const closed = paper([fill('10:00:00', 'BUY', 5, 100), fill('10:01:00', 'SELL', -5, 102)]);
  const live = { ...group(open)[0], id: 'live', accountType: undefined, rawExecutions: [], notes: 'Live trade' };
  const first = parser.upsertImportedTrades([live], group(open));
  assert.equal(first.trades.length, 2);
  const second = parser.upsertImportedTrades(first.trades, group(closed));
  assert.equal(second.result.closedOpen, 1);
  assert.equal(second.trades.length, 2);
  assert.equal(second.trades[0].status, 'Open');
  assert.equal(second.trades[0].notes, 'Live trade');
  assert.equal(second.trades[1].pl, 10);
});

test('unsupported instruments and malformed fills produce diagnostics; canceled orders never import', () => {
  const result = parse(paper([fill('10:00:00', 'BUY', 5, 100), fill('10:01:00', 'BUY', 1, 2, 'CALL'), fill('10:02:00', 'SELL', -5, '~')]));
  assert.equal(result.executions.length, 1);
  assert.equal(result.skippedTradeRows.length, 2);
});

test('cash ledger takes priority over filled orders in a combined export', () => {
  const text = `${csv}\nCash Balance\nDATE,TIME,TYPE,REF #,DESCRIPTION,Misc Fees,Commissions & Fees,AMOUNT,BALANCE\n10/29/26,10:00:00,TRD,123,BOT +5 XYZ @100,,, -500,0\nFutures Statements`;
  const result = parse(text);
  assert.equal(result.executions.length, 1);
  assert.equal(result.executions[0].ticker, 'XYZ');
});

test('a long-to-short reversal allocates cash and fees to both lifecycles', () => {
  const ledger = `DATE,TIME,TYPE,REF #,DESCRIPTION,Misc Fees,Commissions & Fees,AMOUNT,BALANCE
10/29/26,10:00:00,TRD,open,BOT +5 XYZ @100,-1,,-500,0
10/29/26,10:01:00,TRD,reverse,SOLD -8 XYZ @101,-1.6,,808,0
10/29/26,10:02:00,TRD,close,BOT +3 XYZ @99,-0.6,,-297,0`;
  const trades = group(ledger);
  assert.deepEqual(trades.map(t => [t.direction, t.size, t.status, cents(t.pl)]), [
    ['Long', 5, 'Closed', 300], ['Short', 3, 'Closed', 480]
  ]);
  const first = parser.upsertImportedTrades([], trades);
  assert.equal(parser.upsertImportedTrades(first.trades, group(ledger)).result.ignored, 2);
});

test('Cam Journal upload handler imports the attached file and reports matching totals', async () => {
  const source = fs.readFileSync(new URL('../public/cam-journal/app.js', import.meta.url), 'utf8');
  const start = source.indexOf('async function importBrokerCsvFile(');
  const end = source.indexOf('\n}', start) + 2;
  let status = '';
  let saves = 0;
  const context = vm.createContext({
    ...parser, trades: [], selectedTradeIds: new Set(),
    save: () => { saves++; }, renderAll() {},
    showImportStatus: text => { status = text; },
    escapeHtml: text => text, money: value => Number(value).toFixed(2), console
  });
  vm.runInContext(source.slice(start, end), context);
  await context.importBrokerCsvFile({ text: async () => csv });
  assert.equal(context.trades.length, 10);
  assert.match(status, /20 executions/);
  assert.match(status, /New: 10/);
  assert.match(status, /Closed in file: 10\. Open in file: 0\. File P\/L: 312\.64/);
  await context.importBrokerCsvFile({ text: async () => csv });
  assert.equal(context.trades.length, 10);
  assert.match(status, /New: 0\. Updated: 0\. Duplicates ignored: 10/);
  assert.equal(saves, 2);
});
