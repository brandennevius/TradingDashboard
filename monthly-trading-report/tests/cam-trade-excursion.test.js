const assert = require('node:assert/strict');
const test = require('node:test');

const {
  initialEntry,
  rFromPrice,
  priceFromR
} = require('../public/cam-journal/trade-excursion-calculator.js');

test('calculates long MFE and MAE from prices', () => {
  const trade = { entry: 100, size: 50, risk: 250, direction: 'Long' };
  assert.equal(rFromPrice(110, trade), 2);
  assert.equal(rFromPrice(95, trade), -1);
});

test('calculates short MFE and MAE from prices', () => {
  const trade = { entry: 100, size: 50, risk: 250, direction: 'Short' };
  assert.equal(rFromPrice(90, trade), 2);
  assert.equal(rFromPrice(105, trade), -1);
});

test('migrates legacy R values back to prices in both directions', () => {
  const longTrade = { entry: 100, size: 50, risk: 250, direction: 'Long' };
  const shortTrade = { entry: 100, size: 50, risk: 250, direction: 'Short' };
  assert.equal(priceFromR(2, longTrade), 110);
  assert.equal(priceFromR(-1, longTrade), 95);
  assert.equal(priceFromR(2, shortTrade), 90);
  assert.equal(priceFromR(-1, shortTrade), 105);
});

test('returns null when required trade data is missing or invalid', () => {
  assert.equal(rFromPrice(110, { entry: 100, size: 50, risk: 0, direction: 'Long' }), null);
  assert.equal(rFromPrice('', { entry: 100, size: 50, risk: 250, direction: 'Long' }), null);
  assert.equal(priceFromR(2, { entry: 100, size: '', risk: 250, direction: 'Long' }), null);
});

test('round trips fractional trade values without losing precision', () => {
  const trade = { entry: 42.125, size: 137.5, risk: 83.2, direction: 'Short' };
  const price = 40.875;
  const r = rFromPrice(price, trade);
  assert.ok(Math.abs(priceFromR(r, trade) - price) < 1e-10);
});

test('averages partial opening fills within 30 seconds and ignores later scale-ins', () => {
  const executions = [
    {date: '2026-10-20', time: '09:30:00', action: 'BOT', qty: 40, price: 100},
    {date: '2026-10-20', time: '09:30:20', action: 'BOT', qty: 60, price: 101},
    {date: '2026-10-20', time: '09:35:00', action: 'BOT', qty: 100, price: 105}
  ];
  const reference = initialEntry(executions, {direction: 'Long'});
  assert.deepEqual(reference, {entry: 100.6, size: 100, fillCount: 2, source: '2 fills within 30 sec'});
});

test('uses sell executions as the initial entry for short trades', () => {
  const executions = [
    {date: '10/20/26', time: '9:30:00 AM', action: 'SOLD', qty: 25, price: 50},
    {date: '10/20/26', time: '9:30:10 AM', action: 'SOLD', qty: 25, price: 49.8},
    {date: '10/20/26', time: '9:31:00 AM', action: 'BOT', qty: 10, price: 49}
  ];
  assert.deepEqual(initialEntry(executions, {direction: 'Short'}), {
    entry: 49.9,
    size: 50,
    fillCount: 2,
    source: '2 fills within 30 sec'
  });
});

test('risk distance keeps excursion R independent from final size and risk', () => {
  const trade = {entry: 100, size: 500, risk: 2000, riskDistance: 2, direction: 'Long'};
  assert.equal(rFromPrice(104, trade), 2);
  assert.equal(priceFromR(-1.5, trade), 97);
});
