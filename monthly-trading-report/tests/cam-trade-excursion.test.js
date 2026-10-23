const assert = require('node:assert/strict');
const test = require('node:test');

const {
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
