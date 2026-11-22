import test from 'node:test';
import assert from 'node:assert/strict';
import { buildAccountEquityHistory, type AccountEquitySnapshot } from '../lib/account-equity-history';
const point = (portfolioTag: string, date: string, equity: number): AccountEquitySnapshot => ({ portfolioTag, date, equity });

test('selected portfolio uses recorded dollar balances within the scope, not a zero-based P&L curve', () => {
  const points = buildAccountEquityHistory([
    point('A', '2026-10-31', 99000), point('B', '2026-11-01', 700000),
    point('A', '2026-11-03', 99500), point('A', '2026-11-01', 100000),
    point('A', '2026-11-04', 101000)
  ], ['A', 'B'], 'A', '2026-11-01', '2026-11-03');
  assert.deepEqual(points, [
    { date: '2026-11-01 23:59:59', accountEquity: 100000 },
    { date: '2026-11-03 23:59:59', accountEquity: 99500 }
  ]);
});

test('all portfolios sum only same-day complete observations, never stale or future values', () => {
  const points = buildAccountEquityHistory([
    point('A', '2026-11-01', 100000), point('B', '2026-11-01', 200),
    point('A', '2026-11-02', 98000), point('B', '2026-11-03', 300),
    point('A', '2026-11-04', 97000), point('B', '2026-11-04', 400),
    point('UNSELECTED', '2026-11-04', 5000000)
  ], ['A', 'B', 'A'], '', '2026-11-01', '2026-11-04');
  assert.deepEqual(points, [
    { date: '2026-11-01 23:59:59', accountEquity: 100200 },
    { date: '2026-11-04 23:59:59', accountEquity: 97400 }
  ]);
});

test('missing history stays absent and invalid balances cannot fabricate an equity line', () => {
  assert.deepEqual(buildAccountEquityHistory([], ['A'], '', '2026-11-01', '2026-11-03'), []);
  assert.deepEqual(buildAccountEquityHistory([point('A', '2026-11-01', NaN)], ['A'], '', '2026-11-01', '2026-11-03'), []);
  assert.deepEqual(buildAccountEquityHistory([point('A', '2026-11-01', 100000)], ['A', 'MISSING'], '', '2026-11-01', '2026-11-03'), []);
});
