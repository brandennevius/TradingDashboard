import { test, expect } from '@playwright/test';

test('position tooltip shows only exposure held at the hovered timestamp', async ({ page }) => {
  await page.request.post('/api/login', { data: { userId: 'branden', password: process.env.E2E_PASSWORD } });
  await page.request.put('/api/settings/branden-portfolios', {
    data: { portfolios: ['Tooltip Test'], defaultPortfolio: 'Tooltip Test' }
  });
  const entry = (date: string, time: string, shares: number) => ({ type: 'ENTRY', date, time, shares, price: 100 });
  const exit = (date: string, time: string, shares: number) => ({ type: 'EXIT', date, time, shares, price: 100 });
  for (const input of [
    { symbol: 'CLOSED_EARLY', status: 'BREAKEVEN', entryDate: '2026-11-01', exitDate: '2026-11-01',
      executions: [entry('2026-11-01', '09:00:00', 10), exit('2026-11-01', '10:00:00', 10)] },
    // Closed now, but still active at the historical point we hover.
    { symbol: 'HELD_THEN', status: 'BREAKEVEN', entryDate: '2026-11-01', exitDate: '2026-11-03',
      executions: [entry('2026-11-01', '11:00:00', 10), exit('2026-11-03', '09:00:00', 10)] },
    { symbol: 'NOT_YET_OPEN', status: 'OPEN', entryDate: '2026-11-03',
      executions: [entry('2026-11-03', '10:00:00', 10)] }
  ]) {
    const created = await page.request.post('/api/trades', {
      data: { ...input, portfolioTag: 'Tooltip Test', shares: 10, avgEntry: 100 }
    });
    expect(created.ok()).toBeTruthy();
  }
  await page.goto('/journal/branden/dashboard?portfolio=Tooltip%20Test');
  await page.getByLabel('Start date', { exact: true }).fill('2026-11-01');
  await page.getByLabel('End date', { exact: true }).fill('2026-11-03');
  await page.getByLabel('Drawdown budget ($)', { exact: true }).fill('10000');
  const chart = page.locator('.position-size-panel');
  await expect(chart.locator('.position-size-legend')).toContainText('HELD_THEN');
  const axis = chart.locator('.recharts-xAxis .recharts-cartesian-axis-line');
  await expect(chart.locator("svg.recharts-surface")).toBeVisible();
  await expect(axis).toBeAttached();
  // Nine categorical timestamps: range start, five fills and three daily ends.
  // Hover index 4: Nov 1 end.
  await chart.locator("svg.recharts-surface").scrollIntoViewIfNeeded();
  const box = await axis.boundingBox();
  expect(box).not.toBeNull();
  const hover = async (index: number) => page.mouse.move(box!.x + box!.width * index / 8, box!.y - 50);
  await hover(4);
  const tooltip = chart.locator('.position-size-tooltip');
  await expect(tooltip).toContainText('2026-11-01 23:59:59');
  await expect(tooltip).toContainText('HELD_THEN');
  await expect(tooltip).not.toContainText('CLOSED_EARLY');
  await expect(tooltip).not.toContainText('NOT_YET_OPEN');
  await expect(tooltip).toContainText('Total: 10.00%');
  // Immediately after HELD_THEN closes, before the next position opens.
  await hover(6);
  await expect(tooltip).toContainText('2026-11-03 09:00:00');
  await expect(tooltip).toContainText('No open positions at this time.');
  await expect(tooltip).toContainText('Total: 0.00%');
  await expect(tooltip).not.toContainText('HELD_THEN');
  // At entry, the new trade is included; the two earlier closed trades are not.
  await hover(7);
  await expect(tooltip).toContainText('NOT_YET_OPEN');
  await expect(tooltip).not.toContainText('CLOSED_EARLY');
  await expect(tooltip).not.toContainText('HELD_THEN');
});
