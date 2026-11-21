import { test, expect, type Page } from '@playwright/test';

async function openTrade(page: Page) {
  await page.request.post('/api/login', { data: { userId: 'branden', password: process.env.E2E_PASSWORD } });
  const response = await page.request.post('/api/trades', { data: { symbol: 'DROP', entryDate: '2026-11-20', status: 'OPEN' } });
  expect(response.ok()).toBeTruthy();
  const { trade } = await response.json();
  await page.goto(`/?journal=branden&tradeId=${trade.id}`);
  await expect(page.getByRole('heading', { name: '#DROP', exact: true })).toBeVisible();
  return trade;
}

async function drop(page: Page, files: { name: string; type: string; size?: number }[]) {
  const dataTransfer = await page.evaluateHandle(items => {
    const dt = new DataTransfer();
    for (const item of items) dt.items.add(new File([new Uint8Array(item.size || 20)], item.name, { type: item.type }));
    return dt;
  }, files);
  const section = page.getByRole('article', { name: 'Trade screenshots' });
  await section.dispatchEvent('dragenter', { dataTransfer });
  await section.dispatchEvent('dragover', { dataTransfer });
  await section.dispatchEvent('drop', { dataTransfer });
  await dataTransfer.dispose();
}

test('desktop drop uploads multiple images automatically and preserves unsaved reviews', async ({ page }) => {
  const trade = await openTrade(page);
  let uploads = 0;
  await page.route(`**/api/trades/${trade.id}/screenshots`, async route => {
    expect(route.request().method()).toBe('POST');
    expect(route.request().headers()['idempotency-key']).toBeTruthy();
    expect(route.request().postDataBuffer()?.toString()).toContain('filename="chart');
    const url = `/drop-test-${++uploads}.png`;
    // Persist the reference in the disposable journal to verify it survives reload.
    await page.request.put(`/api/trades/${trade.id}`, { data: { ...trade, screenshots: Array.from({length: uploads}, (_, i) => `/drop-test-${i + 1}.png`) } });
    await route.fulfill({ json: { url } });
  });
  const review = page.getByLabel('General review (optional)', { exact: false });
  await review.fill('Keep this unsaved review while uploading');
  await drop(page, [{ name: 'chart-a.png', type: 'image/png' }, { name: 'chart-b.jpg', type: 'image/jpeg' }]);
  const section = page.getByRole('article', { name: 'Trade screenshots' });
  await expect(section.getByRole('status')).toHaveText('2 images saved.');
  await expect(section.locator('.trade-edit-screenshot')).toHaveCount(2);
  await expect(review).toHaveValue('Keep this unsaved review while uploading');
  expect(uploads).toBe(2);
  await page.reload();
  await expect(section.locator('.trade-edit-screenshot')).toHaveCount(2);
});

test('rejects invalid and oversized drops without posting; picker still works', async ({ page }) => {
  const trade = await openTrade(page);
  let uploads = 0;
  await page.route(`**/api/trades/${trade.id}/screenshots`, route => {
    uploads++;
    return route.fulfill({ json: { url: '/picker-test.png' } });
  });
  const section = page.getByRole('article', { name: 'Trade screenshots' });
  await drop(page, [{ name: 'notes.txt', type: 'text/plain' }]);
  await expect(section.getByRole('status')).toContainText('No files were uploaded.');
  await drop(page, [{ name: 'huge.png', type: 'image/png', size: 3500001 }]);
  await expect(section.getByRole('status')).toContainText('huge.png');
  expect(uploads).toBe(0);
  const chooser = page.waitForEvent('filechooser');
  await section.locator('.trade-screenshot-upload').click();
  await (await chooser).setFiles({ name: 'chart.png', mimeType: 'image/png', buffer: Buffer.from('test image') });
  await expect(section.getByRole('status')).toHaveText('1 image saved.');
});

test('partial failures keep earlier images visible and allow another upload', async ({ page }) => {
  const trade = await openTrade(page);
  let uploads = 0;
  await page.route(`**/api/trades/${trade.id}/screenshots`, route => {
    uploads++;
    return uploads === 2 ? route.fulfill({ status: 500, json: { error: 'Storage temporarily unavailable.' } })
      : route.fulfill({ json: { url: `/partial-${uploads}.png` } });
  });
  const section = page.getByRole('article', { name: 'Trade screenshots' });
  await drop(page, [{ name: 'a.png', type: 'image/png' }, { name: 'b.png', type: 'image/png' }]);
  await expect(section.getByRole('status')).toHaveText('1 image saved. Storage temporarily unavailable.');
  await expect(section.locator('.trade-edit-screenshot')).toHaveCount(1);
  await drop(page, [{ name: 'retry.png', type: 'image/png' }]);
  await expect(section.getByRole('status')).toHaveText('1 image saved.');
  await expect(section.locator('.trade-edit-screenshot')).toHaveCount(2);
});

test('read-only viewers cannot upload by dropping files', async ({ page }) => {
  const trade = await openTrade(page);
  await page.request.post('/api/login', { data: { userId: 'tim', password: 'tim' } });
  await page.reload();
  const section = page.getByRole('article', { name: 'Trade screenshots' });
  await expect(section).toBeVisible();
  let uploads = 0;
  await page.route(`**/api/trades/${trade.id}/screenshots`, route => { uploads++; return route.abort(); });
  await drop(page, [{ name: 'readonly.png', type: 'image/png' }]);
  await expect(section.locator('.trade-screenshot-upload')).toHaveCount(0);
  await expect(section.getByRole('status')).toHaveCount(0);
  expect(uploads).toBe(0);
});
