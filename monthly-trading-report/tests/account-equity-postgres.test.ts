import test from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { Client } from 'pg';

test('equity projection reads only the requested owner and falls back to statement equity', {
  skip: !process.env.BACKUP_TEST_DATABASE_URL
}, async () => {
  const url = new URL(process.env.BACKUP_TEST_DATABASE_URL!);
  assert.equal(url.hostname, 'localhost');
  assert.match(url.pathname, /^\/trading_dashboard_backup_test(?:_[a-z0-9]+)?$/);
  const schema = `equity_test_${randomUUID().replaceAll('-', '')}`;
  const admin = new Client({ connectionString: url.toString(), ssl: false });
  await admin.connect();
  const previous = process.env.DATABASE_URL;
  let pool: ReturnType<typeof import('../lib/store')['getPool']> = null;
  try {
    await admin.query(`create schema ${schema}`);
    url.searchParams.set('options', `-c search_path=${schema}`);
    url.searchParams.set('application_name', schema);
    process.env.DATABASE_URL = url.toString();
    const store = await import('../lib/store');
    pool = store.getPool();
    assert.ok(pool);
    assert.deepEqual(await store.listBrokerPortfolioEquityHistory('branden'), []);
    await pool.query(`create table broker_portfolio_snapshots (
      user_id text, portfolio_tag text, coverage_date text,
      current_equity numeric, statement_equity numeric, open_positions jsonb
    );
    insert into broker_portfolio_snapshots values
      ('branden','A','2026-11-01',100000,99000,'[{"private":"position data"}]'),
      ('branden','B','2026-11-01',null,200,'[]'),
      ('branden','A','2026-11-02',0,99000,'[]'),
      ('branden','A','2026-11-03',null,null,'[]'),
      ('cam','A','2026-11-01',700000,699000,'[]')`);
    assert.deepEqual(await store.listBrokerPortfolioEquityHistory('branden'), [
      { portfolioTag: 'A', date: '2026-11-01', equity: 100000 },
      { portfolioTag: 'B', date: '2026-11-01', equity: 200 },
      { portfolioTag: 'A', date: '2026-11-02', equity: 0 }
    ]);
  } finally {
    if (pool) await pool.end();
    if (previous === undefined) delete process.env.DATABASE_URL;
    else process.env.DATABASE_URL = previous;
    try { await admin.query(`drop schema if exists ${schema} cascade`); }
    finally { await admin.end(); }
  }
});
