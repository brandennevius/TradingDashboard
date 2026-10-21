import fs from "node:fs/promises";
import path from "node:path";
import { getPool } from "./store";
import { GAUGE_VERSION } from "./market-gauge";
import type { MarketArchive, MarketSnapshot } from "./trade-market-context";

const file = () => path.join(process.cwd(), "data", "market-context-history.json");
const empty = (): MarketArchive => ({ sessions: [], snapshots: [], refreshedAt: null });
let schema: Promise<unknown> | null = null;
async function database() {
  const db = getPool();
  if (db) {
    schema ||= db.query(`create table if not exists market_context_history (
      session_date text not null, version text not null, snapshot jsonb,
      refreshed_at timestamptz not null default now(), primary key (session_date, version))`).catch((e) => { schema = null; throw e; });
    await schema;
  }
  return db;
}

export async function readMarketArchive(): Promise<MarketArchive> {
  const db = await database();
  if (db) {
    const result = await db.query("select session_date, snapshot, refreshed_at from market_context_history where version = $1 order by session_date", [GAUGE_VERSION]);
    return { sessions: result.rows.map((r) => r.session_date), snapshots: result.rows.flatMap((r) => r.snapshot ? [r.snapshot as MarketSnapshot] : []),
      refreshedAt: result.rows.map((r) => new Date(r.refreshed_at).toISOString()).sort().at(-1) || null };
  }
  try { return JSON.parse(await fs.readFile(file(), "utf8")) as MarketArchive; }
  catch (e) { if ((e as NodeJS.ErrnoException).code === "ENOENT") return empty(); throw e; }
}

let writes = Promise.resolve();
export async function saveMarketArchive(sessions: string[], snapshots: MarketSnapshot[]) {
  const db = await database();
  const refreshedAt = new Date().toISOString();
  if (db) {
    const bySession = new Map(snapshots.map((s) => [s.session, s]));
    await db.query(`insert into market_context_history (session_date, version, snapshot, refreshed_at)
      select x.session, $1, x.snapshot, $3::timestamptz
      from jsonb_to_recordset($2::jsonb) as x(session text, snapshot jsonb)
      on conflict (session_date, version) do update set
        snapshot = coalesce(market_context_history.snapshot, excluded.snapshot), refreshed_at = excluded.refreshed_at`,
    [GAUGE_VERSION, JSON.stringify(sessions.map((session) => ({ session, snapshot: bySession.get(session) || null }))), refreshedAt]);
    return;
  }
  const write = writes.then(async () => {
    const current = await readMarketArchive();
    const bySession = new Map(current.snapshots.map((s) => [s.session, s]));
    for (const snapshot of snapshots) if (!bySession.has(snapshot.session)) bySession.set(snapshot.session, snapshot);
    const next = { sessions: [...new Set([...current.sessions, ...sessions])].sort(), snapshots: [...bySession.values()], refreshedAt };
    await fs.mkdir(path.dirname(file()), { recursive: true });
    const tmp = `${file()}.${process.pid}.tmp`;
    await fs.writeFile(tmp, JSON.stringify(next));
    await fs.rename(tmp, file());
  });
  writes = write.catch(() => {});
  await write;
}
