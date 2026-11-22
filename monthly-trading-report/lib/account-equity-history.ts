export type AccountEquitySnapshot = { portfolioTag: string; date: string; equity: number };

/** Recorded statement equity only; never infer balances from trade P&L. */
export function buildAccountEquityHistory(
  snapshots: AccountEquitySnapshot[], portfolios: string[], portfolio: string, start: string, end: string
) {
  const required = [...new Set(portfolio ? [portfolio] : portfolios)];
  if (!required.length || !start || !end || start > end) return [];
  const dates = new Map<string, Map<string, number>>();
  for (const point of snapshots) {
    if (!required.includes(point.portfolioTag) || point.date < start || point.date > end || !Number.isFinite(point.equity)) continue;
    const accounts = dates.get(point.date) || new Map<string, number>();
    accounts.set(point.portfolioTag, point.equity);
    dates.set(point.date, accounts);
  }
  return [...dates].sort(([a], [b]) => a.localeCompare(b)).flatMap(([date, accounts]) => {
    if (!required.every(name => accounts.has(name))) return [];
    return [{ date: `${date} 23:59:59`, accountEquity: required.reduce((sum, name) => sum + accounts.get(name)!, 0) }];
  });
}
