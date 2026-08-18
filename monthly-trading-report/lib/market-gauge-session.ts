export function filterCandlesThroughSession<T extends { time: string }>(candles: T[], sessionDate?: string) {
  return sessionDate ? candles.filter((candle) => candle.time <= sessionDate) : candles;
}

export function hasExactIndexSessionEvidence<T extends { symbol: string; date: string }>(
  regimes: T[],
  sessionDate: string,
  expectedSymbols: readonly string[]
) {
  const symbols = new Set(regimes.filter((item) => item.date === sessionDate).map((item) => item.symbol));
  return symbols.size === expectedSymbols.length && expectedSymbols.every((symbol) => symbols.has(symbol));
}
