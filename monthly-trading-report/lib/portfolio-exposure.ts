export const ACCOUNT_LOSS_THRESHOLD_DOLLARS = 688_000;

export function portfolioExposure(values: Array<number | null>, equity: number | null, floor: number | null) {
  const missingCount = values.filter((value) => value === null || !Number.isFinite(value)).length;
  const dollars = missingCount ? null : values.reduce<number>((sum, value) => sum + Math.abs(value!), 0);
  const validEquity = equity !== null && Number.isFinite(equity) && equity > 0;
  const remainingDrawdown = validEquity && floor !== null && Number.isFinite(floor)
    ? Math.max(0, equity - floor) : null;
  return {
    dollars,
    missingCount,
    remainingDrawdown,
    equityPct: dollars !== null && validEquity ? dollars / equity * 100 : null,
    drawdownPct: dollars !== null && remainingDrawdown !== null && remainingDrawdown > 0
      ? dollars / remainingDrawdown * 100 : null
  };
}
