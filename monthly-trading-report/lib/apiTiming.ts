type ApiTimingMeta = Record<string, string | number | boolean | null | undefined>;

export function createApiTimer(route: string) {
  const startedAt = Date.now();

  return function logApiTiming(status: number, meta: ApiTimingMeta = {}) {
    const durationMs = Date.now() - startedAt;
    const payload = {
      scope: "api-timing",
      route,
      status,
      durationMs,
      ...meta
    };

    if (status >= 500 || durationMs >= 1000) {
      console.warn(payload);
      return;
    }

    console.info(payload);
  };
}
