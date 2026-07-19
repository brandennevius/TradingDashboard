const LEGACY_CAM_SCREENSHOT_PATH = /^\/api\/cam-journal\/screenshots\/([^/?#]+)(?:[?#].*)?$/;
const WATCHLIST_SCREENSHOT_PATH = /^\/api\/watchlists\/screenshots\/([^/?#]+)(?:[?#].*)?$/;

export function watchlistScreenshotDisplayUrl(value: string) {
  const url = String(value || "").trim();
  const legacyMatch = url.match(LEGACY_CAM_SCREENSHOT_PATH);
  if (!legacyMatch) return url;
  return `/api/watchlists/screenshots/${legacyMatch[1]}`;
}

export function watchlistScreenshotFetchUrl(value: string) {
  const url = String(value || "").trim();
  const storedMatch = url.match(LEGACY_CAM_SCREENSHOT_PATH) || url.match(WATCHLIST_SCREENSHOT_PATH);
  return storedMatch ? `/api/watchlists?screenshotId=${encodeURIComponent(storedMatch[1])}` : url;
}
