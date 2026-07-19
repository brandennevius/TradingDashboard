const LEGACY_CAM_SCREENSHOT_PATH = /^\/api\/cam-journal\/screenshots\/([^/?#]+)(?:[?#].*)?$/;

export function watchlistScreenshotDisplayUrl(value: string) {
  const url = String(value || "").trim();
  const legacyMatch = url.match(LEGACY_CAM_SCREENSHOT_PATH);
  if (!legacyMatch) return url;
  return `/api/watchlists/screenshots/${legacyMatch[1]}`;
}
