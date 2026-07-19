import assert from "node:assert/strict";
import test from "node:test";
import { watchlistScreenshotDisplayUrl, watchlistScreenshotFetchUrl } from "../lib/watchlist-screenshot";

test("legacy Cam screenshot URLs use the authenticated watchlist image route", () => {
  assert.equal(
    watchlistScreenshotDisplayUrl("/api/cam-journal/screenshots/62765cc0-95bc-41f2-934f-905f4a003d33"),
    "/api/watchlists/screenshots/62765cc0-95bc-41f2-934f-905f4a003d33"
  );
});

test("current watchlist and external screenshot URLs remain unchanged", () => {
  assert.equal(watchlistScreenshotDisplayUrl("/api/watchlists/screenshots/abc"), "/api/watchlists/screenshots/abc");
  assert.equal(watchlistScreenshotDisplayUrl("https://example.test/chart.png"), "https://example.test/chart.png");
});

test("stored screenshots load through the existing authenticated watchlist request", () => {
  assert.equal(
    watchlistScreenshotFetchUrl("/api/cam-journal/screenshots/62765cc0-95bc-41f2-934f-905f4a003d33"),
    "/api/watchlists?screenshotId=62765cc0-95bc-41f2-934f-905f4a003d33"
  );
  assert.equal(
    watchlistScreenshotFetchUrl("/api/watchlists/screenshots/abc"),
    "/api/watchlists?screenshotId=abc"
  );
  assert.equal(watchlistScreenshotFetchUrl("https://example.test/chart.png"), "https://example.test/chart.png");
});
