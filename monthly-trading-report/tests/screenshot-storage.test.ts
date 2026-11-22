import assert from "node:assert/strict";
import test from "node:test";
import sharp from "sharp";
import { mockR2 } from "./fixtures/mock-r2";
import { copyImageToStorage, encodeNewScreenshot, imageHash, prepareImage, readStoredImage, sharedScreenshotIds, ScreenshotPixelLimitError } from "../lib/screenshot-storage";

test("private immutable uploads verify exact original bytes, retry safely, and fail closed", async () => {
  const r2 = mockR2();
  const previousWrites = process.env.SCREENSHOT_STORAGE_WRITES;
  process.env.SCREENSHOT_STORAGE_WRITES = "enabled";
  try {
    const bytes = Buffer.from([0, 255, 12, 34]);
    const source = await prepareImage("trade_screenshots", "shot-1", "image/png", bytes);
    assert.equal(source.image_data, null);
    assert.equal(source.storage_path, `trade_screenshots/shot-1/${imageHash(bytes)}`);
    assert.deepEqual(await readStoredImage(source), bytes);
    await copyImageToStorage("trade_screenshots", "shot-1", "image/png", bytes);
    assert.equal(r2.objects.size, 1);
    r2.controls.corrupt = true;
    await assert.rejects(readStoredImage(source), /integrity/);
    assert.deepEqual(await readStoredImage({ ...source, image_data: bytes }), bytes);
    await assert.rejects(copyImageToStorage("trade_screenshots", "shot-1", "image/png", bytes), /integrity/);
    await assert.rejects(copyImageToStorage("trade_screenshots", "../bad", "image/png", bytes), /Invalid/);
    process.env.R2_ACCOUNT_ID = "https://untrusted.example";
    await assert.rejects(prepareImage("trade_screenshots", "shot-2", "image/png", bytes), /configured/);
    delete process.env.SCREENSHOT_STORAGE_WRITES;
    assert.deepEqual(await prepareImage("trade_screenshots", "legacy", "image/png", bytes), { image_data: bytes });
  } finally {
    r2.restore();
    if (previousWrites === undefined) delete process.env.SCREENSHOT_STORAGE_WRITES;
    else process.env.SCREENSHOT_STORAGE_WRITES = previousWrites;
  }
});

test("reference discovery covers versioned setup examples and watchlist aliases without accepting remote URLs", () => {
  assert.deepEqual(sharedScreenshotIds({
    versions: [{ examples: ["/api/cam-journal/screenshots/setup-1"] }],
    weekly: [{ shots: ["/api/watchlists/screenshots/watch-1", "/api/cam-journal/screenshots/setup-1"] }],
    ignored: "https://example.com/api/cam-journal/screenshots/no"
  }), ["setup-1", "watch-1"]);
});


test("new uploads are encoded as bounded JPEGs only when R2 writes are enabled", async () => {
  const previous = process.env.SCREENSHOT_STORAGE_WRITES;
  const png = await sharp({ create: { width: 3000, height: 1500, channels: 3, background: "#567" } }).png().toBuffer();
  try {
    delete process.env.SCREENSHOT_STORAGE_WRITES;
    assert.deepEqual(await encodeNewScreenshot("chart.png", "image/png", png),
      { fileName: "chart.png", mimeType: "image/png", bytes: png });
    process.env.SCREENSHOT_STORAGE_WRITES = "enabled";
    const result = await encodeNewScreenshot("chart.png", "image/png", png);
    assert.equal(result.fileName, "chart.jpg");
    assert.equal(result.mimeType, "image/jpeg");
    const { format, width, height } = await sharp(result.bytes).metadata();
    assert.deepEqual({ format, width, height }, { format: "jpeg", width: 2560, height: 1280 });
    await assert.rejects(encodeNewScreenshot("bad.png", "image/png", Buffer.from("invalid")));
  } finally {
    if (previous === undefined) delete process.env.SCREENSHOT_STORAGE_WRITES;
    else process.env.SCREENSHOT_STORAGE_WRITES = previous;
  }
});

test("a small compressed image above 40 million pixels gets an actionable size error", async () => {
  const previous = process.env.SCREENSHOT_STORAGE_WRITES;
  // A valid SVG gives independently chosen dimensions without a huge fixture file.
  const image = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="8001" height="5000"><rect width="100%" height="100%" fill="#567"/></svg>');
  assert.equal(8001 * 5000, 40_005_000);
  assert.ok(image.length < 3_500_000, "File bytes and decoded pixel count are separate limits");
  try {
    process.env.SCREENSHOT_STORAGE_WRITES = "enabled";
    await assert.rejects(encodeNewScreenshot("large.svg", "image/svg+xml", image), error => {
      assert.ok(error instanceof ScreenshotPixelLimitError);
      assert.equal(error.status, 413);
      assert.match(error.message, /40 million pixels/);
      assert.match(error.message, /Crop.*width and height/);
      return true;
    });
    // Existing passthrough mode is unchanged, including its original bytes.
    delete process.env.SCREENSHOT_STORAGE_WRITES;
    assert.deepEqual(await encodeNewScreenshot("large.svg", "image/svg+xml", image),
      { fileName: "large.svg", mimeType: "image/svg+xml", bytes: image });
  } finally {
    if (previous === undefined) delete process.env.SCREENSHOT_STORAGE_WRITES;
    else process.env.SCREENSHOT_STORAGE_WRITES = previous;
  }
});

test("the 40-million-pixel boundary still encodes and unrelated corrupt images are not called oversized", async () => {
  const previous = process.env.SCREENSHOT_STORAGE_WRITES;
  const image = Buffer.from('<svg xmlns="http://www.w3.org/2000/svg" width="8000" height="5000"><rect width="100%" height="100%" fill="#567"/></svg>');
  try {
    process.env.SCREENSHOT_STORAGE_WRITES = "enabled";
    const result = await encodeNewScreenshot("boundary.svg", "image/svg+xml", image);
    const { width, height, format } = await sharp(result.bytes).metadata();
    assert.deepEqual({ width, height, format }, { width: 2560, height: 1600, format: "jpeg" });
    await assert.rejects(encodeNewScreenshot("corrupt.png", "image/png", Buffer.from("not an image")),
      error => !(error instanceof ScreenshotPixelLimitError));
  } finally {
    if (previous === undefined) delete process.env.SCREENSHOT_STORAGE_WRITES;
    else process.env.SCREENSHOT_STORAGE_WRITES = previous;
  }
});
