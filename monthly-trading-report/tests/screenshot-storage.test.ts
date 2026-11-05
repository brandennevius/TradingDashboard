import assert from "node:assert/strict";
import test from "node:test";
import { copyImageToStorage, imageHash, prepareImage, readStoredImage, sharedScreenshotIds } from "../lib/screenshot-storage";

test("private immutable uploads verify exact original bytes, retry safely, and fail closed", async () => {
  const previous = { ...process.env };
  const originalFetch = globalThis.fetch;
  const objects = new Map<string, Buffer>();
  let corrupt = false;
  process.env.SUPABASE_STORAGE_URL = "https://test.supabase.co";
  process.env.SUPABASE_STORAGE_SERVICE_KEY = "synthetic-service-key";
  process.env.SCREENSHOT_STORAGE_WRITES = "enabled";
  globalThis.fetch = async (input, options) => {
    const url = new URL(String(input));
    assert.equal(url.origin, "https://test.supabase.co");
    assert.equal(new Headers(options?.headers).get("authorization"), "Bearer synthetic-service-key");
    assert.equal(options?.redirect, "error");
    if (url.pathname.includes("/bucket/")) return Response.json({ public: false });
    const key = url.pathname.replace("/authenticated/", "/");
    if (options?.method === "POST") {
      assert.equal(new Headers(options.headers).get("x-upsert"), "false");
      if (objects.has(key)) return new Response("exists", { status: 409 });
      objects.set(key, Buffer.from(options.body as Uint8Array));
      return new Response("ok");
    }
    const bytes = objects.get(key);
    return bytes ? new Response(new Uint8Array(corrupt ? Buffer.from("corrupt") : bytes)) : new Response("missing", { status: 404 });
  };
  try {
    const bytes = Buffer.from([0, 255, 12, 34]);
    const source = await prepareImage("trade_screenshots", "shot-1", "image/png", bytes);
    assert.equal(source.image_data, null);
    assert.equal(source.storage_path, `trade_screenshots/shot-1/${imageHash(bytes)}`);
    assert.deepEqual(await readStoredImage(source), bytes);
    await copyImageToStorage("trade_screenshots", "shot-1", "image/png", bytes);
    assert.equal(objects.size, 1);
    corrupt = true;
    await assert.rejects(readStoredImage(source), /integrity/);
    assert.deepEqual(await readStoredImage({ ...source, image_data: bytes }), bytes);
    await assert.rejects(copyImageToStorage("trade_screenshots", "shot-1", "image/png", bytes), /integrity/);
    await assert.rejects(copyImageToStorage("trade_screenshots", "../bad", "image/png", bytes), /Invalid/);
    process.env.SUPABASE_STORAGE_URL = "https://untrusted.example";
    await assert.rejects(prepareImage("trade_screenshots", "shot-2", "image/png", bytes), /configured/);
    delete process.env.SCREENSHOT_STORAGE_WRITES;
    assert.deepEqual(await prepareImage("trade_screenshots", "legacy", "image/png", bytes), { image_data: bytes });
  } finally {
    globalThis.fetch = originalFetch;
    for (const key of ["SUPABASE_STORAGE_URL", "SUPABASE_STORAGE_SERVICE_KEY", "SCREENSHOT_STORAGE_WRITES"]) {
      if (previous[key] === undefined) delete process.env[key]; else process.env[key] = previous[key];
    }
  }
});

test("reference discovery covers versioned setup examples and watchlist aliases without accepting remote URLs", () => {
  assert.deepEqual(sharedScreenshotIds({
    versions: [{ examples: ["/api/cam-journal/screenshots/setup-1"] }],
    weekly: [{ shots: ["/api/watchlists/screenshots/watch-1", "/api/cam-journal/screenshots/setup-1"] }],
    ignored: "https://example.com/api/cam-journal/screenshots/no"
  }), ["setup-1", "watch-1"]);
});
