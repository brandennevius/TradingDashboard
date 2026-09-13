import { createCanvas, loadImage } from "@napi-rs/canvas";
import { getCachedTradeExcursion, getCamJournalScreenshot, getTradeScreenshot, saveCachedTradeExcursion } from "./store";
import { calculateTradeExcursion, excursionInstrument, tradeExcursionInputHash, unavailableTradeExcursion } from "./trade-excursion";
import { fetchTradeIntradayBars } from "./trade-excursion-market-data";
import type { SetupChecklistTemplate, TradeLogEntry } from "./types";
import type { ReviewEvidence, ReviewImage } from "./trade-review-export";

export async function loadReviewExcursion(trade: TradeLogEntry, signal?: AbortSignal) {
  const asOf = trade.status === "OPEN" ? new Intl.DateTimeFormat("en-CA", { timeZone: "America/New_York", year: "numeric", month: "2-digit", day: "2-digit" }).format(new Date()) : trade.exitDate;
  try {
    const cached = await getCachedTradeExcursion(trade.id, trade.userId, tradeExcursionInputHash(trade, asOf));
    if (cached) return cached;
    const instrument = excursionInstrument(trade.symbol);
    if (!instrument.providerSymbol) return unavailableTradeExcursion(trade, "No compatible market-data symbol is configured.", { asOf });
    const marketData = await fetchTradeIntradayBars(instrument, trade.entryDate, asOf, undefined, signal);
    const result = marketData.error
      ? unavailableTradeExcursion(trade, marketData.error, { provider: marketData.provider, interval: marketData.interval, asOf })
      : calculateTradeExcursion(trade, marketData.bars, { provider: marketData.provider, interval: marketData.interval, asOf, instrument });
    await saveCachedTradeExcursion(trade.id, trade.userId, result);
    return result;
  } catch {
    signal?.throwIfAborted();
    return unavailableTradeExcursion(trade, "Market data could not be loaded for this export.", { asOf });
  }
}

type ReviewImageOwner =
  | { kind: "trade" }
  | { kind: "strategy-example"; exampleId: string };

type ReviewImageLoaders = {
  trade: typeof getTradeScreenshot;
  journal: typeof getCamJournalScreenshot;
};

const defaultReviewImageLoaders: ReviewImageLoaders = {
  trade: getTradeScreenshot,
  journal: getCamJournalScreenshot
};

async function mapWithConcurrency<T, R>(items: T[], concurrency: number, worker: (item: T) => Promise<R>) {
  const results = new Array<R>(items.length);
  let nextIndex = 0;
  const runners = Array.from({ length: Math.min(concurrency, items.length) }, async () => {
    while (nextIndex < items.length) {
      const index = nextIndex++;
      results[index] = await worker(items[index]);
    }
  });
  await Promise.all(runners);
  return results;
}

function storedImagePath(value: string) {
  if (!value.startsWith("/")) return [];
  try {
    return new URL(value, "http://review.local").pathname.split("/").filter(Boolean).map(decodeURIComponent);
  } catch {
    return [];
  }
}

// Read only inline images or screenshots owned by the selected trade/example. Never fetch arbitrary URLs.
export async function loadReviewImage(
  value: string,
  label: string,
  trade: TradeLogEntry,
  owner: ReviewImageOwner = { kind: "trade" },
  loaders: ReviewImageLoaders = defaultReviewImageLoaders
): Promise<ReviewImage> {
  let bytes: Buffer;
  const inline = value.match(/^data:image\/(?:png|jpe?g|gif|webp);base64,([A-Za-z0-9+/=\s]+)$/i);
  if (inline) {
    bytes = Buffer.from(inline[1], "base64");
  } else {
    const path = storedImagePath(value);
    if (owner.kind === "trade") {
      const id = path.length === 5 && path[0] === "api" && path[1] === "trades" && path[2] === trade.id && path[3] === "screenshots"
        ? path[4]
        : "";
      const stored = id ? await loaders.trade(id) : null;
      if (!stored || stored.tradeId !== trade.id || stored.userId !== trade.userId) {
        throw new Error(`Cannot read ${label}. Re-upload the image before exporting; it was not skipped.`);
      }
      bytes = stored.imageData;
    } else {
      const id = path.length === 4 && path[0] === "api" && path[1] === "cam-journal" && path[2] === "screenshots"
        ? path[3]
        : "";
      const stored = id ? await loaders.journal(id) : null;
      if (!stored || stored.entityType !== "setup-strategy-example" || stored.entityId !== owner.exampleId) {
        throw new Error(`Cannot read ${label}. Re-upload the image before exporting; it was not skipped.`);
      }
      bytes = stored.imageData;
    }
    if (!bytes.length) {
      throw new Error(`Cannot read ${label}. Re-upload the image before exporting; it was not skipped.`);
    }
  }
  if (bytes.length > 20 * 1024 * 1024) throw new Error(`${label} exceeds 20 MB. Upload a smaller image.`);
  try {
    const image = await loadImage(bytes);
    if (!image.width || !image.height || image.width * image.height > 40_000_000) throw new Error("Image dimensions exceed limit");
    const scale = Math.min(1, 2400 / Math.max(image.width, image.height), Math.sqrt(4_500_000 / (image.width * image.height)));
    const width = Math.max(1, Math.round(image.width * scale));
    const height = Math.max(1, Math.round(image.height * scale));
    const canvas = createCanvas(width, height);
    const context = canvas.getContext("2d");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    const png = canvas.toBuffer("image/png");
    const jpeg = canvas.toBuffer("image/jpeg", 88);
    const normalized = jpeg.length < png.length ? { mime: "image/jpeg", bytes: jpeg } : { mime: "image/png", bytes: png };
    return { label, dataUrl: `data:${normalized.mime};base64,${normalized.bytes.toString("base64")}` };
  } catch {
    throw new Error(`Cannot decode ${label}. Re-upload a readable chart image before exporting.`);
  }
}

export async function collectReviewEvidence(trades: TradeLogEntry[], templates: SetupChecklistTemplate[], signal?: AbortSignal): Promise<ReviewEvidence> {
  const evidence: ReviewEvidence = { images: {}, excursions: {} };
  let imageBytes = 0;
  const countedImages = new Set<string>();
  const imageLoads = new Map<string, Promise<ReviewImage>>();
  const tasks: { tradeId: string; cacheKey: string; load: () => Promise<ReviewImage> }[] = [];
  for (const trade of trades) {
    evidence.images[trade.id] = [];
    for (const [index, screenshot] of trade.screenshots.entries()) {
      tasks.push({
        tradeId: trade.id,
        cacheKey: `trade:${trade.id}:${screenshot}`,
        load: () => loadReviewImage(screenshot, `${trade.symbol} ${trade.entryDate} actual trade chart ${index + 1}`, trade, { kind: "trade" })
      });
    }
    const matched = templates.filter((template) => trade.setupTags.some((tag) => tag.trim().toLowerCase() === template.setupName.trim().toLowerCase()));
    for (const template of matched) {
      for (const example of (template.strategyExamples || []).filter((item) => item.active !== false)) {
        for (const [index, screenshot] of example.screenshots.entries()) {
          tasks.push({
            tradeId: trade.id,
            cacheKey: `strategy-example:${example.id}:${screenshot}`,
            load: () => loadReviewImage(
              screenshot,
              `Comparison example ${example.id}: ${example.symbol} ${example.quality}, ${template.setupName}, chart ${index + 1}`,
              trade,
              { kind: "strategy-example", exampleId: example.id }
            )
          });
        }
      }
    }
  }

  const loadedImages = await mapWithConcurrency(tasks, 4, async (task) => {
    signal?.throwIfAborted();
    let image = imageLoads.get(task.cacheKey);
    if (!image) {
      image = task.load();
      imageLoads.set(task.cacheKey, image);
    }
    return { tradeId: task.tradeId, image: await image };
  });
  for (const { tradeId, image } of loadedImages) {
    if (!countedImages.has(image.dataUrl)) {
      countedImages.add(image.dataUrl);
      imageBytes += Buffer.byteLength(image.dataUrl);
      if (imageBytes > 43 * 1024 * 1024) throw new Error("The unique chart set is too large. Narrow the trade filters or reduce active example charts; no charts were skipped.");
    }
    evidence.images[tradeId].push(image);
  }

  const excursions = await mapWithConcurrency(trades, 4, async (trade) => {
    signal?.throwIfAborted();
    return { tradeId: trade.id, excursion: await loadReviewExcursion(trade, signal) };
  });
  for (const { tradeId, excursion } of excursions) evidence.excursions[tradeId] = excursion;
  return evidence;
}
