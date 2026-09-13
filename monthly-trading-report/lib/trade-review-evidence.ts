import { createCanvas, loadImage } from "@napi-rs/canvas";
import { getCachedTradeExcursion, getTradeScreenshot, saveCachedTradeExcursion } from "./store";
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

// Read only inline images or screenshots owned by the selected trade. Never fetch arbitrary URLs.
export async function loadReviewImage(value: string, label: string, trade: TradeLogEntry, example = false): Promise<ReviewImage> {
  let bytes: Buffer;
  const inline = value.match(/^data:image\/(?:png|jpe?g|gif|webp);base64,([A-Za-z0-9+/=\s]+)$/i);
  if (inline) {
    bytes = Buffer.from(inline[1], "base64");
  } else {
    const id = !example ? value.match(/\/screenshots\/([^/?#]+)/)?.[1] : undefined;
    const stored = id ? await getTradeScreenshot(decodeURIComponent(id)) : null;
    if (!stored || stored.tradeId !== trade.id || stored.userId !== trade.userId) {
      throw new Error(`Cannot read ${label}. Re-upload the image before exporting; it was not skipped.`);
    }
    bytes = stored.imageData;
  }
  if (bytes.length > 20 * 1024 * 1024) throw new Error(`${label} exceeds 20 MB. Upload a smaller image.`);
  try {
    const image = await loadImage(bytes);
    if (!image.width || !image.height || image.width * image.height > 40_000_000) throw new Error("Image dimensions exceed limit");
    const canvas = createCanvas(image.width, image.height);
    canvas.getContext("2d").drawImage(image, 0, 0);
    return { label, dataUrl: `data:image/png;base64,${canvas.toBuffer("image/png").toString("base64")}` };
  } catch {
    throw new Error(`Cannot decode ${label}. Re-upload a readable chart image before exporting.`);
  }
}

export async function collectReviewEvidence(trades: TradeLogEntry[], templates: SetupChecklistTemplate[], signal?: AbortSignal): Promise<ReviewEvidence> {
  const evidence: ReviewEvidence = { images: {}, excursions: {} };
  let imageBytes = 0;
  for (const trade of trades) {
    signal?.throwIfAborted();
    evidence.images[trade.id] = [];
    const add = async (image: ReviewImage) => {
      imageBytes += Buffer.byteLength(image.dataUrl);
      if (imageBytes > 43 * 1024 * 1024) throw new Error("The full chart set is too large. Narrow the trade filters or reduce active example charts; no charts were skipped.");
      evidence.images[trade.id].push(image);
    };
    for (const [index, screenshot] of trade.screenshots.entries()) {
      await add(await loadReviewImage(screenshot, `${trade.symbol} ${trade.entryDate} actual trade chart ${index + 1}`, trade));
    }
    const matched = templates.filter((template) => trade.setupTags.some((tag) => tag.trim().toLowerCase() === template.setupName.trim().toLowerCase()));
    for (const template of matched) {
      for (const example of (template.strategyExamples || []).filter((item) => item.active !== false)) {
        for (const [index, screenshot] of example.screenshots.entries()) {
          await add(await loadReviewImage(screenshot, `Comparison example ${example.id}: ${example.symbol} ${example.quality}, ${template.setupName}, chart ${index + 1}`, trade, true));
        }
      }
    }
    evidence.excursions[trade.id] = await loadReviewExcursion(trade, signal);
  }
  return evidence;
}
