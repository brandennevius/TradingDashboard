export type MarketReviewOcrReviewRow = {
  key: string;
  pdfPage: number;
  label: string;
  rank: number | null;
  rawText: string | null;
  candidateTicker: string | null;
  confidence: number | null;
  reason: string;
};

export type MarketReviewOcrReviewPage = {
  pdfPage: number;
  label: string;
  acceptedTickers: string[];
  reviewRows: MarketReviewOcrReviewRow[];
};

export type MarketReviewOcrCorrection = {
  pdf_page: number;
  label: string;
  tickers: string[];
  reviewed: true;
};

const EXCLUDED_TOKENS = new Set([
  "FAVORITES", "MARKET", "MARKETS", "NAME", "SCREENS", "STOCK", "STOCKS", "SYMBOL", "TICKER"
]);

function record(value: unknown): Record<string, unknown> | null {
  return value && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : null;
}

function integer(value: unknown) {
  return Number.isInteger(value) && Number(value) > 0 ? Number(value) : null;
}

function text(value: unknown) {
  return typeof value === "string" && value.trim() ? value.trim() : null;
}

function number(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function normalizeMarketReviewTicker(value: unknown) {
  const ticker = String(value || "").trim().toUpperCase();
  if (!/^[A-Z][A-Z0-9.-]{0,9}$/.test(ticker) || EXCLUDED_TOKENS.has(ticker)) return null;
  return ticker;
}

export function marketReviewOcrRowKey(pdfPage: number, rank: number | null, index: number) {
  return `${pdfPage}:${rank ?? "unranked"}:${index}`;
}

export function parseMarketReviewOcrReview(value: unknown): MarketReviewOcrReviewPage[] {
  const ocr = record(value);
  const items = ocr && Array.isArray(ocr.items) ? ocr.items : [];
  const pages: MarketReviewOcrReviewPage[] = [];
  const seenPages = new Set<number>();

  for (const rawItem of items) {
    const item = record(rawItem);
    if (!item) continue;
    const pdfPage = integer(item.pdf_page);
    const label = text(item.label);
    if (!pdfPage || !label || seenPages.has(pdfPage)) continue;
    seenPages.add(pdfPage);

    const acceptedTickers: string[] = [];
    if (Array.isArray(item.tickers)) {
      for (const rawTicker of item.tickers) {
        const ticker = normalizeMarketReviewTicker(rawTicker);
        if (ticker && !acceptedTickers.includes(ticker)) acceptedTickers.push(ticker);
      }
    }

    const reviewRows = (Array.isArray(item.review_rows) ? item.review_rows : []).flatMap((rawRow, index) => {
      const row = record(rawRow);
      if (!row) return [];
      const rank = integer(row.rank);
      return [{
        key: marketReviewOcrRowKey(pdfPage, rank, index),
        pdfPage,
        label,
        rank,
        rawText: text(row.raw_text),
        candidateTicker: text(row.candidate_ticker),
        confidence: number(row.confidence),
        reason: text(row.reason) || "OCR_REVIEW_REQUIRED"
      } satisfies MarketReviewOcrReviewRow];
    });

    pages.push({ pdfPage, label, acceptedTickers, reviewRows });
  }

  return pages.sort((left, right) => left.pdfPage - right.pdfPage);
}

export function buildMarketReviewOcrCorrections(
  pages: MarketReviewOcrReviewPage[],
  resolutions: Record<string, string>,
  reviewedPages: Record<number, boolean>
) {
  const errors: string[] = [];
  if (!pages.length) errors.push("No OCR pages were returned for review.");

  const corrections: MarketReviewOcrCorrection[] = pages.map((page) => {
    const tickers = [...page.acceptedTickers];
    for (const row of page.reviewRows) {
      const ticker = normalizeMarketReviewTicker(resolutions[row.key]);
      if (!ticker) {
        errors.push(`Page ${page.pdfPage}${row.rank ? ` rank ${row.rank}` : ""} needs a valid ticker.`);
      } else if (!tickers.includes(ticker)) {
        tickers.push(ticker);
      }
    }
    if (!reviewedPages[page.pdfPage]) errors.push(`Page ${page.pdfPage} has not been marked reviewed.`);
    return { pdf_page: page.pdfPage, label: page.label, tickers, reviewed: true };
  });

  return { ready: errors.length === 0, errors, corrections };
}

export function hasSavedV2MarketReviewCorrections(value: unknown) {
  const ocr = record(value);
  if (!ocr || ocr.status !== "CORRECTED" || ocr.schema_version !== "marketsurge_ocr_v2" || !Array.isArray(ocr.corrections)) return false;
  return ocr.corrections.length > 0 && ocr.corrections.every((raw) => record(raw)?.reviewed === true);
}

export function canRetryMarketReview(status: string, ocr: unknown) {
  return status === "FAILED" || (status === "NEEDS_REVIEW" && hasSavedV2MarketReviewCorrections(ocr));
}
