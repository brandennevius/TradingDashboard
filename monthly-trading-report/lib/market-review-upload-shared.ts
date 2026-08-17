export const MARKET_REVIEW_BLOB_PREFIX = "market-review/source";
export const MARKET_REVIEW_MAX_SOURCE_PDF_BYTES = 20 * 1024 * 1024;

export type MarketReviewUploadDescriptor = {
  upload_id: string;
  session_date: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
};

export type MarketReviewBlobReference = MarketReviewUploadDescriptor & {
  blob_url: string;
  blob_pathname: string;
  blob_content_type: string;
};

export function marketReviewBlobPathnameValue(input: MarketReviewUploadDescriptor) {
  return `${MARKET_REVIEW_BLOB_PREFIX}/${input.session_date}/${input.upload_id}/${input.sha256}.pdf`;
}

export function buildMarketReviewCreatePayload(reference: MarketReviewBlobReference) {
  return { session_date: reference.session_date, marketsurge_pdf: reference };
}
