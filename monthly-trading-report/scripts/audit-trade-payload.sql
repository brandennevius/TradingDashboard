\set ON_ERROR_STOP on
BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SET LOCAL statement_timeout = '60s';
SET LOCAL lock_timeout = '3s';
SELECT current_setting('transaction_read_only') AS read_only, now() AS captured_at;
WITH scopes AS (
  SELECT 'all' AS scope, screenshots FROM public.trade_logs
  UNION ALL
  SELECT 'branden_visible', screenshots FROM public.trade_logs WHERE user_id = 'branden' AND hidden = false
), items AS (
  SELECT scope, item #>> '{}' AS value FROM scopes
  CROSS JOIN LATERAL jsonb_array_elements(CASE WHEN jsonb_typeof(screenshots) = 'array' THEN screenshots ELSE '[]'::jsonb END) item
)
SELECT scope, count(*) AS references,
  count(*) FILTER (WHERE value ~ '^data:image/[^;]+;base64,') AS inline_images,
  coalesce(sum(octet_length(value)) FILTER (WHERE value ~ '^data:image/[^;]+;base64,'),0) AS inline_string_bytes,
  coalesce(max(octet_length(value)) FILTER (WHERE value ~ '^data:image/[^;]+;base64,'),0) AS largest_inline_string_bytes
FROM items GROUP BY scope ORDER BY scope;
SELECT count(*) AS all_trades, count(*) FILTER (WHERE user_id='branden' AND hidden=false) AS branden_visible_trades FROM public.trade_logs;
SELECT 'trade_screenshots' AS table_name, count(*) AS rows,
 count(image_data) AS bytea_rows, coalesce(sum(octet_length(image_data)),0) AS bytea_bytes,
 count(storage_path) AS storage_rows, coalesce(sum(image_bytes),0) AS recorded_image_bytes,
 min(octet_length(image_data)) AS min_bytea_bytes,
 percentile_cont(0.5) WITHIN GROUP (ORDER BY octet_length(image_data)) AS median_bytea_bytes,
 max(octet_length(image_data)) AS max_bytea_bytes
FROM public.trade_screenshots
UNION ALL
SELECT 'cam_journal_screenshots', count(*), count(image_data), coalesce(sum(octet_length(image_data)),0),
 count(storage_path), coalesce(sum(image_bytes),0), min(octet_length(image_data)),
 percentile_cont(0.5) WITHIN GROUP (ORDER BY octet_length(image_data)), max(octet_length(image_data))
FROM public.cam_journal_screenshots;
ROLLBACK;
