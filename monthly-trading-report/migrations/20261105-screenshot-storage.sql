-- Run before deploying the Storage-compatible application. No bytes are removed.
-- Take/verify backup first. Short lock timeout makes a busy database fail safely.
BEGIN;
SET LOCAL lock_timeout = '5s';
ALTER TABLE public.trade_screenshots
  ADD COLUMN IF NOT EXISTS storage_path text,
  ADD COLUMN IF NOT EXISTS image_sha256 text,
  ADD COLUMN IF NOT EXISTS image_bytes bigint,
  ALTER COLUMN image_data DROP NOT NULL;
ALTER TABLE public.cam_journal_screenshots
  ADD COLUMN IF NOT EXISTS storage_path text,
  ADD COLUMN IF NOT EXISTS image_sha256 text,
  ADD COLUMN IF NOT EXISTS image_bytes bigint,
  ALTER COLUMN image_data DROP NOT NULL;
DO $$
DECLARE table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY['trade_screenshots','cam_journal_screenshots'] LOOP
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
      WHERE conrelid = ('public.' || table_name)::regclass AND conname = table_name || '_image_source') THEN
      EXECUTE format('ALTER TABLE public.%I ADD CONSTRAINT %I CHECK
        (image_data IS NOT NULL OR (storage_path IS NOT NULL AND image_sha256 ~ ''^[a-f0-9]{64}$'' AND image_bytes >= 0
        AND image_sha256 IS NOT NULL AND image_bytes IS NOT NULL)) NOT VALID', table_name, table_name || '_image_source');
    END IF;
    EXECUTE format('ALTER TABLE public.%I VALIDATE CONSTRAINT %I', table_name, table_name || '_image_source');
  END LOOP;
END $$;
COMMIT;
