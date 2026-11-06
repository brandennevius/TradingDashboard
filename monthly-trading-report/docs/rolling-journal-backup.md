# Rolling journal backup

Run one nightly backup on the Mac under its signed-in user. The private archive
holds an encrypted database dump and manifest for each restore point. Screenshot
files are encrypted once per SHA-256 and reused across points. PostgreSQL exports
the dump and screenshot references from the same read-only snapshot.

The operator uses Node 24, GnuPG, and PostgreSQL client tools. Keep its launch
configuration and credentials in a private directory outside the checkout.
Provide the database session endpoint, private pgpass file and CA, expected
Supabase project and R2 bucket identity, and bucket credentials. Use
`scripts/run-journal-backup.sh` to pipe the archive passphrase from the Mac login
Keychain. The runner writes its last result and successful time inside the archive.
The Mac must be available and its Keychain accessible for the job to run.

Set `JOURNAL_BACKUP_RETENTION_DAYS=30`. Enable `JOURNAL_BACKUP_ENABLE_PRUNE=true`
only after an isolated restore validates a production restore point. Retention
removes expired restore points and encrypted screenshot files no retained point
references. It does not delete any live R2 objects or current journal records.
The optional `JOURNAL_BACKUP_ENABLE_LEGACY_PRUNE=true` expires only Cam's nightly
revision records under the same policy. Enable it after the old Vercel nightly
schedule is retired. Manual and recovery revisions are outside this policy.

To rehearse recovery, decrypt a point's dump and manifest into a private temporary
directory. Check the dump SHA-256 against the manifest and restore to a disposable
PostgreSQL database. Compare both screenshot tables' R2 references to the manifest.
Decrypt the content-addressed files and compare byte lengths and SHA-256 to the
manifest. Restore objects to their original keys only when coordinating an actual
recovery. Use an isolated database and bucket for a full recovery rehearsal.

The job flags 350 MB of PostgreSQL database size and 7 GB of screenshot bucket
storage. Inspect the recorded last success after the scheduled time. A local
schedule cannot run or report a failure while the Mac is powered off.
