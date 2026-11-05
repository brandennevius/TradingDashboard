// Only routine save/nightly rows are eligible. Recovery, migration and unknown
// reasons are pinned. The newest pre-save snapshot is retained even if old.
export const boundedRevisionPruneSql = `
  delete from cam_journal_revisions
  where (reason = 'save' and id <> coalesce(
    (select max(id) from cam_journal_revisions where reason = 'save'), -1))
     or (reason = 'nightly' and created_at < now() - interval '30 days')
`;

export const boundedHistoryEnabled = () => process.env.CAM_JOURNAL_BOUNDED_HISTORY === "enabled";
