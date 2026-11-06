#!/bin/bash
set -euo pipefail
umask 077
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:$PATH"
config_dir="${JOURNAL_BACKUP_CONFIG_DIR:-$HOME/Library/Application Support/TradingDashboardBackup}"
source "$config_dir/settings.sh"
cd "$(dirname "$0")/.."
log="$JOURNAL_BACKUP_ROOT/last-run.log"
if /usr/bin/security find-generic-password -a "$JOURNAL_BACKUP_KEYCHAIN_ACCOUNT" \
    -s "$JOURNAL_BACKUP_KEYCHAIN_SERVICE" -w "$HOME/Library/Keychains/login.keychain-db" |
    node --env-file="$config_dir/r2.env" --import tsx scripts/rolling-journal-backup.ts > "$log" 2>&1; then
  date -u '+%Y-%m-%dT%H:%M:%SZ' > "$JOURNAL_BACKUP_ROOT/last-success"
else
  /usr/bin/logger -t TradingDashboardBackup 'Journal backup failed. Check the private last-run.log.'
  exit 1
fi
