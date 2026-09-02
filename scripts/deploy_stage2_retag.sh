#!/usr/bin/env bash
# Stage 2 of docs/plans/deployment-2026-09-02.md — re-tag the library.
#
# Reads ZERO MinIO objects: it scores tags from embedding vectors already in
# SQLite. Backs the database up first, dry-runs, shows you the projection, and
# only writes after you confirm.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
say "1/4  Backing up photo.db"
BK="photo.db.bak-retag-$(date +%Y%m%d-%H%M%S)"
sqlite3 photo.db ".backup '$BK'" || { echo "backup FAILED — stopping"; exit 1; }
sqlite3 "$BK" "PRAGMA integrity_check;" | head -1
echo "     $BK"

say "2/4  Current tag state"
sqlite3 photo.db "SELECT COALESCE(source,'(none)'), COUNT(DISTINCT tag)||' labels', COUNT(*)||' rows' FROM tags GROUP BY 1;"
printf "     most common label: "
sqlite3 photo.db "SELECT tag||' on '||COUNT(*)||' photos' FROM tags GROUP BY tag ORDER BY COUNT(*) DESC LIMIT 1;"

say "3/4  Dry run (nothing written)"
.venv/bin/python scripts/retag.py --db photo.db 2>&1 | tail -25

say "4/4  Apply?"
read -r -p "     Write these tags to photo.db? [y/N] " ans
case "$ans" in
  [yY]*)
    .venv/bin/python scripts/retag.py --db photo.db --apply 2>&1 | tail -20
    say "Result"
    sqlite3 photo.db "SELECT COALESCE(source,'(none)'), COUNT(DISTINCT tag)||' labels', COUNT(*)||' rows' FROM tags GROUP BY 1;"
    printf "     most common label now: "
    sqlite3 photo.db "SELECT tag||' on '||COUNT(*)||' photos' FROM tags GROUP BY tag ORDER BY COUNT(*) DESC LIMIT 1;"
    printf '\n\033[32m     Done. Rollback:  sqlite3 photo.db "DELETE FROM tags WHERE source LIKE %s;"\033[0m\n' "'%vocab-v2'"
    echo "     Or restore:      cp $BK photo.db"
    ;;
  *) echo "     Skipped. Nothing written. Backup kept at $BK" ;;
esac
