#!/usr/bin/env bash
#
# SigLIP 2 cutover. Run AFTER a restart has already picked up the thumbnail
# cache fix, so that if search misbehaves you know which change caused it.
#
# Both env vars must flip together. Flipping only CHITRA_EMBED_MODEL leaves the
# sidecar returning 768-d query vectors while the table is still ranked as
# 512-d, and `mat @ q_vec` raises for every user — a total search outage, not a
# degraded one.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

MODEL="google/siglip2-base-patch16-224"
CLIP="openai/clip-vit-base-patch32"
red() { printf '\033[31m%s\033[0m\n' "$*"; }
grn() { printf '\033[32m%s\033[0m\n' "$*"; }
say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/5  Coverage gate"
gap=$(sqlite3 photo.db "SELECT COUNT(*) FROM embeddings c WHERE c.model='$CLIP'
      AND NOT EXISTS (SELECT 1 FROM embeddings s WHERE s.photo_id=c.photo_id AND s.model='$MODEL');")
sqlite3 photo.db "SELECT '     '||model||'  dim='||dim||'  '||COUNT(*)||' rows' FROM embeddings GROUP BY model, dim;"
echo "     photos with CLIP but no SigLIP: $gap"
if [ "$gap" -gt 5 ]; then
  red "     ABORTED — $gap photos would silently vanish from search."
  red "     Run the re-embed to completion first."
  exit 1
elif [ "$gap" -gt 0 ]; then
  echo "     (known: photo 2468's thumbnail sits on a bad sector and is already"
  echo "      unreadable under CLIP too — it is broken today either way)"
fi
grn "     Coverage acceptable."

say "2/5  Backup"
BK="photo.db.bak-cutover-$(date +%Y%m%d-%H%M%S)"
sqlite3 photo.db ".backup '$BK'" || { red "backup failed"; exit 1; }
echo "     $BK  ($(sqlite3 "$BK" 'PRAGMA integrity_check;' | head -1))"

say "3/5  Setting both variables together"
if grep -q '^CHITRA_ACTIVE_EMBED_MODEL=' .env.production 2>/dev/null; then
  sed -i "s|^CHITRA_EMBED_MODEL=.*|CHITRA_EMBED_MODEL=$MODEL|; s|^CHITRA_ACTIVE_EMBED_MODEL=.*|CHITRA_ACTIVE_EMBED_MODEL=$MODEL|" .env.production
  echo "     updated existing entries"
else
  printf '\n# SigLIP 2 cutover %s\nCHITRA_EMBED_MODEL=%s\nCHITRA_ACTIVE_EMBED_MODEL=%s\n' "$(date +%F)" "$MODEL" "$MODEL" >> .env.production
  echo "     appended"
fi
n=$(grep -c '^CHITRA_.*EMBED_MODEL=' .env.production)
[ "$n" = "2" ] || { red "     expected 2 entries, found $n — NOT restarting"; exit 1; }
grn "     both variables set"

say "4/5  Restart"
read -r -p "     Restart now? Search 503s for ~20s. [y/N] " ans
case "$ans" in
  [yY]*) ./scripts/safe_restart.sh ;;
  *) echo "     Skipped. Config is staged; run scripts/safe_restart.sh when ready."; exit 0 ;;
esac

say "5/5  Verify"
sleep 5
echo "     sidecar: $(curl -s --max-time 25 localhost:5101/health 2>/dev/null)"
echo "     api    : $(curl -s --max-time 15 localhost:5000/api/health 2>/dev/null | tr ',' '\n' | grep -E '"status"|embed_status' | tr '\n' ' ')"
if curl -s --max-time 15 localhost:5101/health 2>/dev/null | grep -q '"dim":768'; then
  grn "
     SigLIP is live (dim=768). Search is now ranked on the new model.
     Expect ~82ms searches (was ~27ms) and the sidecar growing toward ~3.2GB."
else
  red "
     Sidecar is NOT reporting dim=768. The cutover did not take."
  echo "     Roll back:  sed -i '/^CHITRA_EMBED_MODEL=/d; /^CHITRA_ACTIVE_EMBED_MODEL=/d' .env.production && ./scripts/safe_restart.sh"
fi
echo "
     Rollback at any time (no re-embed needed, CLIP rows are untouched):
       cd ~/services/chitra && sed -i '/^CHITRA_EMBED_MODEL=/d; /^CHITRA_ACTIVE_EMBED_MODEL=/d' .env.production && ./scripts/safe_restart.sh"
