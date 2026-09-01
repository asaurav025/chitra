#!/usr/bin/env bash
#
# Safe restart for the Chitra API + workers.
#
# The hazard this exists to prevent: stop_workers.sh double-TERMs, which
# cold-kills an in-flight transcode and strands the row in 'processing'
# forever with nothing to retry it. So we refuse to stop while the video
# queue is busy, rather than trusting the operator to remember.
#
# Order is workers-then-API on purpose: the CLIP embedding sidecar starts
# with the workers, and the API needs it for search. Search returns
# 503 search_unavailable for ~20s while CLIP loads. That is by design.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

red()  { printf '\033[31m%s\033[0m\n' "$*"; }
grn()  { printf '\033[32m%s\033[0m\n' "$*"; }
say()  { printf '\n\033[1m== %s\033[0m\n' "$*"; }

say "1/4  Checking nothing is mid-transcode"
vq=$(redis-cli llen rq:queue:video 2>/dev/null || echo "?")
dq=$(redis-cli llen rq:queue:default 2>/dev/null || echo "?")
ff=$(pgrep -x ffmpeg | wc -l)
echo "     video queue:   $vq"
echo "     default queue: $dq"
echo "     ffmpeg procs:  $ff"

if [ "$vq" != "0" ] || [ "$ff" != "0" ]; then
  red "     ABORTED — a transcode is in flight."
  red "     Stopping now would cold-kill it and strand the video permanently."
  red "     Wait for the video queue to drain, then re-run."
  exit 1
fi
grn "     Safe to proceed."

say "2/4  Stopping workers (systemd Restart=always respawns them)"
./stop_workers.sh
echo "     waiting for respawn..."
for i in $(seq 1 30); do
  sleep 2
  n=$(pgrep -fc 'worker.py' 2>/dev/null || echo 0)
  [ "$n" -ge 6 ] && break
done
echo "     worker processes: $(pgrep -fc 'worker.py' 2>/dev/null || echo 0) (expect 6)"

say "3/4  Restarting the API (sudo — will prompt for your password)"
sudo systemctl restart chitra-api || { red "     API restart failed"; exit 1; }
sleep 5

say "4/4  Verifying"
echo "     workers:  $(pgrep -fc 'worker.py' 2>/dev/null || echo 0)  (expect 6)"
echo "     sidecar:  $(pgrep -fc embed_service 2>/dev/null || echo 0)  (expect 1)"
echo "     services: $(systemctl is-active chitra-api chitra-workers | tr '\n' ' ')"
echo "     sidecar health: $(curl -s --max-time 20 localhost:5101/health 2>/dev/null || echo 'not up yet — CLIP still loading, recheck in 20s')"
echo "     api health:     $(curl -s --max-time 10 localhost:5000/api/health 2>/dev/null | head -c 200)"
mem=$(cat /sys/fs/cgroup/system.slice/chitra-api.service/memory.current 2>/dev/null || echo 0)
echo "     API memory: $((mem/1024/1024)) MB  (was ~1287 MB before the fix)"
grn "
Done. If sidecar health is empty, wait ~20s and run:  curl -s localhost:5101/health"
