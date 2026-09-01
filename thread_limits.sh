# shellcheck shell=bash
#
# Thread caps for the BLAS / OpenMP stacks underneath numpy, torch and ONNX
# Runtime. Sourced by start_production.sh and start_workers.sh.
#
# Why this exists: on this 6-core box torch and ONNX Runtime both default to 6
# threads and nothing in the codebase set a count. A CLIP image embed measured
# 153 ms at 6 threads and 68 ms at 3 — the default was 2.25x slower, because
# six threads on six cores contend with each other and with everything else on
# the box. The workers were 81% CPU-throttled at the same time.
#
# Why env vars: torch and the BLAS libraries read these once, at import. By the
# time Python could call torch.set_num_threads() the pools are already built,
# so the launcher is the only place that works.
#
# Usage — pass the default thread count for this process class:
#
#     . ./thread_limits.sh 2
#
# MUST be sourced *after* .env.production. `: "${VAR:=default}"` assigns only
# when VAR is unset or empty, so anything already exported — by
# .env.production, or by a systemd `Environment=` — wins over the default here.
# Sourcing this first would invert that and make the defaults unoverridable.

_chitra_thread_default="${1:?thread_limits.sh must be sourced with a thread count, e.g. '. ./thread_limits.sh 2'}"

: "${OMP_NUM_THREADS:=$_chitra_thread_default}"
: "${MKL_NUM_THREADS:=$_chitra_thread_default}"
: "${OPENBLAS_NUM_THREADS:=$_chitra_thread_default}"
: "${NUMEXPR_NUM_THREADS:=$_chitra_thread_default}"
# Apple Accelerate's variable is VECLIB_MAXIMUM_THREADS (there is no
# VECLIB_NUM_THREADS). Inert on this Linux box; set for correctness on macOS
# dev machines.
: "${VECLIB_MAXIMUM_THREADS:=$_chitra_thread_default}"

export OMP_NUM_THREADS
export MKL_NUM_THREADS
export OPENBLAS_NUM_THREADS
export NUMEXPR_NUM_THREADS
export VECLIB_MAXIMUM_THREADS

unset _chitra_thread_default
