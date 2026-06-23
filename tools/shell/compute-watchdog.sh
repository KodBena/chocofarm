#!/usr/bin/env bash
#
# tools/shell/compute-watchdog.sh — run a compute-bound command and abort it if it stalls.
#
# Detects a wedged/deadlocked process group by sampling *cumulative CPU time*
# across the target and all its descendants. A healthy compute-bound job accrues
# CPU-seconds steadily; a stalled one accrues ~none. When aggregate CPU stays
# below the threshold for too long, the whole tree is terminated (TERM -> KILL).
#
# Portable: uses only `ps` (Linux & macOS). No /proc, no `top` column parsing.
#
# Usage:
#   ./cpu-watchdog.sh <command> [args...]
#
# Tunables (environment overrides):
#   CHECK_INTERVAL=5      # seconds between samples
#   MAX_IDLE_DURATION=20  # seconds below threshold before declaring a stall
#   CPU_THRESHOLD=5       # percent of ONE core, averaged over the interval
#   GRACE_PERIOD=5        # seconds between SIGTERM and SIGKILL
#   ON_STALL=terminate    # terminate | abort | alert
#                         #   terminate: TERM then KILL the tree, exit 124
#                         #   abort:     SIGABRT (core dump) then terminate, exit 134
#                         #   alert:     ring bell, detach child, exit 124
#
# Verified 2026-06-23: a healthy CPU-bound job survives (reaps its real exit code); a stall (sub-threshold
# aggregate CPU) is TERM→KILLed (exit 124); confirmed on the lab_harness+producer tree (no false "cold"
# through the ~30s warmup — the producer child carries the tree's CPU). Use for compute-bound benchmarks
# that can wedge (the leaf-eval lab; the zmq producer pool-warmup that EAGAIN-stalls at very high overcommit).
#
# Public Domain (The Unlicense).
set -euo pipefail

CHECK_INTERVAL="${CHECK_INTERVAL:-5}"
MAX_IDLE_DURATION="${MAX_IDLE_DURATION:-20}"
CPU_THRESHOLD="${CPU_THRESHOLD:-5}"
GRACE_PERIOD="${GRACE_PERIOD:-5}"
ON_STALL="${ON_STALL:-terminate}"

log() { printf '[watchdog] %s\n' "$*" >&2; }

if [ "$#" -eq 0 ]; then
    echo "Usage: $0 <command> [args...]" >&2
    exit 2
fi

# --- Enumerate the PIDs of a process and all of its descendants. ---
# One `ps` snapshot; closure over the ppid relation (handles out-of-order rows).
tree_pids() {
    local root=$1
    ps -A -o pid=,ppid= 2>/dev/null | awk -v root="$root" '
        { ppid[$1] = $2; order[NR] = $1 }
        END {
            intree[root] = 1
            changed = 1
            while (changed) {
                changed = 0
                for (i = 1; i <= NR; i++) {
                    p = order[i]
                    if (!(p in intree) && (ppid[p] in intree)) {
                        intree[p] = 1; changed = 1
                    }
                }
            }
            for (p in intree) if (p ~ /^[0-9]+$/) print p
        }'
}

# --- Sum cumulative CPU-seconds over the target and its descendants. ---
# Parses the TIME column ([DD-]HH:MM:SS[.ss]) robustly on Linux and macOS.
tree_cpu_seconds() {
    local root=$1
    ps -A -o pid=,ppid=,time= 2>/dev/null | awk -v root="$root" '
        function to_sec(t,    a, b, n, i, s, d) {
            d = 0
            if (index(t, "-")) { split(t, a, "-"); d = a[1] + 0; t = a[2] }
            n = split(t, b, ":")
            s = 0
            for (i = 1; i <= n; i++) s = s * 60 + (b[i] + 0)
            return d * 86400 + s
        }
        { ppid[$1] = $2; cpu[$1] = to_sec($3); order[NR] = $1 }
        END {
            intree[root] = 1
            changed = 1
            while (changed) {
                changed = 0
                for (i = 1; i <= NR; i++) {
                    p = order[i]
                    if (!(p in intree) && (ppid[p] in intree)) {
                        intree[p] = 1; changed = 1
                    }
                }
            }
            total = 0
            for (p in intree) total += cpu[p]
            printf "%.3f", total
        }'
}

# --- Terminate the whole tree gracefully: SIGTERM, grace window, then SIGKILL. ---
terminate_tree() {
    local root=$1 pids waited=0
    pids=$(tree_pids "$root") || true
    [ -z "$pids" ] && return 0

    log "sending SIGTERM to process tree of $root"
    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null || true

    while [ "$waited" -lt "$GRACE_PERIOD" ]; do
        kill -0 "$root" 2>/dev/null || break
        sleep 1
        waited=$(( waited + 1 ))
    done

    pids=$(tree_pids "$root") || true
    if [ -n "$pids" ]; then
        log "escalating to SIGKILL"
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null || true
    fi
}

# --- Ensure an interrupted watchdog never leaves the tree running. ---
TARGET_PID=""
cleanup() {
    trap - EXIT INT TERM
    if [ -n "$TARGET_PID" ] && kill -0 "$TARGET_PID" 2>/dev/null; then
        log "watchdog exiting; cleaning up tree of $TARGET_PID"
        terminate_tree "$TARGET_PID"
    fi
}
trap cleanup EXIT INT TERM

# --- Launch the target in the background. ---
"$@" &
TARGET_PID=$!
log "monitoring PID $TARGET_PID and descendants: $*"
log "interval=${CHECK_INTERVAL}s threshold=${CPU_THRESHOLD}%/core max-idle=${MAX_IDLE_DURATION}s on-stall=${ON_STALL}"

idle=0
prev=$(tree_cpu_seconds "$TARGET_PID") || prev=0

while kill -0 "$TARGET_PID" 2>/dev/null; do
    t0=$SECONDS
    sleep "$CHECK_INTERVAL"
    kill -0 "$TARGET_PID" 2>/dev/null || break          # finished cleanly mid-sleep

    elapsed=$(( SECONDS - t0 ))
    [ "$elapsed" -le 0 ] && elapsed=$CHECK_INTERVAL

    cur=$(tree_cpu_seconds "$TARGET_PID") || cur=$prev
    # CPU% of one core = 100 * delta_cpu_seconds / wall_seconds (clamped at 0).
    cpu_pct=$(awk -v a="$prev" -v b="$cur" -v t="$elapsed" \
        'BEGIN { d = b - a; if (d < 0) d = 0; printf "%d", (100.0 * d) / t }') || cpu_pct=0
    prev=$cur

    if [ "$cpu_pct" -lt "$CPU_THRESHOLD" ]; then
        idle=$(( idle + elapsed ))
        log "cold: ${cpu_pct}%/core  (idle ${idle}/${MAX_IDLE_DURATION}s)"
    else
        [ "$idle" -ne 0 ] && log "recovered: ${cpu_pct}%/core"
        idle=0
    fi

    if [ "$idle" -ge "$MAX_IDLE_DURATION" ]; then
        log "STALL DETECTED: aggregate CPU below ${CPU_THRESHOLD}%/core for ${idle}s"
        printf '\a' >&2                                   # audible bell (tmux/terminal)
        case "$ON_STALL" in
            alert)
                log "ON_STALL=alert; detaching and leaving the process running"
                TARGET_PID=""                             # release: cleanup will not kill it
                exit 124
                ;;
            abort)
                log "ON_STALL=abort; sending SIGABRT for a core dump"
                kill -ABRT "$TARGET_PID" 2>/dev/null || true
                sleep 2
                terminate_tree "$TARGET_PID"
                exit 134
                ;;
            terminate | *)
                terminate_tree "$TARGET_PID"
                exit 124
                ;;
        esac
    fi
done

# --- Reap the real exit status of a normally-terminated target. ---
if wait "$TARGET_PID"; then
    code=0
else
    code=$?
fi
TARGET_PID=""
log "process exited normally with code $code"
exit "$code"
