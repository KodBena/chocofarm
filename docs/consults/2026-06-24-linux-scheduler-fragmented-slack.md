# Consult — Linux scheduler tuning for fragmented-slack reclamation on a shared core

**Date:** 2026-06-24
**Commissioned under:** ADR-0014 (outside opinion on a topic orthogonal to the main work — kernel scheduler internals — kept out of the main context).
**Question:** the real-generator investigation found the workload generator-bound, with the inference server ~58% busy on its core and the other ~42% idle but *fragmented*. A 4th generator reniced to nice 19 did not cleanly reclaim that slack (≲5%, ~noise). Beyond CPU pinning and nice values, what *other* Linux scheduler configuration could let a low-priority CPU-bound thread efficiently use the fragmented idle time on a core shared with a bursty, mildly-latency-sensitive thread? Non-portability is acceptable (research box).

The consultant's brief is recorded verbatim below (ADR-0005 Rule 9 — commissioned-review artifacts recorded verbatim, in-tree). All host inspection was read-only; nothing on the host was changed.

---

# Scheduler-tuning brief — fragmented-slack reclamation on a shared core

**Host (verified read-only):** kernel **6.19.12-1-default**, 4 vCPU. **EEVDF** scheduler (any 6.6+ is EEVDF; nice on this kernel is EEVDF, not classic CFS). Build: **`PREEMPT_DYNAMIC`** with **`PREEMPT_VOLUNTARY=y`** baseline, **`CONFIG_HZ=1000`** (1 ms tick), `NO_HZ_FULL=y` capable, `HIGH_RES_TIMERS=y`. **cgroup v2** with the **`cpu` controller present** (`cpuset cpu io memory ...`). **`CONFIG_SCHED_CLASS_EXT=y`** — sched_ext is compiled in and the sysfs interface is live (`state: disabled`, `switch_all` present). `RT_GROUP_SCHED` is **off** (RT/DEADLINE bandwidth is system-wide, not per-cgroup). `debugfs/sched/` is **root-only here** (unreadable as your user) — the `sched_features` toggles and the `preempt` mode file need root to read or write. Your shell's cgroup (`.../tmux-spawn-...scope`) does **not** have `cpu` in its `subtree_control` yet, so per-process `cpu.idle`/`cpu.weight` needs you to enable the controller in the parent slice first.

Diagnosis of why nice-19 gave ≲5%: under EEVDF, nice (and `cpu.weight`) is a **weight on the eligible-virtual-time race, not a runnability gate**. A nice-19 producer still becomes *eligible* and still has a positive lag, so it gets picked in the short gaps — but it also carries a normal **request size / time-slice**, so when the consumer's batch wakes it must wait out the producer's slice or pay a preempt+switch. Weight alone tunes *share*, not *interleave granularity* or *wakeup preemption*. That is the knob class you actually need.

## Option space (what each does for THIS fragmentation problem)

| Knob / policy | How to set | Effect on producer using slack | Effect on consumer burst latency |
|---|---|---|---|
| **`SCHED_IDLE`** (per-task) | `sched_setattr` policy=5, or cgroup `cpu.idle=1` | Producer runs *only* when nothing else is eligible → soaks idle gaps cleanly | Strong protect: consumer wake **always** preempts an IDLE task; near-zero share denial |
| **`SCHED_BATCH`** (per-task) | `sched_setattr` policy=3 | Treated as non-interactive → **no wakeup-preempt credit**, longer effective slices, fewer switches | Protects indirectly: producer won't preempt on its own wakeups, but no hard yield to consumer |
| **cgroup v2 `cpu.idle=1`** | write `1` to producer cgroup's `cpu.idle` | Same semantics as SCHED_IDLE but at cgroup granularity → soaks slack | Same strong protect as SCHED_IDLE |
| **EEVDF `latency-nice` / `sched_runtime` (slice)** on **consumer** | `sched_setattr` `SCHED_FLAG_LATENCY_NICE` (latency_nice −20) and/or small `sched_runtime` request | — | Shrinks consumer's request size → it preempts sooner & is picked promptly when its batch lands; *the* EEVDF lever for "mildly latency-sensitive" |
| **`debugfs/sched/` features** (`NEXT_BUDDY`, `GENTLE_FAIR_SLEEPERS`, `PREEMPT_SHORT`) + `base_slice_ns` | echo to `/sys/kernel/debug/sched/...` (**root**) | Lowering `base_slice_ns` → finer interleave fits short gaps | Tunes preempt aggressiveness; **root-only here, read-blocked as you** |
| **`PREEMPT_DYNAMIC` mode → `full`** | `echo full > /sys/kernel/debug/sched/preempt` or `preempt=full` cmdline (**root**) | — | Build defaults to **voluntary**; switching to **full** preemption lets the consumer's wakeup preempt the producer at any point, not just at voluntary points → tighter burst latency |
| **`SCHED_DEADLINE` on consumer** | `sched_setattr` runtime/deadline/period | — | **Reserves** the consumer's burst (admission-controlled CBS); deadline outranks all CFS/EEVDF → producer provably can't delay it. Bounds verified: period_min 100 µs, max 4.19 s, deadline class admits |
| **`SCHED_DEADLINE`/`SCHED_FIFO` for consumer + producer left CFS** | as above | Producer auto-fills whatever the reserved consumer leaves | Hardest protect; RT throttle (`rt_runtime 950000/1000000`) leaves 5% headroom so a runaway can't fully starve |
| **sched_ext (BPF scheduler)** | load a `.bpf.o`, `switch_all` (**root**) | A custom DSQ policy can encode "fill gaps, yield instantly on consumer wake" exactly | Fully programmable; **compiled in and available here** — biggest lever, biggest effort |

Knobs that **don't** apply: `cpu.max` (caps the producer, doesn't improve interleave); plain `cpu.weight`/nice (share, not granularity — already shown ≲5%); per-cgroup RT bandwidth (`RT_GROUP_SCHED` off).

## Ranked — most worth trying first

1. **`SCHED_IDLE` on the producer** (or cgroup `cpu.idle=1`) — **[established]**. This is the textbook fit: an always-runnable throughput task that must yield instantly to anything else. The producer becomes eligible only when the consumer isn't, and any consumer wakeup preempts it immediately — directly converting "fragmented idle" into producer work *without* touching consumer latency. No root needed for the per-task `sched_setattr` form (an unprivileged task may move *itself* to SCHED_IDLE). **Start here.**

2. **Add EEVDF `latency-nice = −20` (and/or a small `sched_runtime`) on the consumer** — **[established]** that it's the EEVDF latency lever; **[conjecture]** on the magnitude for your specific burst shape. Pairs with #1: SCHED_IDLE guarantees the producer yields, latency-nice makes the consumer's pickup *prompt* when its batch lands in a contended instant. Per-task, no root.

3. **`SCHED_DEADLINE` reservation for the consumer**, producer left as ordinary/idle CFS — **[established]** mechanism, **[conjecture]** on tuning. Hard guarantee: size runtime to one forward + poll, period to the batch cadence; the producer soaks 100% of the complement. More work (admission control, needs root/`CAP_SYS_NICE`, and a bursty/variable forward time makes runtime sizing fiddly) — use only if #1+#2 leave residual jitter.

4. **Switch `PREEMPT_DYNAMIC` to `full`** (root) — **[conjecture]** for your case. Cheap one-line system-wide change that tightens *every* preemption point; helps consumer pickup latency under the voluntary-baseline build. Confounds A/B (affects both threads), so test it isolated, after #1–#2.

**Out of scope but available if you go deep:** sched_ext is compiled in and live here — a ~100-line BPF scheduler could encode the exact "fill-the-gaps, instant-yield" policy and is the ceiling of what this box can do, at correspondingly higher effort.

**Caveats:** debugfs `sched_features`/`base_slice_ns`/`preempt` are **root-only and unreadable as your user** — confirm with `sudo` before relying on them. To use cgroup `cpu.idle`/`cpu.weight` you must first add `cpu` to `cgroup.subtree_control` up the slice chain (currently only `memory pids`); the per-task `sched_setattr` path sidesteps that. All findings above are read-only inspection — **nothing on the host was changed.**
