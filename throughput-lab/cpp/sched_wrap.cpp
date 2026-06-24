// throughput-lab/cpp/sched_wrap.cpp — a tiny audited setuid-free privilege CONFINEMENT helper: set a
//   scheduling policy / attribute on ITSELF, then exec the real command. The maintainer grants ONE small
//   binary the capability — `sudo setcap cap_sys_nice+ep sched_wrap` — so the cap-requiring scheduling
//   experiments (SCHED_FIFO/RR, SCHED_DEADLINE, negative nice, the EEVDF latency slice/latency-nice) run
//   with the privilege confined to this audited helper, and the actual workload runs AS THE USER (not
//   root). This is the least-privilege alternative to `sudo <workload>` (which would run the workload as
//   root) or `setcap` on the python/producer binaries (too broad).
//
//   FAIL LOUD (ADR-0002 / ADR-0013 experiment integrity): if the requested scheduling attribute cannot be
//   set, this EXITS NON-ZERO and does NOT exec — an experiment must never silently run at the wrong
//   priority and be mis-recorded. The only sanctioned outcome is "policy set, then exec" or "loud failure".
//
//   Usage:  sched_wrap [attrs] -- <command> [args...]
//     --policy other|batch|idle|fifo|rr|deadline   (default: leave policy, just apply --nice/--slice)
//     --nice N            nice value for other/batch (negative needs CAP_SYS_NICE or an RLIMIT_NICE bump)
//     --prio N            static RT priority 1..99 for fifo/rr (needs CAP_SYS_NICE)
//     --runtime NS --deadline NS --period NS   sched_deadline params, nanoseconds (needs CAP_SYS_NICE)
//     --slice NS          EEVDF custom time-slice for a fair task (the mainline 6.12+ latency lever):
//                         a SMALLER slice => lower wakeup/scheduling latency, same CPU share
//     --latency-nice N    set sched_latency_nice via SCHED_FLAG_LATENCY_NICE (older/alt latency lever;
//                         may be rejected if this kernel lacks the field — reported loudly, not masked)
//
//   Public Domain (The Unlicense).
#define _GNU_SOURCE
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

// The kernel SchedAttr ABI (sched_setattr(2)). Declared locally because glibc ships no wrapper. The
// trailing latency-nice field is appended by `size`; older kernels that predate it ignore the tail when
// the corresponding flag is unset, and reject it (EINVAL/E2BIG) when set — which is the loud signal we want.
struct SchedAttr {
    uint32_t size;
    uint32_t sched_policy;
    uint64_t sched_flags;
    int32_t  sched_nice;
    uint32_t sched_priority;
    uint64_t sched_runtime;
    uint64_t sched_deadline;
    uint64_t sched_period;
    uint32_t sched_util_min;
    uint32_t sched_util_max;
    int32_t  sched_latency_nice;   // appended; only meaningful with SCHED_FLAG_LATENCY_NICE
};

#ifndef SCHED_FLAG_LATENCY_NICE
#define SCHED_FLAG_LATENCY_NICE 0x80
#endif

int sched_setattr_(pid_t pid, SchedAttr* a, unsigned int flags) {
    return static_cast<int>(syscall(SYS_sched_setattr, pid, a, flags));
}

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "sched_wrap: %s: %s\n", what, std::strerror(errno));
    std::exit(111);   // distinct from the command's own codes; never exec on failure
}

bool streq(const char* a, const char* b) { return std::strcmp(a, b) == 0; }

int policy_of(const char* name) {
    if (streq(name, "other"))    return SCHED_OTHER;
    if (streq(name, "batch"))    return SCHED_BATCH;
    if (streq(name, "idle"))     return SCHED_IDLE;
    if (streq(name, "fifo"))     return SCHED_FIFO;
    if (streq(name, "rr"))       return SCHED_RR;
    if (streq(name, "deadline")) return SCHED_DEADLINE;
    return -1;
}

}  // namespace

int main(int argc, char** argv) {
    SchedAttr a;
    std::memset(&a, 0, sizeof a);
    a.size = sizeof a;
    a.sched_policy = SCHED_OTHER;   // default; --policy overrides
    bool have_policy = false, have_slice = false, have_latnice = false;
    long latnice = 0;

    int i = 1;
    auto need = [&](const char* flag) -> const char* {
        if (i + 1 >= argc) { std::fprintf(stderr, "sched_wrap: %s needs a value\n", flag); std::exit(2); }
        return argv[++i];
    };
    for (; i < argc; ++i) {
        const char* f = argv[i];
        if (streq(f, "--")) { ++i; break; }
        else if (streq(f, "--policy")) {
            int p = policy_of(need("--policy"));
            if (p < 0) { std::fprintf(stderr, "sched_wrap: unknown --policy %s\n", argv[i]); return 2; }
            a.sched_policy = static_cast<uint32_t>(p); have_policy = true;
        } else if (streq(f, "--nice"))     { a.sched_nice = static_cast<int32_t>(std::atol(need("--nice"))); }
        else if (streq(f, "--prio"))       { a.sched_priority = static_cast<uint32_t>(std::atol(need("--prio"))); }
        else if (streq(f, "--runtime"))    { a.sched_runtime = std::strtoull(need("--runtime"), nullptr, 10); }
        else if (streq(f, "--deadline"))   { a.sched_deadline = std::strtoull(need("--deadline"), nullptr, 10); }
        else if (streq(f, "--period"))     { a.sched_period = std::strtoull(need("--period"), nullptr, 10); }
        else if (streq(f, "--slice"))      { a.sched_runtime = std::strtoull(need("--slice"), nullptr, 10); have_slice = true; }
        else if (streq(f, "--latency-nice")) { latnice = std::atol(need("--latency-nice")); have_latnice = true; }
        else { std::fprintf(stderr, "sched_wrap: unknown flag %s\n", f); return 2; }
    }
    if (i >= argc) { std::fprintf(stderr, "sched_wrap: missing -- <command>\n"); return 2; }

    if (have_latnice) { a.sched_flags |= SCHED_FLAG_LATENCY_NICE; a.sched_latency_nice = static_cast<int32_t>(latnice); }
    // The EEVDF custom slice (--slice) rides sched_runtime on a FAIR policy; harmless on other/batch.
    (void)have_policy; (void)have_slice;

    if (sched_setattr_(0, &a, 0) != 0) fail("sched_setattr");   // loud; never exec at the wrong priority

    execvp(argv[i], &argv[i]);
    fail("execvp");   // only reached if exec failed
}
