// throughput-lab/scratch/fiber_switch_microbench.cpp
// Purpose: ISOLATE the per-leaf fiber-machinery overhead the current fiber-per-tree producer pays, so a
//   fiber-vs-explicit-state verdict rests on a MEASURED number (ADR-0009), not an assertion. This is a
//   SCRATCH probe (not a committed bench) for the work-stealing/explicit-state soundness investigation.
//
//   It measures the TWO costs an explicit-state (Option B) cursor would eliminate, in isolation from the
//   search compute (which is identical either way and NOT recoverable):
//     (1) PER-LEAF CONTEXT SWITCH: one leaf = one round-trip boost.context swap (search yields to driver,
//         driver resumes search). We time N such round-trips through a trivial fiber that only yields.
//     (2) PER-DECISION STACK ALLOC: TreeState::start() builds a fresh protected_fixedsize_stack(512 KiB)
//         every decision (mmap + guard page; munmap on destruction) and runs to the first yield. We time
//         N such alloc+enter+exit+free cycles.
//   A "leaf round-trip" here is the SAME two-way swap fiber_leaf.hpp's YieldingNetEvaluator::predict incurs
//   (ch_.caller = resume()) plus the driver's resume_with (fib = resume()). No search, no env, no wire.
//
//   Output: ns/leaf-switch and ns/decision-stack-alloc, so the per-leaf overhead can be put NEXT TO the
//   measured per-leaf wall time of the real producer (a leaf's intrinsic search+wire cost) for the ratio
//   that decides whether the overhead is worth a refactor.
// Public Domain (The Unlicense).
#include <boost/context/fiber.hpp>
#include <boost/context/protected_fixedsize_stack.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace ctx = boost::context;
using Clock = std::chrono::steady_clock;

// (1) PER-LEAF SWITCH. A fiber that loops yielding `leaves_per_decision` times then returns, mirroring a
// search that parks at L leaves per decision. Each yield is one round-trip swap (the leaf cost). We keep
// ONE long-lived fiber and pump it, so this isolates the SWITCH cost from the stack-alloc cost.
static double bench_leaf_switch(std::int64_t total_leaves) {
    volatile std::int64_t sink = 0;
    ctx::fiber caller_slot;  // the channel's `caller` analog
    // The fiber yields forever; the driver resumes it total_leaves times.
    ctx::fiber f{std::allocator_arg, ctx::protected_fixedsize_stack(512 * 1024),
                 [&](ctx::fiber&& c) {
                     caller_slot = std::move(c);
                     for (;;) {
                         sink += 1;  // a trivial "leaf body" (the search would compute here)
                         caller_slot = std::move(caller_slot).resume();  // yield to driver (== predict's swap)
                     }
                     return std::move(caller_slot);
                 }};
    f = std::move(f).resume();  // run to first yield
    const auto t0 = Clock::now();
    for (std::int64_t i = 0; i < total_leaves; ++i)
        f = std::move(f).resume();  // driver-side resume (== resume_with's swap)
    const auto t1 = Clock::now();
    (void)sink;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(total_leaves);
}

// (2) PER-DECISION STACK ALLOC. Mirror TreeState::start(): build a fresh protected_fixedsize_stack(512 KiB)
// fiber, enter it (it touches a little stack then yields once = "first parked leaf"), then let it be
// destroyed (munmap). One iteration == one decision's start() alloc+enter+park+free.
static double bench_decision_stack(std::int64_t total_decisions) {
    volatile std::int64_t sink = 0;
    const auto t0 = Clock::now();
    for (std::int64_t i = 0; i < total_decisions; ++i) {
        ctx::fiber caller_slot;
        ctx::fiber f{std::allocator_arg, ctx::protected_fixedsize_stack(512 * 1024),
                     [&](ctx::fiber&& c) {
                         // touch some stack like a real descent would (a few KiB), then park at first leaf
                         volatile char probe[4096];
                         for (int k = 0; k < 4096; k += 512) probe[k] = static_cast<char>(k);
                         sink += probe[0];
                         caller_slot = std::move(c);
                         caller_slot = std::move(caller_slot).resume();  // park at first leaf
                         return std::move(caller_slot);
                     }};
        f = std::move(f).resume();  // run to first park (== start()'s resume())
        // f and its stack are destroyed at scope exit -> munmap (the per-decision free TreeState::start incurs
        // when it reassigns `fib` next decision).
    }
    const auto t1 = Clock::now();
    (void)sink;
    return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(total_decisions);
}

int main(int argc, char** argv) {
    std::int64_t leaves = (argc > 1) ? std::atoll(argv[1]) : 20'000'000;
    std::int64_t decisions = (argc > 2) ? std::atoll(argv[2]) : 2'000'000;

    // warm up both paths (page-fault the first stacks, prime the icache)
    (void)bench_leaf_switch(100'000);
    (void)bench_decision_stack(10'000);

    // 3 interleaved replicates, report median (robust-benchmark-statistics: timings are right-skewed)
    double sw[3], st[3];
    for (int r = 0; r < 3; ++r) {
        sw[r] = bench_leaf_switch(leaves);
        st[r] = bench_decision_stack(decisions);
    }
    auto med3 = [](double a, double b, double c) {
        double hi = a > b ? a : b, lo = a > b ? b : a;
        return c > hi ? hi : (c < lo ? lo : c);
    };
    double sw_med = med3(sw[0], sw[1], sw[2]);
    double st_med = med3(st[0], st[1], st[2]);
    std::printf("leaf_switch_ns_median=%.2f  (reps: %.2f %.2f %.2f, N=%lld/rep)\n",
                sw_med, sw[0], sw[1], sw[2], static_cast<long long>(leaves));
    std::printf("decision_stack_ns_median=%.2f  (reps: %.2f %.2f %.2f, N=%lld/rep)\n",
                st_med, st[0], st[1], st[2], static_cast<long long>(decisions));
    return 0;
}
