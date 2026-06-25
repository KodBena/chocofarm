// cpp/parity/quantity_elision_ab.cpp
// Purpose: the SINGLE-SOURCE empirical zero-cost elision A/B oracle for Quantity<Tag,Rep> (quantity.hpp).
//   The hot kernel `hot` is written ONCE, parameterised on the accumulator type by the macro CHOCO_AB_TYPED:
//   compiled with -DCHOCO_AB_TYPED=1 the accumulator is the strong type Quantity<ProbeTag,uint32_t>
//   (construct / .value() / same-domain += / defaulted <=> ); with =0 it is the bare uint32_t the phantom
//   wraps. Building the SAME source twice (only the typedef differs) removes any TU-to-TU instruction-
//   scheduling nondeterminism, so the two `hot` disassemblies are BYTE-IDENTICAL iff the phantom is
//   genuinely zero-cost — the maintainer's hard gate (objdump diff empty). extern "C" => stable symbol.
//
//   The sweep iterates by POINTER (a single induction variable, no separate index), so the only register
//   the prologue zeroes is the accumulator — this removes the order-free pair of independent zeroing xors
//   (acc=0 and i=0) whose arbitrary scheduling is the sole non-cost source of jitter in an index-based
//   loop. With one zeroing the typed and raw schedules coincide exactly.
//
// Public Domain (The Unlicense).
#include <bit>
#include <cstdint>
#include <span>

#if CHOCO_AB_TYPED
#  include "chocofarm/quantity.hpp"
namespace {
struct ProbeTag {};
using Acc = chocofarm::Quantity<ProbeTag, std::uint32_t>;
}  // namespace
template <> struct chocofarm::quantity_additive<ProbeTag> : std::true_type {};
#  define WRAP(x) Acc{(x)}
#  define UNWRAP(a) (a).value()
#else
using Acc = std::uint32_t;
#  define WRAP(x) (x)
#  define UNWRAP(a) (a)
#endif

extern "C" std::uint32_t hot(const std::uint64_t* bits, const std::uint64_t* mask, std::size_t n,
                             std::uint32_t off, std::uint32_t thr) {
    std::span<const std::uint64_t> b(bits, n), m(mask, n);
    Acc acc = WRAP(0u);
    const std::uint64_t* mp = m.data();
    for (std::uint64_t w : b)
        acc += WRAP(static_cast<std::uint32_t>(std::popcount(w & *mp++)));
    acc += WRAP(off);
    return (acc > WRAP(thr)) ? UNWRAP(acc) : 0u;
}
