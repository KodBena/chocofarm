// cpp/include/chocofarm/quantity.hpp
// Purpose: the ONE home (ADR-0012 P1) for the codebase-wide ZERO-COST PHANTOM-TYPE machinery —
//   `Quantity<Tag, Rep>`, a single-member strong typedef that mints a distinct, non-interconvertible
//   integer domain per `Tag`. This is the Band-1 (solver-agnostic / general) reusable substrate; the
//   DOMAIN instantiations (WorldCount/WorldRank, etc.) live in their own domain headers (world.hpp) and
//   include this one. The point (ADR-0000): an illegal integer-domain MIX — a count used as a rank, a raw
//   int passed where a typed quantity is owed, a width/sign mismatch, two unrelated domains added — is made
//   UNREPRESENTABLE at compile time, not guarded at runtime. ADR-0012 P8: the typed signature IS the SSOT.
//
//   ZERO-COST is the hard gate, not an aspiration. The struct is a single `Rep` member, trivially
//   copyable + standard-layout + the same sizeof as `Rep` (the static_asserts below pin all three), every
//   member is constexpr + [[nodiscard]] + noexcept, and there is NO virtual / no padding / no extra state.
//   Under -O3 it lowers to the bare `Rep`: proven STATICALLY by the elision asserts here and EMPIRICALLY by
//   the objdump A/B oracle (cpp/parity/quantity_elision_*.cpp + the A/B check), whose hot-function
//   instruction sequences over Quantity<Tag,uint32_t> and over raw uint32_t are byte-identical.
//
//   A pure leaf header (only <bit>/<compare>/<concepts>/<cstdint>/<type_traits>) so it stays cycle-free.
//
// Public Domain (The Unlicense).
#pragma once

#include <bit>
#include <compare>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace chocofarm {

// The reps a phantom quantity may wrap: an integral type, motivated per-domain at the `using` site (sign +
// width chosen there, never "int by default" — ADR-0000 rule 1). Constrains the template so a nonsensical
// Rep (a float, a pointer) is a hard compile error at the alias, not a silent miscompile.
template <class R>
concept QuantityRep = std::integral<R>;

// The zero-cost strong-type machinery. `Tag` makes `Quantity<A, R>` and `Quantity<B, R>` DISTINCT,
// non-interconvertible types — the phantom. Construction from the raw rep is EXPLICIT (a boundary ACL the
// reader must write out — ADR-0012 P2: a boundary translates, it does not coerce silently); `.value()` is
// the explicit unwrap a TRUE boundary uses (a std::span size, the wire/Python seam, a stdlib call wanting
// the primitive — ADR-0000 item 5: every crossing is named + visible). Comparison is the defaulted
// three-way over the single rep (total order + equality WITHIN a domain); cross-domain compares do not
// compile (distinct Tag => distinct type). No implicit conversion to/from `Rep` exists in either direction.
template <class Tag, QuantityRep Rep>
struct Quantity {
    Rep v = 0;

    using rep_type = Rep;
    using tag_type = Tag;

    constexpr Quantity() noexcept = default;

    // EXPLICIT raw -> domain: the ONLY way a raw integer becomes a typed quantity is to name the type.
    explicit constexpr Quantity(Rep raw) noexcept : v(raw) {}

    // EXPLICIT domain -> raw: the named unwrap a true boundary uses.
    [[nodiscard]] constexpr Rep value() const noexcept { return v; }

    // Same-domain total order + equality. Cross-domain (distinct Tag) does NOT compile.
    [[nodiscard]] friend constexpr bool operator==(Quantity, Quantity) noexcept = default;
    [[nodiscard]] friend constexpr auto operator<=>(Quantity, Quantity) noexcept = default;
};

// ---- OPT-IN, concept-gated same-domain arithmetic (ADR-0000: offer ops ONLY where they make sense) ----
//
// Arithmetic is NOT blanket-enabled: a generic `Quantity` has only construction/unwrap/compare. A domain
// turns ON the operation that makes SENSE for its tag by specializing one trait to true — so "a Count + a
// Count is a Count" is enabled for the count tag, while a tag where addition is meaningless (a pure opaque
// id) silently offers none. The operators are constrained on the trait, so an un-enabled `+` is a hard
// compile error, not a silent miscompile. Cross-tag mixing never compiles regardless (distinct type).

// Enable same-domain `+`/`+=` (a quantity plus a quantity of the SAME tag yields that tag — additive
// monoid). A domain specializes `quantity_additive<Tag>` to derive from std::true_type.
template <class Tag>
struct quantity_additive : std::false_type {};
template <class Tag>
inline constexpr bool quantity_additive_v = quantity_additive<Tag>::value;

// Enable affine index ops (`Q + Rep` / `Q - Rep` -> Q; `Q - Q` -> Rep): an INDEX/RANK domain, where a
// quantity plus a RAW offset stays the same domain and the gap between two is a raw count. A domain
// specializes `quantity_affine<Tag>` to true.
template <class Tag>
struct quantity_affine : std::false_type {};
template <class Tag>
inline constexpr bool quantity_affine_v = quantity_affine<Tag>::value;

template <class Tag, QuantityRep Rep>
    requires quantity_additive_v<Tag>
[[nodiscard]] constexpr Quantity<Tag, Rep> operator+(Quantity<Tag, Rep> a, Quantity<Tag, Rep> b) noexcept {
    return Quantity<Tag, Rep>{static_cast<Rep>(a.value() + b.value())};
}
template <class Tag, QuantityRep Rep>
    requires quantity_additive_v<Tag>
constexpr Quantity<Tag, Rep>& operator+=(Quantity<Tag, Rep>& a, Quantity<Tag, Rep> b) noexcept {
    a = a + b;
    return a;
}

// Affine: rank + raw offset -> rank; rank - raw offset -> rank; rank - rank -> raw gap. The raw operand is
// the SAME Rep (no foreign-width mixing). These are the named, visible index crossings (ADR-0000 item 5).
template <class Tag, QuantityRep Rep>
    requires quantity_affine_v<Tag>
[[nodiscard]] constexpr Quantity<Tag, Rep> operator+(Quantity<Tag, Rep> a, Rep off) noexcept {
    return Quantity<Tag, Rep>{static_cast<Rep>(a.value() + off)};
}
template <class Tag, QuantityRep Rep>
    requires quantity_affine_v<Tag>
[[nodiscard]] constexpr Quantity<Tag, Rep> operator-(Quantity<Tag, Rep> a, Rep off) noexcept {
    return Quantity<Tag, Rep>{static_cast<Rep>(a.value() - off)};
}
template <class Tag, QuantityRep Rep>
    requires quantity_affine_v<Tag>
[[nodiscard]] constexpr Rep operator-(Quantity<Tag, Rep> a, Quantity<Tag, Rep> b) noexcept {
    return static_cast<Rep>(a.value() - b.value());
}

// ---- STATIC proof of elision (ADR-0009: a perf/equivalence claim carries its substantiation in-line) ----
// These pin the three properties the EMPIRICAL objdump A/B then confirms lower to the bare Rep. A probe
// instantiation (uint32_t is the live world-domain rep; the property is rep-agnostic by construction).
namespace detail {
struct ElisionProbeTag {};
using ElisionProbe = Quantity<ElisionProbeTag, std::uint32_t>;
static_assert(sizeof(ElisionProbe) == sizeof(std::uint32_t),
              "Quantity must be the SAME SIZE as its Rep (a single member, no padding) — else not zero-cost.");
static_assert(alignof(ElisionProbe) == alignof(std::uint32_t),
              "Quantity must have the SAME ALIGNMENT as its Rep — else it perturbs layout at the seam.");
static_assert(std::is_trivially_copyable_v<ElisionProbe>,
              "Quantity must be trivially copyable — the precondition for register-passing / memcpy elision.");
static_assert(std::is_standard_layout_v<ElisionProbe>,
              "Quantity must be standard-layout — so it has the same ABI/representation as the bare Rep.");
static_assert(std::is_trivially_copy_constructible_v<ElisionProbe> &&
                  std::is_trivially_move_constructible_v<ElisionProbe> &&
                  std::is_trivially_destructible_v<ElisionProbe>,
              "Quantity copy/move/dtor must be trivial — register-pass + no cleanup, the zero-cost core.");
}  // namespace detail

}  // namespace chocofarm
