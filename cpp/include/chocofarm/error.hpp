// cpp/include/chocofarm/error.hpp
// Purpose: the small boundary Error type for the C++ runner's fallible shell (ADR-0012 P9 rule 5).
//   A recoverable boundary failure — an unreachable redis, a missing weight payload, a malformed
//   manifest or instance file — is reported as a [[nodiscard]] std::expected<T, Error> returned by
//   value, NEVER a thrown exception or a sentinel/nullable pointer. `Error` carries only the
//   human-readable diagnostic (the same message the as-merged code threw via std::runtime_error),
//   so the imperative shell can print it loudly (ADR-0002) at the boundary. A genuine INVARIANT
//   violation (an impossible state / programmer bug) stays an assert/abort, not an Error — Error is
//   reserved for the recoverable, expected boundary conditions an operator or upstream causes.
//
// Public Domain (The Unlicense).
#pragma once

#include <string>
#include <utility>

namespace chocofarm {

// A boundary failure's diagnostic. Value type (cheap to move, returned by value in std::expected).
struct Error {
    std::string message;
};

// Convenience: build an Error from a message (so call sites read std::unexpected(make_error(...))).
[[nodiscard]] inline Error make_error(std::string message) {
    return Error{std::move(message)};
}

}  // namespace chocofarm
