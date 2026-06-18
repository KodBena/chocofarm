// cpp/include/chocofarm/serve.hpp
// Purpose: the persistent --serve control loop — the C++ side of the ActorTransport (chocofarm/az/
//   actor_transport.py). It runs the C++ Gumbel actor as a long-lived process that reads control_spec
//   JSON-line messages from `in`, holds the env + net + policy LIVE across generations, and writes one
//   JSON-line reply per message to `out`. This is what makes ONLINE RECONFIGURATION possible: a HOT
//   config change (m/n_sims/c_*) rebuilds the policy without tearing down the env or the process; an
//   INSTANCE change (instance/faces) is a loud reject (a new experiment). The serialization contract is
//   control_spec (the SSOT both sides derive); this loop is the subprocess-pipe TRANSPORT impl behind the
//   Python ActorTransport seam — a ZeroMQ daemon would be a second impl with no change to this dispatch.
//
//   It reuses the proven episode loop (runner.hpp run_episodes), reloading the net only on a version
//   change (the version gate, independent of the config_epoch gate), so weights/results stay on the redis
//   bytes-store exactly as the one-shot runner has them (P7). It is ADDITIVE to the one-shot runner —
//   the parity fixtures and the --instance/--episodes CLI path are untouched.
//
// Public Domain (The Unlicense).
#pragma once

#include <iosfwd>
#include <string>

#include "chocofarm/transport.hpp"  // RedisClient — the weight-read / result-write seam

namespace chocofarm {

// The wire-path startup knobs (their ONE home is RuntimeConfig; they arrive as --serve STARTUP args, never
// ActorConfig — ADR-0012 P1 / Q6). When `infer_endpoint` is non-empty, generate dispatches to the WIRE
// path (run_episodes_wire_batched: every Gumbel-AZ leaf resolved REMOTELY on the JAX InferenceServer over a
// DEALER socket, NO local NetForward reload / policy build); when empty, generate runs the SERIAL
// run_episodes against a locally-reloaded NetForward (binary dispatch — Override O-2: no local-batched
// branch). `pool_threads`/`pool_batch` size the wire pool (fibers_per_thread = ceil(batch/threads), derived
// in RuntimeConfig); they default to RuntimeConfig::from_env's host-sized defaults when 0.
struct ServeOptions {
    std::string infer_endpoint;  // empty -> serial; ipc:// (or tcp://) -> wire path
    int pool_threads = 0;        // 0 -> RuntimeConfig default (env CHOCO_POOL_THREADS / 4)
    int pool_batch = 0;          // 0 -> RuntimeConfig default (env CHOCO_POOL_BATCH / 32)
};

// Run the control loop until a `shutdown` message or stdin EOF. `redis` is the weight/result transport;
// `run` is the redis weight-key namespace (the --serve --run <id> startup arg, session-fixed); `opts`
// carries the wire-path startup knobs (a non-empty `infer_endpoint` selects the wire generation path);
// `in`/`out` are the control channel (std::cin/std::cout in production, injectable streams in tests).
// Returns 0 on a clean exit. The loop itself never throws — every boundary failure (bad JSON, a missing
// config field, a missing weight payload, a failed generate) is a typed control_spec error REPLY, not an abort.
[[nodiscard]] int serve(RedisClient& redis, const std::string& run, const ServeOptions& opts,
                        std::istream& in, std::ostream& out);

}  // namespace chocofarm
