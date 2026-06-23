// throughput-lab/cpp/main.cpp
// Purpose: the producer entry point — the thin IMPERATIVE SHELL (ADR-0012 P9 functional-core /
//   imperative-shell). It parses CLI flags into a tlab::ProducerConfig (an ACL: parse-validate-translate
//   the argv view ONCE at this boundary, never coerce — a bad flag is a loud stderr error + non-zero exit,
//   ADR-0002), calls tlab::run_producer (all the load-generation + transport work), and prints the
//   per-thread ProducerStats in a human- AND harness-readable form. NO load-generation or transport logic
//   lives here — those are producer.cpp / boundary_*.cpp.
// Public Domain (The Unlicense).
//
//   CLI surface:
//     --topology   <per-thread|coalescing>   (BoundaryTopology; default per-thread)
//     --mode       <decoupled|coupled>        (ProducerMode; default decoupled)
//     --threads    <N>                        (producer threads; default 1)
//     --rate       <hz>                       (per-thread target emission rate R; default 1000)
//     --rows       <B>                        (rows per batch; default 1 = single-leaf)
//     --in-dim     <D>                        (feature width; default 241 = Stage-A)
//     --seconds    <S>                        (measured run duration; default 5)
//     --calib-seconds <S>                     (STEP 1 calibration window; default 0.2)
//     --endpoint   <ipc://...|tcp://...>      (server ZMQ endpoint; default ipc:///tmp/tlab-infer.sock)
//     --recv-timeout-ms <ms>                  (bounds Boundary recv/poll; default 5000)
//     --help                                  (print usage and exit 0)
//
//   The summary is printed as aligned human lines PLUS one machine-readable "RESULT key=value ..." line
//   per thread and one "AGGREGATE key=value ..." line, so harness/run_lab.py can parse the throughput
//   without screen-scraping the prose (ADR-0009: the measured number is surfaced unambiguously).

#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <expected>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "producer.hpp"

namespace {

// ---- a tiny typed-optional CLI ACL (no nullable-pointer/sentinel — P9) --------------------------
// Parse argv into key->value strings (every flag here takes one value, except --help). An unknown flag,
// a flag missing its value, or a malformed value is a loud error (returned as a message), never a silent
// default that misreports what ran (ADR-0002).

struct ParseError {
    std::string message;
};

[[nodiscard]] std::optional<long> parse_long(std::string_view s) {
    long v = 0;
    const char* begin = s.data();
    const char* end = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(begin, end, v);
    if (ec != std::errc{} || ptr != end) return std::nullopt;
    return v;
}

[[nodiscard]] std::optional<double> parse_double(std::string_view s) {
    // std::from_chars for double is available in the libstdc++ this toolchain ships; use it for a clean
    // locale-independent parse (no strtod global-locale surprise).
    double v = 0.0;
    const char* begin = s.data();
    const char* end = s.data() + s.size();
    auto [ptr, ec] = std::from_chars(begin, end, v);
    if (ec != std::errc{} || ptr != end) return std::nullopt;
    return v;
}

[[nodiscard]] std::expected<tlab::ProducerConfig, ParseError> parse_args(int argc, char** argv) {
    tlab::ProducerConfig cfg;   // defaults from producer.hpp

    auto need_value = [&](int& i) -> std::expected<std::string_view, ParseError> {
        if (i + 1 >= argc)
            return std::unexpected(ParseError{std::string("flag ") + argv[i] + " needs a value"});
        return std::string_view(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            return std::unexpected(ParseError{"--help"});   // handled specially by the caller (exit 0)
        } else if (flag == "--topology") {
            auto v = need_value(i);
            if (!v) return std::unexpected(v.error());
            if (*v == "per-thread")      cfg.topology = tlab::BoundaryTopology::PerThread;
            else if (*v == "coalescing") cfg.topology = tlab::BoundaryTopology::Coalescing;
            else return std::unexpected(ParseError{std::string("--topology must be per-thread|coalescing, got ") +
                                                   std::string(*v)});
        } else if (flag == "--mode") {
            auto v = need_value(i);
            if (!v) return std::unexpected(v.error());
            if (*v == "decoupled")    cfg.mode = tlab::ProducerMode::Decoupled;
            else if (*v == "coupled") cfg.mode = tlab::ProducerMode::Coupled;
            else return std::unexpected(ParseError{std::string("--mode must be decoupled|coupled, got ") +
                                                   std::string(*v)});
        } else if (flag == "--threads") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto n = parse_long(*v);
            if (!n || *n < 1) return std::unexpected(ParseError{"--threads must be an integer >= 1"});
            cfg.n_threads = static_cast<int>(*n);
        } else if (flag == "--rate") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto r = parse_double(*v);
            if (!r || *r <= 0.0) return std::unexpected(ParseError{"--rate must be a positive number (hz)"});
            cfg.target_rate_hz = *r;
        } else if (flag == "--rows") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto b = parse_long(*v);
            if (!b || *b < 1) return std::unexpected(ParseError{"--rows must be an integer >= 1"});
            cfg.rows_per_batch = static_cast<tlab::wire::count_t>(*b);
        } else if (flag == "--in-dim") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto d = parse_long(*v);
            if (!d || *d < 1) return std::unexpected(ParseError{"--in-dim must be an integer >= 1"});
            cfg.in_dim = static_cast<tlab::wire::count_t>(*d);
        } else if (flag == "--seconds") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto s = parse_double(*v);
            if (!s || *s <= 0.0) return std::unexpected(ParseError{"--seconds must be a positive number"});
            cfg.run_seconds = *s;
        } else if (flag == "--calib-seconds") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto s = parse_double(*v);
            if (!s || *s <= 0.0) return std::unexpected(ParseError{"--calib-seconds must be a positive number"});
            cfg.calib_window_seconds = *s;
        } else if (flag == "--endpoint") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            cfg.endpoint = std::string(*v);
        } else if (flag == "--recv-timeout-ms") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto ms = parse_long(*v);
            if (!ms) return std::unexpected(ParseError{"--recv-timeout-ms must be an integer"});
            cfg.recv_timeout_ms = static_cast<int>(*ms);
        } else if (flag == "--send-queue-mb") {
            auto v = need_value(i);  if (!v) return std::unexpected(v.error());
            auto mb = parse_long(*v);
            if (!mb || *mb < 16 || *mb > 1024)
                return std::unexpected(ParseError{"--send-queue-mb must be an integer in [16, 1024] (<=1G)"});
            cfg.send_queue_bytes = static_cast<std::size_t>(*mb) << 20;
        } else {
            return std::unexpected(ParseError{std::string("unknown flag: ") + std::string(flag)});
        }
    }
    return cfg;
}

void print_usage(std::ostream& os) {
    os << "tlab-producer — calibrated synthetic-load producer for throughput-lab\n"
       << "usage: tlab-producer [flags]\n"
       << "  --topology   <per-thread|coalescing>   (default per-thread)\n"
       << "  --mode       <decoupled|coupled>       (default decoupled)\n"
       << "  --threads    <N>                       (default 1)\n"
       << "  --rate       <hz>                      (per-thread target emission rate; default 1000)\n"
       << "  --rows       <B>                       (rows per batch; default 1)\n"
       << "  --in-dim     <D>                       (feature width; default 241)\n"
       << "  --seconds    <S>                       (measured run duration; default 5)\n"
       << "  --calib-seconds <S>                    (STEP 1 calibration window; default 0.2)\n"
       << "  --endpoint   <ipc://...|tcp://...>     (default ipc:///tmp/tlab-infer.sock)\n"
       << "  --recv-timeout-ms <ms>                 (default 5000)\n"
       << "  --send-queue-mb <MB>                   (outstanding-send byte budget cap; default 256, max 1024)\n"
       << "  --help                                 (this help)\n";
}

const char* topology_name(tlab::BoundaryTopology t) {
    return t == tlab::BoundaryTopology::PerThread ? "per-thread" : "coalescing";
}
const char* mode_name(tlab::ProducerMode m) {
    return m == tlab::ProducerMode::Decoupled ? "decoupled" : "coupled";
}

}  // namespace

int main(int argc, char** argv) {
    auto parsed = parse_args(argc, argv);
    if (!parsed) {
        if (parsed.error().message == "--help") {
            print_usage(std::cout);
            return EXIT_SUCCESS;
        }
        std::cerr << "tlab-producer: " << parsed.error().message << "\n\n";
        print_usage(std::cerr);
        return EXIT_FAILURE;
    }
    const tlab::ProducerConfig cfg = *parsed;

    // Echo the run configuration so the report is self-describing (what was REQUESTED, next to ACHIEVED).
    std::cout << "tlab-producer: topology=" << topology_name(cfg.topology)
              << " mode=" << mode_name(cfg.mode) << " threads=" << cfg.n_threads
              << " rate=" << cfg.target_rate_hz << "hz/thread rows=" << cfg.rows_per_batch
              << " in_dim=" << cfg.in_dim << " seconds=" << cfg.run_seconds
              << " endpoint=" << cfg.endpoint << "\n";

    auto result = tlab::run_producer(cfg);
    if (!result) {
        std::cerr << "tlab-producer: run failed: " << result.error().message
                  << (result.error().is_timeout ? " (timeout — slow/absent server?)" : "") << "\n";
        return EXIT_FAILURE;
    }
    const std::vector<tlab::ProducerStats>& stats = *result;

    // ---- per-thread report -----------------------------------------------------------------------
    double agg_requested = 0.0, agg_achieved = 0.0;
    std::uint64_t agg_sent = 0, agg_recv = 0;
    bool any_overhead_bound = false;

    std::cout << "\n--- per-thread results ---\n";
    for (std::size_t t = 0; t < stats.size(); ++t) {
        const auto& s = stats[t];
        agg_requested += s.requested_rate_hz;
        agg_achieved += s.achieved_rate_hz;
        agg_sent += s.batches_sent;
        agg_recv += s.replies_recv;
        any_overhead_bound = any_overhead_bound || s.overhead_bound;

        std::cout << "thread " << t << ":  calib=" << s.calib.ops_per_sec << " ops/s"
                  << "  requested=" << s.requested_rate_hz << "hz"
                  << "  achieved=" << s.achieved_rate_hz << "hz"
                  << "  sent=" << s.batches_sent << "  recv=" << s.replies_recv
                  << "  lat(us) mean=" << s.mean_reply_latency_us
                  << " p50=" << s.p50_reply_latency_us << " p99=" << s.p99_reply_latency_us
                  << (s.overhead_bound ? "  [OVERHEAD-BOUND]" : "") << "\n";

        // Machine-readable line (one per thread) for the harness.
        std::cout << "RESULT thread=" << t
                  << " calib_ops_per_sec=" << s.calib.ops_per_sec
                  << " requested_hz=" << s.requested_rate_hz
                  << " achieved_hz=" << s.achieved_rate_hz
                  << " batches_sent=" << s.batches_sent
                  << " replies_recv=" << s.replies_recv
                  << " lat_mean_us=" << s.mean_reply_latency_us
                  << " lat_p50_us=" << s.p50_reply_latency_us
                  << " lat_p99_us=" << s.p99_reply_latency_us
                  << " overhead_bound=" << (s.overhead_bound ? 1 : 0) << "\n";
    }

    // ---- aggregate -------------------------------------------------------------------------------
    std::cout << "\n--- aggregate (" << stats.size() << " threads) ---\n"
              << "requested total: " << agg_requested << " hz\n"
              << "achieved total:  " << agg_achieved << " hz\n"
              << "batches sent:    " << agg_sent << "\n"
              << "replies recv:    " << agg_recv << "\n"
              << (any_overhead_bound ? "NOTE: at least one thread was OVERHEAD-BOUND "
                                       "(requested rate exceeded its emit ceiling)\n"
                                     : "");
    std::cout << "AGGREGATE threads=" << stats.size()
              << " topology=" << topology_name(cfg.topology)
              << " mode=" << mode_name(cfg.mode)
              << " requested_total_hz=" << agg_requested
              << " achieved_total_hz=" << agg_achieved
              << " batches_sent=" << agg_sent
              << " replies_recv=" << agg_recv
              << " any_overhead_bound=" << (any_overhead_bound ? 1 : 0) << "\n";

    return EXIT_SUCCESS;
}
