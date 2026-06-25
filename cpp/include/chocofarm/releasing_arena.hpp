// cpp/include/chocofarm/releasing_arena.hpp
// Purpose: ReleasingArena — a per-decision search-tree arena whose grown chunks are RETURNED TO THE OS
//   when the decision ends, so a parked fiber's resident tracks its CURRENT decision's tree size, not its
//   lifetime-maximum (the O(fibers) OOM dissolution; ADR-0000 make-the-illegal-state-unrepresentable).
//
//   THE DEFECT IT DISSOLVES (RCA tlab_finding #23/#26; massif-attributed, ADR-0009): the Gumbel node pool
//   was served from a per-policy std::pmr::monotonic_buffer_resource over an INLINE member buffer. A
//   monotonic resource frees nothing until release(), and its DEFAULT upstream (new_delete_resource ->
//   glibc) RETAINS the freed large blocks in the process heap rather than munmap'ing them (the R4 null
//   result: the glibc-trim env knobs cut <4% — the footprint is the resource's own retained high-water,
//   not glibc fragmentation). With the fiber driver holding EVERY one of K fibers' trees live, each fiber
//   settled at its deepest-EVER decision's high-water and held it for life: resident = threads*K*max_tree
//   -> ~2.4 GiB/producer at K=1024/n_sims=256, and four coincident producers exceed the 8 GiB box (OOM
//   SIGKILL). The high-water is UNBOUNDED in the fiber population even though throughput is server-bound.
//
//   THE STRUCTURAL CHANGE: keep the monotonic resource's hot path (a small INLINE floor sized for the
//   COMMON shallow decision -> zero syscalls on the overwhelming majority of decisions), but give it an
//   UPSTREAM that mmap()s each overflow block and munmap()s it on deallocate. release() (already called at
//   the START of every run_search, gumbel.cpp) then returns the prior decision's overflow chunks to the OS.
//   A parked fiber holds ONLY its current decision's live tree (the search re-reads the node pool across
//   resume_with, so the CURRENT tree must persist while parked -- and it does: release() runs only at the
//   NEXT decision's start, by which point the fiber's prior search has fully unwound). So per-fiber resident
//   becomes the CURRENT decision's tree, not the lifetime max; episodic fibers sit mostly at shallow belief
//   depths, so the coincident high-water collapses. The node graph + every value is BYTE-IDENTICAL: only
//   the allocator's upstream differs (ADR-0012 P6 — an allocator swap perturbs no value; the parity gates
//   + the Option-A proof re-validate this).
//
//   WHY mmap/munmap and not a free-list: munmap RETURNS pages to the OS deterministically (MADV-free is not
//   enough -- the pages must leave the process RSS so four producers' coincident peaks sum within budget).
//   The blocks are large (>= one decision's overflow, KiB..low-MiB), for which mmap is the right primitive
//   and the per-deep-decision syscall is negligible against that decision's compute. Shallow decisions that
//   fit the inline floor never reach here, so the hot path is unchanged.
//
// Public Domain (The Unlicense).
#pragma once

#include <sys/mman.h>
#include <unistd.h>  // sysconf / _SC_PAGESIZE

#include <cstddef>
#include <memory_resource>
#include <new>

namespace chocofarm {

// An mmap-backed std::pmr::memory_resource: each block is its own MAP_PRIVATE|MAP_ANONYMOUS mapping,
// returned to the OS by munmap on deallocate. Intended as the UPSTREAM of a monotonic_buffer_resource
// (the per-decision search arena): the monotonic resource batches the node pool's many small allocations
// into a few large upstream blocks, and this upstream makes the monotonic resource's release() actually
// shrink the process RSS (vs new_delete_resource, which hands large frees back to glibc, which retains
// them). Stateless beyond the page rounding; safe to share (each (ptr,bytes) pair round-trips its own
// mapping). The size is stashed in a header word ahead of the returned pointer so deallocate can munmap
// the exact mapping length even though pmr passes the monotonic resource's REQUESTED bytes (which the
// monotonic resource preserves on the round-trip, but the header keeps this self-contained + asserts).
class MmapUpstream final : public std::pmr::memory_resource {
  public:
    MmapUpstream() = default;

  private:
    // The page size, read once. A block's true mapping length is rounded up to a page (mmap maps whole
    // pages anyway); storing the rounded length in the header lets deallocate munmap exactly what mmap got.
    static std::size_t page() {
        static const std::size_t p = static_cast<std::size_t>(::sysconf(_SC_PAGESIZE));
        return p;
    }
    static std::size_t round_up(std::size_t n, std::size_t a) { return (n + a - 1) & ~(a - 1); }

    // One header page word ahead of the user pointer carries the total mapping length, so deallocate is
    // self-describing (munmaps the exact length) regardless of the requested bytes pmr replays. We over-
    // align the returned pointer to the requested alignment within the first page.
    struct Header {
        std::size_t map_len;  // total munmap length (header + payload, page-rounded)
        std::size_t pad;      // keep the header a 2-word (16B) unit so payload start is 16-aligned
    };

    void* do_allocate(std::size_t bytes, std::size_t alignment) override {
        const std::size_t pg = page();
        const std::size_t hdr = sizeof(Header);
        // payload must satisfy `alignment`; we place it at offset hdr from the mapping base. The base is
        // page-aligned (mmap guarantees), and hdr is 16; for the search arena alignment is <= alignof(max_
        // align_t) (16) so offset hdr already satisfies it. Guard louder-than-silent for any larger ask.
        if (alignment > hdr) {
            // Round the payload offset up to `alignment` and grow the map accordingly (rare; the node
            // pool's max alignment is 16). Keeps the contract total: any alignment is honored, not assumed.
            const std::size_t off = round_up(hdr, alignment);
            const std::size_t map_len = round_up(off + bytes, pg);
            void* base = ::mmap(nullptr, map_len, PROT_READ | PROT_WRITE,
                                MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (base == MAP_FAILED) throw std::bad_alloc();
            auto* h = reinterpret_cast<Header*>(static_cast<std::byte*>(base) + off - hdr);
            h->map_len = map_len;
            h->pad = off;  // payload offset from base, so deallocate recovers base
            return static_cast<std::byte*>(base) + off;
        }
        const std::size_t map_len = round_up(hdr + bytes, pg);
        void* base = ::mmap(nullptr, map_len, PROT_READ | PROT_WRITE,
                            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (base == MAP_FAILED) throw std::bad_alloc();
        auto* h = static_cast<Header*>(base);
        h->map_len = map_len;
        h->pad = hdr;  // payload offset == hdr
        return static_cast<std::byte*>(base) + hdr;
    }

    void do_deallocate(void* p, std::size_t /*bytes*/, std::size_t /*alignment*/) override {
        if (!p) return;
        // recover the header: it sits sizeof(Header) before the payload only in the common (alignment<=hdr)
        // case; for the over-aligned case `pad` carries the offset, so the header is always at p - hdr with
        // h->pad recording the true base offset. Read map_len + base, then munmap.
        auto* h = reinterpret_cast<Header*>(static_cast<std::byte*>(p) - sizeof(Header));
        const std::size_t off = h->pad;
        const std::size_t map_len = h->map_len;
        void* base = static_cast<std::byte*>(p) - off;
        ::munmap(base, map_len);
    }

    bool do_is_equal(const std::pmr::memory_resource& o) const noexcept override {
        return this == &o;  // each mapping is self-describing, but equality is identity (stateful-ish)
    }
};

}  // namespace chocofarm
