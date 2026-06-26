#!/bin/bash
# DMZ lint — flags BARE-int domain leaks (struct fields + function params typed int/size_t/uNN_t) that
# escape the phantom-type discipline (ADR-0000/ADR-0012). AST-anchored (clang-query over the compile DB),
# source-verified (only bare-int SPELLINGS, not the domain aliases World/count_t or Quantity<>), and
# allowlisted two ways: by DMZ file (the strong-type machinery's Rep, domain aliases, the wire codec, the
# bitmask storage) and by a per-line `// NOLINT(dmz...)` marker (greppable, self-justifying — the home for
# the measured-hot exceptions the loop-mod A/B established). Exit 1 on any un-allowlisted holdout.
# Companion: .clang-tidy carries modernize-loop-convert (the generator-fed loop antipattern) + the
# sign-compare guard. Usage: tools/lint/dmz-lint.sh   (run from the repo root; needs cpp/build compile DB)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
DB=cpp/build
[ -f "$DB/compile_commands.json" ] || { echo "no compile DB at $DB (configure -DCMAKE_EXPORT_COMPILE_COMMANDS=ON)"; exit 2; }
TUS=(cpp/src/env.cpp cpp/src/env_zdd.cpp cpp/src/features.cpp cpp/src/gumbel.cpp cpp/src/gumbel_cursor.cpp
     cpp/src/ismcts.cpp cpp/src/search_runtime.cpp cpp/src/instance.cpp)
DMZ_RE='(quantity|world|domains|proc_domains|wire_spec|collected_set)\.hpp'
BARE='\b(unsigned|signed|int|long|short|char|size_t|u?int(8|16|32|64)_t)\b'
tmp=$(mktemp)
for tu in "${TUS[@]}"; do
  [ -f "$tu" ] || continue
  clang-query -p "$DB" "$tu" -f tools/lint/dmz.clang-query 2>/dev/null \
    | grep -oE "$ROOT/(cpp|throughput-lab)/[^ :]+\.(hpp|cpp):[0-9]+:[0-9]+" | grep -vE "/build/"
done | sort -u > "$tmp"
violations=0
while IFS= read -r loc; do
  [ -z "$loc" ] && continue
  file="${loc%%:*}"; rest="${loc#*:}"; ln="${rest%%:*}"
  rel="${file#$ROOT/}"
  echo "$rel" | grep -qE "$DMZ_RE" && continue          # DMZ file: raw int legitimate
  src="$(sed -n "${ln}p" "$file")"
  echo "$src" | grep -q 'NOLINT(dmz' && continue         # measured/justified exception
  echo "$src" | grep -qE "$BARE" || continue             # only bare-int spellings (skip domain aliases/Quantity)
  printf "  %s:%s\t%s\n" "$rel" "$ln" "$(echo "$src" | sed 's/^ *//' | cut -c1-72)"
  violations=$((violations+1))
done < "$tmp"
rm -f "$tmp"
echo "DMZ lint: $violations un-allowlisted bare-int field/param holdout(s)."
[ "$violations" -eq 0 ]
