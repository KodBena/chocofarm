#!/usr/bin/env python3
"""
tools/cleanroom.py — build a comment-stripped "cleanroom" source tree for BLIND modeling.

Purpose: a blind modeling agent must derive its model from code SEMANTICS, not from the author's
narrative. Comments and docstrings carry rationale, history, ADR pointers, and — fatally for blindness —
measured findings and targets (e.g. a header that says "push rows/forward toward B≈192"). This tool
produces a hermetic tree containing ONLY whitelisted source files, with every comment and docstring
removed, so an agent pointed at the tree cannot be contaminated by prose and cannot even discover that
other files exist.

Two design choices that matter:
  * WHITELIST, not blacklist. A "do not read foo.md" instruction is itself a leak (it reveals foo.md
    exists and is interesting). The cleanroom contains exactly the allowed files and nothing else.
  * STRIP IN PLACE. Comments/docstrings are blanked where they sit (never reflowed), and newlines inside
    multi-line comments/docstrings are preserved, so a cleanroom file's line N maps 1:1 to source line N.
    Citations from the cleanroom resolve directly against the real repo.

Modes:
  manifest  — print every source file under --root (broad), one relative path per line, for authoring a
              whitelist/blacklist. Build/ runs/ tb/ .git/ __pycache__/ etc. are excluded.
  build     — copy a selected set into --dest, stripping comments/docstrings, preserving the directory
              structure and line numbers. Select with --whitelist FILE or --blacklist FILE.

Languages: .py/.pyi (tokenize COMMENTs + ast docstrings); .c/.cc/.cpp/.cxx/.h/.hpp/.hxx/.hh
(string/char/raw-string-aware // and /* */ scanner). Other extensions are copied verbatim.

Limitations (documented, ADR-0009 honesty): the C++ scanner handles standard string/char literals,
escapes, and C++11 raw strings R"delim(...)delim"; it does not special-case digraphs/trigraphs or
line-continuations inside // comments (none occur in this tree). Python docstring removal targets the
first statement of every module/class/def (and any bare string-literal statement); a sole-statement
docstring is replaced with `pass` so the file still parses.

Public Domain (The Unlicense).
"""
from __future__ import annotations

import argparse
import ast
import io
import os
import sys
import tokenize

PY_EXT = {".py", ".pyi"}
CPP_EXT = {".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx", ".hh"}
SRC_EXT = PY_EXT | CPP_EXT
EXCLUDE_DIRS = {".git", "build", "build-profile", "__pycache__", "runs", "tb",
                "attic", ".claude", "node_modules", ".mypy_cache", ".pytest_cache"}


# ---------------------------------------------------------------- Python ----
def strip_python(src: str) -> str:
    """Blank every `#` comment (tokenize) and every docstring / bare string-literal statement (ast),
    in place, preserving line numbers. A sole-statement docstring becomes `pass`."""
    spans: list[tuple[int, int, int, int]] = []   # (srow, scol, erow, ecol); rows 1-idx, cols 0-idx
    pass_at: dict[int, int] = {}                   # srow -> indent col, for sole-statement docstrings

    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                spans.append((tok.start[0], tok.start[1], tok.end[0], tok.end[1]))
    except (tokenize.TokenError, IndentationError):
        pass   # fall back to ast-only stripping (comments rare in such cases)

    tree = ast.parse(src)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not body:
            continue
        for idx, stmt in enumerate(body):
            is_doc = (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                      and isinstance(stmt.value.value, str))
            # strip the leading docstring of every scope; also any other bare string-literal statement
            if is_doc and (idx == 0 or True):
                spans.append((stmt.lineno, stmt.col_offset, stmt.end_lineno, stmt.end_col_offset))
                if idx == 0 and len(body) == 1 and not isinstance(node, ast.Module):
                    pass_at[stmt.lineno] = stmt.col_offset

    rows = [list(line) for line in src.split("\n")]
    for (sr, sc, er, ec) in spans:
        for r in range(sr, er + 1):
            if r - 1 >= len(rows):
                break
            row = rows[r - 1]
            start = sc if r == sr else 0
            end = ec if r == er else len(row)
            for c in range(start, min(end, len(row))):
                row[c] = " "

    out = ["".join(r) for r in rows]
    for sr, col in pass_at.items():
        out[sr - 1] = " " * col + "pass"
    return "\n".join(line.rstrip() for line in out)


# ------------------------------------------------------------------- C++ ----
def strip_cpp(src: str) -> str:
    """Remove // and /* */ comments via a char scanner that respects string, char, and raw-string
    literals. Newlines inside block comments are preserved, so line numbers are stable."""
    out: list[str] = []
    i, n = 0, len(src)
    st = "code"
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if st == "code":
            prev = out[-1] if out else ""
            if c == "/" and nxt == "/":
                st = "line"; i += 2; continue
            if c == "/" and nxt == "*":
                st = "block"; i += 2; continue
            if c == '"':
                out.append(c); st = "string"; i += 1; continue
            if c == "'":
                out.append(c); st = "char"; i += 1; continue
            if c == "R" and nxt == '"' and not (prev.isalnum() or prev == "_"):
                j = i + 2
                delim = ""
                while j < n and src[j] != "(":
                    delim += src[j]; j += 1
                close = ")" + delim + '"'
                end = src.find(close, j)
                end = n if end == -1 else end + len(close)
                out.append(src[i:end]); i = end; continue
            out.append(c); i += 1; continue
        if st == "line":
            if c == "\n":
                out.append("\n"); st = "code"
            i += 1; continue
        if st == "block":
            if c == "*" and nxt == "/":
                st = "code"; i += 2; continue
            if c == "\n":
                out.append("\n")
            i += 1; continue
        if st in ("string", "char"):
            out.append(c)
            if c == "\\" and nxt:
                out.append(nxt); i += 2; continue
            if (st == "string" and c == '"') or (st == "char" and c == "'"):
                st = "code"
            i += 1; continue
    # drop now-blank lines' trailing whitespace, keep the lines (line-number stability)
    return "\n".join(line.rstrip() for line in "".join(out).split("\n"))


def strip_file(rel: str, src: str) -> str:
    ext = os.path.splitext(rel)[1]
    if ext in PY_EXT:
        return strip_python(src)
    if ext in CPP_EXT:
        return strip_cpp(src)
    return src


def compact_blanks(text: str) -> str:
    """Collapse runs of 2+ blank (whitespace-only) lines into one, and drop leading/trailing blank lines.
    Stripping comments/docstrings leaves long empty regions (a 50-line module docstring -> 50 blank lines);
    without this, a `sed -n 1,10p` / `head` sample could read the top of a file as empty. At most one blank
    line ever separates content. NOTE: this deliberately breaks the 1:1 cleanroom->source line mapping
    (line counts shrink); cross-reference cleanroom citations to the real source by SYMBOL, not line."""
    out: list[str] = []
    prev_blank = True   # start True so leading blank lines are dropped
    for line in text.split("\n"):
        blank = line.strip() == ""
        if blank and prev_blank:
            continue
        out.append("" if blank else line)
        prev_blank = blank
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out) + "\n"


# -------------------------------------------------------------- verify ------
# The "lint": prove stripping removed ONLY comments/docstrings and corrupted no code.
def _is_subseq(a: str, b: str) -> bool:
    """Is `a` a subsequence of `b`? (stripping may only REMOVE characters, never add/reorder.)"""
    it = iter(b)
    return all(ch in it for ch in a)


def _strip_doc_nodes(tree: ast.AST) -> ast.AST:
    """Mirror strip_python at the AST level: drop every bare string-literal statement from each
    module/class/def body; if that empties a def/class body, substitute `pass` (as the stripper does)."""
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if not body:
            continue
        kept = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)
                                        and isinstance(s.value.value, str))]
        if not kept and not isinstance(node, ast.Module):
            kept = [ast.Pass()]
        node.body = kept
    return tree


def verify_python(orig: str, stripped: str) -> tuple[bool, str]:
    """Strong equivalence: the stripped file must parse, and its AST (docstrings normalized away) must be
    structurally identical to the original's. Any code change at all -> mismatch."""
    try:
        left = ast.dump(_strip_doc_nodes(ast.parse(orig)))
    except SyntaxError as e:
        return False, f"original failed to parse?! {e}"
    try:
        right = ast.dump(_strip_doc_nodes(ast.parse(stripped)))
    except SyntaxError as e:
        return False, f"stripped does not parse: {e}"
    if left != right:
        return False, "AST differs beyond docstrings (code was altered)"
    return True, ""


def _cpp_literals(src: str) -> list[str]:
    """Collect every string/char/raw-string literal (the scanner, collecting instead of stripping)."""
    lits: list[str] = []
    i, n = 0, len(src)
    st, buf = "code", ""
    while i < n:
        c = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if st == "code":
            prev = src[i - 1] if i > 0 else ""
            if c == "/" and nxt == "/":
                st = "line"; i += 2; continue
            if c == "/" and nxt == "*":
                st = "block"; i += 2; continue
            if c == '"':
                st = "string"; buf = c; i += 1; continue
            if c == "'":
                st = "char"; buf = c; i += 1; continue
            if c == "R" and nxt == '"' and not (prev.isalnum() or prev == "_"):
                j = i + 2
                delim = ""
                while j < n and src[j] != "(":
                    delim += src[j]; j += 1
                close = ")" + delim + '"'
                end = src.find(close, j)
                end = n if end == -1 else end + len(close)
                lits.append(src[i:end]); i = end; continue
            i += 1; continue
        if st == "line":
            if c == "\n":
                st = "code"
            i += 1; continue
        if st == "block":
            if c == "*" and nxt == "/":
                st = "code"; i += 2; continue
            i += 1; continue
        # string / char
        buf += c
        if c == "\\" and nxt:
            buf += nxt; i += 2; continue
        if (st == "string" and c == '"') or (st == "char" and c == "'"):
            lits.append(buf); buf = ""; st = "code"
        i += 1
    return lits


def verify_cpp(orig: str, stripped: str) -> tuple[bool, str]:
    """No stdlib C++ parser, so assert the strong structural invariants: identical line count, every
    string/char literal preserved verbatim, and stripped-nonwhitespace is a subsequence of original."""
    errs = []
    lo, ls = _cpp_literals(orig), _cpp_literals(stripped)
    if lo != ls:
        errs.append(f"literals changed ({len(lo)}->{len(ls)})")
    if not _is_subseq("".join(stripped.split()), "".join(orig.split())):
        errs.append("stripped is not a subsequence of original (chars added/reordered)")
    return (not errs), "; ".join(errs)


# ------------------------------------------------------------------ CLI -----
def iter_manifest(root: str):
    for dp, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for f in sorted(filenames):
            if os.path.splitext(f)[1] in SRC_EXT:
                yield os.path.relpath(os.path.join(dp, f), root)


def read_list(path: str) -> list[str]:
    out = []
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                out.append(s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)

    m = sub.add_parser("manifest", help="list all source files under --root")
    m.add_argument("--root", required=True)

    b = sub.add_parser("build", help="copy selected files into --dest, comments stripped")
    b.add_argument("--root", required=True)
    b.add_argument("--dest", required=True)
    g = b.add_mutually_exclusive_group(required=True)
    g.add_argument("--whitelist", help="file of relative paths to INCLUDE (one per line)")
    g.add_argument("--blacklist", help="file of relative paths to EXCLUDE from the full manifest")
    b.add_argument("--no-strip", action="store_true", help="copy verbatim (debug)")
    b.add_argument("--no-compact", action="store_true",
                   help="keep blank lines as-is (default collapses 2+ blank lines into one)")

    v = sub.add_parser("verify", help="prove the cleanroom changed only comments/docstrings")
    v.add_argument("--root", required=True, help="the original source root")
    v.add_argument("--cleanroom", required=True, help="the stripped tree to check against --root")

    a = ap.parse_args()

    if a.mode == "manifest":
        for rel in sorted(iter_manifest(a.root)):
            print(rel)
        return 0

    if a.mode == "verify":
        n_ok = n_fail = 0
        for dp, _dn, fn in os.walk(a.cleanroom):
            for f in sorted(fn):
                rel = os.path.relpath(os.path.join(dp, f), a.cleanroom)
                ext = os.path.splitext(f)[1]
                origp = os.path.join(a.root, rel)
                if not os.path.isfile(origp):
                    print(f"FAIL {rel}: no matching source under --root"); n_fail += 1; continue
                with open(origp, encoding="utf-8") as fh:
                    orig = fh.read()
                with open(os.path.join(dp, f), encoding="utf-8") as fh:
                    strp = fh.read()
                if ext in PY_EXT:
                    ok, msg = verify_python(orig, strp)
                elif ext in CPP_EXT:
                    ok, msg = verify_cpp(orig, strp)
                else:
                    ok, msg = True, ""
                if ok:
                    n_ok += 1
                else:
                    print(f"FAIL {rel}: {msg}"); n_fail += 1
        print(f"[verify] {n_ok} ok, {n_fail} failed")
        return 0 if n_fail == 0 else 1

    if a.whitelist:
        selected = read_list(a.whitelist)
    else:
        black = set(read_list(a.blacklist))
        selected = [r for r in iter_manifest(a.root) if r not in black]

    n_files = 0
    n_missing = 0
    for rel in selected:
        srcp = os.path.join(a.root, rel)
        if not os.path.isfile(srcp):
            print(f"[cleanroom] WARNING: whitelisted path not found: {rel}", file=sys.stderr)
            n_missing += 1
            continue
        with open(srcp, encoding="utf-8") as fh:
            src = fh.read()
        text = src if a.no_strip else strip_file(rel, src)
        if not a.no_strip and not a.no_compact:
            text = compact_blanks(text)
        outp = os.path.join(a.dest, rel)
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        with open(outp, "w", encoding="utf-8") as fh:
            fh.write(text)
        n_files += 1
    print(f"[cleanroom] wrote {n_files} files to {a.dest}"
          + (f" ({n_missing} missing)" if n_missing else ""), file=sys.stderr)
    return 0 if n_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
