#!/usr/bin/env python3
"""Find argparse flags that are ACCEPTED BUT NEVER READ.

This codebase has produced this bug repeatedly, and it is the worst kind:
the flag parses, the run completes, the exit code is 0, and the requested
behaviour simply never happened. Known instances:

  * ``--warm-start-equity-trunk`` (prob5) -- declared but ``_build_network``
    never read it, so training ran from a RANDOM init and a nonexistent
    checkpoint path trained happily, because the file was never opened.
  * ``--sage-plies`` -- dropped on play_models' parallel path, silently
    giving 0-ply Sage.

The check: every ``add_argument("--foo-bar", ...)`` creates ``args.foo_bar``.
If ``.foo_bar`` (or ``getattr(x, "foo_bar")``) appears nowhere in the tree,
nothing consumes the flag.

Usage
-----
    python3 tools/audit_argparse_dests.py [REPO] [--compare GITREF]

    REPO       repo root to scan (default: the repo containing this file)
    --compare  also diff the flag surface of shared files against a git ref,
               e.g. ``--compare repro``. Path-keyed, so it says nothing about
               files that MOVED between the two trees -- check renames
               separately (basename matching is a cheap first pass).

Read-only: walks files and, for --compare, runs ``git show``. Safe to point
at a checkout that has a training run in flight.

Caveat
------
Consumption is detected textually. A program that splats its namespace
(``train(**vars(args))``) consumes dests without any literal ``.dest``
read, and its flags would be reported here as false positives. The scan
warns when it sees that pattern. The inverse -- a dest that is read only
in dead code -- is not detected at all; this finds "never referenced",
not "never effective".
"""
import argparse
import ast
import os
import re
import subprocess
import sys

SKIP_DIRS = {".git", "__pycache__", ".venv", "node_modules", ".pytest_cache"}


def py_files(root):
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                out.append(os.path.join(dirpath, fn))
    return sorted(out)


def flags_in(src):
    """{dest: '--flag'} for every add_argument call carrying a long option."""
    found = {}
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return found
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and fn.attr == "add_argument"):
            continue
        longs = [a.value for a in node.args
                 if isinstance(a, ast.Constant) and isinstance(a.value, str)
                 and a.value.startswith("--")]
        if not longs:
            continue
        action = None
        dest = None
        for kw in node.keywords:
            if kw.arg == "dest" and isinstance(kw.value, ast.Constant):
                dest = kw.value.value
            if kw.arg == "action" and isinstance(kw.value, ast.Constant):
                action = kw.value.value
        if action == "help":
            continue
        if dest is None:
            dest = longs[0][2:].replace("-", "_")
        found[dest] = longs[0]
    return found


def _has_vars_splat(src):
    """True if the module really calls f(**vars(...)) -- parsed, not grepped,
    so prose mentioning the pattern (like this file's own docstring) does not
    trip it."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        for kw in node.keywords:
            if kw.arg is not None:
                continue                      # not a ** splat
            v = kw.value
            if (isinstance(v, ast.Call) and isinstance(v.func, ast.Name)
                    and v.func.id == "vars"):
                return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("repo", nargs="?", default=None)
    ap.add_argument("--compare", default=None, metavar="GITREF")
    a = ap.parse_args()

    repo = os.path.abspath(
        a.repo or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    files = py_files(repo)
    blobs = {}
    for p in files:
        try:
            blobs[p] = open(p, encoding="utf-8", errors="replace").read()
        except OSError:
            pass
    allsrc = "\n".join(blobs.values())

    splat = [os.path.relpath(p, repo) for p, s in blobs.items()
             if _has_vars_splat(s)]
    if splat:
        print("WARNING: namespace splatting found -- dests in these files are "
              "consumed without a literal attribute read, so they may show up "
              "below as false positives:")
        for p in splat:
            print(f"    {p}")
        print()

    print(f"=== flags declared but dest never read anywhere under {repo} ===")
    n = 0
    for p in sorted(blobs):
        for dest, flag in sorted(flags_in(blobs[p]).items()):
            attr = re.compile(r"\.%s\b" % re.escape(dest))
            gat = re.compile(r"""getattr\([^,]+,\s*["']%s["']""" % re.escape(dest))
            if not attr.search(allsrc) and not gat.search(allsrc):
                print(f"  {os.path.relpath(p, repo):48s} {flag}  (dest={dest})")
                n += 1
    print(f"  -> {n} suspicious")

    if not a.compare:
        return 1 if n else 0

    print()
    print(f"=== flag surface vs {a.compare} (same-path files only) ===")
    for p in sorted(blobs):
        rel = os.path.relpath(p, repo)
        here = flags_in(blobs[p])
        r = subprocess.run(["git", "-C", repo, "show", f"{a.compare}:{rel}"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            if here:
                print(f"  {rel}: absent at this path on {a.compare} "
                      f"(moved? renamed?) -- {len(here)} flags here")
            continue
        there = flags_in(r.stdout)
        only_ref = sorted(set(there.values()) - set(here.values()))
        only_here = sorted(set(here.values()) - set(there.values()))
        if only_ref or only_here:
            print(f"  {rel}:")
            if only_ref:
                print(f"      only on {a.compare}: {' '.join(only_ref)}")
            if only_here:
                print(f"      only here        : {' '.join(only_here)}")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
