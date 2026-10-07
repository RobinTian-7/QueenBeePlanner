#!/usr/bin/env bash
# Fetch the Silo-Bench data at a pinned upstream commit into
# third_party/acl26-silo-bench (needs git and network access).
# Usage: scripts/fetch_silo_bench.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party/acl26-silo-bench"
URL="https://github.com/jwyjohn/acl26-silo-bench"
PIN="e74127782ed1c42fff474249961f022c063d76f2"

if [[ -d "$DEST/benchmarks" ]]; then
  echo "Silo-Bench data already present: $DEST/benchmarks"
  exit 0
fi
if [[ -e "$DEST" || -L "$DEST" ]]; then
  # Only an empty directory is replaced.
  if ! rmdir "$DEST" 2>/dev/null; then
    echo "error: $DEST exists but has no benchmarks/ directory; move it away and re-run" >&2
    exit 1
  fi
fi
if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required" >&2
  exit 1
fi

mkdir -p "$(dirname "$DEST")"
TMP="$(mktemp -d "$(dirname "$DEST")/.acl26-silo-bench.XXXXXX")"
trap 'rm -rf "$TMP"' EXIT

git -C "$TMP" init -q
git -C "$TMP" remote add origin "$URL"
git -C "$TMP" fetch -q --depth 1 origin "$PIN"
git -C "$TMP" checkout -q FETCH_HEAD
HEAD_SHA="$(git -C "$TMP" rev-parse HEAD)"
if [[ "$HEAD_SHA" != "$PIN" ]]; then
  echo "error: fetched $HEAD_SHA, expected $PIN" >&2
  exit 1
fi
if [[ ! -d "$TMP/benchmarks" ]]; then
  echo "error: commit $PIN has no benchmarks/ directory" >&2
  exit 1
fi

mv "$TMP" "$DEST"
trap - EXIT
echo "Silo-Bench $PIN -> $DEST"
echo "data: $DEST/benchmarks (default location; set SILO_BENCH_DIR to use another copy)"
