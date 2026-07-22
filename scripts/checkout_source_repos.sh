#!/usr/bin/env bash
#
# checkout_source_repos.sh — fetch the public Daml source repositories the
# benchmark was built from, pinned to the exact commits that were evaluated.
#
# Only needed for the optional **Daml syntax pass-rate** metric, which shells out
# to the Daml toolchain against each example's source project (its daml.yaml).
# The headline **exact-match** metric needs none of this — the prefix/suffix/target
# are embedded in the dataset.
#
# Repos are cloned into sources/<repo>/ at the benchmark root. Dataset file paths
# are repo-relative (<repo>/...), which the syntax check resolves under sources/.
#
# Usage:
#   scripts/checkout_source_repos.sh           # all repos
#   scripts/checkout_source_repos.sh splice    # a single repo
#
# These are public, Apache-2.0 repositories (see NOTICE). ~several GB total when
# fully checked out.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT_DIR/sources"

# repo  url  commit  (commit = the snapshot the benchmark was evaluated against)
REPOS=$(cat <<'EOF'
daml              https://github.com/digital-asset/daml.git              c69d4fdda3fb132aeb06a52562ea51f91171a53d
daml-finance      https://github.com/digital-asset/daml-finance.git      155f931b6ebe7d3662fd72788cb17f0bfb5a7ba6
splice            https://github.com/hyperledger-labs/splice.git         e04a9d54a4385bf5a72200087ed3f120f0b51157
canton            https://github.com/digital-asset/canton.git            bb8170fb7ba00312ed425f2ddfc624e8125f9717
ex-models         https://github.com/digital-asset-archive/ex-models.git ca6818c10923360f7158e03a4c50dabe156f5419
account-hierarchy https://github.com/SynfiniDLT/account-hierarchy.git    407a020bd7b76ed044a3e169c621048bf2faf22e
EOF
)

want="${1:-}"

checkout_one() {
  local name="$1" url="$2" sha="$3"
  local dir="$DEST/$name"
  if [ -d "$dir/.git" ]; then
    echo "[$name] already present at $dir — skipping (delete it to re-fetch)"
    return 0
  fi
  echo "[$name] fetching $url @ ${sha:0:10} -> $dir"
  mkdir -p "$dir"
  git -C "$dir" init -q
  git -C "$dir" remote add origin "$url" 2>/dev/null || true
  # Try a cheap single-commit fetch first; fall back to a full fetch if the
  # server refuses fetch-by-sha.
  if git -C "$dir" fetch --depth 1 origin "$sha" 2>/dev/null; then
    git -C "$dir" checkout -q FETCH_HEAD
  else
    echo "[$name] shallow fetch-by-sha unsupported; fetching full history…"
    git -C "$dir" fetch -q origin
    git -C "$dir" checkout -q "$sha"
  fi
  echo "[$name] done"
}

while read -r name url sha; do
  [ -z "$name" ] && continue
  if [ -n "$want" ] && [ "$want" != "$name" ]; then
    continue
  fi
  checkout_one "$name" "$url" "$sha"
done <<< "$REPOS"

echo "All requested source repos are in $DEST/"
