#!/usr/bin/env bash
#
# install_daml.sh — install the Daml SDK (the `daml` assistant), needed only for
# the optional Daml **syntax pass-rate** metric. The exact-match metric does not
# need it.
#
# This installs the Daml *assistant*; it then auto-fetches the specific SDK
# version pinned in each source repo's daml.yaml on first use (the benchmark
# spans several SDK versions, e.g. 1.17.x, 2.10.x and 3.3.x), so you do not pick
# a version here.
#
# Why the assistant and not DPM: the assistant is deprecated in favour of the
# Digital Asset Package Manager (dpm), but the benchmark's source repos are
# pinned to SDK versions that dpm's registry does not provide (dpm only ships
# recent releases). The assistant fetches those exact pinned versions, so it is
# the tool that covers the full range here.
#
# Usage:
#   scripts/install_daml.sh
#
# After install, ensure ~/.daml/bin is on your PATH (the installer prints this).
set -euo pipefail

if command -v daml >/dev/null 2>&1; then
  echo "daml already installed: $(daml version 2>/dev/null | head -1 || echo present)"
  exit 0
fi

echo "Installing the Daml SDK from https://get.daml.com/ …"
curl -sSL https://get.daml.com/ | sh

echo
echo "Done. If 'daml' is not found, add the SDK to your PATH, e.g.:"
echo '  export PATH="$HOME/.daml/bin:$PATH"'
echo "(also requires a JDK 11+; install one if 'daml' reports a missing Java runtime)."
