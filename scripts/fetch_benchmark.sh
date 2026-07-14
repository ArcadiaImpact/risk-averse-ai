#!/usr/bin/env bash
# Vendor the riskaverseAIs benchmark at the commit pinned in config.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['benchmark']['repo'])")
COMMIT=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['benchmark']['commit'])")
DIR=$(python3 -c "import yaml;print(yaml.safe_load(open('config.yaml'))['benchmark']['vendor_dir'])")
if [ ! -d "$DIR/.git" ]; then
  git clone "$REPO" "$DIR"
fi
git -C "$DIR" fetch -q origin
git -C "$DIR" checkout -q "$COMMIT"
echo "riskaverseAIs vendored at $(git -C "$DIR" rev-parse --short HEAD) in $DIR"
