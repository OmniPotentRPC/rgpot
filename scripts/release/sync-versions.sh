#!/usr/bin/env bash
# Lockstep version sync for rgpot monorepo releases.
# Usage: scripts/release/sync-versions.sh <semver>
# Invoked from cog pre_bump_hooks as: scripts/release/sync-versions.sh {{version}}
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VERSION="${1:?usage: sync-versions.sh <semver>}"

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
  echo "error: refusing non-semver version: $VERSION" >&2
  exit 1
fi

# meson project version
if [[ -f meson.build ]]; then
  sed -i -E "0,/version: *'[^']*'/s//version: '${VERSION}'/" meson.build
fi

# towncrier config (used by towncrier build --version as well)
if [[ -f towncrier.toml ]]; then
  sed -i -E "s/^(version = )\"[^\"]*\"/\1\"${VERSION}\"/" towncrier.toml
fi

# Rust crate published to crates.io
if [[ -f rgpot-core/Cargo.toml ]]; then
  # Only the package table's version line (first occurrence in file is package).
  sed -i -E "0,/^version = \"[^\"]*\"/s//version = \"${VERSION}\"/" rgpot-core/Cargo.toml
fi

# pixi workspace version (informational; keep aligned)
if [[ -f pixi.toml ]]; then
  sed -i -E "0,/^version = \"[^\"]*\"/s//version = \"${VERSION}\"/" pixi.toml
fi

echo "synced lockstep version -> ${VERSION}"
echo "  meson.build / towncrier.toml / rgpot-core/Cargo.toml / pixi.toml"
