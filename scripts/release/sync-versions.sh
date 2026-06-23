#!/usr/bin/env bash
# Lockstep version sync for rgpot monorepo releases.
# Usage: scripts/release/sync-versions.sh <semver>
# Invoked from cog pre_bump_hooks as: scripts/release/sync-versions.sh {{version}}
#
# Surfaces (must stay aligned for release.yml tag gate + crates.io):
#   meson.build, towncrier.toml, rgpot-core/Cargo.toml [package], pixi.toml [workspace]
# Does NOT touch dependency version pins (e.g. nickel = ">=9.9.9,<10").
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VERSION="${1:?usage: sync-versions.sh <semver>}"

if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+([.-].*)?$ ]]; then
  echo "error: refusing non-semver version: $VERSION" >&2
  exit 1
fi

# meson project version (first version: '...' in project(...))
if [[ -f meson.build ]]; then
  sed -i -E "0,/version: *'[^']*'/s//version: '${VERSION}'/" meson.build
fi

# towncrier stamp (single top-level version key)
if [[ -f towncrier.toml ]]; then
  sed -i -E "s/^(version = )\"[^\"]*\"/\1\"${VERSION}\"/" towncrier.toml
fi

# Rust crate published to crates.io — first ^version = in file is [package]
if [[ -f rgpot-core/Cargo.toml ]]; then
  sed -i -E "0,/^version = \"[^\"]*\"/s//version = \"${VERSION}\"/" rgpot-core/Cargo.toml
fi

# pixi workspace version only (first ^version = is under [workspace]; later lines
# are dependency constraints and must not be rewritten).
if [[ -f pixi.toml ]]; then
  awk -v ver="$VERSION" '
    BEGIN { done = 0 }
    !done && /^version = "/ {
      sub(/version = "[^"]*"/, "version = \"" ver "\"")
      done = 1
    }
    { print }
  ' pixi.toml > pixi.toml.tmp && mv pixi.toml.tmp pixi.toml
fi

echo "synced lockstep version -> ${VERSION}"
echo "  meson.build / towncrier.toml / rgpot-core/Cargo.toml / pixi.toml [workspace]"
