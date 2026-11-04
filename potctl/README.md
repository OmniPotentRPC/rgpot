# potctl

Repo-local **project control plane** for [rgpot](https://github.com/OmniPotentRPC/rgpot).
Not published to crates.io (`publish = false`). Lives beside `rgpot-core` in the
Cargo workspace at the repository root.

## Commands

### Release (lockstep semver)

```bash
cargo run -q -p potctl -- release assert
cargo run -q -p potctl -- release assert --require-changelog 1.2.0
cargo run -q -p potctl -- release sync 1.2.3
cargo run -q -p potctl -- release notes 1.2.0 -o /tmp/notes.md
```

Lockstep surfaces: `meson.build`, `CMakeLists.txt` (`VERSION`), `rgpot-core/Cargo.toml`,
`towncrier.toml`, `pixi.toml` (first `version =` only — workspace key).

### CI helpers (replaces ad-hoc bash in GHA)

```bash
# Required files + lockstep (first step in build/potentials/release-prepare)
cargo run -q -p potctl -- ci preflight
cargo run -q -p potctl -- ci preflight --no-lockstep   # files only

# macOS + pixi: host clang/SDK + conda dep paths (unit-tested plan; no-op on Linux)
if [ "$(uname -s)" = Darwin ] || [ "${RUNNER_OS:-}" = macOS ]; then
  export SDKROOT="$(xcrun --show-sdk-path)"
fi
eval "$(cargo run -q -p potctl -- ci darwin-env)"
meson setup bbdir ${POTCTL_MESON_EXTRA:-} …
```

```bash
pixi r potctl-test      # cargo test -p potctl
pixi r ci-preflight
```

## Wiring

| Consumer | How |
|----------|-----|
| `cog.toml` `pre_bump_hooks` | `cargo run -q -p potctl -- release sync/assert …` |
| `pixi.toml` | `release-*`, `ci-preflight`, `ci-darwin-env`, `potctl-test` |
| `ci-orchestrator.yml` / `release*.yml` | Job `ci-tools`/`tools` → `setup-ci-tools` artifact → jobs `ensure-potctl`/`restore-ci-tools` + `potctl ci preflight`; build matrix legs are one-liners `potctl ci meson-test` / `cmake-test` (darwin-env applied inside potctl) |
| GHA rust-cache | `Swatinem/rust-cache` `shared-key: potctl-ci-tools` |

CI builds **portable/fat** potctl per OS family (`setup-ci-tools`): **Linux** → `x86_64-unknown-linux-musl` (+ crt-static when possible); **macOS** → `lipo` fat binary (aarch64 + x86_64). Artifacts are `ci-tools-<sha>-Linux` / `…-macOS`; `ensure-potctl` restores `…-${{ runner.os }}` (no single binary spans Darwin and Linux — different ABIs). Cross-run: rust-cache.

## Design rule

Anything that is “assert / emit env / rewrite semver / CI step body” belongs in `potctl` with
`cargo test -p potctl`. Nickel (or hand orchestrator until `orchestrator.ncl`) writes the job
graph and matrices only; GHA steps should be thin `potctl ci …` lines.

| `potctl ci` verb | Role |
|------------------|------|
| `preflight` | Repo layout / lockstep sanity |
| `darwin-env` | Print `eval`-able macOS+pixi host SDK exports (low-level; prefer verbs below) |
| `meson-test --rpc … --cache …` | Meson setup + compile + test (orchestrator matrix leg) |
| `cmake-test --rpc … --cache …` | CMake configure + build + ctest |
| `build-test --sys meson\|cmake …` | Same as above via `matrix.sys` |
| `xckernel-test` | Meson `-Dwith_xckernel=true` and suite `xckernel` (`XcKernelGoldens`) |

Meson/cmake/pixi still execute the real builds; potctl only orchestrates argv + env.


## Cosmopolitan APE (`potctl.com`, gating on PR)

See [`cosmo/README.md`](cosmo/README.md) and `.github/workflows/cosmo-potctl.yml`.

One `potctl.com` via Cosmopolitan + community `*-linux-cosmo` targets, built once
on Linux and smoke-tested on Linux and macOS with `ci preflight` / `release assert`.
That workflow is a **gating** PR/release check (failure fails the check). Normal
musl/fat `setup-ci-tools` / `ensure-potctl` remains the default potctl install for
`ci-orchestrator.yml` (build/potentials stages) and `release*.yml` jobs.
