# Dagger × GitHub Actions (rgpot sketch)

This directory is an **illustrative** integration: how rgpot *could* run selected
CI through Dagger while GitHub Actions remains the outer scheduler and **pixi**
remains the reproducibility story for C++/Python/Rust toolchains.

It is **not** a replacement for `ci/gha/` (Nickel → GHA YAML). Merge only if you
want a parallel experiment, not as the sole CI path.

## How well does Dagger integrate with GitHub?

### What works well (real strengths)

| Integration point | Mechanism | Fit for rgpot |
|-------------------|-----------|---------------|
| **Run pipelines on GHA** | [`dagger/dagger-for-github@v8`](https://github.com/dagger/dagger-for-github) installs the CLI, starts an engine on the runner (or attaches to Dagger Cloud), runs `dagger call` / `shell` / `check` | Excellent for *invoking* Dagger from a normal workflow |
| **Job summaries / traces** | Action can emit GitHub job summaries; with `DAGGER_CLOUD_TOKEN` + `--cloud`, traces and cache survive ephemeral runners | Good observability; Cloud is optional but recommended for cache hit rates |
| **PR / branch triggers** | Still normal `on: pull_request` / `push` — Dagger does not replace GHA event model | Same as today; no loss of branch protection / required checks |
| **Secrets** | GHA secrets → action `cloud-token` / env; Dagger can receive secrets as function args | Same threat model as any GHA step; no special GitHub API for Dagger secrets |
| **Local = CI** | Same `dagger call …` on laptop and in the action | Main win if functions are fully containerized |

### Where integration is thin or awkward (honest limits)

| Gap | Detail | rgpot implication |
|-----|--------|-------------------|
| **GHA is still the outer shell** | Tags (`v*`), `workflow_run` (doc preview commenter), `softprops/action-gh-release`, `peaceiris/actions-gh-pages`, Environments/reviewers stay in YAML | Release + docs deploy do not become “pure Dagger” without reimplementing each integration |
| **Not a drop-in for pixi matrices** | Dagger shines at container DAGs; our matrices are meson/cmake × os × rpc/cache with host SDK/clang quirks (`potctl ci darwin-env`) and pixi envs (`devbld`, `metatomicbld`, `tbbld`) | You either (a) run **pixi inside** Dagger containers (duplicate lock/env), (b) `host` exec on the runner (Dagger adds little), or (c) abandon pixi in CI for pure container images (big policy change) |
| **Cosmo / APE builds** | Already custom toolchain + host `rustc` cosmo target; poor fit for stock OCI layers | Likely remains GHA/shell or a dedicated container image you maintain |
| **crates.io / GH Release** | Publish steps use GHA secrets + `softprops`; doable as `dagger call release publish` but still needs a GHA job on `push: tags` | Outer trigger stays GHA |
| **Required checks / CODEOWNERS** | GitHub sees *workflow* success, not Dagger function graph | You still author at least one GHA workflow file (or generate it from Nickel) that calls Dagger |
| **Engine on GHA runners** | Ephemeral runner starts engine every job unless Dagger Cloud; cold starts + layer pulls add minutes | For heavy C++ compiles, Cloud cache or large self-hosted runners matter more than the action version |

### Bottom line

- **GitHub integration quality: good as a *step runner*** — `dagger/dagger-for-github@v8` is mature, documented, and fine for PR/main jobs.
- **GitHub integration quality: weak as a *full platform replacement*** — you still need GHA (or another orchestrator) for events, permissions, and first-party deploy/release actions.
- **Pixi coexistence: complementary, not automatic** — best pattern is “Dagger orchestrates *containers*; pixi installs *inside* the container or image you define,” accepting duplicated env maintenance unless you fully image-bake pixi envs.

## Recommended hybrid (if we ever pursue this)

1. **Keep** `ci/gha/` Nickel for all GHA structure (including a thin `dagger-ci.yml` generator).
2. **Add** a Dagger module under `ci/dagger/` for *subset* jobs that are container-friendly (e.g. `rgpot-core` tests on Linux, or a single meson debug build in a fixed image).
3. **Do not** move release/gh-pages/cosmo/metatomic torch layout into Dagger first.
4. **Do not** drop pixi; pass `pixi install` / `pixi run` as the command inside the function, or build an image with pixi + lockfile baked in.

## Local experiment (optional)

```bash
# Requires dagger CLI: https://docs.dagger.io/install
cd ci/dagger
# After module exists:
# dagger call -m . ci-smoke
```

The workflow [`.github/workflows/dagger-ci.yml`](../../.github/workflows/dagger-ci.yml) is `workflow_dispatch` + optional PR path-filter only, so it does not gate the repo until you promote it.

## References

- [Dagger: GitHub Actions integration](https://docs.dagger.io/getting-started/ci-integrations/github-actions/)
- [dagger/dagger-for-github](https://github.com/dagger/dagger-for-github) (`@v8`)
