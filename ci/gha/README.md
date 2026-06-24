# CI workflows (Nickel source of truth)

**Every** GitHub Actions workflow under `.github/workflows/` is generated from
`ci/gha/*.ncl`. Do **not** hand-edit the YAML; change Nickel and regenerate.

## Layout

| Path | Role |
|------|------|
| [`lib/pins.ncl`](lib/pins.ncl) | Action / pixi / runner pins (bump once, all workflows follow) |
| [`lib/steps.ncl`](lib/steps.ncl) | Shared step/job builders (`checkout`, `setup_pixi`, `ensure_potctl`, `ci-tools` job, …) |
| `build.ncl` | Build matrix (meson/cmake × os × rpc/cache, rust, client bridge) |
| `potentials.ncl` | xtb/tblite + metatomic feature matrix |
| `release.ncl` | Tag `v*` publish (gate → confidence → crates.io → GH Release) |
| `release_prepare.ncl` | Cog/towncrier dry-run on PR / `workflow_dispatch` |
| `prek.ncl`, `ci_docs.ncl`, `docs_quality.ncl`, `towncrier.ncl` | Quality / docs / fragments |
| `cosmo_potctl.ncl`, `cosmo_potctl_spike.ncl` | Cosmopolitan APE potctl + legacy alias |
| `ci_doc_commenter.ncl` | PR doc preview comment |
| [`gen.sh`](gen.sh) | Export all of the above to `.github/workflows/` |
| `workflow.ncl` | Deprecated alias → `import "build.ncl"` (compat only) |

## Regenerate

```bash
pixi r -e cigen gen-gha
# equivalent: bash ci/gha/gen.sh   (needs `nickel` on PATH)
```

Format before commit (`prek` `nickel-format` hook, or all at once):

```bash
pixi r -e cigen nickel-fmt
# or: find ci/gha -name '*.ncl' -print0 | xargs -0 nickel format
```

Requires the `cigen` pixi environment (`nickel >= 9.9.9`).

## Why Nickel for everything (not only the build matrix)

Hand-maintaining eleven YAML files duplicated the same pins (`checkout@v6`,
`setup-pixi@v0.9.4`, `pixi-version`, `setup-ci-tools` / `ensure-potctl` patterns).
That is how `workflow.ncl` drifted behind `build.yml` during the potctl/lockstep
work: someone fixed YAML in place and never regenerated.

Generating **all** workflows from one library means:

1. **One pin table** — bump `lib/pins.ncl` once (e.g. `cache@v5` everywhere, including
   cosmo/lychee caches that still said `v4` in old hand YAML).
2. **One potctl CI story** — `job_ci_tools_os_matrix`, `ensure_potctl`,
   `restore_ci_tools_linux`, `darwin_env_prelude` live in `lib/steps.ncl`.
3. **Refactors are mechanical** — add a step helper, re-export; no eleven-file grep.
4. **Review focus** — PRs touch `ci/gha/*.ncl`; generated YAML is expected noise
   (or verified by CI via `gen-gha` + `git diff --exit-code` if you add a check later).

Shell-heavy steps (cosmo toolchain install, metatomic torch layout) stay as
multiline strings inside the relevant `*.ncl` file; Nickel is not trying to
express bash — only structure, reuse, and pins.

## Optional follow-up

- prek/CI job: `pixi r -e cigen gen-gha && git diff --exit-code .github/workflows`
  so drift cannot land on `main`.
