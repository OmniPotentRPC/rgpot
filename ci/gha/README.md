# CI workflows (Nickel source of truth)

**Every** GitHub Actions workflow under `.github/workflows/` is generated from
`ci/gha/*.ncl`. **Do not hand-edit the YAML.** Change Nickel, then regenerate.

## Design (Dagger vs Nickel)

We evaluated replacing GHA execution with **Dagger** and chose **not** to migrate
now: the problem was pin/step **duplication and drift**, not GHA as a runner.
Nickel owns the workflow graph, matrices, and shared steps; bash stays as
multiline strings in the owning module. See implementer/goal notes or the
decision summary in this section’s history: *Nickel library + generate GHA*
unless a future goal revisits Dagger as an additional local runner.

## Layout

| Path | Role |
|------|------|
| [`lib/pins.ncl`](lib/pins.ncl) | Action / pixi / runner pins (bump once, all workflows follow) |
| [`lib/steps.ncl`](lib/steps.ncl) | Step/job builders (`checkout_*`, `setup_pixi*`, `ensure_potctl`, `job_ci_tools_os_matrix`, `job_tools_linux`, `darwin_env_prelude`, …) |
| `build.ncl`, `potentials.ncl`, `release.ncl`, … | One module per workflow (compose builders + job-specific bash) |
| [`gen.sh`](gen.sh) | Export all modules → `.github/workflows/*.yml` |
| [`check-drift.sh`](check-drift.sh) | `gen.sh` then `git diff --exit-code .github/workflows` |
| `workflow.ncl` | Deprecated alias → `import "build.ncl"` |

## Commands

```bash
# Regenerate all workflows
pixi r -e cigen gen-gha

# Format Nickel sources
pixi r -e cigen nickel-fmt

# Fail if YAML does not match Nickel (also prek hook `gha-nickel-drift`)
pixi r -e cigen gha-drift
```

Requires the `cigen` pixi environment (`nickel >= 9.9.9`).

## Idiomatic usage

1. **Pins only in `lib/pins.ncl`** — never scatter `actions/checkout@vN` in workflow modules.
2. **Shared steps only in `lib/steps.ncl`** — functions return step/job records; merge with `&` for names/`if` conditions.
3. **Workflow modules are thin** — `let S = import "lib/steps.ncl" in` then compose jobs; keep cosmo/metatomic/release bash as `m%"..."%` strings in that file only.
4. **YAML is output** — if a job is wrong, fix `.ncl` and `gen-gha`; do not patch `.yml` in the PR.

## Drift enforcement

| Gate | How |
|------|-----|
| Local / prek | Hook `gha-nickel-drift` runs when `ci/gha/**/*.ncl` or `.github/workflows/**/*.yml` change |
| Manual / CI | `pixi r -e cigen gha-drift` (or `bash ci/gha/check-drift.sh` with `nickel` on PATH) |

## Tracking `ci/gha/lib/`

Root `.gitignore` has `lib/`; `ci/gha/lib/` is explicitly un-ignored (`!ci/gha/lib/`).
Confirm with `git ls-files ci/gha/lib`.
