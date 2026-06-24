# CI workflow generation (Nickel)

Source of truth for the main **Build Matrix** workflow:

| File | Role |
|------|------|
| [`workflow.ncl`](workflow.ncl) | Nickel program (modular steps/jobs) |
| [`../../.github/workflows/build.yml`](../../.github/workflows/build.yml) | Generated YAML — do not hand-edit |

## Regenerate

```bash
pixi r -e cigen gen-gha
# or: nickel export --format yaml ci/gha/workflow.ncl -o .github/workflows/build.yml
```

Format before commit (prek `nickel-format` hook also runs this):

```bash
pixi r -e cigen nickel format ci/gha/workflow.ncl
```

Requires the `cigen` pixi environment (`nickel >= 9.9.9`).

## Design notes (post potctl / lockstep era)

- **`ci-tools` job** builds portable/fat `potctl` once per OS (`setup-ci-tools` artifact).
- Consumer jobs call **`ensure-potctl`** (restore + `potctl ci preflight`).
- Meson/CMake scripts use **`eval "$(potctl ci darwin-env)"`** instead of inlined macOS/pixi env blocks (logic lives in `potctl/src/ci.rs`, unit-tested).
- Action pins: `checkout@v6`, `setup-pixi@v0.9.4`, `cache@v5` (match other workflows).

Other workflows (`release.yml`, `potentials.yml`, cosmo) are hand-maintained YAML; only the combinatorial build matrix is Nickel-generated.
