# Module placeholder

When you run `dagger init` / add a real module here, expose at least:

```text
ci-smoke()           # e.g. cargo test -p potctl + cargo nextest -p rgpot-core in a rust image
                     # or: container with pixi + lockfile, then `pixi run` meson setup/test (linux only)
rust-test(all_features: bool)
# later, optional:
# meson-linux(rpc: bool, cache: bool)  # only if imaged pixi/devbld is acceptable
```

Pixi coexistence patterns:

1. **Bake pixi + `pixi install -e devbld` into an OCI image** (best Dagger ergonomics; image rebuild when lock changes).
2. **Mount repo + run pixi on host** from a thin function (little Dagger benefit; runner still needs pixi like today).
3. **Drop pixi in Dagger path** — pure distro packages/cargo in container (diverges from local pixi story; only acceptable for a *subset* job).

Until the module exists, `dagger-ci.yml` will fail at `dagger call` — that is expected for this sketch PR.
