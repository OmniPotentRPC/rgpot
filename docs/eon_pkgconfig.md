# eOn packaging: pkg-config install of rgpot

eOn resolves **installed** rgpot via `dependency('rgpot')` before the Meson
subproject wrap (`subprojects/rgpot.wrap`). Engines stay **runtime dlopen**
(no torch / metatomic / xtb at link time).

## Build + install profile (matches eOn wrap options)

```bash
meson setup build-eon-export \
  -Dwith_rpc_client_only=true \
  -Dpure_lib=true \
  -Dwith_tests=false \
  -Dwith_examples=false \
  -Dwith_xtb=false \
  -Dwith_metatomic=false \
  -Dwith_rpc=false \
  --prefix=/usr/local
meson compile -C build-eon-export
meson install -C build-eon-export
# exports: libnwchempot, libcpmdpot, libptlrpc, include/rgpot/**, rgpot.pc
```

Optional engine plugins (install separately / set `*_ENGINE` env at run time):

| Engine | Env | Linked into eOn? |
|--------|-----|------------------|
| `libnwchemc.so` | `NWCHEMC_LIBRARY` / `RGPOT_NWCHEMC_ENGINE` | no |
| `libcpmdc.so` | `CPMDC_LIBRARY` / `RGPOT_CPMDC_ENGINE` | no |
| `libmetatomic_engine.so` | `RGPOT_METATOMIC_ENGINE` | no |
| `libxtb_engine.so` | `RGPOT_XTB_ENGINE` | no (thin `XTBDlopen` is always in the host link; engine is runtime-only) |

## eOn configure

```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
meson setup bbdir -Dwith_rgpot=true
# uses pkg-config 'rgpot' when present; else falls back to subproject wrap
```

Wrap fallback still uses:

```
with_rpc_client_only=true, pure_lib=true, with_tests=false, with_examples=false
```
