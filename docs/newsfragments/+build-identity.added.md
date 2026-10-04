`Capabilities` carries `buildVersion` and `buildRevision` (appended at `@18`
and `@19`), filled by the Rust and C++ producers, and `rgpot_source_revision()`
returns the source revision of the loaded library. The revision comes from
`RGPOT_SOURCE_REVISION` or the git checkout and is empty when neither exists.
