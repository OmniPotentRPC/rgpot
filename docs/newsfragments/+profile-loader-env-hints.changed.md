With an empty path, `rgpot::abi::ProfileLoader` and the Rust
`rgpot_core::profile::library_candidates` try only the `<PREFIX>_LIBRARY` and
`RGPOT_<PREFIX>_ENGINE` values that are set and not empty, and try the default
library names only when neither is set. A hint that does not open now fails
with a message naming each library tried instead of loading a default one.
