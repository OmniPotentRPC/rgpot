// MIT License
// Copyright 2023--present rgpot developers

//! Version of the loaded library, for callers to check against the header
//! they compiled with.
//!
//! `rgpot.h` stamps `RGPOT_VERSION` and `RGPOT_VERSION_MAJOR` / `_MINOR` /
//! `_PATCH` at generation time; these functions report the same numbers
//! from the shared object actually loaded. The C ABI follows semver: a
//! caller built against major M runs on any library of major M whose minor
//! is at least the header's.
//!
//! ```c
//! if (rgpot_version_major() != RGPOT_VERSION_MAJOR ||
//!     rgpot_version_minor() < RGPOT_VERSION_MINOR) {
//!   fprintf(stderr, "rgpot %s loaded, built against %s\n",
//!           rgpot_version(), RGPOT_VERSION);
//!   abort();
//! }
//! ```

use std::os::raw::c_char;

static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");

/// Full version string of the loaded library, e.g. `"3.4.0"`.
///
/// The pointer refers to static storage: never free it.
#[no_mangle]
pub extern "C" fn rgpot_version() -> *const c_char {
    VERSION.as_ptr().cast()
}

static REVISION: &str = concat!(env!("RGPOT_SOURCE_REVISION"), "\0");

/// Source revision the loaded library was built from, or an empty string.
///
/// The value is `RGPOT_SOURCE_REVISION` when that is set, otherwise the git
/// short hash of the checkout. A tarball build leaves it empty. The pointer
/// refers to static storage: never free it.
#[no_mangle]
pub extern "C" fn rgpot_source_revision() -> *const c_char {
    REVISION.as_ptr().cast()
}

/// Major version of the loaded library.
#[no_mangle]
pub extern "C" fn rgpot_version_major() -> u32 {
    parse(env!("CARGO_PKG_VERSION_MAJOR"))
}

/// Minor version of the loaded library.
#[no_mangle]
pub extern "C" fn rgpot_version_minor() -> u32 {
    parse(env!("CARGO_PKG_VERSION_MINOR"))
}

/// Patch version of the loaded library.
#[no_mangle]
pub extern "C" fn rgpot_version_patch() -> u32 {
    parse(env!("CARGO_PKG_VERSION_PATCH"))
}

const fn parse(s: &str) -> u32 {
    let b = s.as_bytes();
    let mut v = 0u32;
    let mut i = 0;
    while i < b.len() {
        v = v * 10 + (b[i] - b'0') as u32;
        i += 1;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;

    #[test]
    fn version_string_matches_the_crate() {
        let s = unsafe { CStr::from_ptr(rgpot_version()) };
        assert_eq!(s.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
        let joined = format!(
            "{}.{}.{}",
            rgpot_version_major(),
            rgpot_version_minor(),
            rgpot_version_patch()
        );
        assert!(env!("CARGO_PKG_VERSION").starts_with(&joined));
    }

    #[test]
    fn source_revision_is_the_build_revision() {
        let got = unsafe { CStr::from_ptr(rgpot_source_revision()) }
            .to_str()
            .unwrap();
        assert_eq!(got, env!("RGPOT_SOURCE_REVISION"));
        assert!(got.bytes().all(|b| b.is_ascii_alphanumeric()), "{got:?}");
    }

    /// The committed header must carry this crate's version: cbindgen
    /// stamps it only on regeneration and a release bump stamps it with
    /// potctl, so drift means one of the two was skipped.
    #[test]
    fn committed_header_matches_the_crate() {
        let header =
            std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/include/rgpot.h"))
                .unwrap();
        let want = format!("#define RGPOT_VERSION \"{}\"", env!("CARGO_PKG_VERSION"));
        assert!(header.contains(&want), "rgpot.h lacks {want}");
        for f in [
            "rgpot_version(void)",
            "rgpot_source_revision(void)",
            "rgpot_version_major(void)",
            "rgpot_version_minor(void)",
            "rgpot_version_patch(void)",
        ] {
            assert!(header.contains(f), "rgpot.h lacks {f}");
        }
    }
}
