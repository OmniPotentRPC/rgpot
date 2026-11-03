//! The RGPOT_VERSION block of include/rgpot.h.
//!
//! build.rs writes this block for cbindgen, and ci/version/rgpot_version.py
//! writes the same text for cogapp. Both read MAJOR, MINOR and PATCH from
//! the numeric core of the version, before any pre-release or build
//! suffix, so 3.6.0-rc.1 gives PATCH 0. potctl's tests run both writers on
//! the same versions and fail when they differ.

/// MAJOR, MINOR and PATCH of a semver string such as `3.6.0-rc.1+build.7`.
pub fn numeric_core(version: &str) -> Option<(u64, u64, u64)> {
    let core = version.split(['-', '+']).next()?;
    let mut parts = core.split('.').map(|p| p.parse::<u64>().ok());
    match (parts.next(), parts.next(), parts.next(), parts.next()) {
        (Some(Some(major)), Some(Some(minor)), Some(Some(patch)), None) => {
            Some((major, minor, patch))
        }
        _ => None,
    }
}

/// The four RGPOT_VERSION macros for `version`, one per line.
pub fn header_macros(version: &str) -> String {
    let (major, minor, patch) = numeric_core(version)
        .unwrap_or_else(|| panic!("release version is not semver: {version:?}"));
    format!(
        "#define RGPOT_VERSION \"{version}\"\n\
         #define RGPOT_VERSION_MAJOR {major}\n\
         #define RGPOT_VERSION_MINOR {minor}\n\
         #define RGPOT_VERSION_PATCH {patch}"
    )
}
