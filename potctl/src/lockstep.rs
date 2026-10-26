//! Lockstep semver surfaces for the rgpot monorepo
//! (meson / CMake / cargo / towncrier / pixi / pyproject).

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub type Result<T> = std::result::Result<T, String>;

pub fn repo_root() -> Result<PathBuf> {
    let mut d = env::current_dir().map_err(|e| e.to_string())?;
    for _ in 0..12 {
        if d.join("meson.build").is_file()
            && d.join("pixi.toml").is_file()
            && d.join("rgpot-core").join("Cargo.toml").is_file()
        {
            return Ok(d);
        }
        if !d.pop() {
            break;
        }
    }
    Err("could not find rgpot repo root (meson.build + pixi.toml + rgpot-core/) from cwd".into())
}

pub fn semver_ok(v: &str) -> bool {
    let v = v.strip_prefix(['v', 'V']).unwrap_or(v);
    if v.is_empty() {
        return false;
    }
    let bytes = v.as_bytes();
    let mut i = 0usize;
    let mut dots = 0u32;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == 0 {
        return false;
    }
    while i < bytes.len() && bytes[i] == b'.' {
        dots += 1;
        i += 1;
        let start = i;
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i == start {
            return false;
        }
    }
    if dots != 2 {
        return false;
    }
    if i == bytes.len() {
        return true;
    }
    if bytes[i] == b'.' || bytes[i] == b'-' {
        i += 1;
        if i >= bytes.len() {
            return false;
        }
        while i < bytes.len() {
            let c = bytes[i] as char;
            if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                i += 1;
            } else {
                return false;
            }
        }
        return true;
    }
    false
}

pub fn strip_v(v: &str) -> &str {
    v.strip_prefix(['v', 'V']).unwrap_or(v)
}

fn read_text(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))
}

fn write_text(path: &Path, content: &str) -> Result<()> {
    fs::write(path, content).map_err(|e| format!("write {}: {e}", path.display()))
}

fn line_indent(line: &str) -> &str {
    let n = line.len() - line.trim_start().len();
    &line[..n]
}

fn line_eol(line: &str) -> &str {
    if line.ends_with("\r\n") {
        "\r\n"
    } else if line.ends_with('\n') {
        "\n"
    } else {
        ""
    }
}

fn between_quotes(s: &str) -> Option<&str> {
    let a = s.find('"')?;
    let b = s.rfind('"')?;
    if b > a {
        Some(&s[a + 1..b])
    } else {
        None
    }
}

fn between_single_quotes(s: &str) -> Option<&str> {
    let a = s.find('\'')?;
    let b = s.rfind('\'')?;
    if b > a {
        Some(&s[a + 1..b])
    } else {
        None
    }
}

pub fn read_meson_version(root: &Path) -> Result<String> {
    for line in read_text(&root.join("meson.build"))?.lines() {
        let s = line.trim();
        if s.starts_with("version:") {
            if let Some(q) = between_single_quotes(s) {
                return Ok(q.to_string());
            }
        }
    }
    Err("meson.build: no version: line".into())
}

pub fn read_cmake_version(root: &Path) -> Result<String> {
    for line in read_text(&root.join("CMakeLists.txt"))?.lines() {
        let s = line.trim();
        if let Some(rest) = s.strip_prefix("VERSION ") {
            let tok = rest.split_whitespace().next().unwrap_or("");
            if tok.starts_with(|c: char| c.is_ascii_digit()) {
                return Ok(tok.to_string());
            }
        }
    }
    Err("CMakeLists.txt: no VERSION line".into())
}

fn read_first_toml_version(root: &Path, rel: &str) -> Result<String> {
    let path = root.join(rel);
    for line in read_text(&path)?.lines() {
        let s = line.trim();
        if s.starts_with("version = \"") {
            if let Some(q) = between_quotes(s) {
                return Ok(q.to_string());
            }
        }
    }
    Err(format!("{rel}: no version = \"...\" line"))
}

/// Table name of a TOML header line: `[project]` -> `project`,
/// `[[tool.towncrier.type]]` -> `tool.towncrier.type`.
fn toml_table_name(line: &str) -> Option<&str> {
    let s = line.trim();
    let inner = s.strip_prefix('[')?.strip_suffix(']')?;
    let inner = match inner.strip_prefix('[') {
        Some(i) => i.strip_suffix(']')?,
        None => inner,
    };
    Some(inner.trim())
}

/// `version = "..."` belonging to `[section]` only, so dependency pins and
/// other tables cannot be mistaken for the package version.
fn toml_section_version(text: &str, section: &str) -> Option<String> {
    let mut here = false;
    for line in text.lines() {
        if let Some(name) = toml_table_name(line) {
            here = name == section;
            continue;
        }
        if here && line.trim().starts_with("version = \"") {
            if let Some(q) = between_quotes(line.trim()) {
                return Some(q.to_string());
            }
        }
    }
    None
}

fn read_toml_section_version(root: &Path, rel: &str, section: &str) -> Result<String> {
    let text = read_text(&root.join(rel))?;
    toml_section_version(&text, section)
        .ok_or_else(|| format!("{rel}: no version = \"...\" under [{section}]"))
}

pub fn read_cargo_version(root: &Path) -> Result<String> {
    read_first_toml_version(root, "rgpot-core/Cargo.toml")
}

pub fn read_towncrier_version(root: &Path) -> Result<String> {
    read_first_toml_version(root, "towncrier.toml")
}

pub fn read_pixi_workspace_version(root: &Path) -> Result<String> {
    read_toml_section_version(root, "pixi.toml", "workspace")
}

pub fn read_pyproject_version(root: &Path) -> Result<String> {
    read_toml_section_version(root, "pyproject.toml", "project")
}

/// Generated C header whose RGPOT_VERSION macros stamp the ABI.
const HEADER_REL: &str = "rgpot-core/include/rgpot.h";

/// RGPOT_VERSION from the generated C header, or None when the tree has
/// no header (a scratch root in tests).
pub fn read_header_version(root: &Path) -> Result<Option<String>> {
    let p = root.join(HEADER_REL);
    if !p.is_file() {
        return Ok(None);
    }
    for line in read_text(&p)?.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("#define RGPOT_VERSION ") {
            if let Some(v) = between_quotes(rest) {
                return Ok(Some(v.to_string()));
            }
        }
    }
    Err(format!("{HEADER_REL}: no #define RGPOT_VERSION line"))
}

/// Rewrite the four RGPOT_VERSION macros. cbindgen writes them from
/// CARGO_PKG_VERSION, but only when the header is regenerated, so a bump
/// must stamp them too or the header keeps the previous release.
fn replace_header_versions(text: &str, version: &str) -> String {
    let core = version.split(['-', '+']).next().unwrap_or(version);
    let mut parts = core.split('.');
    let major = parts.next().unwrap_or("0");
    let minor = parts.next().unwrap_or("0");
    let patch = parts.next().unwrap_or("0");
    let mut out = String::with_capacity(text.len());
    for line in text.split_inclusive('\n') {
        let eol = line_eol(line);
        let body = &line[..line.len() - eol.len()];
        let replaced = match body.trim_start() {
            l if l.starts_with("#define RGPOT_VERSION_MAJOR ") => {
                Some(format!("#define RGPOT_VERSION_MAJOR {major}"))
            }
            l if l.starts_with("#define RGPOT_VERSION_MINOR ") => {
                Some(format!("#define RGPOT_VERSION_MINOR {minor}"))
            }
            l if l.starts_with("#define RGPOT_VERSION_PATCH ") => {
                Some(format!("#define RGPOT_VERSION_PATCH {patch}"))
            }
            l if l.starts_with("#define RGPOT_VERSION ") => {
                Some(format!("#define RGPOT_VERSION \"{version}\""))
            }
            _ => None,
        };
        match replaced {
            Some(r) => {
                out.push_str(&r);
                out.push_str(eol);
            }
            None => out.push_str(line),
        }
    }
    out
}

pub fn changelog_section(root: &Path, ver: &str) -> Result<String> {
    let path = root.join("CHANGELOG.md");
    if !path.is_file() {
        return Ok(String::new());
    }
    let needle = format!("## [{ver}]");
    let mut grab = false;
    let mut lines = Vec::new();
    for line in read_text(&path)?.lines() {
        if line.starts_with(&needle) {
            grab = true;
            lines.push(line.to_string());
            continue;
        }
        if grab && line.starts_with("## [") {
            break;
        }
        if grab {
            lines.push(line.to_string());
        }
    }
    let mut out = lines.join("\n");
    if !out.is_empty() && !out.ends_with('\n') {
        out.push('\n');
    }
    Ok(out)
}

fn replace_first_meson_version(text: &str, version: &str) -> String {
    let mut done = false;
    let mut out = String::with_capacity(text.len() + 8);
    for line in text.split_inclusive('\n') {
        let s = line.trim_start_matches([' ', '\t']);
        if !done && s.starts_with("version:") {
            let ind = line_indent(line.trim_end_matches(['\r', '\n']));
            let eol = line_eol(line);
            out.push_str(ind);
            out.push_str(&format!("version: '{version}',"));
            out.push_str(eol);
            done = true;
        } else {
            out.push_str(line);
        }
    }
    out
}

fn replace_first_cmake_version(text: &str, version: &str) -> String {
    let mut done = false;
    let mut out = String::with_capacity(text.len() + 8);
    for line in text.split_inclusive('\n') {
        let trimmed = line.trim_start_matches([' ', '\t']);
        if !done && trimmed.starts_with("VERSION ") {
            let rest = &trimmed["VERSION ".len()..];
            let tok = rest.split_whitespace().next().unwrap_or("");
            if tok.starts_with(|c: char| c.is_ascii_digit()) {
                let ind = line_indent(line.trim_end_matches(['\r', '\n']));
                let eol = line_eol(line);
                out.push_str(ind);
                out.push_str("VERSION ");
                out.push_str(version);
                out.push_str(eol);
                done = true;
                continue;
            }
        }
        out.push_str(line);
    }
    out
}

fn replace_first_toml_version_line(text: &str, version: &str) -> String {
    let mut done = false;
    let mut out = String::with_capacity(text.len() + 8);
    for line in text.split_inclusive('\n') {
        if !done && line.trim().starts_with("version = \"") {
            let ind = line_indent(line.trim_end_matches(['\r', '\n']));
            let eol = line_eol(line);
            out.push_str(ind);
            out.push_str(&format!("version = \"{version}\""));
            out.push_str(eol);
            done = true;
        } else {
            out.push_str(line);
        }
    }
    out
}

fn replace_toml_section_version_line(text: &str, section: &str, version: &str) -> String {
    let mut done = false;
    let mut here = false;
    let mut out = String::with_capacity(text.len() + 8);
    for line in text.split_inclusive('\n') {
        let bare = line.trim_end_matches(['\r', '\n']);
        if let Some(name) = toml_table_name(bare) {
            here = name == section;
            out.push_str(line);
            continue;
        }
        if !done && here && bare.trim().starts_with("version = \"") {
            out.push_str(line_indent(bare));
            out.push_str(&format!("version = \"{version}\""));
            out.push_str(line_eol(line));
            done = true;
            continue;
        }
        out.push_str(line);
    }
    out
}

pub fn sync_versions(root: &Path, version: &str) -> Result<()> {
    let version = strip_v(version);
    if !semver_ok(version) {
        return Err(format!("refusing non-semver version: {version}"));
    }

    let meson_p = root.join("meson.build");
    if meson_p.is_file() {
        let t = read_text(&meson_p)?;
        write_text(&meson_p, &replace_first_meson_version(&t, version))?;
    }

    let cmake_p = root.join("CMakeLists.txt");
    if cmake_p.is_file() {
        let t = read_text(&cmake_p)?;
        write_text(&cmake_p, &replace_first_cmake_version(&t, version))?;
    }

    for rel in ["towncrier.toml", "rgpot-core/Cargo.toml"] {
        let p = root.join(rel);
        if p.is_file() {
            let t = read_text(&p)?;
            write_text(&p, &replace_first_toml_version_line(&t, version))?;
        }
    }

    for (rel, section) in [("pixi.toml", "workspace"), ("pyproject.toml", "project")] {
        let p = root.join(rel);
        if p.is_file() {
            let t = read_text(&p)?;
            write_text(&p, &replace_toml_section_version_line(&t, section, version))?;
        }
    }

    let header_p = root.join(HEADER_REL);
    if header_p.is_file() {
        let t = read_text(&header_p)?;
        write_text(&header_p, &replace_header_versions(&t, version))?;
    }

    println!("synced lockstep version -> {version}");
    println!(
        "  meson.build / CMakeLists.txt / towncrier.toml / rgpot-core/Cargo.toml / pixi.toml [workspace] / pyproject.toml [project] / rgpot.h"
    );
    Ok(())
}

pub fn assert_lockstep(root: &Path, expected: Option<&str>, require_changelog: bool) -> Result<()> {
    let meson_v = read_meson_version(root)?;
    let cmake_v = read_cmake_version(root)?;
    let cargo_v = read_cargo_version(root)?;
    let town_v = read_towncrier_version(root)?;
    let pixi_v = read_pixi_workspace_version(root)?;
    let pyproj_v = read_pyproject_version(root)?;
    let header_v = read_header_version(root)?;
    let exp = expected.map(strip_v).filter(|s| !s.is_empty());

    println!("lockstep surfaces:");
    println!("  meson.build      = {meson_v}");
    println!("  CMakeLists.txt   = {cmake_v}");
    println!("  Cargo.toml       = {cargo_v}");
    println!("  towncrier.toml   = {town_v}");
    println!("  pixi.toml[ws]    = {pixi_v}");
    println!("  pyproject[proj]  = {pyproj_v}");
    if let Some(h) = &header_v {
        println!("  rgpot.h          = {h}");
    }

    let refv = &meson_v;
    for (name, val) in [
        ("cmake", &cmake_v),
        ("cargo", &cargo_v),
        ("towncrier", &town_v),
        ("pixi", &pixi_v),
        ("pyproject", &pyproj_v),
    ] {
        if val != refv {
            return Err(format!("{name} version ({val}) != meson ({refv})"));
        }
    }
    if let Some(h) = &header_v {
        if h != refv {
            return Err(format!("rgpot.h version ({h}) != meson ({refv})"));
        }
    }

    if let Some(e) = exp {
        println!("  expected (tag)   = {e}");
        if refv != e {
            return Err(format!("lockstep version ({refv}) != expected ({e})"));
        }
    }

    if require_changelog {
        let ver = exp.unwrap_or(refv.as_str());
        let section = changelog_section(root, ver)?;
        if section.trim().is_empty() {
            return Err(format!(
                "no CHANGELOG.md section for {ver} (run towncrier build via cog bump first)"
            ));
        }
        println!("  CHANGELOG.md     = section present for {ver}");
    }

    println!("lockstep ok ({refv})");
    Ok(())
}

pub fn extract_changelog(root: &Path, ver_raw: &str, outfile: Option<&Path>) -> Result<()> {
    let ver = strip_v(ver_raw);
    let section = changelog_section(root, ver)?;
    if section.trim().is_empty() {
        return Err(format!("no CHANGELOG.md section for {ver}"));
    }
    if let Some(p) = outfile {
        write_text(p, &section)?;
    } else {
        let mut stdout = io::stdout().lock();
        stdout
            .write_all(section.as_bytes())
            .map_err(|e| e.to_string())?;
        if !section.ends_with('\n') {
            stdout.write_all(b"\n").map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const PYPROJECT: &str = concat!(
        "[project]\n",
        "name = \"rgpot\"\n",
        "version = \"2.5.3\"\n",
        "dependencies = [\n",
        "  \"numpy>=1.24\",\n",
        "]\n",
        "\n",
        "[project.optional-dependencies]\n",
        "metatomic = [\n",
        "  \"vesin>=0.6.0\",\n",
        "]\n",
        "\n",
        "[tool.some-plugin]\n",
        "version = \"9.9.9\"\n",
    );

    fn scratch_root(tag: &str) -> PathBuf {
        let root = env::temp_dir().join(format!("potctl-lockstep-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("rgpot-core")).unwrap();
        fs::write(
            root.join("meson.build"),
            "project(\n    'rgpot',\n    'cpp',\n    version: '2.5.3',\n)\n",
        )
        .unwrap();
        fs::write(
            root.join("CMakeLists.txt"),
            "project(\n  rgpot\n  VERSION 2.5.3\n)\n",
        )
        .unwrap();
        fs::write(
            root.join("rgpot-core/Cargo.toml"),
            "[package]\nname = \"rgpot-core\"\nversion = \"2.5.3\"\n",
        )
        .unwrap();
        fs::write(
            root.join("towncrier.toml"),
            "[tool.towncrier]\nname = \"rgpot\"\nversion = \"2.5.3\"\n",
        )
        .unwrap();
        fs::write(
            root.join("pixi.toml"),
            "[workspace]\nname = \"rgpot\"\nversion = \"2.5.3\"\n",
        )
        .unwrap();
        fs::write(root.join("pyproject.toml"), PYPROJECT).unwrap();
        root
    }

    #[test]
    fn semver_basic() {
        assert!(semver_ok("1.2.0"));
        assert!(semver_ok("v1.2.0"));
        assert!(semver_ok("1.2.0-rc.1"));
        assert!(!semver_ok("not-a-version"));
        assert!(!semver_ok("1.2"));
    }

    #[test]
    fn toml_table_name_handles_plain_and_array_headers() {
        assert_eq!(toml_table_name("[project]"), Some("project"));
        assert_eq!(toml_table_name("  [workspace]  "), Some("workspace"));
        assert_eq!(
            toml_table_name("[[tool.towncrier.type]]"),
            Some("tool.towncrier.type")
        );
        assert_eq!(toml_table_name("version = \"1.0.0\""), None);
    }

    #[test]
    fn pyproject_version_comes_from_project_table() {
        assert_eq!(
            toml_section_version(PYPROJECT, "project").as_deref(),
            Some("2.5.3")
        );
        assert_eq!(
            toml_section_version(PYPROJECT, "tool.some-plugin").as_deref(),
            Some("9.9.9")
        );
        assert_eq!(toml_section_version(PYPROJECT, "build-system"), None);
    }

    #[test]
    fn section_write_touches_only_the_named_table() {
        let out = replace_toml_section_version_line(PYPROJECT, "project", "3.0.0");
        assert!(out.contains("[project]\nname = \"rgpot\"\nversion = \"3.0.0\"\n"));
        assert!(out.contains("\"vesin>=0.6.0\""));
        assert!(out.contains("[tool.some-plugin]\nversion = \"9.9.9\"\n"));
        assert_eq!(
            toml_section_version(&out, "project").as_deref(),
            Some("3.0.0")
        );
    }

    #[test]
    fn sync_writes_pyproject_project_version() {
        let root = scratch_root("sync");
        sync_versions(&root, "v3.1.4").unwrap();
        assert_eq!(read_pyproject_version(&root).unwrap(), "3.1.4");
        assert_eq!(read_pixi_workspace_version(&root).unwrap(), "3.1.4");
        assert_eq!(read_meson_version(&root).unwrap(), "3.1.4");
        let text = fs::read_to_string(root.join("pyproject.toml")).unwrap();
        assert!(text.contains("\"vesin>=0.6.0\""));
        assert!(text.contains("[tool.some-plugin]\nversion = \"9.9.9\""));
        assert_lockstep(&root, Some("3.1.4"), false).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    const HEADER: &str = concat!(
        "#include <stdint.h>\n",
        "\n",
        "#define RGPOT_VERSION \"2.5.3\"\n",
        "#define RGPOT_VERSION_MAJOR 2\n",
        "#define RGPOT_VERSION_MINOR 5\n",
        "#define RGPOT_VERSION_PATCH 3\n",
        "\n",
        "const char *rgpot_version(void);\n",
    );

    #[test]
    fn sync_stamps_the_header_macros() {
        let root = scratch_root("header");
        fs::create_dir_all(root.join("rgpot-core/include")).unwrap();
        fs::write(root.join(HEADER_REL), HEADER).unwrap();
        sync_versions(&root, "4.0.1-rc.2").unwrap();
        let text = fs::read_to_string(root.join(HEADER_REL)).unwrap();
        assert!(text.contains("#define RGPOT_VERSION \"4.0.1-rc.2\"\n"));
        assert!(text.contains("#define RGPOT_VERSION_MAJOR 4\n"));
        assert!(text.contains("#define RGPOT_VERSION_MINOR 0\n"));
        assert!(text.contains("#define RGPOT_VERSION_PATCH 1\n"));
        assert!(text.contains("const char *rgpot_version(void);\n"));
        assert_eq!(read_header_version(&root).unwrap().as_deref(), Some("4.0.1-rc.2"));
        assert_lockstep(&root, Some("4.0.1-rc.2"), false).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn assert_catches_header_drift() {
        let root = scratch_root("header-drift");
        fs::create_dir_all(root.join("rgpot-core/include")).unwrap();
        fs::write(root.join(HEADER_REL), HEADER.replace("\"2.5.3\"", "\"2.5.2\"")).unwrap();
        let err = assert_lockstep(&root, None, false).unwrap_err();
        assert!(err.contains("rgpot.h"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn assert_catches_pyproject_drift() {
        let root = scratch_root("drift");
        let drifted = replace_toml_section_version_line(PYPROJECT, "project", "2.5.2");
        fs::write(root.join("pyproject.toml"), &drifted).unwrap();
        let err = assert_lockstep(&root, None, false).unwrap_err();
        assert!(err.contains("pyproject"), "{err}");
        assert!(err.contains("2.5.2"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn assert_reports_missing_pyproject_project_version() {
        let root = scratch_root("missing");
        fs::write(
            root.join("pyproject.toml"),
            "[build-system]\nrequires = []\n",
        )
        .unwrap();
        let err = assert_lockstep(&root, None, false).unwrap_err();
        assert!(err.contains("pyproject.toml"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }
}
