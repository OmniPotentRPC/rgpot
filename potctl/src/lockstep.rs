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

pub fn read_cargo_version(root: &Path) -> Result<String> {
    read_first_toml_version(root, "rgpot-core/Cargo.toml")
}

pub fn read_towncrier_version(root: &Path) -> Result<String> {
    read_first_toml_version(root, "towncrier.toml")
}

pub fn read_pixi_workspace_version(root: &Path) -> Result<String> {
    read_first_toml_version(root, "pixi.toml")
}

pub fn read_pyproject_version(root: &Path) -> Result<String> {
    read_first_toml_version(root, "pyproject.toml")
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

    for rel in [
        "towncrier.toml",
        "rgpot-core/Cargo.toml",
        "pixi.toml",
        "pyproject.toml",
    ] {
        let p = root.join(rel);
        if p.is_file() {
            let t = read_text(&p)?;
            write_text(&p, &replace_first_toml_version_line(&t, version))?;
        }
    }

    println!("synced lockstep version -> {version}");
    println!(
        "  meson.build / CMakeLists.txt / towncrier.toml / rgpot-core/Cargo.toml / pixi.toml [workspace] / pyproject.toml"
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
    let exp = expected.map(strip_v).filter(|s| !s.is_empty());

    println!("lockstep surfaces:");
    println!("  meson.build      = {meson_v}");
    println!("  CMakeLists.txt   = {cmake_v}");
    println!("  Cargo.toml       = {cargo_v}");
    println!("  towncrier.toml   = {town_v}");
    println!("  pixi.toml[ws]    = {pixi_v}");
    println!("  pyproject.toml   = {pyproj_v}");

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

    #[test]
    fn semver_basic() {
        assert!(semver_ok("1.2.0"));
        assert!(semver_ok("v1.2.0"));
        assert!(semver_ok("1.2.0-rc.1"));
        assert!(!semver_ok("not-a-version"));
        assert!(!semver_ok("1.2"));
    }
}
