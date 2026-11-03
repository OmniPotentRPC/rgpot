//! Release version surfaces for the rgpot monorepo.
//!
//! `[workspace.package] version` in the root Cargo.toml is the only
//! hand-written copy. The Cargo members inherit it, cargo writes it into
//! Cargo.lock, and cogapp regenerates the files in ci/version/files.txt
//! from it. `sync` writes the canonical copy; `assert` checks the rest.

use std::env;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

#[path = "../../rgpot-core/version_macros.rs"]
mod version_macros;

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

/// Workspace members whose `[package]` inherits the release version.
const MEMBERS: [&str; 2] = ["rgpot-core", "rgpot-pair"];

/// The canonical release version, `[workspace.package]` in the root Cargo.toml.
pub fn read_workspace_version(root: &Path) -> Result<String> {
    read_toml_section_version(root, "Cargo.toml", "workspace.package")
}

/// Error unless `[package]` of a member says `version.workspace = true`.
fn check_member_inherits(root: &Path, member: &str) -> Result<()> {
    let rel = format!("{member}/Cargo.toml");
    let text = read_text(&root.join(&rel))?;
    let mut here = false;
    for line in text.lines() {
        if let Some(name) = toml_table_name(line) {
            here = name == "package";
            continue;
        }
        let s: String = line.chars().filter(|c| !c.is_whitespace()).collect();
        if here && s == "version.workspace=true" {
            return Ok(());
        }
        if here && s.starts_with("version=") {
            return Err(format!(
                "{rel}: [package] writes its own version; use version.workspace = true"
            ));
        }
    }
    Err(format!("{rel}: [package] has no version.workspace = true"))
}

/// Version that Cargo.lock records for `name`, or None when the tree has
/// no Cargo.lock or the lock has no entry for it.
fn read_lock_version(root: &Path, name: &str) -> Result<Option<String>> {
    let p = root.join("Cargo.lock");
    if !p.is_file() {
        return Ok(None);
    }
    let needle = format!("name = \"{name}\"");
    let text = read_text(&p)?;
    let mut lines = text.lines();
    while let Some(line) = lines.next() {
        if line.trim() == needle {
            if let Some(next) = lines.next() {
                if next.trim().starts_with("version = \"") {
                    return Ok(between_quotes(next.trim()).map(str::to_owned));
                }
            }
        }
    }
    Ok(None)
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

/// How to bring every generated copy back in step with Cargo.toml.
const REGENERATE: &str = "run pixi r -e version release-cog and cargo update --workspace";

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

/// Error unless the four RGPOT_VERSION macros of the C header are the
/// block that rgpot-core/build.rs and cogapp write for `canon`.
fn check_header_macros(root: &Path, canon: &str) -> Result<()> {
    let p = root.join(HEADER_REL);
    if !p.is_file() {
        return Ok(());
    }
    if version_macros::numeric_core(canon).is_none() {
        return Err(format!("release version is not semver: {canon}"));
    }
    let text = read_text(&p)?;
    for want in version_macros::header_macros(canon).lines() {
        let name: Vec<&str> = want.split_whitespace().take(2).collect();
        let found = text
            .lines()
            .map(str::trim)
            .find(|line| line.split_whitespace().take(2).eq(name.iter().copied()));
        match found {
            Some(line) if line == want => {}
            Some(line) => {
                return Err(format!(
                    "{HEADER_REL}: `{line}` should be `{want}`; {REGENERATE}"
                ))
            }
            None => return Err(format!("{HEADER_REL}: no {} line", name.join(" "))),
        }
    }
    Ok(())
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
    let p = root.join("Cargo.toml");
    let t = read_text(&p)?;
    if toml_section_version(&t, "workspace.package").is_none() {
        return Err("Cargo.toml: no version = \"...\" under [workspace.package]".into());
    }
    write_text(
        &p,
        &replace_toml_section_version_line(&t, "workspace.package", version),
    )?;
    println!("wrote [workspace.package] version = {version} in Cargo.toml");
    println!("  regenerate the copies: pixi r -e version release-cog");
    Ok(())
}

pub fn assert_lockstep(root: &Path, expected: Option<&str>, require_changelog: bool) -> Result<()> {
    let canon = read_workspace_version(root)?;
    let exp = expected.map(strip_v).filter(|s| !s.is_empty());

    println!("release version (Cargo.toml [workspace.package]) = {canon}");
    for member in MEMBERS {
        if root.join(member).join("Cargo.toml").is_file() {
            check_member_inherits(root, member)?;
            println!("  {member:<26} = inherited");
        }
    }

    let mut copies = vec![
        ("meson.build".to_owned(), read_meson_version(root)?),
        ("CMakeLists.txt".to_owned(), read_cmake_version(root)?),
        ("towncrier.toml".to_owned(), read_towncrier_version(root)?),
        ("pixi.toml".to_owned(), read_pixi_workspace_version(root)?),
        ("pyproject.toml".to_owned(), read_pyproject_version(root)?),
    ];
    if let Some(h) = read_header_version(root)? {
        copies.push((HEADER_REL.to_owned(), h));
    }
    for member in MEMBERS {
        if let Some(v) = read_lock_version(root, member)? {
            copies.push((format!("Cargo.lock {member}"), v));
        }
    }
    for (name, val) in &copies {
        println!("  {name:<26} = {val}");
        if val != &canon {
            return Err(format!(
                "{name} version ({val}) != Cargo.toml [workspace.package] ({canon}); {REGENERATE}"
            ));
        }
    }
    check_header_macros(root, &canon)?;

    if let Some(e) = exp {
        println!("  {:<26} = {e}", "expected (tag)");
        if canon != e {
            return Err(format!("release version ({canon}) != expected ({e})"));
        }
    }

    if require_changelog {
        let ver = exp.unwrap_or(canon.as_str());
        let section = changelog_section(root, ver)?;
        if section.trim().is_empty() {
            return Err(format!(
                "no CHANGELOG.md section for {ver} (run towncrier build via cog bump first)"
            ));
        }
        println!("  {:<26} = section present for {ver}", "CHANGELOG.md");
    }

    println!("lockstep ok ({canon})");
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
            root.join("Cargo.toml"),
            "[workspace]\nmembers = [\"rgpot-core\"]\n\n[workspace.package]\nversion = \"2.5.3\"\n",
        )
        .unwrap();
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
            "[package]\nname = \"rgpot-core\"\nversion.workspace = true\n",
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
    fn sync_writes_only_the_workspace_version() {
        let root = scratch_root("sync");
        sync_versions(&root, "v3.1.4").unwrap();
        assert_eq!(read_workspace_version(&root).unwrap(), "3.1.4");
        assert_eq!(read_pyproject_version(&root).unwrap(), "2.5.3");
        assert_eq!(read_meson_version(&root).unwrap(), "2.5.3");
        let text = fs::read_to_string(root.join("rgpot-core/Cargo.toml")).unwrap();
        assert!(text.contains("version.workspace = true"));
        let err = assert_lockstep(&root, Some("3.1.4"), false).unwrap_err();
        assert!(err.contains("meson.build"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn sync_refuses_a_tree_without_a_workspace_version() {
        let root = scratch_root("no-canon");
        fs::write(root.join("Cargo.toml"), "[workspace]\nmembers = []\n").unwrap();
        let err = sync_versions(&root, "3.1.4").unwrap_err();
        assert!(err.contains("[workspace.package]"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn assert_accepts_a_tree_in_step() {
        let root = scratch_root("in-step");
        assert_lockstep(&root, Some("2.5.3"), false).unwrap();
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn assert_rejects_a_member_with_its_own_version() {
        let root = scratch_root("member");
        fs::write(
            root.join("rgpot-core/Cargo.toml"),
            "[package]\nname = \"rgpot-core\"\nversion = \"2.5.3\"\n",
        )
        .unwrap();
        let err = assert_lockstep(&root, None, false).unwrap_err();
        assert!(err.contains("version.workspace"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn assert_catches_lockfile_drift() {
        let root = scratch_root("lock");
        fs::write(
            root.join("Cargo.lock"),
            "[[package]]\nname = \"rgpot-core\"\nversion = \"2.5.2\"\n",
        )
        .unwrap();
        let err = assert_lockstep(&root, None, false).unwrap_err();
        assert!(err.contains("Cargo.lock"), "{err}");
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
    fn assert_catches_header_drift() {
        let root = scratch_root("header-drift");
        fs::create_dir_all(root.join("rgpot-core/include")).unwrap();
        fs::write(
            root.join(HEADER_REL),
            HEADER.replace("\"2.5.3\"", "\"2.5.2\""),
        )
        .unwrap();
        let err = assert_lockstep(&root, None, false).unwrap_err();
        assert!(err.contains("rgpot.h"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn assert_catches_a_wrong_numeric_header_macro() {
        let root = scratch_root("header-patch");
        fs::create_dir_all(root.join("rgpot-core/include")).unwrap();
        fs::write(root.join(HEADER_REL), HEADER).unwrap();
        assert_lockstep(&root, None, false).unwrap();
        fs::write(
            root.join(HEADER_REL),
            HEADER.replace("RGPOT_VERSION_PATCH 3", "RGPOT_VERSION_PATCH 3-rc"),
        )
        .unwrap();
        let err = assert_lockstep(&root, None, false).unwrap_err();
        assert!(err.contains("RGPOT_VERSION_PATCH 3-rc"), "{err}");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn header_macros_take_the_numeric_patch_of_a_pre_release() {
        assert_eq!(version_macros::numeric_core("3.6.0-rc.1"), Some((3, 6, 0)));
        assert_eq!(
            version_macros::numeric_core("3.6.0+build.7"),
            Some((3, 6, 0))
        );
        assert_eq!(version_macros::numeric_core("3.6"), None);
        assert!(
            version_macros::header_macros("3.6.0-rc.1").ends_with("#define RGPOT_VERSION_PATCH 0")
        );
    }

    /// Versions on which the cbindgen and cogapp writers must agree.
    const HEADER_CASES: [&str; 6] = [
        "3.6.0",
        "3.6.0-rc.1",
        "3.6.0-rc.1+build.7",
        "10.20.30+meta",
        "1.0.0-0.3.7",
        "1.0.0-x-y-z.--",
    ];

    fn cogapp_header_macros(version: Option<&str>) -> String {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let script = root.join("ci/version/rgpot_version.py");
        let python = env::var("PYTHON").unwrap_or_else(|_| "python3".into());
        let out = std::process::Command::new(&python)
            .arg(&script)
            .arg("header-macros")
            .args(version)
            .output()
            .unwrap_or_else(|e| panic!("run {python} {}: {e}", script.display()));
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap().trim_end().to_owned()
    }

    #[test]
    fn header_macros_match_the_cogapp_writer() {
        for version in HEADER_CASES {
            assert_eq!(
                cogapp_header_macros(Some(version)),
                version_macros::header_macros(version),
                "{version}"
            );
        }
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
        let canon = read_workspace_version(root).unwrap();
        assert_eq!(
            cogapp_header_macros(None),
            version_macros::header_macros(&canon)
        );
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
