#!/usr/bin/env python3
"""Regenerate schema artifacts from the pinned canonical file and diff them.

argv: <repo root> [capnp] [build dir] [canonical Potentials.capnp]

The two vendored copies must be byte-identical and must carry the file id
`@0xbd1f89fa17369103`. When a canonical path is given (the meson subproject
file, after the wrap's diff_files patch), both copies must match it. C++
bindings are regenerated from that file and compared with the bindings
regenerated from each vendored copy and with the files the build wrote.
A CodeGeneratorRequest is the input every language backend sees, including
Rust. A checked-in generated header or Rust module must match the
regeneration.
"""

from __future__ import annotations

import difflib
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FILE_ID = "@0xbd1f89fa17369103;"


def _struct_body(lines: list[str], name: str, rel: Path) -> list[str]:
    needle = f"struct {name} {{"
    try:
        start = lines.index(needle)
    except ValueError as exc:
        raise AssertionError(f"{rel} must expose {name}") from exc
    depth = 0
    body: list[str] = []
    for line in lines[start:]:
        body.append(line)
        depth += line.count("{") - line.count("}")
        if depth == 0:
            return body
    raise AssertionError(f"{rel} has unclosed struct {name}")


def _assert_expr_potspec(schema_path: Path, lines: list[str]) -> None:
    rel = schema_path
    expr_term = "\n".join(_struct_body(lines, "ExprTerm", rel))
    if not re.search(r"name\s+@0\s*:\s*Text", expr_term):
        raise AssertionError(f"{rel} ExprTerm.name must be Text")
    if not re.search(r"pot\s+@1\s*:\s*PotSpec", expr_term):
        raise AssertionError(f"{rel} ExprTerm.pot must be PotSpec")

    expr_params = "\n".join(_struct_body(lines, "ExprParams", rel))
    if not re.search(r"expression\s+@0\s*:\s*Text", expr_params):
        raise AssertionError(f"{rel} ExprParams.expression must be Text")
    if not re.search(r"terms\s+@1\s*:\s*List\(ExprTerm\)", expr_params):
        raise AssertionError(f"{rel} ExprParams.terms must be List(ExprTerm)")

    pot_spec = "\n".join(_struct_body(lines, "PotSpec", rel))
    if not re.search(r"expr\s+@0\s*:\s*ExprParams", pot_spec):
        raise AssertionError(f"{rel} PotSpec.expr must be ExprParams")
    if not re.search(r"none\s+@1\s*:\s*Void", pot_spec):
        raise AssertionError(f"{rel} PotSpec.none must be Void (capnp unions need two arms)")
    if re.search(r"^\s+(lj|d3|d4|nwchem)\s+@", pot_spec, re.MULTILINE):
        raise AssertionError(f"{rel} PotSpec must not invent dummy leaf arms")

    cfg = "\n".join(_struct_body(lines, "PotentialConfig", rel))
    if re.search(r"^\s+(expr|sum|list)\s+@", cfg, re.MULTILINE):
        raise AssertionError(f"{rel} PotentialConfig must not grow expr/sum/list arms")


def _norm(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


def _regenerate_cpp(root: Path, capnp: str, schema: Path, out: Path) -> dict[str, bytes]:
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "CppCore" / "rgpot" / "rpc" / "capnp_compile.py"),
            capnp,
            str(out),
            str(schema.parent),
            str(schema),
        ],
        check=True,
    )
    return {p.name: p.read_bytes() for p in sorted(out.iterdir()) if p.is_file()}


def _request(capnp: str, schema: Path) -> bytes:
    return subprocess.run(
        [capnp, "compile", "-o-", f"--src-prefix={schema.parent}", str(schema)],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def _try_rust(capnp: str, schema: Path, out: Path) -> dict[str, bytes] | None:
    out.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [capnp, "compile", f"-orust:{out}", f"--src-prefix={schema.parent}", str(schema)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        return None
    files = {p.name: p.read_bytes() for p in sorted(out.iterdir()) if p.is_file()}
    if not files:
        return None
    return files


def _assert_generated(root: Path, capnp: str, schema: Path, build_dir: Path | None) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        alone = tmp_path / "src" / "Potentials.capnp"
        alone.parent.mkdir()
        shutil.copyfile(schema, alone)
        fresh = _regenerate_cpp(root, capnp, alone, tmp_path / "cpp")
        if set(fresh) != {"Potentials.capnp.cpp", "Potentials.capnp.h"}:
            raise AssertionError(f"regenerated C++ artifacts are {sorted(fresh)}")
        request = _request(capnp, alone)
        rust = _try_rust(capnp, alone, tmp_path / "rust")

        for copy in (
            root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp",
            root / "rgpot-core" / "schema" / "Potentials.capnp",
        ):
            copy_dir = tmp_path / copy.parent.name
            copy_schema = copy_dir / "Potentials.capnp"
            copy_dir.mkdir()
            shutil.copyfile(copy, copy_schema)
            regenerated = _regenerate_cpp(root, capnp, copy_schema, copy_dir / "gen")
            for name, data in fresh.items():
                if regenerated.get(name) != data:
                    raise AssertionError(
                        f"regenerated {name} from {copy.relative_to(root)} "
                        "differs from the canonical schema"
                    )
            if _request(capnp, copy_schema) != request:
                raise AssertionError(
                    f"CodeGeneratorRequest from {copy.relative_to(root)} "
                    "differs from the canonical schema"
                )
            copy_rust = _try_rust(capnp, copy_schema, copy_dir / "rust")
            if (rust is None) != (copy_rust is None) or rust != copy_rust:
                raise AssertionError(
                    f"regenerated Rust bindings from {copy.relative_to(root)} differ"
                )

        if build_dir is not None:
            built = build_dir / "CppCore" / "rgpot" / "rpc"
            for name, data in fresh.items():
                artifact = built / name
                if not artifact.is_file():
                    raise AssertionError(f"{artifact} was not generated by the build")
                if artifact.read_bytes() != data:
                    raise AssertionError(
                        f"{artifact} differs from bindings regenerated from the canonical schema"
                    )

        skip_parts = {".git", "bbdir", "subprojects", "target", "cargo-target"}
        if build_dir is not None:
            skip_parts.add(build_dir.name)
        for pattern in ("Potentials.capnp.h", "Potentials.capnp.cpp", "Potentials_capnp.rs"):
            for path in root.rglob(pattern):
                if any(part in skip_parts for part in path.parts):
                    continue
                if pattern == "Potentials_capnp.rs":
                    if rust is None or pattern not in rust and "Potentials_capnp.rs" not in rust:
                        raise AssertionError(
                            f"checked-in {path} has no regenerated Rust bindings to compare"
                        )
                    fresh_rust = next(iter(rust.values()))
                    if path.read_bytes() != fresh_rust:
                        raise AssertionError(f"checked-in {path} differs from regeneration")
                elif path.read_bytes() != fresh[pattern]:
                    raise AssertionError(f"checked-in {path} differs from regeneration")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    capnp = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else shutil.which("capnp")
    build_dir = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] else None
    canonical = Path(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] else None
    cpp_schema = root / "CppCore" / "rgpot" / "rpc" / "Potentials.capnp"
    rust_schema = root / "rgpot-core" / "schema" / "Potentials.capnp"

    cpp_text = cpp_schema.read_bytes()
    rust_text = rust_schema.read_bytes()
    if _norm(cpp_text) != _norm(rust_text):
        diff = difflib.unified_diff(
            rust_text.decode().splitlines(),
            cpp_text.decode().splitlines(),
            fromfile=str(rust_schema.relative_to(root)),
            tofile=str(cpp_schema.relative_to(root)),
            lineterm="",
            n=3,
        )
        print("\n".join(list(diff)[:120]))
        raise AssertionError("rgpot-core bundled schema must match C++ RPC schema")

    source = canonical if canonical is not None else cpp_schema
    if canonical is not None and _norm(canonical.read_bytes()) != _norm(cpp_text):
        raise AssertionError(
            "vendored Potentials.capnp differs from the pinned canonical schema"
        )
    text = source.read_text(encoding="utf-8")
    if FILE_ID not in text:
        raise AssertionError(f"{source} must carry file id {FILE_ID}")
    if "buildVersion" not in text or "buildRevision" not in text:
        raise AssertionError(f"{source} must carry buildVersion and buildRevision")
    if "getCapabilities @12" not in text:
        raise AssertionError(f"{source} must carry getCapabilities @12")

    lines = text.splitlines()
    if "enum CPMDSectionKind {" not in lines:
        raise AssertionError(f"{source} must expose CPMDSectionKind")
    _assert_expr_potspec(source, lines)

    if capnp is None:
        raise AssertionError("capnp is required to regenerate and diff the artifacts")
    _assert_generated(root, capnp, source, build_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
