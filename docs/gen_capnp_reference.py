#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Write docs/orgmode/reference/capnp_schema.org from the effective schema.

The effective schema is ``Potentials.capnp`` with the local diffs of
``CppCore/rgpot/rpc/local/series`` applied, which is the file the build
compiles and serves.  Every struct, interface and enum becomes a section;
every field, union arm, enumerant and method becomes a table row, with its
ordinal, type, default and the comment that the schema carries.

    python docs/gen_capnp_reference.py          rewrite the page
    python docs/gen_capnp_reference.py --check  fail when the page is stale

The page is generated, so it is never edited by hand: change the schema
comment and run this script again.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "CppCore/rgpot/rpc"))

import schema_ledger  # noqa: E402

PAGE = "docs/orgmode/reference/capnp_schema.org"
SCHEMA = "CppCore/rgpot/rpc/Potentials.capnp"
SERIES = "CppCore/rgpot/rpc/local"

GROUPS = (
    ("Wire messages and the service", ("ForceInput", "PotentialResult", "Capabilities", "Potential")),
    ("Configuration carrier", ("PotentialConfig", "PotSpec", "PotentialSpec", "ExprTerm", "ExprParams", "CommonMethodSpec")),
    ("NWChem input", ("NWChem",)),
    ("CPMD input", ("CPMD",)),
)

_DECL = re.compile(r"^\s*(struct|interface|enum)\s+(\w+)")


def effective_schema(root: Path = ROOT) -> str:
    text = (root / SCHEMA).read_text(encoding="utf-8")
    return schema_ledger.apply_series(text, schema_ledger.read_series(root / SERIES))


def _comment_of(line: str) -> str | None:
    text = schema_ledger._strip_comment(line)
    if len(text) < len(line):
        return line[len(text) + 1 :].strip()
    return None


def _clean(text: str) -> str:
    text = re.sub(r"@(brief|return|param\s+\w+)\s*", "", text)
    text = text.replace("|", "/")
    text = re.sub(r"`([^`]*)`", r"=\1=", text)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = text.replace("[[", "[ [").replace("\\", "/")
    return " ".join(text.split())


def _statement_comments(lines: list[str], start: int) -> str:
    """Trailing comments from the line a statement starts on to the one that ends it."""
    pieces = []
    i = start - 1
    while i < len(lines):
        c = _comment_of(lines[i])
        if c:
            pieces.append(c)
        if ";" in schema_ledger._strip_comment(lines[i]):
            break
        i += 1
    return " ".join(pieces)


def _leading_comments(lines: list[str], start: int) -> list[str]:
    """The block of ``#`` lines directly above the 1-based line ``start``."""
    block: list[str] = []
    i = start - 2
    while i >= 0 and lines[i].strip().startswith("#"):
        block.append(lines[i].strip().lstrip("#").strip())
        i -= 1
    block.reverse()
    return block


def _prose(block: list[str]) -> str:
    keep = []
    for raw in block:
        if raw.startswith("---") or re.match(r"@(struct|interface|enum)\b", raw):
            continue
        if raw.startswith("@field"):
            continue
        keep.append(raw)
    return _clean(" ".join(keep))


def _default(lines: list[str], start: int) -> str:
    text = ""
    i = start - 1
    while i < len(lines):
        stripped = schema_ledger._strip_comment(lines[i])
        text += " " + stripped.strip()
        if ";" in stripped:
            break
        i += 1
    depth = 0
    for k, ch in enumerate(text):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            value = text[k + 1 :].split(";")[0].strip()
            return value
    return ""


def _scope_kind(entries) -> str:
    kinds = {e.kind for e in entries}
    if "enumerant" in kinds:
        return "enum"
    if "method" in kinds:
        return "interface"
    return "struct"


def _scope_decl_line(lines: list[str], scope: str, first_entry_line: int) -> int | None:
    leaf = scope.split(".")[-1]
    found = None
    for number in range(1, first_entry_line):
        m = _DECL.match(lines[number - 1])
        if m and m.group(2) == leaf:
            found = number
    return found


OPERATION_METHODS = {
    "energy": "calculateEnergy",
    "forces": "calculateForces",
    "gradient": "calculateGradient",
    "hessian": "calculateHessian",
    "dipole": "calculateDipole",
    "polarizability": "calculatePolarizability",
    "quadrupole": "calculateQuadrupole",
    "stress": "calculateStress",
    "optimize": "calculateOptimize",
    "frequencies": "calculateFrequencies",
    "hessianVector": "calculateHvp",
}

# What the schema leaves unsaid and a reader can read off the neighbours.
NOTES = {
    "PotentialResult.componentsValid": "True when energyComponents holds a harvested vector.",
    "PotentialResult.chargeValid": "True when chargeIntegrals holds a harvested vector.",
    "PotentialResult.multiStateValid": "True when multiStateEnergies holds a harvested vector.",
    "PotentialResult.mdTrajectoryValid": "True when mdTrajectoryRow holds a harvested row.",
    "CommonMethodSpec.charge": "Net charge of the system, in units of the elementary charge.",
    "CommonMethodSpec.smearing": "Occupation smearing; the kind and the width are in Smearing.",
    "CommonMethodSpec.Smearing.kind": "The smearing function; none switches smearing off.",
    "CommonMethodSpec.Smearing.widthEv": "The smearing width in eV.",
    "ExprTerm.name": "Name that the expression text uses for this term.",
    "ExprTerm.pot": "The potential that evaluates the term.",
    "ExprParams.expression": "Expression over the term names, evaluated by the Lepton parser.",
    "ExprParams.terms": "The named terms the expression refers to.",
}


def derived_description(scope: str, entry, kind: str) -> str:
    """A description for a row whose schema line carries no comment."""
    leaf = scope.split(".")[-1]
    note = NOTES.get(f"{scope}.{entry.name}")
    if note:
        return note
    if entry.kind == "arm":
        typ = entry.type[1:] if entry.type.startswith(":") else entry.type
        if typ == "Void":
            return "Selects this arm, which carries no data."
        return f"Selects this arm of the union, which carries a ={typ}=."
    if entry.kind == "enumerant":
        if scope == "Capabilities.Operation" and entry.name in OPERATION_METHODS:
            return f"The operation that =Potential.{OPERATION_METHODS[entry.name]}= serves."
        return f"The ={entry.name}= value of =enum {leaf}=."
    if entry.kind == "method":
        return ""
    if entry.name == "directives":
        return "Further directives of the stanza, each a keyword with its arguments."
    return "No comment in the schema."


def _cell(text: str) -> str:
    return text.replace("|", "/") if text else ""


def _code(text: str) -> str:
    return f"={text}=" if text else ""


def _group_of(scope: str) -> int:
    top = scope.split(".")[0]
    for index, (_title, prefixes) in enumerate(GROUPS):
        if any(top == p or (p in ("NWChem", "CPMD") and top.startswith(p)) for p in prefixes):
            return index
    return len(GROUPS)


def render(root: Path = ROOT) -> str:
    text = effective_schema(root)
    lines = text.splitlines()
    parsed = schema_ledger.parse_schema(text)
    by_scope: dict[str, list[tuple[schema_ledger.Entry, int]]] = {}
    for entry, line in parsed:
        by_scope.setdefault(entry.scope, []).append((entry, line))

    out: list[str] = [
        "#+COMMENT: Copyright 2023--present rgpot developers",
        "#+TITLE: Cap'n Proto schema",
        "#+OPTIONS: toc:nil todo:nil num:nil author:nil title:nil ^:nil",
        "#+STARTUP: content",
        "",
        "* Cap'n Proto schema",
        "",
        "This page is generated by =docs/gen_capnp_reference.py= from the schema the",
        "build compiles, which is =CppCore/rgpot/rpc/Potentials.capnp= with the",
        "diffs of =CppCore/rgpot/rpc/local/series= applied.",
        "Change a comment in the schema, then run the script again; do not edit this",
        "page.",
        "",
        "Cap'n Proto names a field, union arm, enumerant or method on the wire by its",
        "ordinal, so a row keeps its ordinal for good.",
        "The type column is as written in the schema, and the default column is empty",
        "when the schema gives none (zero, empty or false by the Cap'n Proto rule).",
        "A union arm is written =name @N :Type=; exactly one arm is set.",
        "A row that reads \"No comment in the schema\" has none upstream, so the",
        "comment has to be added to the pinned schema before this page can say more.",
        "The meaning of each =PotentialConfig= arm and the ordering of the overlay are",
        "in :doc:`potential_config`.",
        "",
    ]
    grouped: dict[int, list[str]] = {}
    for scope in by_scope:
        grouped.setdefault(_group_of(scope), []).append(scope)
    titles = [title for title, _ in GROUPS] + ["Native potential parameters"]
    for index in sorted(grouped):
        out.append(f"** {titles[index]}")
        out.append("")
        for scope in grouped[index]:
            entries = by_scope[scope]
            kind = _scope_kind([e for e, _ in entries])
            decl = _scope_decl_line(lines, scope, entries[0][1])
            summary = _prose(_leading_comments(lines, decl)) if decl else ""
            out.append(f"*** {kind} {scope}")
            out.append("")
            if summary:
                out.extend(_sentences(summary))
                out.append("")
            if kind == "interface":
                out.append("| Method | Ordinal | Signature | Description |")
            elif kind == "enum":
                out.append("| Enumerant | Ordinal | Description |")
            else:
                out.append("| Member | Ordinal | Type | Default | Description |")
            out.append("|-")
            for entry, line in entries:
                name = f"{scope}.{entry.name}"
                comment = _statement_comments(lines, line)
                if not comment:
                    comment = " ".join(_leading_comments(lines, line))
                    if comment.lstrip().startswith("---"):
                        comment = ""
                description = _clean(comment) or derived_description(scope, entry, kind)
                if kind == "interface":
                    out.append(
                        f"| {_code(name)} | {entry.ordinal} | {_code(_cell(entry.type))} | {_cell(description)} |"
                    )
                elif kind == "enum":
                    out.append(f"| {_code(name)} | {entry.ordinal} | {_cell(description)} |")
                else:
                    default = _default(lines, line) if entry.kind == "field" else ""
                    typ = entry.type[1:] if entry.type.startswith(":") else entry.type
                    label = f"{entry.kind} {typ}" if entry.kind == "arm" else typ
                    out.append(
                        f"| {_code(name)} | {entry.ordinal} | {_code(_cell(label))} | "
                        f"{_code(_cell(default))} | {_cell(description)} |"
                    )
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z=])", text)
    return [p for p in parts if p]


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=lambda p: Path(p).resolve(), default=ROOT)
    parser.add_argument("--check", action="store_true", help="fail when the page is stale")
    args = parser.parse_args(argv)
    page = args.root / PAGE
    text = render(args.root)
    if args.check:
        if not page.is_file() or page.read_text(encoding="utf-8") != text:
            print(f"{PAGE} is stale; run python docs/gen_capnp_reference.py", file=sys.stderr)
            return 1
        return 0
    page.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
