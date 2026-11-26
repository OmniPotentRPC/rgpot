#!/usr/bin/env python3
# Copyright 2023--present rgpot developers
"""Inventory the public surface and report where the documentation covers it.

The surfaces are

``potential``
    Every ``rgpot::PotType`` enumerator.  A potential is covered when a
    how-to page under ``docs/orgmode/howto/potentials-*.org`` has a section
    headed by its name in ``=...=`` marks.
``c_abi``
    Every function and type of the C headers listed in ``doccheck.C_HEADERS``.
``cpp``
    Every public C++ class, struct, enum and free function ``doccheck``
    discovers.
``python``
    Every name of ``rgpot`` and ``rgpot.ase``, with the members of its classes.
``capnp``
    Every struct, interface, enum, field, union arm, enumerant and method of
    the vendored Cap'n Proto schema the build compiles.
    The generated page ``reference/capnp_schema.org`` lists each.
``rust``
    Every ``pub`` item of ``rgpot-core`` and ``rgpot-pair``, which the
    rustdoc pages of the Sphinx build render from the doc comments, and every
    ``potctl`` command.
``tool``
    The command-line programs the build installs.

An item has a reference *entry* when one of those pages documents it, is
*mentioned* when its name occurs in some other page, and is otherwise
*missing*.  ``report`` prints the three counts per surface; ``errors`` lists
the missing items, which the ``inventory`` check of ``doccheck.py`` turns into
a failure.

The module uses the standard library only.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "CppCore/rgpot/rpc"))

import doccheck  # noqa: E402

CATEGORIES = ("potential", "c_abi", "cpp", "python", "capnp", "rust", "tool")

POT_TYPES_HEADER = "CppCore/rgpot/pot_types.hpp"
POTENTIAL_PAGES = "docs/orgmode/howto/potentials-*.org"
CAPNP_PAGE = "docs/orgmode/reference/capnp_schema.org"
CAPNP_SCHEMA = "CppCore/rgpot/rpc/Potentials.capnp"
RUST_CRATES = ("rgpot-core", "rgpot-pair")
SPHINX_CONF = "docs/source/conf.py"
POTCTL_MAIN = "potctl/src/main.rs"
POTCTL_PAGE = "docs/orgmode/reference/cli.org"
# Programs the build produces for users, all documented on one page.
TOOLS = {
    name: "docs/orgmode/reference/cli.org"
    for name in doccheck.PROGRAMS
}


@dataclass(frozen=True)
class Item:
    category: str
    name: str
    source: str
    kind: str = ""


# ------------------------------------------------------------ doc text


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def doc_sources(root: Path = ROOT) -> dict[str, str]:
    """Relative path -> text of every documentation page."""
    return {doccheck.rel(p, root): _read(p) for p in doccheck.doc_pages(root)}


_TOKENS: dict[int, tuple[str, set[str]]] = {}


def _token_set(text: str) -> set[str]:
    cached = _TOKENS.get(id(text))
    if cached is not None and cached[0] is text:
        return cached[1]
    tokens = set(re.findall(r"\w+", text))
    tokens.update(re.findall(r"\w+(?:\.\w+)+", text))
    _TOKENS[id(text)] = (text, tokens)
    return tokens


def _word(name: str, text: str) -> bool:
    """True when ``name`` occurs in ``text`` as a whole word or dotted name."""
    if re.fullmatch(r"\w+(?:\.\w+)*", name):
        return name in _token_set(text)
    return re.search(rf"(?<![\w.]){re.escape(name)}(?!\w)", text) is not None


def _mention_text(root: Path, skip: set[str]) -> str:
    return "\n".join(t for p, t in doc_sources(root).items() if p not in skip)


# ------------------------------------------------------------ potentials


def pot_types(root: Path = ROOT) -> list[str]:
    text = doccheck.strip_c_comments(_read(root / POT_TYPES_HEADER))
    body = text.split("enum class PotType", 1)[1].split("}", 1)[0]
    body = body.split("{", 1)[1]
    names = [part.split("=")[0].strip() for part in body.split(",")]
    return [n for n in names if n and n != "UNKNOWN"]


def potential_page_text(root: Path = ROOT) -> str:
    return "\n".join(_read(p) for p in sorted(root.glob(POTENTIAL_PAGES)))


def potential_sections(root: Path = ROOT) -> dict[str, str]:
    """PotType name -> text of the how-to section headed by =Name=."""
    sections: dict[str, str] = {}
    for page in sorted(root.glob(POTENTIAL_PAGES)):
        lines = _read(page).splitlines()
        current = None
        level = 0
        for line in lines:
            m = re.match(r"^(\*+)\s+(.*)$", line)
            if m:
                depth = len(m.group(1))
                names = re.findall(r"=([A-Za-z0-9_]+)=", m.group(2))
                if current is not None and depth <= level:
                    current = None
                if names and depth >= 2:
                    current = names[0]
                    level = depth
                    sections.setdefault(current, "")
                    # a heading may name several aliases of one potential
                    for alias in names[1:]:
                        sections.setdefault(alias, "")
                    continue
            if current is not None:
                sections[current] += line + "\n"
    return sections


# ----------------------------------------------------------- C/C++/Python


def _reference_covered(root: Path) -> set[str]:
    entries = doccheck.reference_entries(root)
    covered = {e.name for e in entries if not e.name.startswith(".")}
    cpp = doccheck.cpp_public_names(root)
    py = doccheck.py_public_names(root)
    for entry in entries:
        if not entry.name.startswith("."):
            continue
        for target in doccheck._applies_to(entry):
            sep = "::" if "::" in target else "."
            member = f"{target}{sep}{entry.name[1:]}"
            if member in py or member in cpp:
                covered.add(member)
    return covered


# ------------------------------------------------------------- Cap'n Proto


def capnp_effective(root: Path = ROOT) -> str:
    return _read(root / CAPNP_SCHEMA)


def capnp_entries(root: Path = ROOT):
    import schema_ledger

    return schema_ledger.parse_schema(capnp_effective(root))


def capnp_scopes(root: Path = ROOT) -> list[str]:
    seen: list[str] = []
    for entry, _ in capnp_entries(root):
        if entry.scope not in seen:
            seen.append(entry.scope)
    return seen


def capnp_page_names(root: Path = ROOT) -> tuple[set[str], set[str]]:
    """(scopes, Scope.member names) the generated schema page lists."""
    page = root / CAPNP_PAGE
    if not page.is_file():
        return set(), set()
    scopes: set[str] = set()
    members: set[str] = set()
    for line in _read(page).splitlines():
        hm = re.match(r"^\*{3,4}\s+(?:struct|interface|enum|union)\s+(\S+)\s*$", line)
        if hm:
            scopes.add(hm.group(1))
            continue
        rm = re.match(r"^\|\s*=([\w.]+)=\s*\|", line)
        if rm:
            members.add(rm.group(1))
    return scopes, members


# -------------------------------------------------------------------- Rust

_RUST_ITEM = re.compile(
    r'^\s*pub\s+(?:unsafe\s+|extern\s+"C"\s+|const\s+|async\s+)*'
    r"(fn|struct|enum|trait|type|const|static|mod|union)\s+(\w+)"
)


def rust_items(root: Path = ROOT, crate: str = "rgpot-core") -> list[tuple[str, str, str, bool]]:
    """(file, kind, name, documented) for each ``pub`` item outside tests."""
    out = []
    for path in sorted((root / crate / "src").rglob("*.rs")):
        lines = _read(path).splitlines()
        tests_from = len(lines)
        for i, line in enumerate(lines):
            if re.match(r"\s*#\[cfg\(test\)\]", line):
                tests_from = i
                break
        for i, line in enumerate(lines[:tests_from]):
            m = _RUST_ITEM.match(line)
            if not m or re.match(r"\s*pub\((crate|super|self)", line):
                continue
            kind, name = m.groups()
            j = i - 1
            while j >= 0:
                above = lines[j].strip()
                if above.startswith("#["):
                    j -= 1
                elif above.endswith(")]"):
                    while j >= 0 and not lines[j].strip().startswith("#["):
                        j -= 1
                    j -= 1
                else:
                    break
            documented = j >= 0 and lines[j].strip().startswith(("///", "*/", "#[doc"))
            if kind == "mod" and not documented:
                target = [path.parent / f"{name}.rs", path.parent / name / "mod.rs"]
                documented = any(t.is_file() and "//!" in _read(t) for t in target)
            out.append((doccheck.rel(path, root), kind, name, documented))
    return out


def sphinx_rust_crates(root: Path = ROOT) -> set[str]:
    text = _read(root / SPHINX_CONF)
    m = re.search(r"rust_crates\s*=\s*\{(.*?)\}", text, re.S)
    if not m:
        return set()
    return {
        k.replace("_", "-") for k in re.findall(r'"([\w-]+)"\s*:', m.group(1))
    }


_CLAP_ENUM = re.compile(r"enum\s+(\w+)\s*\{(.*?)\n\}", re.S)


def potctl_commands(root: Path = ROOT) -> list[str]:
    """``release sync`` style names of every potctl subcommand."""
    names: dict[str, list[str]] = {}
    texts = [_read(root / "potctl/src/main.rs")]
    for m in _CLAP_ENUM.finditer("\n".join(texts)):
        variants = re.findall(r"^    ([A-Z]\w*)\s*[,{(]", m.group(2), re.M)
        names[m.group(1)] = variants

    def kebab(word: str) -> str:
        return re.sub(r"(?<!^)(?=[A-Z])", "-", word).lower()

    out = []
    for top in names.get("Commands", []):
        group = top
        sub = names.get(f"{top}Cmd")
        if sub:
            out.extend(f"{kebab(group)} {kebab(v)}" for v in sub)
        else:
            out.append(kebab(group))
    return out


# --------------------------------------------------------------- inventory


def inventory(root: Path = ROOT) -> list[tuple[Item, str]]:
    """Every public item with its coverage: entry, mention or missing."""
    sources = doc_sources(root)
    out: list[tuple[Item, str]] = []

    # potentials
    sections = potential_sections(root)
    skip = {p for p in sources if re.match(r"docs/orgmode/howto/potentials-", p)}
    other = _mention_text(root, skip)
    for name in pot_types(root):
        item = Item("potential", name, POT_TYPES_HEADER, "enumerator")
        if name in sections:
            out.append((item, "entry"))
        elif _word(name, other):
            out.append((item, "mention"))
        else:
            out.append((item, "missing"))

    # C, C++ and Python names
    covered = _reference_covered(root)
    prose = _mention_text(root, set())
    cnames = doccheck.c_public_names(root)
    cppnames = doccheck.cpp_public_names(root)
    pynames = doccheck.py_public_names(root)

    def classify(item: Item, short: str, name: str) -> tuple[Item, str]:
        if name in covered:
            return item, "entry"
        if _word(short, prose):
            return item, "mention"
        return item, "missing"

    for name, info in sorted(cnames.items()):
        out.append(classify(Item("c_abi", name, info.header, info.kind), name, name))
    for name, info in sorted(cppnames.items()):
        out.append(
            classify(Item("cpp", name, info.header, info.kind), name.split("::")[-1], name)
        )
    for name, info in sorted(pynames.items()):
        out.append(classify(Item("python", name, doccheck.PY_INIT, info.kind), name.split(".")[-1], name))

    # Cap'n Proto
    scopes, members = capnp_page_names(root)
    capnp_text = "\n".join(t for p, t in sources.items() if p != CAPNP_PAGE)
    for scope in capnp_scopes(root):
        item = Item("capnp", scope, CAPNP_SCHEMA, "scope")
        if scope in scopes:
            out.append((item, "entry"))
        elif _word(scope.split(".")[-1], capnp_text):
            out.append((item, "mention"))
        else:
            out.append((item, "missing"))
    for entry, _line in capnp_entries(root):
        name = f"{entry.scope}.{entry.name}"
        item = Item("capnp", name, CAPNP_SCHEMA, entry.kind)
        if name in members:
            out.append((item, "entry"))
        elif _word(name, capnp_text):
            out.append((item, "mention"))
        else:
            out.append((item, "missing"))

    # Rust
    crates = sphinx_rust_crates(root)
    for crate in RUST_CRATES:
        for path, kind, name, documented in rust_items(root, crate):
            item = Item("rust", f"{crate}::{name}", path, kind)
            out.append((item, "entry" if documented and crate in crates else "missing"))
    page = root / POTCTL_PAGE
    page_text = _read(page) if page.is_file() else ""
    for command in potctl_commands(root):
        item = Item("rust", f"potctl {command}", POTCTL_MAIN, "command")
        if _word(f"potctl {command}", page_text) or _word(f"={command}=", page_text):
            out.append((item, "entry"))
        elif _word(f"potctl {command}", prose) or _word(command, prose):
            out.append((item, "mention"))
        else:
            out.append((item, "missing"))

    # tools
    for tool, page_name in TOOLS.items():
        item = Item("tool", tool, page_name, "program")
        page = root / page_name
        text = _read(page) if page.is_file() else ""
        if re.search(rf"(?m)^\*+\s+.*={re.escape(tool)}=", text):
            out.append((item, "entry"))
        elif _word(tool, prose):
            out.append((item, "mention"))
        else:
            out.append((item, "missing"))
    return out


def counts(items: list[tuple[Item, str]]) -> dict[str, dict[str, int]]:
    table = {c: {"entry": 0, "mention": 0, "missing": 0} for c in CATEGORIES}
    for item, status in items:
        table[item.category][status] += 1
    return table


def report(items: list[tuple[Item, str]]) -> str:
    table = counts(items)
    lines = [
        f"{'surface':<10} {'items':>6} {'entry':>6} {'mention':>8} {'missing':>8} {'covered':>8}",
    ]
    totals = {"entry": 0, "mention": 0, "missing": 0}
    for cat in CATEGORIES:
        row = table[cat]
        n = sum(row.values())
        pct = 100.0 * row["entry"] / n if n else 100.0
        lines.append(
            f"{cat:<10} {n:>6} {row['entry']:>6} {row['mention']:>8} "
            f"{row['missing']:>8} {pct:>7.1f}%"
        )
        for key in totals:
            totals[key] += row[key]
    n = sum(totals.values())
    pct = 100.0 * totals["entry"] / n if n else 100.0
    lines.append(
        f"{'all':<10} {n:>6} {totals['entry']:>6} {totals['mention']:>8} "
        f"{totals['missing']:>8} {pct:>7.1f}%"
    )
    return "\n".join(lines)


def errors(root: Path = ROOT) -> list[str]:
    """One message per public item without a reference entry."""
    out = []
    for item, status in inventory(root):
        if status == "entry":
            continue
        what = {
            "potential": "no how-to section under docs/orgmode/howto/potentials-*.org for",
            "capnp": f"{CAPNP_PAGE} has no entry for",
            "rust": "no rustdoc or reference page for",
            "tool": "no reference heading for",
        }.get(item.category, "no reference entry for")
        out.append(f"{what} {item.category} {item.name} ({item.source})")
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=lambda p: Path(p).resolve(), default=ROOT)
    parser.add_argument("--missing", action="store_true", help="list the missing items")
    args = parser.parse_args(argv)
    items = inventory(args.root)
    print(report(items))
    if args.missing:
        for item, status in items:
            if status != "entry":
                print(f"{status:8} {item.category:9} {item.name}  ({item.source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
