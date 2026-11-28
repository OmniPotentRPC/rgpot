#!/usr/bin/env python3
"""Append-only ordinal ledger for Potentials.capnp.

Cap'n Proto identifies a field, union arm, enumerant or method on the wire
by its ordinal. An ordinal may never be renumbered, reused or given another
type, and peers (gpridge, eOn, rgconf and its other consumers) only agree
when every copy keeps one numbering. Potentials.ordinals records each
ordinal once as

    <scope> @<ordinal> <kind> <name> <type> <origin>

kind is field, arm (union member), group, enumerant or method. origin is
upstream (declared by the pinned schema) or local (rgpot shipped it before
the pinned schema adopted it).

The schema is rgconf's crates/rgconf-core/schema/potential.capnp at the
commit schema.pin names, vendored byte for byte as Potentials.capnp with
the common.capnp it imports beside it, in CppCore/rgpot/rpc and in
rgpot-core/schema. Through rgpot 3.8 the pin was potentials-schema and rgpot
compiled it with an add-only series of local diffs; rgconf now declares
every one of those members at the ordinal rgpot gave it, so the series is
gone and every local ledger entry is retired by a later upstream line for
the same entry. A new member is added to rgconf first and arrives here with
the next pin.

check fails when

* a ledger line changes or disappears relative to --base (append only),
* a local entry has no later upstream line retiring it,
* a vendored copy declares an entry the ledger lacks, or differs from the
  ledger in any entry,
* the two copies of a vendored file differ,
* the pinned schema (given with --upstream, or fetched with --fetch)
  drops, renumbers or retypes a ledger entry or declares one the ledger
  lacks, or a vendored file is not byte for byte the pinned file.

append adds the entries the pinned schema declares and the ledger lacks, at
the end of the ledger, and retires a local entry the pinned schema declares
unchanged by adding the same entry again as upstream; the later line wins.
It never rewrites an existing line.
"""

from __future__ import annotations

import argparse
import configparser
import dataclasses
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

LEDGER = Path("CppCore/rgpot/rpc/Potentials.ordinals")
PIN = Path("CppCore/rgpot/rpc/schema.pin")
SCHEMA_DIRS = (
    Path("CppCore/rgpot/rpc"),
    Path("rgpot-core/schema"),
)
SCHEMA_NAME = "Potentials.capnp"
COMMON_NAME = "common.capnp"
VENDORED = tuple(d / SCHEMA_NAME for d in SCHEMA_DIRS)
VENDORED_COMMON = tuple(d / COMMON_NAME for d in SCHEMA_DIRS)
DIVERGES = "diverges from the pinned rgconf schema"

ORDINAL = re.compile(r"@\d+\b")


@dataclasses.dataclass(frozen=True)
class Entry:
    scope: str
    ordinal: int
    kind: str
    name: str
    type: str
    origin: str = ""

    @property
    def key(self) -> tuple[str, int]:
        return (self.scope, self.ordinal)

    def same(self, other: "Entry") -> bool:
        return (self.scope, self.ordinal, self.kind, self.name, self.type) == (
            other.scope,
            other.ordinal,
            other.kind,
            other.name,
            other.type,
        )

    def describe(self) -> str:
        return f"{self.scope} @{self.ordinal} {self.kind} {self.name} {self.type}"

    def line(self) -> str:
        return f"{self.describe()} {self.origin}"


class LedgerError(Exception):
    pass


# ---------------------------------------------------------------- parsing


def _strip_comment(line: str) -> str:
    quoted = False
    escaped = False
    for i, ch in enumerate(line):
        if escaped:
            escaped = False
        elif ch == "\\" and quoted:
            escaped = True
        elif ch == '"':
            quoted = not quoted
        elif ch == "#" and not quoted:
            return line[:i]
    return line


def _strip_annotations(text: str) -> str:
    """Drop $annotation and $annotation(...) applications."""
    out = []
    i = 0
    while i < len(text):
        if text[i] != "$":
            out.append(text[i])
            i += 1
            continue
        i += 1
        while i < len(text) and (text[i].isalnum() or text[i] in "._"):
            i += 1
        if i < len(text) and text[i] == "(":
            depth = 0
            while i < len(text):
                depth += text[i] == "("
                depth -= text[i] == ")"
                i += 1
                if depth == 0:
                    break
    return "".join(out)


def _statements(text: str):
    """Yield (token, statement, line) with token one of ';', '{', '}'."""
    buf: list[str] = []
    start = 0
    depth = 0
    quoted = False
    for lineno, raw in enumerate(text.splitlines(), 1):
        for ch in _strip_comment(raw) + "\n":
            if quoted:
                buf.append(ch)
                if ch == '"' and (len(buf) < 2 or buf[-2] != "\\"):
                    quoted = False
                continue
            if ch == '"':
                quoted = True
            elif ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            elif depth == 0 and ch in ";{}":
                yield ch, " ".join("".join(buf).split()), start
                buf = []
                start = 0
                continue
            if not start and not ch.isspace():
                start = lineno
            buf.append(ch)


_DECL = re.compile(r"^(struct|enum|interface)\s+(\w+)")
_GROUP = re.compile(r"^(\w+)\s*(?:@(\d+)\s*)?:\s*(union|group)$")
_FIELD = re.compile(r"^(\w+)\s*@(\d+)\s*:\s*(.+)$")
_ENUMERANT = re.compile(r"^(\w+)\s*@(\d+)$")
_METHOD = re.compile(r"^(\w+)\s*@(\d+)\s*(\(.*)$")


def _field_type(rest: str) -> str:
    depth = 0
    for i, ch in enumerate(rest):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            rest = rest[:i]
            break
    return ":" + "".join(rest.split())


def parse_schema(text: str) -> list[tuple[Entry, int]]:
    """Every ordinal a Cap'n Proto schema declares, with its source line."""
    # frame: kind, ordinal owner scope, name prefix, decl path
    stack: list[tuple[str, str, str, str]] = [("file", "", "", "")]
    found: list[tuple[Entry, int]] = []
    for token, stmt, line in _statements(text):
        kind, owner, prefix, path = stack[-1]
        stmt = " ".join(_strip_annotations(stmt).split())
        if token == "}":
            if stmt:
                raise LedgerError(f"line {line}: unterminated statement {stmt!r}")
            if len(stack) == 1:
                raise LedgerError(f"line {line}: unbalanced closing brace")
            stack.pop()
            continue
        if token == "{":
            decl = _DECL.match(stmt)
            group = _GROUP.match(stmt)
            if decl:
                name = f"{path}.{decl.group(2)}" if path else decl.group(2)
                stack.append((decl.group(1), name, "", name))
            elif stmt == "union" and kind in ("struct", "group", "union"):
                stack.append(("union", owner, prefix, path))
            elif group and kind in ("struct", "group", "union"):
                gname, gord, gkind = group.groups()
                if gord is not None:
                    container = "arm" if kind == "union" else "group"
                    found.append((Entry(owner, int(gord), container, prefix + gname, gkind), line))
                stack.append(("union" if gkind == "union" else "group", owner, f"{prefix}{gname}.", path))
            else:
                stack.append(("other", owner, prefix, path))
            continue
        if not stmt or kind in ("file", "other"):
            continue
        if stmt.split()[0] in ("using", "const", "annotation"):
            continue
        if kind in ("struct", "group", "union"):
            m = _FIELD.match(stmt)
            if m:
                container = "arm" if kind == "union" else "field"
                found.append(
                    (
                        Entry(
                            owner, int(m.group(2)), container, prefix + m.group(1), _field_type(m.group(3))
                        ),
                        line,
                    )
                )
            elif ORDINAL.search(stmt):
                raise LedgerError(f"line {line}: cannot read {stmt!r}")
        elif kind == "enum":
            m = _ENUMERANT.match(stmt)
            if not m:
                raise LedgerError(f"line {line}: cannot read enumerant {stmt!r}")
            found.append((Entry(owner, int(m.group(2)), "enumerant", m.group(1), "-"), line))
        elif kind == "interface":
            m = _METHOD.match(stmt)
            if not m:
                raise LedgerError(f"line {line}: cannot read method {stmt!r}")
            found.append(
                (Entry(owner, int(m.group(2)), "method", m.group(1), "".join(m.group(3).split())), line)
            )
    if len(stack) != 1:
        raise LedgerError(f"unclosed {stack[-1][0]} {stack[-1][3]}")
    return found


def schema_entries(text: str, label: str) -> list[Entry]:
    try:
        found = [entry for entry, _ in parse_schema(text)]
    except LedgerError as exc:
        raise LedgerError(f"{label}: {exc}") from exc
    _unique(found, label)
    return found


def _unique(entries: list[Entry], label: str) -> None:
    keys: dict[tuple[str, int], Entry] = {}
    names: dict[tuple[str, str], Entry] = {}
    for entry in entries:
        if entry.key in keys:
            raise LedgerError(
                f"{label}: {entry.describe()} reuses the ordinal of {keys[entry.key].describe()}"
            )
        twin = names.get((entry.scope, entry.name))
        if twin is not None:
            raise LedgerError(f"{label}: {entry.describe()} reuses the name of {twin.describe()}")
        keys[entry.key] = entry
        names[(entry.scope, entry.name)] = entry


# ----------------------------------------------------------------- ledger

_LEDGER_LINE = re.compile(r"^(\S+) @(\d+) (field|arm|group|enumerant|method) (\S+) (\S+) (upstream|local)$")


def parse_ledger(text: str, label: str) -> list[Entry]:
    entries = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LEDGER_LINE.match(line)
        if not m:
            raise LedgerError(f"{label}:{lineno}: malformed ledger line {raw!r}")
        scope, ordinal, kind, name, kind_type, origin = m.groups()
        entries.append(Entry(scope, int(ordinal), kind, name, kind_type, origin))
    seen: dict[tuple[str, int], Entry] = {}
    for entry in entries:
        old = seen.get(entry.key)
        if old is not None and not _retires(old, entry):
            raise LedgerError(f"{label}: {entry.line()} reuses the ordinal of {old.line()}")
        seen[entry.key] = entry
    _unique(resolve_ledger(entries), label)
    return entries


def _retires(old: Entry, new: Entry) -> bool:
    """A later upstream line for a local entry retires it once upstream adopts it."""
    return old.origin == "local" and new.origin == "upstream" and old.same(new)


def resolve_ledger(entries: list[Entry]) -> list[Entry]:
    """One entry per ordinal: the last line for it, at the place of the first."""
    latest = {e.key: e for e in entries}
    out = []
    for entry in entries:
        if latest.get(entry.key) is not None:
            out.append(latest.pop(entry.key))
    return out


LEDGER_HEADER = """\
# Append-only ordinal ledger for Potentials.capnp (see schema_ledger.py).
# <scope> @<ordinal> <kind> <name> <type> <origin>
# upstream: declared by the pinned schema (potentials-schema through rgpot 3.8,
# rgconf's potential.capnp after).
# local: shipped by rgpot before the pinned schema adopted it; a later
# upstream line retires it. Never edit or delete a line; append only.
"""


# ------------------------------------------------------------------- inputs


@dataclasses.dataclass
class Pin:
    url: str
    revision: str
    schema: str
    common: str


def read_pin(root: Path) -> Pin:
    parser = configparser.ConfigParser()
    if not parser.read(root / PIN, encoding="utf-8"):
        raise LedgerError(f"{PIN} is missing")
    section = parser["rgconf"]
    return Pin(section["url"], section["revision"], section["schema"], section["common"])


def _git(*args: str, cwd: Path | None = None) -> str:
    # The schema is UTF-8; the locale's code page (cp1252 on Windows) would
    # turn its non-ASCII comments into different text.
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
    ).stdout


def fetch_pinned(pin: Pin) -> tuple[str, str]:
    """The pinned potential.capnp and common.capnp, fetched from the pin url."""
    with tempfile.TemporaryDirectory() as tmp:
        _git("init", "-q", cwd=Path(tmp))
        _git("fetch", "-q", "--depth", "1", pin.url, pin.revision, cwd=Path(tmp))
        head = _git("rev-parse", "FETCH_HEAD", cwd=Path(tmp)).strip()
        if head != pin.revision:
            raise LedgerError(f"fetched {head}, {PIN} pins {pin.revision}")
        return (
            _git("show", f"{pin.revision}:{pin.schema}", cwd=Path(tmp)),
            _git("show", f"{pin.revision}:{pin.common}", cwd=Path(tmp)),
        )


def base_ledger(root: Path, base: str) -> str | None:
    try:
        _git("rev-parse", "--verify", "--quiet", f"{base}^{{commit}}", cwd=root)
    except subprocess.CalledProcessError as exc:
        raise LedgerError(f"base revision {base} is not a commit here") from exc
    try:
        return _git("show", f"{base}:{LEDGER.as_posix()}", cwd=root)
    except subprocess.CalledProcessError:
        return None


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def pinned_texts(args: argparse.Namespace, pin: Pin) -> tuple[str, str | None, str] | None:
    """(potential.capnp, common.capnp or None, source) of the pin, if given."""
    if args.upstream:
        common = args.upstream_common or args.upstream.parent / COMMON_NAME
        common_text = _read(common) if common.is_file() else None
        return _read(args.upstream), common_text, str(args.upstream)
    if getattr(args, "fetch", False):
        schema, common = fetch_pinned(pin)
        return (
            schema.replace("\r\n", "\n"),
            common.replace("\r\n", "\n"),
            f"{pin.url} at {pin.revision}",
        )
    return None


# ------------------------------------------------------------------- checks


def check_retired(ledger_lines: list[Entry]) -> list[str]:
    """Every local entry has a later upstream line for the same entry."""
    latest = {e.key: e for e in ledger_lines}
    return [
        f"local {e.describe()} is not retired: rgpot no longer carries local schema members; "
        "add it to rgconf's potential.capnp, re-pin and run append"
        for e in resolve_ledger(ledger_lines)
        if latest[e.key].origin == "local"
    ]


def check_upstream(ledger: list[Entry], upstream: list[Entry]) -> list[str]:
    errors = []
    up_key = {e.key: e for e in upstream}
    led_key = {e.key: e for e in ledger}
    for entry in ledger:
        theirs = up_key.get(entry.key)
        if theirs is None:
            errors.append(f"pinned schema no longer declares {entry.line()}")
        elif not theirs.same(entry):
            errors.append(f"pinned schema has {theirs.describe()}, ledger has {entry.line()}")
    for entry in upstream:
        if entry.key not in led_key:
            errors.append(f"pinned schema declares {entry.describe()}, not in the ledger (run append)")
    return errors


def check_append_only(base: list[Entry], current: list[Entry]) -> list[str]:
    errors = []

    def ident(e: Entry) -> tuple[tuple[str, int], str]:
        return (e.key, e.origin)

    position = {ident(e): i for i, e in enumerate(current)}
    old_keys = {ident(e) for e in base}
    last = -1
    for old in base:
        at = position.get(ident(old))
        if at is None:
            at = position.get((old.key, "upstream" if old.origin == "local" else "local"))
        if at is None:
            errors.append(f"ledger entry disappeared: {old.line()}")
        elif current[at] != old:
            errors.append(f"ledger entry changed: {old.line()} is now {current[at].line()}")
        elif at < last:
            errors.append(f"ledger entry moved: {old.line()}")
        else:
            last = at
    for entry in current[: last + 1]:
        if ident(entry) not in old_keys:
            errors.append(f"ledger entry inserted before existing ones: {entry.line()} (append at the end)")
    return errors


def check_copy(ledger: list[Entry], copy: list[Entry], label: str) -> list[str]:
    errors = []
    mine = {e.key: e for e in copy}
    led = {e.key: e for e in ledger}
    for entry in ledger:
        have = mine.get(entry.key)
        if have is None:
            errors.append(f"{label} lacks {entry.line()}")
        elif not have.same(entry):
            errors.append(f"{label} has {have.describe()}, ledger has {entry.line()}")
    for entry in copy:
        if entry.key not in led:
            errors.append(f"{label} declares {entry.describe()}, not in the ledger (run append)")
    return errors


def run_check(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    pin = read_pin(root)
    ledger_path = args.ledger or root / LEDGER
    lines = parse_ledger(ledger_path.read_text(encoding="utf-8"), str(ledger_path))
    ledger = resolve_ledger(lines)
    errors: list[str] = []
    notes: list[str] = []

    if args.base:
        old = base_ledger(root, args.base)
        if old is None:
            notes.append(f"{LEDGER} does not exist at {args.base}; append-only check skipped")
        else:
            errors += check_append_only(parse_ledger(old, f"{args.base}:{LEDGER}"), lines)
    errors += check_retired(lines)

    texts = {rel: _read(root / rel) for rel in VENDORED}
    commons = {rel: _read(root / rel) for rel in VENDORED_COMMON}
    for rel, text in texts.items():
        errors += check_copy(ledger, schema_entries(text, str(rel)), str(rel))
    for group in (texts, commons):
        (first_rel, first), *rest = group.items()
        for rel, text in rest:
            if text != first:
                errors.append(f"{rel} differs from {first_rel}")

    pinned = pinned_texts(args, pin)
    if pinned is None:
        notes.append("no pinned schema given; the comparison with the rgconf pin is skipped")
    else:
        schema_text, common_text, source = pinned
        notes.append(f"pinned schema: {source}")
        errors += check_upstream(ledger, schema_entries(schema_text, "pinned schema"))
        for rel, text in texts.items():
            if text != schema_text:
                errors.append(f"{rel} {DIVERGES} {pin.revision[:8]}")
        if common_text is not None:
            for rel, text in commons.items():
                if text != common_text:
                    errors.append(f"{rel} {DIVERGES} {pin.revision[:8]}")

    for note in notes:
        print(f"note: {note}", flush=True)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        print(f"{len(errors)} ordinal ledger violation(s)", file=sys.stderr)
        return 1
    print(f"ordinal ledger OK: {len(ledger)} entries")
    return 0


def run_append(args: argparse.Namespace) -> int:
    root = args.root.resolve()
    pin = read_pin(root)
    ledger_path = args.ledger or root / LEDGER
    text = ledger_path.read_text(encoding="utf-8") if ledger_path.exists() else LEDGER_HEADER
    ledger = resolve_ledger(parse_ledger(text, str(ledger_path)))
    pinned = pinned_texts(args, pin)
    schema_text = pinned[0] if pinned is not None else fetch_pinned(pin)[0]
    known = {e.key: e for e in ledger}
    fresh = []
    for entry in schema_entries(schema_text, "pinned schema"):
        old = known.get(entry.key)
        if old is None:
            fresh.append(dataclasses.replace(entry, origin="upstream"))
            continue
        if not old.same(entry):
            raise LedgerError(
                f"pinned schema has {entry.describe()}, ledger has {old.line()}; "
                "append will not rewrite it"
            )
        if old.origin == "local":
            fresh.append(dataclasses.replace(entry, origin="upstream"))
    if fresh:
        if not text.endswith("\n"):
            text += "\n"
        text += "".join(e.line() + "\n" for e in fresh)
        ledger_path.write_text(text, encoding="utf-8", newline="\n")
    print(f"appended {len(fresh)} entries to {ledger_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("check", "append"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
        cmd.add_argument("--ledger", type=Path)
        source = cmd.add_mutually_exclusive_group()
        source.add_argument("--upstream", type=Path, help="the pinned rgconf potential.capnp")
        source.add_argument("--fetch", action="store_true", help="fetch the pinned revision from the pin url")
        cmd.add_argument(
            "--upstream-common",
            type=Path,
            help="the pinned common.capnp (default: common.capnp beside --upstream)",
        )
    sub.choices["check"].add_argument("--base", help="git revision whose ledger must be a prefix of this one")
    args = parser.parse_args(argv)
    try:
        return {"check": run_check, "append": run_append}[args.command](args)
    except (LedgerError, OSError, subprocess.CalledProcessError, KeyError) as exc:
        detail = getattr(exc, "stderr", None)
        print(f"error: {exc}{': ' + detail.strip() if detail else ''}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    os.environ.setdefault("GIT_TERMINAL_PROMPT", "0")
    raise SystemExit(main())
