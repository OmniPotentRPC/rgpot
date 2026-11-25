#!/usr/bin/env python3
"""Compare the tests a configured build registered with registry.txt.

registry.txt lists every test meson and CMake register, the features that
enable it and its options. Each build system runs this script as the
test_list_parity test and passes the features it was configured with:

  check_test_registry.py meson BUILD_DIR FEATURE=0|1...
  check_test_registry.py cmake BUILD_DIR --ctest CTEST FEATURE=0|1...
  check_test_registry.py lint
  check_test_registry.py selftest

It fails when a registered test has no entry, when an entry whose gate
holds was not registered, when a test was registered although its gate
does not hold, when CMake registered a meson-only entry, and when a
test's timeout, suite, working directory, environment, serial flag,
should_fail flag or command arguments differ from its entry. Command
arguments include the Catch2 tags a unit or rerun passes. Meson's
introspection does not report should_fail, so only CMake's is compared.
coverage_serial is serial only when the build instruments for coverage.
When ThreadTeamTest.cc is registered, the union of each thread-limit
family's tag filters must list the same tests as `--list-tests` with no
filter, so an untagged case, a new tag, a dropped tag or a deleted shard
fails. lint checks the registry alone.

--list FILE (or RGPOT_TEST_LIST) writes the features that gate the entries
both build systems share, then the shared entries the build registered, so
a meson and a CMake build can be compared: where the features agree, the
tests must too.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = Path(__file__).resolve().with_name("registry.txt")
DEFAULT_TIMEOUT = 30
# Minimum positional arguments each kind takes. Unit, rerun and py entries may
# append command-line arguments for the registered executable or script.
ARITY = {"unit": 2, "fortran": 2, "nlist": 2, "rerun": 1, "custom": 0}
VARIABLE_ARITY = {"unit", "rerun", "py"}
SOURCE_DIRS = {
    "unit": "CppCore/tests",
    "fortran": "CppCore/rgpot/fortran",
    "nlist": "CppCore/examples",
}
PROBE = "probe"
WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.+-]*")


@dataclass
class Entry:
    name: str
    line: int
    gate: list[list[str]]
    kind: str
    args: list[str] = field(default_factory=list)
    timeout: int = DEFAULT_TIMEOUT
    suite: str | None = None
    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    serial: bool = False
    coverage_serial: bool = False
    should_fail: bool = False
    reason: str | None = None

    def gate_text(self) -> str:
        return " ".join(",".join(clause) for clause in self.gate)

    def holds(self, features: dict[str, bool], probe: bool) -> bool:
        """probe stands for an optional program: True asks whether the test
        may be registered, False whether it must be."""
        return all(
            any(probe if atom == PROBE else features[atom] for atom in clause)
            for clause in self.gate
        )


@dataclass
class Test:
    name: str
    timeout: float | None
    suites: set[str]
    workdir: str | None
    env: dict[str, str]
    serial: bool
    should_fail: bool | None
    command: list[str] = field(default_factory=list)


def parse_options(entry: Entry, words: list[str], where: str) -> list[str]:
    errors = []
    options = False
    for word in words:
        key, eq, value = word.partition("=")
        if not eq and word not in ("serial", "should_fail", "coverage_serial"):
            if options:
                errors.append(f"{where}: argument {word!r} after an option")
            entry.args.append(word)
            continue
        options = True
        if word in ("serial", "should_fail", "coverage_serial"):
            setattr(entry, word, True)
        elif key == "timeout" and value.isdigit() and int(value) > 0:
            entry.timeout = int(value)
        elif key == "suite" and WORD.fullmatch(value):
            entry.suite = value
        elif key == "workdir" and value:
            entry.workdir = value
        elif key == "env" and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=[^=]*", value):
            var, _, setting = value.partition("=")
            entry.env[var] = setting
        elif word == "fp_contract=off" and entry.kind == "unit":
            pass
        else:
            errors.append(f"{where}: bad option {word!r}")
    return errors


def parse(path: Path) -> tuple[dict[str, Entry], list[str]]:
    entries: dict[str, Entry] = {}
    reasons: dict[str, str] = {}
    errors: list[str] = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        where = f"{path.name}:{number}"
        # CMake reads the file as a list of lines, comments included.
        if ";" in line or line.count("[") != line.count("]"):
            errors.append(f"{where}: ';' or an unmatched bracket splits CMake's list of lines")
            continue
        if not line or line.startswith("#"):
            continue
        if '"' in line:
            errors.append(f"{where}: '\"' quotes words when CMake splits an entry")
            continue
        if line.startswith("@"):
            key, _, text = line[1:].partition(" ")
            if not WORD.fullmatch(key) or not text.strip():
                errors.append(f"{where}: expected @reason text")
            elif key in reasons:
                errors.append(f"{where}: @{key} defined twice")
            reasons[key] = text.strip()
            continue
        fields = [part.strip() for part in line.split("|")]
        if len(fields) != 4:
            errors.append(f"{where}: expected name | gate | kind | reason")
            continue
        name, gate, spec, reason = fields
        words = spec.split()
        if not name or name in entries:
            errors.append(f"{where}: missing or repeated name {name!r}")
            continue
        clauses = [clause.split(",") for clause in gate.split()]
        if not clauses or not all(WORD.fullmatch(a) for c in clauses for a in c):
            errors.append(f"{where}: bad gate {gate!r}")
            continue
        if not words or words[0] not in (*ARITY, "py"):
            errors.append(f"{where}: unknown kind in {spec!r}")
            continue
        entry = Entry(name=name, line=number, gate=clauses, kind=words[0])
        errors += parse_options(entry, words[1:], where)
        want = ARITY.get(entry.kind)
        wrong_arity = (
            not entry.args
            if want is None
            else len(entry.args) < want
            or (entry.kind not in VARIABLE_ARITY and len(entry.args) != want)
        )
        if wrong_arity:
            errors.append(f"{where}: wrong number of arguments for {entry.kind}")
        elif entry.kind in SOURCE_DIRS:
            source = ROOT / SOURCE_DIRS[entry.kind] / entry.args[1]
            if not source.is_file():
                errors.append(f"{where}: no {source.relative_to(ROOT)}")
        elif entry.kind == "py" and not (ROOT / entry.args[0]).is_file():
            errors.append(f"{where}: no {entry.args[0]}")
        if entry.kind == "unit" and entry.workdir is None:
            entry.workdir = "."
        if entry.workdir not in (None, ".") and not (ROOT / entry.workdir).is_dir():
            errors.append(f"{where}: no directory {entry.workdir}")
        if reason:
            if not reason.startswith("@"):
                errors.append(f"{where}: a reason is @name")
            entry.reason = reason[1:]
        entries[name] = entry
    for entry in entries.values():
        where = f"{path.name}:{entry.line}"
        if entry.reason is not None and entry.reason not in reasons:
            errors.append(f"{where}: @{entry.reason} is not defined")
        if entry.kind == "rerun":
            target = entries.get(entry.args[0])
            if target is None or target.kind != "unit":
                errors.append(f"{where}: rerun names no unit entry")
            else:
                # A rerun whose gate can hold while its target's fails names
                # an executable the build did not make.
                own = {frozenset(clause) for clause in entry.gate}
                missing = [c for c in target.gate if frozenset(c) not in own]
                if missing:
                    text = " ".join(",".join(c) for c in missing)
                    errors.append(
                        f"{where}: rerun gate lacks '{text}' from {target.name}"
                    )
        if entry.name.startswith("ThreadTeamTest") and entry.suite != "threadteam":
            errors.append(
                f"{where}: a ThreadTeamTest shard belongs to suite threadteam"
            )
    unused = set(reasons) - {e.reason for e in entries.values()}
    errors += [f"{path.name}: @{key} is not used" for key in sorted(unused)]
    return entries, errors


def meson_tests(build: Path) -> list[Test]:
    info = build / "meson-info"
    project = json.loads((info / "intro-projectinfo.json").read_text())
    name = project["descriptive_name"]
    tests = []
    for test in json.loads((info / "intro-tests.json").read_text()):
        suites = [s for s in test["suite"] if s == name or s.startswith(name + ":")]
        if not suites:
            continue
        tests.append(
            Test(
                name=test["name"],
                timeout=test["timeout"],
                suites={s.partition(":")[2] for s in suites if ":" in s},
                workdir=test["workdir"],
                env=test["env"],
                serial=not test["is_parallel"],
                should_fail=None,
                command=[str(part) for part in test["cmd"]],
            )
        )
    return tests


def inside(path: str | Path, directory: Path) -> bool:
    return Path(os.path.realpath(path)).is_relative_to(os.path.realpath(directory))


def _cmake_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    i, n = 0, len(text)
    while i < n:
        if text[i].isspace():
            i += 1
            continue
        if text[i] == '"':
            i += 1
            buf: list[str] = []
            while i < n and text[i] != '"':
                if text[i] == "\\" and i + 1 < n:
                    i += 1
                buf.append(text[i])
                i += 1
            i += 1
            tokens.append("".join(buf))
            continue
        bracket = re.match(r"\[(=*)\[", text[i:])
        if bracket:
            close = f"]{bracket.group(1)}]"
            start = i + bracket.end()
            end = text.find(close, start)
            if end == -1:
                tokens.append(text[start:])
                break
            tokens.append(text[start:end])
            i = end + len(close)
            continue
        j = i
        while j < n and not text[j].isspace():
            j += 1
        tokens.append(text[i:j])
        i = j
    return tokens


def ctest_testfile_commands(build: Path) -> dict[str, list[str]]:
    commands: dict[str, list[str]] = {}
    pending = [build]
    seen: set[Path] = set()
    while pending:
        directory = pending.pop(0).resolve()
        if directory in seen:
            continue
        seen.add(directory)
        testfile = directory / "CTestTestfile.cmake"
        if not testfile.is_file():
            continue
        for raw in testfile.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if line.startswith("add_test(") and line.endswith(")"):
                tokens = _cmake_tokens(line[len("add_test(") : -1])
                if len(tokens) >= 2:
                    commands[tokens[0]] = tokens[1:]
            elif line.startswith("subdirs(") and line.endswith(")"):
                for sub in _cmake_tokens(line[len("subdirs(") : -1]):
                    pending.append(directory / sub)
    return commands


def cmake_tests(build: Path, ctest: str) -> list[Test]:
    shown = subprocess.run(
        [ctest, "--show-only=json-v1"],
        cwd=build,
        check=True,
        capture_output=True,
        text=True,
    )
    data = json.loads(shown.stdout)
    graph = data["backtraceGraph"]
    fallback_commands = ctest_testfile_commands(build)
    tests = []
    for test in data["tests"]:
        # Tests a fetched dependency registers are defined under the build tree.
        defined_in = graph["files"][graph["nodes"][test["backtrace"]]["file"]]
        if not inside(defined_in, ROOT) or inside(defined_in, build):
            continue
        props = {p["name"]: p["value"] for p in test.get("properties", [])}
        env = dict(item.partition("=")[::2] for item in props.get("ENVIRONMENT", []))
        raw_cmd = test.get("command") or fallback_commands.get(test["name"], [])
        tests.append(
            Test(
                name=test["name"],
                timeout=props.get("TIMEOUT"),
                suites=set(props.get("LABELS", [])),
                workdir=props.get("WORKING_DIRECTORY"),
                env=env,
                serial=bool(props.get("RUN_SERIAL", False)),
                should_fail=bool(props.get("WILL_FAIL", False)),
                command=[str(part) for part in raw_cmd],
            )
        )
    return tests


def same_arg(want: str, got: str) -> bool:
    if want == "@capnp@":
        return os.path.isfile(got) and os.access(got, os.X_OK)
    if want == got:
        return True
    if want.startswith(os.sep) or got.startswith(os.sep):
        return os.path.realpath(want) == os.path.realpath(got)
    return False


def expected_arguments(entry: Entry, entries: dict[str, Entry]) -> tuple[str | None, list[str]]:
    """Executable basename (None when the program is the interpreter) and the
    arguments after it. Python entries start the argument list with the script."""
    if entry.kind == "unit":
        return entry.args[0], entry.args[2:]
    if entry.kind == "rerun":
        base = entries[entry.args[0]]
        return base.args[0], entry.args[1:]
    if entry.kind in ("fortran", "nlist"):
        return entry.args[0], entry.args[2:]
    script = ROOT / entry.args[0]
    tail = [str(script)]
    for arg in entry.args[1:]:
        if arg == "@root@":
            tail.append(str(ROOT))
        else:
            tail.append(arg)
    return None, tail


def argument_problems(entry: Entry, test: Test, entries: dict[str, Entry]) -> list[str]:
    if entry.kind == "custom":
        return []
    if not test.command:
        return ["has no command to compare with the registry"]
    exe, want = expected_arguments(entry, entries)
    if exe is not None:
        got_exe = Path(test.command[0]).name
        if got_exe not in (exe, exe + ".exe"):
            return [f"executable {got_exe}, the registry says {exe}"]
        got = test.command[1:]
    else:
        got = test.command[1:]
    if len(got) != len(want) or any(not same_arg(w, g) for w, g in zip(want, got)):
        return [f"arguments {got}, the registry says {want}"]
    return []


def want_serial(entry: Entry, coverage: bool) -> bool:
    return entry.serial or (entry.coverage_serial and coverage)


def compare_options(entry: Entry, test: Test, build: Path, entries: dict[str, Entry], coverage: bool) -> list[str]:
    problems = []
    timeout_scale = int(test.env.get("RGPOT_TEST_TIMEOUT_SCALE", "1"))
    expected_timeout = entry.timeout * timeout_scale
    if test.timeout is None or float(test.timeout) != expected_timeout:
        problems.append(
            f"timeout {test.timeout}, the registry says {entry.timeout}"
            f" times sanitizer scale {timeout_scale}"
        )
    suites = {entry.suite} if entry.suite else set()
    if test.suites != suites:
        problems.append(f"suite {sorted(test.suites)}, the registry says {sorted(suites)}")
    if entry.workdir is None:
        if test.workdir is not None and not inside(test.workdir, build):
            problems.append(f"runs in {test.workdir}, not in a build directory")
    else:
        want = ROOT / entry.workdir
        if test.workdir is None or os.path.realpath(test.workdir) != os.path.realpath(want):
            problems.append(f"runs in {test.workdir}, the registry says {want}")
    for var, setting in entry.env.items():
        if test.env.get(var) != setting:
            problems.append(f"{var}={test.env.get(var)!r}, the registry says {setting!r}")
    serial = want_serial(entry, coverage)
    if test.serial != serial:
        if entry.coverage_serial and not entry.serial:
            problems.append(
                f"serial is {test.serial}, coverage_serial is serial only when "
                f"coverage is on (this build has coverage={coverage})"
            )
        else:
            problems.append(f"serial is {test.serial}, the registry says {entry.serial}")
    if test.should_fail is not None and test.should_fail != entry.should_fail:
        problems.append(
            f"should_fail is {test.should_fail}, the registry says {entry.should_fail}"
        )
    problems += argument_problems(entry, test, entries)
    return [f"{entry.name}: {problem}" for problem in problems]


TAG_FILTER = re.compile(r"\[[A-Za-z0-9_]+\]")


def thread_team_unit(entries: dict[str, Entry]) -> Entry | None:
    found = [
        entry
        for entry in entries.values()
        if entry.kind == "unit" and len(entry.args) > 1 and entry.args[1] == "ThreadTeamTest.cc"
    ]
    if len(found) != 1:
        return None
    return found[0]


def shard_filter(entry: Entry, unit: Entry) -> list[str] | None:
    """Catch2 tag arguments for one registration of ThreadTeamTest.cc."""
    if entry.kind == "unit" and entry.name == unit.name:
        extra = entry.args[2:]
    elif entry.kind == "rerun" and entry.args and entry.args[0] == unit.name:
        extra = entry.args[1:]
    else:
        return None
    return [arg for arg in extra if arg.startswith("[")]


def parse_catch_list(text: str) -> list[str]:
    names = []
    for line in text.splitlines():
        if line.startswith("  ") and not line.startswith("   "):
            name = line[2:].strip()
            if name and not name.startswith("["):
                names.append(name)
    return names


# A cross build's exe_wrapper, run in front of every test executable.
EXE_WRAPPER: list[str] = []


def catch_list(exe: str, env: dict[str, str], *specs: str) -> tuple[list[str], str]:
    merged = os.environ.copy()
    merged.update(env)
    proc = subprocess.run(
        [*EXE_WRAPPER, exe, "--list-tests", *specs],
        check=False,
        capture_output=True,
        text=True,
        env=merged,
    )
    # Catch2 exits 2 when a spec matches nothing. That is an empty shard.
    if proc.returncode not in (0, 2):
        detail = (proc.stderr or proc.stdout).strip()
        return [], f"{exe} --list-tests {' '.join(specs)} exited {proc.returncode}: {detail}"
    return parse_catch_list(proc.stdout), ""


def family_label(limit: str) -> str:
    if limit == "":
        return "no thread limit"
    return f"OMP_THREAD_LIMIT={limit}"


def check_thread_team_shards(
    entries: dict[str, Entry],
    tests: list[Test],
    features: dict[str, bool],
) -> list[str]:
    unit = thread_team_unit(entries)
    if unit is None:
        return ["ThreadTeamTest.cc: expected one unit entry"]
    registered = {test.name: test for test in tests}
    built = registered.get(unit.name)
    if built is None:
        return []
    problems = []
    families: dict[str, list[tuple[Entry, str]]] = {}
    for entry in entries.values():
        tags = shard_filter(entry, unit)
        if tags is None or not entry.holds(features, probe=False):
            continue
        if len(tags) != 1 or not TAG_FILTER.fullmatch(tags[0]):
            problems.append(f"{entry.name}: expected one Catch2 tag, got {tags}")
            continue
        families.setdefault(entry.env.get("OMP_THREAD_LIMIT", ""), []).append((entry, tags[0]))
    if problems:
        return problems
    family_tags = {
        limit: {tag for _, tag in members} for limit, members in sorted(families.items())
    }
    if len(family_tags) > 1:
        baseline_limit, baseline_tags = next(iter(family_tags.items()))
        for limit, tags in family_tags.items():
            if tags != baseline_tags:
                problems.append(
                    f"ThreadTeamTest.cc {family_label(limit)} shard tags differ from "
                    f"{family_label(baseline_limit)}"
                )
    if problems:
        return problems
    exe = built.command[0] if built.command else ""
    if not exe or not os.path.isfile(exe):
        return []
    full, error = catch_list(exe, built.env)
    if error:
        return [error]
    full_set = set(full)
    if len(full) != len(full_set):
        problems.append("ThreadTeamTest.cc: --list-tests repeats a test name")
    for limit, members in sorted(families.items()):
        label = family_label(limit)
        selected: dict[str, set[str]] = {}
        for entry, tag in members:
            names, error = catch_list(exe, built.env, tag)
            if error:
                problems.append(error)
                continue
            selected[tag] = set(names)
        if any(tag not in selected for _, tag in members):
            continue
        union: set[str] = set()
        seen: dict[str, str] = {}
        for tag, names in selected.items():
            for name in names:
                if name in seen:
                    problems.append(
                        f"ThreadTeamTest.cc {label}: {name} matches {seen[name]} and {tag}"
                    )
                else:
                    seen[name] = tag
            union |= names
        missing = sorted(full_set - union)
        extra = sorted(union - full_set)
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if extra:
                detail.append("extra " + ", ".join(extra))
            filters = "+".join(tag for _, tag in members)
            problems.append(
                f"ThreadTeamTest.cc {label}: {filters} --list-tests is not the "
                f"unfiltered list ({'; '.join(detail)})"
            )
    return problems


def compare(
    entries: dict[str, Entry],
    tests: list[Test],
    features: dict[str, bool],
    system: str,
    build: Path,
    coverage: bool,
) -> list[str]:
    used = {a for e in entries.values() for c in e.gate for a in c} - {PROBE}
    absent = sorted(used - set(features))
    if absent:
        return [f"{system} did not pass the features {', '.join(absent)}"]
    problems = []
    registered: dict[str, Test] = {}
    for test in tests:
        if test.name in registered:
            problems.append(f"{test.name}: registered twice")
        registered[test.name] = test
    for name, test in sorted(registered.items()):
        entry = entries.get(name)
        if entry is None:
            problems.append(f"{name}: registered, but registry.txt has no entry")
            continue
        if system == "cmake" and entry.reason is not None:
            problems.append(f"{name}: CMake registered a meson-only entry (@{entry.reason})")
        elif not entry.holds(features, probe=True):
            problems.append(f"{name}: registered, but its gate '{entry.gate_text()}' fails")
        problems += compare_options(entry, test, build, entries, coverage)
    for name, entry in sorted(entries.items()):
        if name in registered or (system == "cmake" and entry.reason is not None):
            continue
        if entry.holds(features, probe=False):
            problems.append(
                f"{name}: gate '{entry.gate_text()}' holds, but {system} did not register it"
            )
    problems += check_thread_team_shards(entries, tests, features)
    return problems


def build_is_coverage(system: str, build: Path) -> bool:
    if system == "meson":
        options = json.loads((build / "meson-info" / "intro-buildoptions.json").read_text())
        for option in options:
            if option.get("name") == "b_coverage":
                return bool(option.get("value"))
        return False
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        return False
    flags = []
    build_type = ""
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("CMAKE_BUILD_TYPE:"):
            build_type = line.split("=", 1)[1]
        elif line.startswith("CMAKE_CXX_FLAGS"):
            flags.append(line.split("=", 1)[1])
    if build_type == "Coverage":
        return True
    blob = " ".join(flags)
    return any(token in blob for token in ("--coverage", "-fprofile-arcs", "-ftest-coverage"))


def parse_features(words: list[str]) -> tuple[dict[str, bool], list[str]]:
    features, errors = {}, []
    for word in words:
        name, eq, value = word.partition("=")
        if not eq or value not in ("0", "1"):
            errors.append(f"expected FEATURE=0|1, got {word!r}")
        features[name] = value == "1"
    return features, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("system", choices=("meson", "cmake", "lint", "selftest"))
    parser.add_argument("rest", nargs="*", metavar="FEATURE=0|1")
    parser.add_argument("--ctest", default="ctest")
    parser.add_argument("--list", type=Path, default=os.environ.get("RGPOT_TEST_LIST"))
    parser.add_argument("--exe-wrapper", action="append", default=[], metavar="WORD")
    return parser


def parse_cli(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    # CMake passes --ctest between the build directory and the FEATURE=0|1
    # words. A '*' positional stops at the first option and then rejects
    # everything after it, so the documented order needs intermixed parsing.
    args = parser.parse_intermixed_args(argv)
    if args.system in ("lint", "selftest"):
        if args.rest:
            parser.error(f"{args.system} takes no positional arguments: {' '.join(args.rest)}")
        args.build = None
        args.features = []
    else:
        if not args.rest:
            parser.error(f"{args.system} needs the build directory")
        args.build = Path(args.rest[0])
        args.features = args.rest[1:]
    del args.rest
    return args


def selftest() -> int:
    import contextlib
    import io
    import shutil
    import stat
    import tempfile
    import unittest

    class CliAndParitySelfTest(unittest.TestCase):
        def test_cmake_accepts_features_after_ctest_flag(self) -> None:
            args = parse_cli(
                [
                    "cmake",
                    "/work/build",
                    "tests=1",
                    "--ctest",
                    "/usr/bin/ctest",
                    "core=0",
                    "python=1",
                ]
            )
            self.assertEqual(args.system, "cmake")
            self.assertEqual(args.build, Path("/work/build"))
            self.assertEqual(args.ctest, "/usr/bin/ctest")
            self.assertEqual(args.features, ["tests=1", "core=0", "python=1"])

        def test_intermixed_options_anywhere_in_argv(self) -> None:
            args = parse_cli(
                [
                    "meson",
                    "/work/meson-build",
                    "tests=1",
                    "--list",
                    "/work/out.txt",
                    "core=1",
                ]
            )
            self.assertEqual(args.system, "meson")
            self.assertEqual(args.build, Path("/work/meson-build"))
            self.assertEqual(args.list, Path("/work/out.txt"))
            self.assertEqual(args.features, ["tests=1", "core=1"])

            before = parse_cli(
                [
                    "--ctest",
                    "custom-ctest",
                    "cmake",
                    "/work/cmake-build",
                    "tests=0",
                ]
            )
            self.assertEqual(before.ctest, "custom-ctest")
            self.assertEqual(before.build, Path("/work/cmake-build"))
            self.assertEqual(before.features, ["tests=0"])

        def test_missing_build_or_extra_lint_args_fail(self) -> None:
            for bad in (
                ["cmake"],
                ["cmake", "--ctest", "/usr/bin/ctest"],
                ["meson"],
                ["lint", "tests=1"],
                ["selftest", "extra"],
            ):
                with self.subTest(bad=bad):
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as ctx:
                            parse_cli(bad)
                    self.assertEqual(ctx.exception.code, 2)

        def test_cmake_main_with_intermixed_ctest_and_features(self) -> None:
            entries, problems = parse(REGISTRY)
            self.assertEqual(problems, [])
            all_atoms = sorted(
                {a for e in entries.values() for c in e.gate for a in c} - {PROBE}
            )
            with tempfile.TemporaryDirectory() as tmp:
                build = Path(tmp) / "build"
                build.mkdir()
                fake_ctest = Path(tmp) / "fake_ctest.py"
                payload = {
                    "backtraceGraph": {
                        "files": [str(ROOT / "CppCore" / "tests" / "tests.cmake")],
                        "nodes": [{"file": 0}],
                    },
                    "tests": [],
                }
                fake_ctest.write_text(
                    f"#!{sys.executable}\n"
                    "import json, sys\n"
                    "assert sys.argv[1:] == ['--show-only=json-v1']\n"
                    f"print(json.dumps({payload!r}))\n"
                )
                fake_ctest.chmod(fake_ctest.stat().st_mode | stat.S_IXUSR)
                out_list = Path(tmp) / "cmake-list.txt"
                feature_args = [f"{atom}=0" for atom in all_atoms]
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = main(
                        [
                            "cmake",
                            str(build),
                            "--ctest",
                            str(fake_ctest),
                            *feature_args[:3],
                            "--list",
                            str(out_list),
                            *feature_args[3:],
                        ]
                    )
                self.assertEqual(rc, 0)
                self.assertTrue(out_list.is_file())
                self.assertIn("feature tests=0\n", out_list.read_text())

        def test_spaced_cmake_test_name_is_one_test(self) -> None:
            """CTest must keep a registry name that contains spaces.

            The project includes `tests.cmake`, which includes
            `RgpotTestRegistry.cmake` and then calls `rgpot_add_test`.
            Each `include()` pops the policy scope it pushed. CTest quotes
            a name only when CMP0110 is NEW on the directory stack at
            generation. The probe copies `cmake_minimum_required` from the
            top level `CMakeLists.txt`, which is what sets that stack. A
            bare `VERSION 3.16` leaves the name as `CuH2`.
            """
            cmake = shutil.which("cmake")
            ctest = shutil.which("ctest")
            self.assertIsNotNone(cmake, "cmake is required")
            self.assertIsNotNone(ctest, "ctest is required")
            listed = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
            requirement = re.search(
                r"^cmake_minimum_required\(VERSION [^)]+\)",
                listed,
                re.MULTILINE,
            )
            self.assertIsNotNone(requirement)
            registry_cmake = ROOT / "cmake" / "RgpotTestRegistry.cmake"
            with tempfile.TemporaryDirectory() as tmp:
                src = Path(tmp) / "src"
                src.mkdir()
                (src / "CMakeLists.txt").write_text(
                    f"{requirement.group(0)}\n"
                    "project(spaced_name LANGUAGES NONE)\n"
                    'include("${CMAKE_CURRENT_LIST_DIR}/caller.cmake")\n',
                    encoding="utf-8",
                )
                (src / "caller.cmake").write_text(
                    f'include("{registry_cmake.as_posix()}")\n'
                    "set(RGPOT_TEST_FEATURE_tests TRUE)\n"
                    "set(RGPOT_TEST_FEATURE_core TRUE)\n"
                    "set(RGPOT_TEST_FEATURE_xtensor TRUE)\n"
                    "enable_testing()\n"
                    "rgpot_add_test(\"CuH2 Utils Test\""
                    " COMMAND \"${CMAKE_COMMAND}\" -E true)\n",
                    encoding="utf-8",
                )
                build = Path(tmp) / "build"
                configured = subprocess.run(
                    [cmake, "-S", str(src), "-B", str(build)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(
                    configured.returncode,
                    0,
                    configured.stdout + configured.stderr,
                )
                shown = subprocess.run(
                    [ctest, "--test-dir", str(build), "--show-only=json-v1"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(shown.returncode, 0, shown.stderr)
                names = [test["name"] for test in json.loads(shown.stdout)["tests"]]
            self.assertEqual(names, ["CuH2 Utils Test"])

        def test_ctest_testfile_fallback_and_unbuilt_shards(self) -> None:
            entries, problems = parse(REGISTRY)
            self.assertEqual(problems, [])
            with tempfile.TemporaryDirectory() as tmp:
                build = Path(tmp) / "build"
                sub = build / "sub"
                sub.mkdir(parents=True)
                (build / "CTestTestfile.cmake").write_text(
                    'add_test([=[ThreadTeamTest_slices]=] "/missing/thread_team_test" "[slices]")\n'
                    'subdirs("sub")\n',
                    encoding="utf-8",
                )
                (sub / "CTestTestfile.cmake").write_text(
                    'add_test([=[PluginTest]=] "/missing/PluginTest")\n',
                    encoding="utf-8",
                )
                commands = ctest_testfile_commands(build)
                self.assertEqual(
                    commands["ThreadTeamTest_slices"],
                    ["/missing/thread_team_test", "[slices]"],
                )
                self.assertEqual(commands["PluginTest"], ["/missing/PluginTest"])
                tests = [
                    Test(
                        name="ThreadTeamTest_slices",
                        timeout=480,
                        suites={"core"},
                        workdir=str(ROOT),
                        env={},
                        serial=False,
                        should_fail=False,
                        command=commands["ThreadTeamTest_slices"],
                    )
                ]
                features = {"tests": True, "core": True}
                self.assertEqual(check_thread_team_shards(entries, tests, features), [])

    suite = unittest.defaultTestLoader.loadTestsFromTestCase(CliAndParitySelfTest)
    result = unittest.TextTestRunner(stream=sys.stdout, verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_cli(argv)
    if args.system == "selftest":
        return selftest()
    EXE_WRAPPER[:] = args.exe_wrapper

    entries, problems = parse(REGISTRY)
    features, errors = parse_features(args.features)
    problems += errors
    tests: list[Test] = []
    if args.system != "lint" and not problems:
        if args.system == "meson":
            tests = meson_tests(args.build)
        else:
            tests = cmake_tests(args.build, args.ctest)
        coverage = build_is_coverage(args.system, args.build)
        problems += compare(entries, tests, features, args.system, args.build, coverage)
        if args.list:
            shared = {e.name for e in entries.values() if e.reason is None}
            gating = {a for n in shared for c in entries[n].gate for a in c} - {PROBE}
            lines = [f"feature {a}={int(features[a])}" for a in sorted(gating & set(features))]
            lines += [f"test {name}" for name in sorted(t.name for t in tests if t.name in shared)]
            args.list.write_text("".join(f"{line}\n" for line in lines))
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            f"{len(problems)} problem(s) against {REGISTRY.relative_to(ROOT)}. A new"
            " test needs an entry there; a test both build systems can run is"
            " registered in both.",
            file=sys.stderr,
        )
        return 1
    if args.system == "lint":
        print(f"{len(entries)} entries")
    else:
        print(f"{args.system}: {len(tests)} tests registered, all match {len(entries)} entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
