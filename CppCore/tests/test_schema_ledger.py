#!/usr/bin/env python3
"""Self-test for CppCore/rgpot/rpc/schema_ledger.py on small schemas."""

from __future__ import annotations

import contextlib
import io
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CppCore" / "rgpot" / "rpc"))

import schema_ledger as L  # noqa: E402

UPSTREAM = textwrap.dedent(
    """\
    @0xbd1f89fa17369103;

    struct ForceInput {
      pos @0 :List(Float64);
      lengthUnit @1 :Text = "angstrom";  # @brief ignored @7 in a comment
    }

    struct XTBParams {
      method @0 :Text = "gfn2";
    }

    struct MorseParams {
      de @0 :Float64 = 0.7102;
      re @1 :Float64 = 2.897;
    }

    struct PotentialConfig {
      union {
        none  @0 :Void;
        xtb   @2 :XTBParams;
        morse @3 :MorseParams;
      }
      common @1 :Text;
      extra :group {
        flag @4 :Bool;
      }
    }

    enum Kind {
      first @0;
      second @1;
    }

    interface Potential {
      calculate @0 (fip :ForceInput) -> (energy :Float64);
    }
    """
)

COMMON = textwrap.dedent(
    """\
    @0xd782940f9a21b301;

    struct OptBool {
      union {
        unset @0 :Void;
        value @1 :Bool;
      }
    }
    """
)

# ForceInput.tag as rgpot shipped it before the pinned schema adopted it.
TAG = "ForceInput @2 field tag :Text"
WITH_TAG = UPSTREAM.replace(
    '  lengthUnit @1 :Text = "angstrom";  # @brief ignored @7 in a comment\n',
    '  lengthUnit @1 :Text = "angstrom";  # @brief ignored @7 in a comment\n  tag @2 :Text;\n',
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "commit.gpgsign=false", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=root,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


class LedgerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.upstream = self.root / "pin" / L.SCHEMA_NAME
        self.upstream.parent.mkdir()
        pin = self.root / L.PIN
        pin.parent.mkdir(parents=True, exist_ok=True)
        pin.write_text(
            "[rgconf]\nurl = https://example.invalid/rgconf.git\n"
            f"revision = {'0' * 40}\nschema = schema/potential.capnp\ncommon = schema/common.capnp\n",
            encoding="utf-8",
        )
        self.write_tree(UPSTREAM)
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 0)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def write_tree(self, upstream: str, common: str = COMMON) -> None:
        self.upstream.write_text(upstream, encoding="utf-8")
        (self.upstream.parent / L.COMMON_NAME).write_text(common, encoding="utf-8")
        self.set_vendored(upstream, common)

    def set_vendored(self, text: str, common: str = COMMON) -> None:
        for rel, body in (*((r, text) for r in L.VENDORED), *((r, common) for r in L.VENDORED_COMMON)):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text(body, encoding="utf-8")

    def vendored(self) -> str:
        return (self.root / L.VENDORED[0]).read_text(encoding="utf-8")

    def ledger(self) -> Path:
        return self.root / L.LEDGER

    def run_tool(self, *args: str) -> int:
        self.err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(self.err):
            return L.main([args[0], "--root", str(self.root), *args[1:]])

    def check(self, *extra: str) -> int:
        return self.run_tool("check", "--upstream", str(self.upstream), *extra)

    def assertRejected(self, needle: str, *extra: str) -> None:
        self.assertEqual(self.check(*extra), 1, self.err.getvalue())
        self.assertIn(needle, self.err.getvalue())

    def test_seeded_ledger_passes(self) -> None:
        self.assertEqual(self.check(), 0, self.err.getvalue())
        self.assertEqual(self.run_tool("check"), 0, self.err.getvalue())
        lines = self.ledger().read_text(encoding="utf-8")
        self.assertIn("PotentialConfig @2 arm xtb :XTBParams upstream\n", lines)
        self.assertIn("PotentialConfig @4 field extra.flag :Bool upstream\n", lines)
        self.assertIn("Kind @1 enumerant second - upstream\n", lines)
        self.assertIn("Potential @0 method calculate (fip:ForceInput)->(energy:Float64) upstream\n", lines)
        self.assertNotIn("@7", lines)

    def test_renumbered_arm_in_vendored_copy(self) -> None:
        self.set_vendored(
            self.vendored()
            .replace("xtb   @2 :XTBParams", "xtb   @5 :XTBParams")
            .replace("    morse @3", "    morse @2")
            .replace("    xtb   @5", "    xtb   @3")
        )
        self.assertRejected(
            "has PotentialConfig @2 arm morse :MorseParams, ledger has PotentialConfig @2 arm xtb"
        )

    def test_retyped_field(self) -> None:
        self.set_vendored(self.vendored().replace("de @0 :Float64", "de @0 :Float32"))
        self.assertRejected("has MorseParams @0 field de :Float32")

    def test_vendored_copy_diverges_from_the_pin(self) -> None:
        self.set_vendored(self.vendored() + "# a local edit\n")
        self.assertRejected(f"{L.VENDORED[0]} {L.DIVERGES}")

    def test_vendored_common_diverges_from_the_pin(self) -> None:
        self.set_vendored(self.vendored(), COMMON + "# a local edit\n")
        self.assertRejected(f"{L.VENDORED_COMMON[0]} {L.DIVERGES}")

    def test_vendored_copies_differ(self) -> None:
        (self.root / L.VENDORED[1]).write_text(self.vendored() + "\n", encoding="utf-8")
        self.assertEqual(self.run_tool("check"), 1)
        self.assertIn(f"{L.VENDORED[1]} differs from {L.VENDORED[0]}", self.err.getvalue())
        self.set_vendored(self.vendored())
        (self.root / L.VENDORED_COMMON[1]).write_text(COMMON + "\n", encoding="utf-8")
        self.assertEqual(self.run_tool("check"), 1)
        self.assertIn(f"{L.VENDORED_COMMON[1]} differs from {L.VENDORED_COMMON[0]}", self.err.getvalue())

    def test_vendored_member_needs_the_ledger(self) -> None:
        # Without a pin to compare with, the ledger still catches a member the
        # copies declare and the ledger does not.
        self.set_vendored(WITH_TAG)
        self.assertEqual(self.run_tool("check"), 1)
        self.assertIn(f"declares {TAG}, not in the ledger", self.err.getvalue())

    def test_new_pinned_entry_needs_append(self) -> None:
        self.write_tree(WITH_TAG)
        self.assertRejected(f"pinned schema declares {TAG}, not in the ledger (run append)")
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 0)
        self.assertTrue(self.ledger().read_text(encoding="utf-8").endswith(f"{TAG} upstream\n"))
        self.assertEqual(self.check(), 0, self.err.getvalue())

    def test_pin_drops_a_ledger_entry(self) -> None:
        older = UPSTREAM.replace("    morse @3 :MorseParams;\n", "")
        self.write_tree(older)
        self.assertRejected("pinned schema no longer declares PotentialConfig @3 arm morse")

    def test_unretired_local_entry(self) -> None:
        text = self.ledger().read_text(encoding="utf-8")
        self.ledger().write_text(text + f"{TAG} local\n", encoding="utf-8")
        self.write_tree(WITH_TAG)
        self.assertRejected(f"local {TAG} is not retired")

    @unittest.skipIf(shutil.which("git") is None, "git is not installed")
    def test_pin_adopting_a_local_field_retires_it(self) -> None:
        # The move from potentials-schema plus the local series to rgconf:
        # the ledger holds a local entry, the new pin declares it unchanged.
        text = self.ledger().read_text(encoding="utf-8")
        self.ledger().write_text(text + f"{TAG} local\n", encoding="utf-8")
        _git(self.root, "init", "-q")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", "base")

        self.write_tree(WITH_TAG)
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 0)
        lines = self.ledger().read_text(encoding="utf-8")
        self.assertIn(f"{TAG} local\n", lines)
        self.assertTrue(lines.endswith(f"{TAG} upstream\n"))
        self.assertEqual(self.check("--base", "HEAD"), 0, self.err.getvalue())

        self.ledger().write_text(lines + f"{TAG} local\n", encoding="utf-8")
        self.assertEqual(self.check(), 1)
        self.assertIn("reuses the ordinal of", self.err.getvalue())

    def test_append_refuses_to_rewrite(self) -> None:
        self.write_tree(UPSTREAM.replace("de @0 :Float64", "de @0 :Float32"))
        self.assertEqual(self.run_tool("append", "--upstream", str(self.upstream)), 1)
        self.assertIn("append will not rewrite it", self.err.getvalue())

    def test_crlf_checkout_passes(self) -> None:
        # git with core.autocrlf=true checks every text file out with CRLF.
        files = [
            *(self.root / rel for rel in (*L.VENDORED, *L.VENDORED_COMMON)),
            self.ledger(),
            self.root / L.PIN,
            self.upstream,
        ]
        for path in files:
            path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        self.assertIn(b"\r\n", (self.root / L.VENDORED[0]).read_bytes())
        self.assertEqual(self.check(), 0, self.err.getvalue())

    @unittest.skipIf(shutil.which("git") is None, "git is not installed")
    def test_ledger_is_append_only_against_base(self) -> None:
        _git(self.root, "init", "-q")
        _git(self.root, "add", "-A")
        _git(self.root, "commit", "-q", "-m", "base")
        original = self.ledger().read_text(encoding="utf-8")

        self.ledger().write_text(original + "MorseParams @9 field spare :Bool local\n", encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertNotIn("ledger entry", self.err.getvalue())

        changed = original.replace(
            "PotentialConfig @2 arm xtb :XTBParams upstream",
            "PotentialConfig @2 arm tblite :XTBParams upstream",
        )
        self.ledger().write_text(changed, encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn("ledger entry changed: PotentialConfig @2 arm xtb", self.err.getvalue())

        dropped = original.replace("Kind @1 enumerant second - upstream\n", "")
        self.ledger().write_text(dropped, encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn("ledger entry disappeared: Kind @1 enumerant second - upstream", self.err.getvalue())

        lines = original.splitlines(keepends=True)
        body = [ln for ln in lines if not ln.startswith("#")]
        moved = [ln for ln in lines if ln.startswith("#")] + body[1:] + body[:1]
        self.ledger().write_text("".join(moved), encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn("ledger entry moved", self.err.getvalue())

        inserted = (
            [ln for ln in lines if ln.startswith("#")] + ["Kind @2 enumerant third - upstream\n"] + body
        )
        self.ledger().write_text("".join(inserted), encoding="utf-8")
        self.assertEqual(self.run_tool("check", "--base", "HEAD"), 1)
        self.assertIn(
            "ledger entry inserted before existing ones: Kind @2 enumerant third", self.err.getvalue()
        )
        self.assertNotIn("ledger entry moved", self.err.getvalue())

        self.ledger().write_text(original, encoding="utf-8")
        self.assertEqual(
            self.run_tool("check", "--base", "HEAD", "--upstream", str(self.upstream)), 0, self.err.getvalue()
        )


def _pinned(tmp: Path) -> Path:
    """The pinned rgconf potential.capnp, with common.capnp beside it."""
    pin = L.read_pin(ROOT)
    try:
        schema, common = L.fetch_pinned(pin)
    except (OSError, subprocess.CalledProcessError) as exc:
        if os.environ.get("CI"):
            raise
        raise unittest.SkipTest(f"cannot fetch {pin.url} at {pin.revision}: {exc}") from exc
    (tmp / L.COMMON_NAME).write_text(common, encoding="utf-8", newline="\n")
    path = tmp / L.SCHEMA_NAME
    path.write_text(schema, encoding="utf-8", newline="\n")
    return path


class RepositoryTest(unittest.TestCase):
    def test_vendored_copies_are_the_pinned_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            pinned = _pinned(Path(tmp))
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = L.main(["check", "--root", str(ROOT), "--upstream", str(pinned)])
            self.assertEqual(code, 0, err.getvalue())

    def test_vendored_copies_match_the_ledger(self) -> None:
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            code = L.main(["check", "--root", str(ROOT)])
        self.assertEqual(code, 0, err.getvalue())

    def test_no_local_series_remains(self) -> None:
        for directory in L.SCHEMA_DIRS:
            self.assertFalse((ROOT / directory / "local").exists(), directory)
        self.assertFalse((ROOT / "subprojects" / "potentials-schema.wrap").exists())

    def test_potential_config_arms_keep_their_ordinals(self) -> None:
        ledger = L.resolve_ledger(L.parse_ledger((ROOT / L.LEDGER).read_text(encoding="utf-8"), "ledger"))
        arms = {e.ordinal: (e.name, e.type, e.origin) for e in ledger if e.scope == "PotentialConfig"}
        self.assertEqual(arms[6], ("xtb", ":XTBParams", "upstream"))
        self.assertEqual(arms[10], ("morse", ":MorseParams", "upstream"))
        # rgpot's own arms keep the ordinals it shipped them at; rgconf's
        # arms follow them.
        self.assertEqual(arms[18], ("composite", ":CompositeParams", "upstream"))
        self.assertEqual(arms[21], ("mullerBrown", ":MullerBrownParams", "upstream"))
        self.assertEqual(arms[30], ("plugin", ":PluginPotentialParams", "upstream"))
        self.assertEqual(arms[31], ("orca", ":ORCAParams", "upstream"))

    def test_parser_reads_every_ordinal_line(self) -> None:
        # Each line that declares an ordinal outside a comment must be an entry.
        for rel in L.VENDORED:
            text = (ROOT / rel).read_text(encoding="utf-8")
            entries = {line for _, line in L.parse_schema(text)}
            for number, raw in enumerate(text.splitlines(), 1):
                code = L._strip_comment(raw)
                if L.ORDINAL.search(code) and not code.lstrip().startswith("@0x"):
                    self.assertIn(number, entries, f"{rel}:{number}: {raw.strip()}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
