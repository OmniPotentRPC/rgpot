"""The documentation checks fail on planted mistakes.

Each test writes a small page with one wrong statement or one broken example
and asserts that doccheck reports it.
"""

from __future__ import annotations

import os
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "docs"))

import doccheck  # noqa: E402


def _page(text: str, name: str = "page.org") -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="doccheck-"))
    page = tmp / name
    page.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
    return page


def _entry(name: str, signature: str, lang: str, extra: str = "") -> str:
    return f"""
    * Page

    *** ={name}=

    #+attr_rgpot: signature
    #+begin_src {lang}
    {signature}
    #+end_src

    {extra}- Inputs :: x.
    - Units :: none.
    - Outputs :: x.
    - Errors :: none.
    """


def _about(errors: list[str], name: str) -> list[str]:
    return [e for e in errors if name in e and not e.startswith("no reference entry")]


class PlantedReferenceMistakes(unittest.TestCase):
    def test_correct_c_entry_passes(self):
        page = _page(_entry("rgpot_abi_stamp", "rgpot_abi_stamp_t rgpot_abi_stamp(void);", "c"))
        self.assertEqual(_about(doccheck.reference_errors(ROOT, [page]), "rgpot_abi_stamp"), [])

    def test_wrong_c_return_type(self):
        page = _page(_entry("rgpot_abi_stamp", "int32_t rgpot_abi_stamp(void);", "c"))
        errors = _about(doccheck.reference_errors(ROOT, [page]), "rgpot_abi_stamp")
        self.assertTrue(any("differs from" in e for e in errors), errors)

    def test_wrong_python_default(self):
        name = next(
            n for n, v in doccheck.py_public_names(ROOT).items()
            if n.endswith(".hvp") and v.params
        )
        params = doccheck.py_public_names(ROOT)[name].params
        wrong = ", ".join(p.name for p in params) + ", bogus=1"
        page = _page(_entry(name, f"hvp({wrong})", "python"))
        errors = _about(doccheck.reference_errors(ROOT, [page]), name)
        self.assertTrue(any("but the source has" in e for e in errors), errors)

    def test_member_claimed_on_a_class_without_it(self):
        names = doccheck.py_public_names(ROOT)
        classes = {n.split(".")[1] for n, v in names.items() if v.kind == "class"}
        lacking = sorted(c for c in classes if f"rgpot.{c}.hvp" not in names)
        self.assertTrue(lacking, "every class has hvp; pick another member")
        page = _page(
            f"""
            * Page

            *** =.hvp=

            - Applies to :: rgpot.{lacking[0]}
            - Inputs :: x.
            - Units :: none.
            - Outputs :: x.
            - Errors :: none.
            """
        )
        errors = doccheck.reference_errors(ROOT, [page])
        self.assertIn(f"rgpot.{lacking[0]} has no member hvp", "\n".join(errors))

    def test_missing_field(self):
        page = _page(
            """
            * Page

            *** =rgpot_abi_stamp_t=

            - Inputs :: none.
            - Outputs :: the stamp.
            - Errors :: TODO.
            """
        )
        errors = _about(doccheck.reference_errors(ROOT, [page]), "rgpot_abi_stamp_t")
        text = "\n".join(errors)
        self.assertIn("missing Units", text)
        self.assertIn("Errors is a placeholder", text)

    def test_name_that_is_not_public(self):
        page = _page(_entry("rgpot_no_such_function", "void rgpot_no_such_function(void);", "c"))
        errors = doccheck.reference_errors(ROOT, [page])
        self.assertIn("rgpot_no_such_function is not a public name", "\n".join(errors))


class PlantedFactMistakes(unittest.TestCase):
    def _facts(self, text: str) -> str:
        return "\n".join(doccheck.facts_errors(ROOT, [_page(text)]))

    def test_unknown_meson_option(self):
        self.assertIn("-Dwith_nonsense is not a meson option", self._facts("Pass =-Dwith_nonsense=true=.\n"))

    def test_bad_meson_value(self):
        self.assertIn("-Dwith_rpc=maybe is not a valid value", self._facts("Pass =-Dwith_rpc=maybe=.\n"))

    def test_unknown_cmake_option(self):
        self.assertIn("-DRGPOT_WITH_NONSENSE is not a CMake option", self._facts("Pass =-DRGPOT_WITH_NONSENSE=ON=.\n"))

    def test_missing_path(self):
        self.assertIn("path CppCore/rgpot/NoSuch.hpp does not exist", self._facts("See =CppCore/rgpot/NoSuch.hpp=.\n"))

    def test_unknown_pixi_task(self):
        self.assertIn("pixi task no-such-task does not exist", self._facts("Run =pixi r no-such-task=.\n"))

    def test_foreign_tokens_are_declared_per_page(self):
        text = "#+rgpot_foreign: -Dwith_rgpot\nIn eOn, pass =-Dwith_rgpot=true=.\n"
        self.assertEqual(self._facts(text), "")

    def test_wrong_default_in_build_options_page(self):
        options = doccheck.meson_options(ROOT)
        cmake = doccheck.cmake_options(ROOT)
        page = (ROOT / doccheck.BUILD_OPTIONS_PAGE).read_text(encoding="utf-8")
        planted = page.replace("| =with_rpc= | boolean | =false= |", "| =with_rpc= | boolean | =true= |")
        self.assertNotEqual(page, planted)
        tmp = Path(tempfile.mkdtemp(prefix="doccheck-"))
        target = tmp / doccheck.BUILD_OPTIONS_PAGE
        target.parent.mkdir(parents=True)
        target.write_text(planted, encoding="utf-8")
        errors = doccheck.build_option_table_errors(tmp, options, cmake)
        self.assertIn("with_rpc defaults to false, not true", "\n".join(errors))


class PlantedBrokenExamples(unittest.TestCase):
    def _blocks(self, text: str, name: str = "page.org"):
        page = _page(text, name)
        return doccheck.page_blocks(page)

    def _static(self, text: str, name: str = "page.org") -> str:
        return "\n".join(doccheck.check_blocks_static(ROOT, self._blocks(text, name)))

    def test_unclassified_block(self):
        text = "#+begin_src python\nimport rgpot\n#+end_src\n"
        self.assertIn("missing classification", self._static(text))

    def test_python_name_that_does_not_exist(self):
        text = "#+attr_rgpot: fragment\n#+begin_src python\nimport rgpot\nrgpot.NoSuchPot()\n#+end_src\n"
        self.assertIn("NoSuchPot", self._static(text))

    def test_c_function_that_does_not_exist(self):
        text = "#+attr_rgpot: fragment\n#+begin_src c\nrgpot_no_such_call(0);\n#+end_src\n"
        self.assertIn("rgpot_no_such_call", self._static(text))

    def test_bash_syntax_error(self):
        text = "#+attr_rgpot: syntax\n#+begin_src bash\nif then fi (\n#+end_src\n"
        self.assertNotEqual(self._static(text), "")

    def test_markdown_classification(self):
        text = "<!-- rgpot-doc: fragment -->\n```python\nimport rgpot\nrgpot.NoSuchPot()\n```\n"
        self.assertIn("NoSuchPot", self._static(text, "page.md"))

    def test_skip_needs_a_reason(self):
        text = "#+attr_rgpot: skip\n#+begin_src text\nx\n#+end_src\n"
        self.assertIn("skip needs a reason", self._static(text))

    @unittest.skipUnless(
        os.environ.get("RGPOT_DOC_BUILD_DIR"), "RGPOT_DOC_BUILD_DIR is not set"
    )
    def test_failing_run_block(self):
        build = Path(os.environ["RGPOT_DOC_BUILD_DIR"])
        text = "#+attr_rgpot: run\n#+begin_src python\nimport rgpot\nassert rgpot.LJPot is None\n#+end_src\n"
        results = list(doccheck.execute_blocks(build, ROOT, self._blocks(text)))
        self.assertEqual([status for _, status, _ in results], ["fail"])


if __name__ == "__main__":
    unittest.main()
