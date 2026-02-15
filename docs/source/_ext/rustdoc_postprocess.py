"""Post-process sphinxcontrib-rust RST output via pandoc.

sphinxcontrib-rust dumps rustdoc (markdown) doc-comments verbatim into RST
directive bodies.  This extension runs *after* sphinxcontrib_rust's
``builder-inited`` handler (priority 600 > default 500), walks the generated
``.rst`` files, extracts the markdown content blocks, and converts them to
proper RST through ``pandoc -f markdown -t rst``.
"""

import re
import subprocess
import textwrap
from pathlib import Path

from sphinx.application import Sphinx
from sphinx.util import logging

_log = logging.getLogger(__name__)

# Matches an indented markdown fenced code block:
#   <indent>```lang
#   ...code...
#   <indent>```
_FENCE_RE = re.compile(
    r"^(?P<indent>[ ]+)```(?P<lang>\w*)\s*\n"
    r"(?P<body>.*?)"
    r"^(?P=indent)```[ ]*$",
    re.MULTILINE | re.DOTALL,
)

# Matches an indented markdown table (header + separator + rows).
_TABLE_RE = re.compile(
    r"^(?P<indent>[ ]+)\|.+\|[ ]*\n"  # header row
    r"(?P=indent)\|[-| :]+\|[ ]*\n"  # separator row
    r"(?:(?P=indent)\|.+\|[ ]*\n)+",  # data rows
    re.MULTILINE,
)

# Matches an indented markdown ATX heading (## Heading).
_HEADING_RE = re.compile(
    r"^(?P<indent>[ ]+)(?P<hashes>#{1,6})[ ]+(?P<text>.+)$",
    re.MULTILINE,
)

# Matches markdown inline code (`code`) that is NOT already double-backtick RST.
# Handles the common case where `code`<letter> breaks RST inline markup rules.
_INLINE_CODE_RE = re.compile(
    r"(?<!`)(`)((?!`)(?:[^`\n])+)\1(?!`)",
)


def _pandoc(markdown: str) -> str:
    """Convert a markdown fragment to RST via pandoc."""
    try:
        result = subprocess.run(
            ["pandoc", "-f", "markdown-smart", "-t", "rst", "--wrap=none"],
            input=markdown,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback: return original text unchanged.
    return markdown


def _convert_fences(content: str) -> str:
    """Convert markdown code fences to RST code-block directives."""

    def _replace(m: re.Match) -> str:
        indent = m.group("indent")
        lang = m.group("lang") or "none"
        body = m.group("body")

        body_indent = indent + "   "
        lines = []
        for line in body.split("\n"):
            stripped = line.rstrip()
            if stripped:
                # Preserve relative indentation: strip the fence indent,
                # then prepend the code-block body indent.
                if stripped.startswith(indent):
                    stripped = stripped[len(indent) :]
                lines.append(body_indent + stripped)
            else:
                lines.append("")
        # Trim trailing blanks.
        while lines and not lines[-1].strip():
            lines.pop()

        body_text = "\n".join(lines)
        return f"{indent}.. code-block:: {lang}\n\n{body_text}\n"

    return _FENCE_RE.sub(_replace, content)


def _convert_tables(content: str) -> str:
    """Convert markdown tables to RST list-tables via pandoc."""

    def _replace(m: re.Match) -> str:
        indent = m.group("indent")
        table_md = textwrap.dedent(m.group(0))
        rst = _pandoc(table_md).rstrip("\n")
        # Re-indent the pandoc output to match the original indentation.
        lines = [indent + line if line.strip() else "" for line in rst.split("\n")]
        return "\n".join(lines) + "\n"

    return _TABLE_RE.sub(_replace, content)


def _convert_inline_code(content: str) -> str:
    """Convert markdown inline code to RST double-backtick literals.

    Markdown uses `code` while RST uses ``code``.  When the closing backtick
    is immediately followed by a word character (e.g. `Vec`s), RST's inline
    markup recognition rules fail.  Converting to double-backtick form fixes
    this and also gives proper literal rendering.

    We skip lines inside ``.. code-block::`` directives and other lines that
    are part of RST directive syntax (starting with ``:`` or ``..``).
    """

    def _process_line(line: str) -> str:
        stripped = line.lstrip()
        # Skip directive lines, option lines, and already-RST markup.
        if stripped.startswith("..") or stripped.startswith(":"):
            return line
        return _INLINE_CODE_RE.sub(r"``\2``", line)

    return "\n".join(_process_line(l) for l in content.split("\n"))


def _convert_headings(content: str) -> str:
    """Convert markdown ATX headings to bold labels.

    RST section headings cannot appear inside directive bodies, so we
    convert ``## Heading`` to ``**Heading**`` which renders as bold.
    """

    def _replace(m: re.Match) -> str:
        indent = m.group("indent")
        text = m.group("text").strip()
        return f"{indent}**{text}**"

    return _HEADING_RE.sub(_replace, content)


def _postprocess(app: Sphinx) -> None:
    """Walk generated RST files and convert markdown fragments."""
    crates_dir = Path(app.srcdir) / "crates"
    if not crates_dir.exists():
        return

    for rst_file in sorted(crates_dir.rglob("*.rst")):
        original = rst_file.read_text(encoding="utf-8")
        converted = original
        converted = _convert_fences(converted)
        converted = _convert_tables(converted)
        converted = _convert_headings(converted)
        converted = _convert_inline_code(converted)
        if converted != original:
            _log.info(
                "[rustdoc_postprocess] Converted markdown in %s",
                rst_file.relative_to(app.srcdir),
            )
            rst_file.write_text(converted, encoding="utf-8")


def setup(app: Sphinx):
    app.connect("builder-inited", _postprocess, priority=600)
    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
