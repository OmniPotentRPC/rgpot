#!/usr/bin/env python3
"""The soversion ledger keeps the lines shipped at the merge-base with main.

main can append soversions after the branch point. Those lines belong to
layouts this branch may not contain, so a shorter ledger than main's tip
is not a dropped line. A line that was in the merge-base ledger must
still be here, and the platform table for that soversion must still match
the table recorded with it.
"""
from __future__ import annotations

import io
import subprocess
import sys
from contextlib import redirect_stderr
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "CppCore" / "tests"))

import check_cxx_soversion as chk  # noqa: E402


def git(*args: str) -> str:
    run = subprocess.run(
        ["git", "-C", str(ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return run.stdout.strip()


def expect_drop_rejected(rows: list[tuple[int, str]], layout: str) -> None:
    err = io.StringIO()
    try:
        with redirect_stderr(err):
            chk.check_against_base(rows, layout)
    except SystemExit as exc:
        if exc.code not in (None, 0) and "dropped a shipped" in err.getvalue():
            return
    chk.fail("a ledger with a shipped line removed was accepted")


def main() -> None:
    layout = chk.LAYOUT.read_text()
    rows = chk.parse_ledger(chk.LEDGER.read_text())
    soversion = chk.parse_soversion(layout)
    live = chk.fingerprint(layout)
    if rows[-1] != (soversion, live):
        chk.fail(
            f"ledger ends at {rows[-1]}, live soversion {soversion} "
            f"fingerprint {live}"
        )

    tip = chk.base_rev()
    if tip is None:
        chk.fail("origin/main is required to check the shipped ledger")
    shipped = git("merge-base", "HEAD", tip)
    shipped_layout = chk.git_show(shipped, "CppCore/rgpot/abi/CxxLayout.hpp")
    shipped_ledger = chk.git_show(shipped, "CppCore/rgpot/abi/cxx_soversion_ledger.txt")
    if shipped_layout is None or shipped_ledger is None:
        chk.fail(f"{shipped} has no soversion ledger")
    shipped_rows = chk.parse_ledger(shipped_ledger)
    if rows[: len(shipped_rows)] != shipped_rows:
        chk.fail("ledger does not keep the soversions shipped at the merge-base")
    if chk.parse_soversion(shipped_layout) == soversion:
        if chk.parse_table(layout) != chk.parse_table(shipped_layout):
            chk.fail(
                f"kCxxSoversionTable for soversion {soversion} does not match "
                f"the table at {shipped}"
            )

    # Fails while the history check uses main's tip, which has later
    # soversions for layouts this branch did not merge.
    chk.check_against_base(rows, layout)
    expect_drop_rejected(shipped_rows[:-1], layout)
    print(
        f"soversion {soversion} matches the ledger shipped at {shipped[:12]}"
    )


if __name__ == "__main__":
    main()
