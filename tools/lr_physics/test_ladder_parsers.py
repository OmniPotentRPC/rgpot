#!/usr/bin/env python3
"""Read lr_phys_bench ladder lines with and without the active field."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import report_from_log
import tip_base_panel

LOAD = "loadavg=0.1,0.2,0.3 nproc=8"
LOG = f"""run hostname=h date=d cpu=c
strong ewald n=128 threads=2 repeats=3 median_us=10 iqr7_us=1 energy_eV=-1.5 {LOAD}
samples_us 9 10 11
strong pme n=256 threads=8 active=4 repeats=3 median_us=20 iqr7_us=2 energy_eV=-2.5 {LOAD}
samples_us 19 20 21
weak128 dsf n=512 threads=4 active=4 repeats=3 median_us=30 iqr7_us=3 energy_eV=-3.5 {LOAD}
samples_us 29 30 31
matched d4 n=64 repeats=3 threads=1 median_us=40 active=1 iqr7_us=4 energy_eV=-4.5 {LOAD}
samples_us 39 40 41
point ladder=strong method=pme n=256 threads=8 grid=32 32 32 spread_us=1
"""


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bench.log"
        path.write_text(LOG, encoding="utf-8")
        _, rows = report_from_log.parse(path)
        got = [(r["ladder"], r["method"], r["n"], r["threads"], r["active"], r["median_s"]) for r in rows]
        want = [
            ("strong", "ewald", 128, 2, "", 10e-6),
            ("strong", "pme", 256, 8, "4", 20e-6),
            ("weak128", "dsf", 512, 4, "4", 30e-6),
            ("matched", "d4", 64, 1, "1", 40e-6),
        ]
        require(got == want, f"report_from_log rows {got}")
        panel = tip_base_panel._medians(path)
        require(
            panel
            == [
                ("strong", "ewald", "128", "2", "", "10"),
                ("strong", "pme", "256", "8", "4", "20"),
                ("weak128", "dsf", "512", "4", "4", "30"),
                ("matched", "d4", "64", "1", "1", "40"),
            ],
            f"tip_base_panel rows {panel}",
        )
    print("ladder parsers read every line")


if __name__ == "__main__":
    main()
