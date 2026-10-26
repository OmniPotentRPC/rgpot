"""EMT and the cell-list EAM are in rgpot, with the copper energy pin."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_emt_and_eam_have_configs_and_the_copper_pin():
    header = (ROOT / "CppCore/rgpot/EMT/EMTPot.hpp").read_text(encoding="utf-8")
    assert "struct EMTConfig" in header
    assert "bool rasmussen = false" in header
    eam = (ROOT / "CppCore/rgpot/EAM/EAM.h").read_text(encoding="utf-8")
    assert "EAM_STANDALONE" in eam
    pin = (ROOT / "CppCore/tests/EMTPotTest.cc").read_text(encoding="utf-8")
    assert "5.129167" in pin
    assert "1.914263" in pin
    bench = (ROOT / "CppCore/tests/PotBench.cc").read_text(encoding="utf-8")
    assert "Benchmark: EMTPot force, Cu tetrahedron" in bench
    assert "Benchmark: EAMPot force, two aluminium atoms" in bench
