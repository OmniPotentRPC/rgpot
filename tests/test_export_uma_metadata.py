"""Metadata the UMA AOTI exporters embed for rgpot's UmaPot contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
ase = pytest.importorskip("ase")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_uma_aoti as exp  # noqa: E402


def _hcn():
    return exp.HCN.copy()


def test_counts_of_is_exact_and_sorted():
    atoms = ase.Atoms("C2H4", positions=[[0, 0, i] for i in range(6)])
    assert exp.counts_of(atoms) == {1: 4, 6: 2}
    assert list(exp.counts_of(atoms)) == [1, 6]
    assert exp.counts_of(_hcn()) == {1: 1, 6: 1, 7: 1}


def test_runtime_metadata_carries_the_contract():
    atoms = _hcn()
    n = len(atoms)
    example = [torch.zeros(n, 3), torch.zeros(n, dtype=torch.long)]
    meta = exp.runtime_metadata(
        Path("hcn.pt2"),
        10.0,
        300,
        example,
        torch.float32,
        task_name="omol",
        charge=-1,
        spin=2,
        z_set=exp.z_set_of(atoms),
        counts=exp.counts_of(atoms),
        label="hcn",
        model="uma-s-1p1",
    )
    assert meta["charge"] == -1
    assert meta["spin"] == 2
    assert meta["task_name"] == "omol"
    assert meta["z_set"] == [1, 6, 7]
    assert meta["natoms"] == 3
    # Embedded as a string (aot_inductor.metadata is str -> str); the
    # C++ side reads it back as {Z: n} pairs.
    assert json.loads(meta["counts"]) == {"1": 1, "6": 1, "7": 1}
    assert meta["model"] == "uma-s-1p1"
    assert meta["torch_version"] == torch.__version__
    assert isinstance(meta["fairchem_version"], str)
    assert meta["fairchem_version"]


def test_baker_dedup_keeps_compositions_that_share_an_element_set(
    tmp_path, monkeypatch
):
    import export_baker_uma_aoti as baker

    structures = {
        "01_c2h2": "C2H2",
        "02_c2h4": "C2H4",
        "03_h2cc": "H2C2",
        "04_ch3o": "CH3O",
        "05_ch3o_closed": "CH3O",
    }
    for label in structures:
        (tmp_path / label).mkdir()
        (tmp_path / label / "reactant.con").write_text("")

    def fake_load(path):
        formula = structures[Path(path).parent.name]
        n = len(ase.Atoms(formula))
        return ase.Atoms(formula, positions=[[0, 0, i] for i in range(n)])

    monkeypatch.setattr(baker, "load_atoms", fake_load)
    jobs = baker.collect_jobs(tmp_path)
    labels = [j["label"] for j in jobs]
    # C2H2 and C2H4 share z_set [1, 6] and are different packages; H2C2
    # is C2H2 again. CH3O doublet and singlet differ in spin.
    assert labels == ["01_c2h2", "02_c2h4", "04_ch3o", "05_ch3o_closed"]
    assert jobs[0]["counts"] == {1: 2, 6: 2}
    assert jobs[1]["counts"] == {1: 4, 6: 2}
    assert jobs[1]["z_set"] == [1, 6]
