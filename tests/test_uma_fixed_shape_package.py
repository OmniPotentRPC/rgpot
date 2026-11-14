"""A fixed-shape UMA package and a dynamic one of the same model agree.

UmaPot takes the atom and edge counts of a package whose metadata records
none from the input checks in its AOTI wrapper source. These tests pin
that torch still writes those checks for the spring model in
``scripts/export_tiny_uma_contract.py``, and that its fixed-shape and
dynamic packages return the eager energy and forces. They skip when torch
or AOTInductor is absent.
"""

from __future__ import annotations

import re
import sys
import zipfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch._inductor")
if not hasattr(torch._C, "_aoti"):
    pytest.skip("torch has no AOTInductor loader", allow_module_level=True)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import export_tiny_uma_contract as tiny  # noqa: E402

SHAPE_KEYS = set(tiny._SHAPE_KEYS)


def _tetra(dtype=torch.float64):
    pos = torch.tensor(
        [
            [12.5, 12.5, 12.5],
            [13.6, 12.5, 12.5],
            [12.5, 13.7, 12.5],
            [12.5, 12.5, 13.4],
        ],
        dtype=dtype,
    )
    example = list(tiny._cluster_example(dtype, 4, 12))
    example[0] = pos
    return example


def _loader(path: Path):
    return torch._C._aoti.AOTIModelPackageLoader(str(path), "model", False, 1, -1)


def _wrapper_source(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [
            n
            for n in archive.namelist()
            if "/data/aotinductor/model/" in n and n.endswith(".wrapper.cpp")
        ]
        assert len(names) == 1, names
        return archive.read(names[0]).decode()


def _check_body(src: str, index: int) -> str:
    start = src.index(f"static void check_input_{index}(")
    return src[start : src.index("\n}", start)]


@pytest.fixture(scope="module")
def packages(tmp_path_factory):
    root = tmp_path_factory.mktemp("pair")
    fixed = root / "pair-fixed.pt2"
    dynamic = root / "pair-dynamic.pt2"
    tiny.export_pair(fixed, "float64", dynamic=False)
    tiny.export_pair(dynamic, "float64", dynamic=True)
    return fixed, dynamic


def test_fixed_package_records_no_shape_key(packages):
    fixed, dynamic = packages
    assert not SHAPE_KEYS & set(_loader(fixed).get_metadata())
    meta = _loader(dynamic).get_metadata()
    assert meta["nedges_min"] == "1"
    assert meta["nedges_max"] == "4096"


def test_wrapper_source_checks_the_compiled_shapes(packages):
    fixed, dynamic = packages
    src = _wrapper_source(fixed)
    pos = _check_body(src, 0)
    edges = _check_body(src, 4)
    assert re.search(r"if \(4 != \w+_size\[0\]\)", pos)
    assert re.search(r"if \(3 != \w+_size\[1\]\)", pos)
    assert re.search(r"if \(2 != \w+_size\[0\]\)", edges)
    assert re.search(r"if \(12 != \w+_size\[1\]\)", edges)

    edges = _check_body(_wrapper_source(dynamic), 4)
    assert re.search(r"if \(\w+_size\[1\] < 2\)", edges)
    assert re.search(r"if \(\w+_size\[1\] > 4096\)", edges)


def test_fixed_and_dynamic_packages_agree(packages):
    fixed, dynamic = packages
    example = _tetra()
    with torch.no_grad():
        eager_e, eager_f = tiny.PairEnergy().double().eval()(*example)
    fixed_e, fixed_f = _loader(fixed).run(example)
    dyn_e, dyn_f = _loader(dynamic).run(example)
    assert float(eager_e) > 0.01
    torch.testing.assert_close(fixed_e, eager_e, rtol=0, atol=1e-12)
    torch.testing.assert_close(fixed_f, eager_f, rtol=0, atol=1e-12)
    torch.testing.assert_close(dyn_e, fixed_e, rtol=0, atol=1e-12)
    torch.testing.assert_close(dyn_f, fixed_f, rtol=0, atol=1e-12)


def test_spring_forces_are_minus_the_energy_gradient():
    example = _tetra()
    pos = example[0].clone().requires_grad_(True)
    example[0] = pos
    energy, forces = tiny.PairEnergy().double().eval()(*example)
    (grad,) = torch.autograd.grad(energy, pos)
    torch.testing.assert_close(forces, -grad, rtol=0, atol=1e-12)
