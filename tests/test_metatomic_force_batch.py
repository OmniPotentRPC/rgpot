"""MetatomicPot owns a single-forward batch and the renamed engine."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_metatomic_overrides_the_batch_hook():
    header = (ROOT / "CppCore/rgpot/MetatomicPot/MetatomicPot.hpp").read_text(
        encoding="utf-8"
    )
    assert "void forceBatchImpl" in header
    assert ".batched = true" in header
    source = (ROOT / "CppCore/rgpot/MetatomicPot/MetatomicPot.cc").read_text(
        encoding="utf-8"
    )
    start = source.index("void MetatomicPot::forceBatchImpl")
    body = source[start : source.index("\nmetatensor_torch::TensorBlock", start)]
    assert "m_model.forward" in body
    assert "energy_values.sum().backward()" in body
    pack = (ROOT / "scripts/rgpot_pack_multi_abi_engines.sh").read_text(
        encoding="utf-8"
    )
    assert "librgpot_metatomic_engine.so" in pack
    assert "ln -sfn librgpot_metatomic_engine.so" in pack
    loader = (ROOT / "python/rgpot/__init__.py").read_text(encoding="utf-8")
    assert "librgpot_metatomic_engine.so" in loader
    assert "libmetatomic_engine.so" in loader
