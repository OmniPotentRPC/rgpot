# MIT License
# Copyright 2023--present rgpot developers
"""Pytest coverage for the rgbind policy header."""

from __future__ import annotations

import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import _rgbind_test

FIXTURE = [
    (0, 10, False),
    (2, 10, False),
    (1, 20, True),
    (4, 10, False),
    (3, 30, False),
    (5, 20, True),
]
WORKERS = (1, 2, 3, 8)


def _canonical(workers: int) -> str:
    return _rgbind_test.schedule_canonical(FIXTURE, workers)


def test_version() -> None:
    assert _rgbind_test.RGBIND_VERSION == 1


def test_claim_released_across_calls() -> None:
    session = _rgbind_test.Session()
    assert session.work(0) == 1
    assert session.work(0) == 1


def test_claim_raii_on_throw() -> None:
    session = _rgbind_test.Session()
    with pytest.raises(RuntimeError, match="boom"):
        session.work_throw()
    assert session.work(0) == 1


def test_claim_contention_eight_threads() -> None:
    session = _rgbind_test.Session()
    barrier = threading.Barrier(8)
    errors: list[str] = []
    oks = 0
    lock = threading.Lock()

    def once() -> None:
        nonlocal oks
        barrier.wait()
        try:
            session.work(30)
        except RuntimeError as exc:
            with lock:
                errors.append(str(exc))
        else:
            with lock:
                oks += 1

    threads = [threading.Thread(target=once) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert oks >= 1
    assert oks + len(errors) == 8
    for text in errors:
        assert text == "Session is in use by another thread"


def test_claim_pool_eight() -> None:
    session = _rgbind_test.Session()
    start = threading.Barrier(8)

    def once() -> str:
        start.wait()
        try:
            session.work(20)
        except RuntimeError as exc:
            return str(exc)
        return "ok"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _i: once(), range(8)))
    assert "ok" in results
    for item in results:
        assert item == "ok" or item == "Session is in use by another thread"


def test_worker_exception_is_reraised() -> None:
    def boom() -> None:
        raise ValueError("from-worker")

    with pytest.raises(ValueError, match="from-worker"):
        _rgbind_test.call_on_worker(boom)


def test_worker_success() -> None:
    seen: list[int] = []

    def ok() -> None:
        seen.append(1)

    _rgbind_test.call_on_worker(ok)
    assert seen == [1]


def test_main_trampoline_signals() -> None:
    def ok() -> None:
        return None

    assert _rgbind_test.call_on_main(ok, True) == 0


def test_schedule_hash_matches_across_workers(capsys: pytest.CaptureFixture[str]) -> None:
    hashes = []
    canonicals = []
    for workers in WORKERS:
        text = _canonical(workers)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        hashes.append(digest)
        canonicals.append(text)
        print(f"schedule_sha256 workers={workers} {digest}")
    assert len(set(hashes)) == 1
    assert len(set(canonicals)) == 1
    captured = capsys.readouterr()
    assert "schedule_sha256 workers=8" in captured.out


def test_schedule_order() -> None:
    text = _canonical(1)
    assert text == "10 0 0 2 4\n20 1 1 5\n30 0 3\n"


def test_schedule_rejects_zero_workers() -> None:
    with pytest.raises(ValueError):
        _rgbind_test.schedule_canonical(FIXTURE, 0)


def test_shape_2d_message() -> None:
    arr = np.zeros((2, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=r"positions: expected shape \(4, 3\), got \(2, 3\)"):
        _rgbind_test.check_f64_2d(arr, 4, 3, "positions")


def test_shape_1d_and_3d_messages() -> None:
    vec = np.zeros((2,), dtype=np.float64)
    with pytest.raises(ValueError, match=r"forces: expected shape \(4,\), got \(2,\)"):
        _rgbind_test.check_f64_1d(vec, 4, "forces")
    cube = np.zeros((1, 1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match=r"cell: expected shape \(2, 2, 2\), got \(1, 1, 1\)"):
        _rgbind_test.check_f64_3d(cube, 2, 2, 2, "cell")


def test_shape_ok() -> None:
    arr = np.zeros((4, 3), dtype=np.float64)
    _rgbind_test.check_f64_2d(arr, 4, 3, "positions")


def test_dtype_and_order_are_type_errors() -> None:
    with pytest.raises(TypeError):
        _rgbind_test.check_f64_2d(np.zeros((4, 3), dtype=np.float32), 4, 3, "positions")
    fortran = np.asfortranarray(np.zeros((4, 3), dtype=np.float64))
    with pytest.raises(TypeError):
        _rgbind_test.check_f64_2d(fortran, 4, 3, "positions")


def test_echo_preserves_specials() -> None:
    values = np.array([0.0, -0.0, np.nan, np.inf, -np.inf, np.nextafter(0.0, 1.0)], dtype=np.float64)
    out = _rgbind_test.echo_f64(values)
    assert out.dtype == np.float64
    assert out.tobytes() == values.tobytes()
    assert np.signbit(out[1])


def test_status_table() -> None:
    kind, message = _rgbind_test.map_status(1, "bad build")
    assert kind == "value_error"
    assert message == "bad build"
    kind, message = _rgbind_test.map_status(4, "")
    assert kind == "type_error"
    kind, message = _rgbind_test.map_status(6, "")
    assert kind == "not_converged"
    kind, message = _rgbind_test.map_status(9, "", "Session")
    assert kind == "runtime_error"
    assert message == "Session is in use by another thread"
    kind, message = _rgbind_test.map_status(8, "")
    assert kind == "numerical"
    kind, message = _rgbind_test.map_status(10, "")
    assert kind == "not_implemented"
    kind, message = _rgbind_test.map_status(11, "thread local")
    assert kind == "runtime_error"
    assert message == "thread local"


def test_raise_status() -> None:
    assert _rgbind_test.raise_status(0) is True
    assert _rgbind_test.raise_status(6) is False
    with pytest.raises(ValueError, match="bad build"):
        _rgbind_test.raise_status(1, "bad build")
    with pytest.raises(TypeError):
        _rgbind_test.raise_status(5)
    with pytest.raises(_rgbind_test.NumericalError):
        _rgbind_test.raise_status(8, "nan")
    assert issubclass(_rgbind_test.NumericalError, ArithmeticError)
    with pytest.raises(NotImplementedError, match="compiled-out feature"):
        _rgbind_test.raise_status(10)
    with pytest.raises(RuntimeError, match="Session is in use by another thread"):
        _rgbind_test.raise_status(9, "", "Session")
    with pytest.raises(RuntimeError, match="thread local"):
        _rgbind_test.raise_status(11, "thread local")
