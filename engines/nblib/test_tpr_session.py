#!/usr/bin/env python3
# MIT License
"""Check the TPR session against mdrun and against its own invariants.

usage: test_tpr_session.py ENGINE.so FILE.tpr SOLUTE [FORCES.txt [CONSTRAINTS]]

SOLUTE is the number of leading massive atoms that are not rigid water.
FORCES.txt holds mdrun's forces, one "x y z" line (kJ/mol/nm) per TPR atom.
"""
import ctypes as C
import math
import sys

EV = 96.48533212331
KB = 8.617333262e-5


def main():
    lib = C.CDLL(sys.argv[1])
    ops = (C.c_char * 512)()
    ctx, n = C.c_void_p(), C.c_size_t()
    err = C.create_string_buffer(512)
    spec = ("tpr " + sys.argv[2]).encode()
    create = lib.rgpot_md_session_engine_create

    def call(text, options):
        return create(text, C.c_size_t(len(text)), options, C.byref(ops),
                      C.byref(ctx), C.byref(n), err, C.c_size_t(512))

    assert call(b"tpr /nonexistent.tpr", b"") != 0 and b"no TPR" in err.value
    assert call(spec, b"threads=x") != 0 and b"threads=N" in err.value
    assert call(spec, b"threads=2") == 0, err.value
    slot = lambda k: C.cast(C.addressof(ops), C.POINTER(C.c_void_p))[k]
    n3 = 3 * n.value
    vec = C.c_double * n3
    x, v, f, m = vec(), vec(), vec(), (C.c_double * n.value)()
    u, k = C.c_double(), C.c_double()
    P = C.POINTER(C.c_double)
    tail = [C.c_char_p, C.c_size_t]
    state = C.CFUNCTYPE(C.c_int, C.c_void_p, P, P, P, C.c_size_t, P, P, *tail)(slot(5))
    masses = C.CFUNCTYPE(C.c_int, C.c_void_p, P, C.c_size_t, *tail)(slot(6))
    setx = C.CFUNCTYPE(C.c_int, C.c_void_p, P, C.c_size_t, *tail)(slot(1))
    nve = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_double, C.c_size_t, *tail)(slot(4))
    lang = C.CFUNCTYPE(C.c_int, C.c_void_p, C.c_double, C.c_double, C.c_double,
                       P, P, P, P, C.c_size_t, C.c_size_t, C.c_size_t,
                       C.c_uint64, C.POINTER(C.c_uint64), *tail)(slot(3))
    destroy = C.CFUNCTYPE(None, C.c_void_p)(slot(7))

    def ok(code):
        assert code == 0, err.value

    def get():
        ok(state(ctx, x, v, f, n3, C.byref(u), C.byref(k), err, 512))

    ok(masses(ctx, m, n.value, err, 512))
    get()
    assert min(m) > 0
    start = list(x)
    if len(sys.argv) > 4:
        ref = [list(map(float, line.split())) for line in open(sys.argv[4])]
        water = 4 if len(ref) != n.value else 1
        solute = int(sys.argv[3])
        idx = [i for i in range(len(ref)) if i < solute or (i - solute) % water != 3 or water == 1]
        worst = max(abs(f[3 * j + d] * EV * 10.0 - ref[i][d])
                    for j, i in enumerate(idx) for d in range(3))
        print("largest force difference from mdrun", worst, "kJ/mol/nm")
        assert worst < 0.05

    # Energy conservation with every constraint held.
    ok(nve(ctx, 0.0005, 20, err, 512))
    get()
    e0 = u.value + k.value
    ok(nve(ctx, 0.0005, 400, err, 512))
    get()
    print("NVE drift over 0.2 ps", u.value + k.value - e0, "eV")
    assert abs(u.value + k.value - e0) < 0.05

    # The plane holds and the temperature matches the constrained degrees of
    # freedom; the normal translates whole rigid waters (massive atoms after
    # the solute, three per molecule).
    solute = int(sys.argv[3])
    normal = vec()
    for i in range(solute, n.value):
        normal[3 * i] = m[i] * (1.0 if ((i - solute) // 3) % 2 else -0.5)
    norm = math.sqrt(sum(a * a for a in normal))
    for j in range(n3):
        normal[j] /= norm
    point = vec(*x)
    hits = C.c_uint64()
    kt = KB * 300
    ok(lang(ctx, kt, 5.0, 0.001, point, normal, None, None, 0, n3, 500, 11,
            C.byref(hits), err, 512))
    total, count = 0.0, 0
    for _ in range(40):
        ok(lang(ctx, kt, 5.0, 0.001, point, normal, None, None, 0, n3, 25, 11,
                C.byref(hits), err, 512))
        get()
        total += k.value
        count += 1
    offset = sum(normal[j] * (x[j] - point[j]) for j in range(n3))
    dof = n3 - int(sys.argv[5] if len(sys.argv) > 5 else 0) - 1
    print("plane offset", offset, "T/T0", 2 * total / count / (dof * kt))
    assert abs(offset) < 1e-9
    if len(sys.argv) > 5:
        assert abs(2 * total / count / (dof * kt) - 1) < 0.04

    # A split pair is refused, and the state survives the refusal.
    bad = vec(*start)
    bad[3 * (n.value - 1)] += 20.0
    assert setx(ctx, bad, n3, err, 512) != 0 and b"periodic images" in err.value
    destroy(ctx)
    print("ok")


main()
