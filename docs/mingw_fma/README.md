<!-- Copyright 2023--present rgpot developers -->
# mingw-w64 `fma` on x86-64 is not a fused multiply-add

`fma` in mingw-w64's `libmingwex` returns wrong results on x86-64. msvcrt
has no `fma`, so MinGW programs get this one, linked statically. rgpot
works around it with `detail::fma_soft` in `CppCore/rgpot/simd/exact_fma.hpp`.
This note and `fma_repro.c` are the material for an upstream report; nothing
has been filed.

## Versions

- GCC 13.2.0 (Ubuntu `gcc-mingw-w64-x86-64-posix` 13.2.0-6ubuntu1+26.1,
  `x86_64-w64-mingw32-gcc-posix --version` prints `13-posix`)
- mingw-w64 runtime 11.0.1 (Ubuntu `mingw-w64-x86-64-dev` 11.0.1-3build1),
  member `lib64_libmingwex_a-fma.o` of `libmingwex.a`
- binutils 2.41.90.20240122, wine 9.0 (Ubuntu 9.0~repack-4build3)

The x86 code in `mingw-w64-crt/math/fma.c` is unchanged since the
2019-12-12 commit "crt/fma{,f}.c: Implement FMA for `double` and `float`
properly". The only later commit touches ARM64EC, so 12.0.0, 13.0.0 and
master (as of 2026-11-14) have the same code, and master's file compiled
natively gives the same three results.

## Cases

| | x | y | z | correct | libmingwex |
|---|---|---|---|---|---|
| 1 | `0x3ff0000000000000` (1) | `0x3ff0000000000001` | `0x3ff0000000000001` | `0x4000000000000001` | `0x4000000000000000` |
| 2 | `0x3ff0000000000001` | `0x3ff0000000000001` | `0x400921fb54442d18` (pi) | `0x401090fdaa22168d` | `0x401090fdaa22168c` |
| 3 | `0x0000000000000000` (0) | `0x7ff0000000000000` (inf) | `0x7ff8000000000000` (qNaN) | `0x7ff8000000000000` | `0xfff8000000000000` |

The correct column is the exact result rounded to nearest even, computed
with rationals. It is also what glibc's `fma` and the FMA3 `vfmadd`
instruction return.

1. 1 * (1 + 2^-52) + (1 + 2^-52) = 2 + 2^-51, which is representable, so
   any fused result is exact. Even the unfused `x*y+z` gets this one right.
   `fma.c` splits y into 1 + 2^-52 and adds the partial products to z one
   at a time. z + 1 = 2 + 2^-52 is a tie and rounds to 2. Adding 2^-52
   ties again and rounds to 2 again.
2. (1 + 2^-52)^2 = 1 + 2^-51 + 2^-104. Adding pi puts the sum 2^-104 above
   a tie, so it rounds up. `fma.c` adds 1 (exact), then 2^-52 + 2^-52,
   which lands exactly on the tie and rounds to even, down. The 2^-104 term
   comes last and is lost. Result: one ulp low.
3. The result has to be a NaN, and it is one. But `fma.c` first computes
   `x * y + z` and returns it if it is not finite. On SSE2, 0 * inf gives
   the default NaN (sign set), and `addsd` returns the first NaN operand,
   so z's NaN is dropped. IEEE 754-2019 6.2.3 says the result should be one
   of the input NaNs; FMA3 and glibc return z. This one is a quality
   issue, not a wrong number.

Cases 1 and 2 have the same cause: each addition in

```c
ret = z;
ret += xhi.f * yhi.f;
ret += xhi.f * ylo.f + xlo.f * yhi.f;
ret += xlo.f * ylo.f;
```

rounds on its own. Splitting the inputs into 26-bit halves makes the
products exact (up to the 1-ulp error the source comment admits for
`xlo * ylo`), but the sums are not. Getting a single rounding needs error-free
transformations for the sums as well, for example 2Sum with a final
round-to-odd step (Boldo and Melquiond), or an integer implementation like
musl's `src/math/fma.c`.

## Reproducing

With wine or on Windows:

```sh
x86_64-w64-mingw32-gcc -O2 -fno-builtin -o fma_repro.exe fma_repro.c
wine ./fma_repro.exe
```

```
case 1 (exact result): fma(3ff0000000000000, 3ff0000000000001, 3ff0000000000001) want 4000000000000001 got 4000000000000000 WRONG
case 2 (tie broken by low product bits): fma(3ff0000000000001, 3ff0000000000001, 400921fb54442d18) want 401090fdaa22168d got 401090fdaa22168c WRONG
case 3 (0 * inf + qNaN): fma(0000000000000000, 7ff0000000000000, 7ff8000000000000) want 7ff8000000000000 got fff8000000000000 WRONG
```

The exit status is the number of wrong cases (3 here). `-fno-builtin` and
the `volatile` inputs stop GCC from folding `fma` at compile time with
MPFR, which would hide the library. `objdump -d fma_repro.exe` shows a call
to `fma` and no `vfmadd`.

Without wine, convert the archive member to ELF and link it into a native
build. This rules out wine, x87 precision and msvcrt:

```sh
ar x "$(x86_64-w64-mingw32-gcc -print-file-name=libmingwex.a)" lib64_libmingwex_a-fma.o
objcopy -I pe-x86-64 -O elf64-x86-64 --redefine-sym fma=mingwex_fma \
  lib64_libmingwex_a-fma.o mingwex_fma.o
cc -O2 -fno-builtin -no-pie -Wl,-z,noexecstack -DFMA=mingwex_fma \
  -o repro fma_repro.c mingwex_fma.o -lm
./repro
```

That gives the same three lines. `cc -O2 -fno-builtin fma_repro.c -lm`
against glibc, and `cc -O2 -mfma` (which emits `vfmadd`), print `ok` for
all three.
