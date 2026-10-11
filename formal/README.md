# Formal statements

| Theorem | Anchor | Precondition |
| --- | --- | --- |
| `Rpot.twoSum_telescopes` | `two_sum` in `validation/eft/eft_diff.cc` | Exact integer arithmetic. Each step is the integer form of Knuth's TwoSum. |

`twoSum_telescopes` shows that the error term restores `a + b` when every operation is exact.
It does not model rounding.
The binary64 statement, including subnormals, signed zero and ties, is checked by `validation/eft/eft_diff.cc` against binary128.
A Lean proof of that floating-point statement is open.
Dekker's product split is checked in the same program against `std::fma` for normal inputs in `[2^-400, 2^400]` whose product is normal.
Its floating-point proof is open.
The split is not claimed for subnormals or for results that overflow.

The constants of the fused exp live in `rgpot-core/src/fp_constants.rs`.
`scripts/export_constants.py --check` compares them with `CppCore/rgpot/simd/exp.hpp` and `validation/golden/exp_constants.json`.

Ogita, Rump and Oishi, SIAM J. Sci. Comput. 26, 1955 (2005), doi:10.1137/030601818.
Boldo and Muller, IEEE Trans. Comput. 60, 176 (2011), doi:10.1109/tc.2010.139.
