/-
Knuth's TwoSum, the exact-arithmetic skeleton.
Ogita, Rump and Oishi, SIAM J. Sci. Comput. 26, 1955 (2005),
doi:10.1137/030601818.
The binary64 rounding argument is not proved here.
-/

namespace Rpot

/-- Steps of TwoSum. Each operation is exact integer arithmetic. -/
def twoSum (a b : Int) : Int × Int :=
  let s := a + b
  let a' := s - b
  let b' := s - a'
  let da := a - a'
  let db := b - b'
  (s, da + db)

/-- In exact arithmetic the error term restores the sum. -/
theorem twoSum_telescopes (a b : Int) :
    (twoSum a b).1 + (twoSum a b).2 = a + b := by
  simp only [twoSum]
  omega

end Rpot
