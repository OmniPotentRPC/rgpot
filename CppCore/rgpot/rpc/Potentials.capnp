@0xbd1f89fa17369103;

# Design kanged from eOn [1] v4 C-style structs

struct ForceInput {
 pos    @0 :List(Float64); # natoms * 3
 atmnrs @1 :List(Int32);   # natoms
 box    @2 :List(Float64); # 9 (3x3)
}

struct PotentialResult {
  energy @0: Float64;
  forces @1: List(Float64); # natoms * 3
}

interface Potential {
  calculate @0 (fip :ForceInput)
    -> (result :PotentialResult);
}

# [1] https://eondocs.org
