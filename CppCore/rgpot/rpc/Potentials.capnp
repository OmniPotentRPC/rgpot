@0xbd1f89fa17369103;

# Design kanged from eOn v4 C-style structs
# TODO(rg): Should be more object oriented

struct ForceInput {
 natm   @0 :Int32; # TODO(rg): Do we really need this..
 pos    @1 :List(Float64);
 atmnrs @2 :List(Int32);
 box    @3 :List(Float64);
}

struct PotentialResult {
  energy @0: Float64;
  forces @1: List(Float64);
}

interface CuH2Pot {
  calculate @0 (fip :ForceInput)
    -> (result :PotentialResult);
}
