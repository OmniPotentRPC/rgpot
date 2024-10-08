@0xbd1f89fa17369103;

struct Position {
  x @0 :Float64;
  y @1 :Float64;
  z @2 :Float64;
}

struct AtomMatrixRPC {
  positions @0 :List(Position);
}

struct BoxMatrix {
  box @0: List(Vector);
}

struct Vector {
  x @0: Float64;
  y @1: Float64;
  z @2: Float64;
}

struct AtomTypes {
  atomTypes @0: List(Int32);
}

struct PotentialResult {
  energy @0: Float64;
  forces @1: List(ForceVector);
}

struct ForceVector {
  x @0: Float64;
  y @1: Float64;
  z @2: Float64;
}

interface CuH2Pot {
  calculate @0 (positions :AtomMatrixRPC, atomTypes :AtomTypes, boxMatrix :BoxMatrix)
    -> (result :PotentialResult);
}
