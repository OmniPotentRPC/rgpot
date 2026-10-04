@0xd2f4b6a8c0e1f357;

# Capabilities as published before buildVersion and buildRevision were
# appended. The wire layout of every ordinal below matches Potentials.capnp;
# test_capabilities_wire.py decodes messages across the two definitions.

struct Capabilities {
  backendName    @0 :Text;  # ABI prefix, e.g. "nwchemc" or "cpmdc".
  backendVersion @1 :Text;  # `<p>_version()` string.
  abiVersion     @2 :Int32; # `<p>_abi_version()` value.
  available      @3 :Bool;  # `<p>_available()`: embed shell linked in.
  operations     @4 :List(Operation); # Calculate operations the ABI serves.
  loweredCommonFields @5 :List(Text); # CommonMethodSpec field names the overlay lowers.
  configKinds    @6 :List(Text); # PotentialConfig arms accepted, e.g. "nwchem".
  schemaVersion  @7 :Text;  # Potentials.capnp release the backend compiled against.
  protocolFamily @8 :Text;  # Stable RPC family, e.g. "rgpot.potentials".
  protocolMajor  @9 :UInt16; # Wire-incompatible changes increment this value.
  protocolMinor  @10 :UInt16; # Additive wire-compatible changes increment this value.
  schemaId       @11 :Text; # Stable Cap'n Proto schema identity.
  bridgeAbiMajor @12 :UInt16; # eindir objective ABI major revision.
  bridgeAbiMinor @13 :UInt16; # eindir objective ABI minor revision.
  bridgeLayout   @14 :UInt32; # eindir objective layout revision.
  dlpackMajor    @15 :UInt16; # DLPack callback major revision.
  dlpackMinor    @16 :UInt16; # DLPack callback minor revision.
  bridgeFeatures @17 :UInt64; # eindir bridge feature bitset.

  enum Operation {
    energy         @0;
    forces         @1;
    gradient       @2;
    hessian        @3;
    dipole         @4;
    polarizability @5;
    quadrupole     @6;
    stress         @7;
    optimize       @8;
    frequencies    @9;
  }
}
