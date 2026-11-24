The molecular-dynamics session entry points and operation tables add optional
neighbour-query and neighbour-setter tail slots while preserving compatibility
with older callers. The slots are on the C++ session operation table. They are
not exports of the rgpot-core C ABI, so that minor is unchanged and the C++
soversion stays 9.
