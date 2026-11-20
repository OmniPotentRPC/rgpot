<!-- Copyright 2023--present rgpot developers -->
The RPC client and server, and the C++ trampoline of `PotentialHandle`, no longer pass an unrequested stress through: a force-only request never carries one, and a stress request to a potential that cannot compute it fails with a clear message.
