<!-- Copyright 2023--present rgpot developers -->
`potserv` and `rgpot_rpc_server_start` now serve `Potential.neighbours` over Cap'n Proto RPC using `rgpot::nlist::build`, advertising `Capabilities.Operation.neighbors` at ordinal 11 and populating `PotentialResult.neighborList` when `ForceInput.neighbors` is set.
The C bridge (`pot_neighbours`), the Rust core (`RpcClient::neighbours`, `rgpot_rpc_neighbours`), the C++ wrapper (`rgpot::RpcClient::neighbours`) and `rgpot.ase.RgpotRpcCalculator.neighbours` expose the call to clients.
