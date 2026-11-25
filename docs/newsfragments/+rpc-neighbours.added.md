`potserv` and `rgpot_rpc_server_start` now serve `Potential.neighbours` over
Cap'n Proto RPC using `rgpot::nlist::build`, advertising
`Capabilities.Operation.neighbors` at ordinal 11 and populating
`PotentialResult.neighborList` when `ForceInput.neighbors` is set. The C bridge
(`pot_neighbours`), the Rust core (`RpcClient::neighbours`,
`rgpot_rpc_neighbours`), the C++ wrapper (`rgpot::RpcClient::neighbours`) and
`rgpot.ase.RgpotRpcCalculator.neighbours` expose the call to clients.
`rgpot_rpc_neighbours` and `pot_neighbours` write the shift, vector and distance
columns only when the request asks for them, so a buffer for a column that is
not requested may be NULL or of any size.
