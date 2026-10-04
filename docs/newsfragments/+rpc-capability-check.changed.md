The Rust `RpcClient` and the `pot_calculate` bridge call
`Potential.getCapabilities` before the first `calculate` and refuse a server
with incompatible capabilities. A server that does not implement the call is
refused with an error naming `Potential.getCapabilities` and the upgrade path;
the rgpot servers (`potserv`, `rgpot_rpc_server_start`) implement it. Servers
built on `Potentials.capnp` elsewhere need the call before an rgpot client
built from this release can use them.
