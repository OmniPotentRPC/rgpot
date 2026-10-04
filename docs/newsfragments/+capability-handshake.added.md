Backends loaded through the potential profile are checked before dispatch.
`ProfileSession::load` and `rgpot::abi::checked_load` read the backend's
`Capabilities` and refuse a different protocol family or major, a lower
protocol minor than required, another schema identity, an incompatible eindir
bridge ABI, layout or feature set, another DLPack revision, or a missing
energy or forces operation. Metadata a backend leaves unset is not checked.
