`rgpot_potential_set_peer_capabilities` installs the peer's `Capabilities`
message on a callback handle. An incompatible peer makes
`rgpot_potential_calculate`, `eindir_objective_eval` and
`eindir_objective_grad` fail before the callback runs, with
`peer refused: <first mismatch>` in `rgpot_last_error()`. A handle that never
receives a message keeps the callback contract.
