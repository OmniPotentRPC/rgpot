`rgpot_md_session_create` and `CAbiMdSession::open_engine` set
`ops.struct_size` to `sizeof(rgpot_md_session_ops)` as the vtable byte capacity
before calling `rgpot_md_session_engine_create`, and the OpenMM session engine
writes at most `min(capacity, sizeof(rgpot_md_session_ops))` bytes while
reporting the filled byte count in `ops->struct_size`. This prevents a newer
engine plugin from writing past an older host's operation table.
