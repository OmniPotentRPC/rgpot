A negative return code of `RGPOT_ENGINE_ERR_LOCK` (`-2`) now reports
`EngineError::Lock` with the operation name and poisons the handle like
other fatal codes. Handles of an engine that declares `reentrancy` 2 now
serialize their native calls process wide, so two handles on the same
engine library no longer run native code at the same time. The
`libpotlibc_engine` plugin frees the handle in `rgpot_engine_destroy` even
when its lock cannot be taken.
