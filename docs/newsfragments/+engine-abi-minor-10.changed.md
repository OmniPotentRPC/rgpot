The generic engine C ABI adds `rgpot_md_session_engine_create` for in-process
molecular-dynamics session engines and bumps `RGPOT_ENGINE_ABI_MINOR` and
`ENGINE_ABI_MINOR` to 10. Minor 9 is used by plugin-unload compatibility work.
No existing engine layout or call changes, so hosts at minor 10 continue to
serve plugins of minor 0 through 9 while older hosts refuse minor 10 plugins.
