<!-- Copyright 2023--present rgpot developers -->

The generic engine C ABI minor is now 7, in `RGPOT_ENGINE_ABI_MINOR` and
`ENGINE_ABI_MINOR`, for the additive return code `RGPOT_ENGINE_ERR_LOCK` (-2).
An engine at minor 7 serves a host built against minor 7 or newer; a host built
against minor 6 refuses that newer engine.
