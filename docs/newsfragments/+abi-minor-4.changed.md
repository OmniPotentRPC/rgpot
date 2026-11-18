The C ABI minor of `rgpot-core` is now 4, in `RGPOT_ABI_VERSION_MINOR`, for
the additive engine and RPC Hessian-vector exports. A caller built against
minor 0, 1, 2 or 3 still matches; a caller built against minor 4 is refused by an
older library.
