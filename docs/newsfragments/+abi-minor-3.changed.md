The C ABI minor of `rgpot-core` is now 3, in `RGPOT_ABI_VERSION_MINOR`, for
the additive engine and RPC Hessian-vector exports. A caller built against
minor 0, 1 or 2 still matches; a caller built against minor 3 is refused by an
older library.
