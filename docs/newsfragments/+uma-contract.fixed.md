UmaPot checks the charge, spin, task and composition embedded in an AOTI
package against its `UmaConfig` and the input atoms on every force call, and
throws `rgpot::UmaContractError` on a mismatch. Without the check, a config
whose charge or spin differs from the package's evaluates a different
potential energy surface and reports no error.
