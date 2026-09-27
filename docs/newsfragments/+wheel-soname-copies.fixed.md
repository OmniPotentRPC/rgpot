Linux wheels import without a system OpenBLAS. auditwheel repaired the real librgpot library but not the librgpot.so.3 and librgpot.so copies that _core loads, so those still named libopenblas.so.0.
