The open Lennard-Jones cluster kernel allocates its scratch buffer
with ``_aligned_malloc`` on Windows. MSVC does not provide
``std::aligned_alloc``.
