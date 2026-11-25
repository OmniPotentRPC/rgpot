The dxtb engine no longer trusts the reply size its worker announces. A reply
larger than the request can produce, or a status message over 1 MiB, fails
the call with return code -1 and drops the worker instead of allocating and
reading it.
