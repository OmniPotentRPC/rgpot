A non-periodic OpenMM session no longer wraps neighbour queries through the
placeholder box. A neighbour buffer shorter than the list is an error from
that session, and the error names the pair count the caller must provide.
