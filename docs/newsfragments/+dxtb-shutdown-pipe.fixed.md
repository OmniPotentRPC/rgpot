The dxtb worker now finishes shutdown before the host closes its end of the
socket. A broken pipe during that shutdown is a clean close, and a broken
pipe during a live evaluation is still reported.
