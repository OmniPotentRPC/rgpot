Engine calls now release the GIL before they take the per-handle call lock.
Concurrent force, HVP, jet or batch calls on one handle from a Python host
that holds the GIL, such as the dxtb per-instance engine, no longer deadlock
when a call returns. The component readers drop the GIL before that lock too.
