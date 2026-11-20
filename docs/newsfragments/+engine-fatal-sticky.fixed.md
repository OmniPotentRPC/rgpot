A fatal engine failure now stays fatal in the Rust engine session. The first
fatal callback failure is kept instead of being replaced by a later recoverable
one, a callback lock poisoned by an earlier panic is a fatal failure and no
callback runs again, and an engine return code below zero is always reported as
fatal even when the host recorded a recoverable failure first. A second fatal
report returns the failure that was stored first.
