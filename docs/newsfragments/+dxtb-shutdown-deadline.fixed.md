Destroying a dxtb engine no longer waits forever on a worker that does not
acknowledge shutdown or does not exit. The host waits a bounded time for the
worker, then kills and reaps it.
