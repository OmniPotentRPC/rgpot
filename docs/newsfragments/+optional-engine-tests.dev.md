Metatomic and potserv tests skip only when their path variables are unset. A variable set to a missing path fails the test, and `capnp` is imported only when a potserv test runs.
