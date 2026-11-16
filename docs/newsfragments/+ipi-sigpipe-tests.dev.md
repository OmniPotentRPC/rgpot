The i-PI socket helpers gain `sendNoSignal`, the one `send()` used by
`IPIPot`, `IPIDriver` and the mock peers in `IPIPotTest` and
`IPIDriverTest`. The `IPIPotTest` mock driver now sets `SO_NOSIGPIPE` on
its socket through `disableSigpipe`, as the library sockets and the
`IPIDriverTest` server already did, so a server that closes mid-exchange
cannot kill the test process where `MSG_NOSIGNAL` is missing. New cases
close the peer and then make the other side write, with `SIGPIPE` at its
default action: the server's next `STATUS` and its `EXIT`, and the
driver's `FORCEREADY`, each fail as "disconnected".
