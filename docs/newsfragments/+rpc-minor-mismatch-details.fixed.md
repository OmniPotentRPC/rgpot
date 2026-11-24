The Python RPC calculator no longer replaces a protocol minor mismatch with a generic calculate failure.
It raises `ProtocolMinorMismatchError`, a `RuntimeError` whose message and `required_minor`, `server_minor` and `client_minor` attributes report the minor the request needs, the one the server advertised and the one the client implements.
