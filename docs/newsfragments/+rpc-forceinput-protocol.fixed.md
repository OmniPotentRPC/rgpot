RPC clients now check the advertised protocol minor before sending the orbital-key `ForceInput` fields.
Those fields first appeared in minor 2, and a keyed request to an older server fails with the required and advertised minors.
`ForceInput.neighbors` needs no minor above 0, and requests without an orbital key stay valid against a minor-1 server.
