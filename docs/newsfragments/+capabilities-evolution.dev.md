`CapabilitiesEvolution` encodes a message with the schema from before
`buildVersion` and `buildRevision` and decodes it with the vendored schema,
then encodes both fields and decodes that message with the older schema.
