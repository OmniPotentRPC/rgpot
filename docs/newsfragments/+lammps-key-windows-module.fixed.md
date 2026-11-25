On Windows, the `LammpsPot` result-cache key now hashes the final path of the
module that loaded, read from the loader, instead of the requested path. A
backend selected through the environment passes an empty path, so two such
backends with the same version, ABI revision and configuration used to share
one key.
