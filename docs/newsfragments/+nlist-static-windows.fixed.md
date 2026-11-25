`Neighbours.hpp` no longer marks `Neighbours::nextGeneration`
`__declspec(dllimport)` for consumers on Windows, so a program linked to a
static `librgpot` resolves it.
A consumer of the shared library still resolves it through the import library.
