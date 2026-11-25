A `Neighbours::update` that rejects a cell or a coordinate, or finds the
generation counter exhausted, throws before anything changes, so a built
object keeps its list and generation and a default-constructed object stays
empty without a generation.
An allocation that fails part way through an update leaves a
default-constructed object with a cleared generation, so the next update
rebuilds instead of reusing a partial list, and a copy assignment that throws
leaves its target unchanged.
A moved-from `Neighbours` is left as a default-constructed one with a cleared
generation, so its next update builds from its own coordinates instead of
reading coordinates it no longer holds.
