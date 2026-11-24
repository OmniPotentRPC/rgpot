<!-- Copyright 2023--present rgpot developers -->

A `Neighbours::update` that throws preserves any prior valid state and
generation intact, and leaves a default-constructed object empty without
assigning a generation.
A moved-from `Neighbours` is left as a default-constructed one with a cleared
generation, so its next update builds from its own coordinates instead of
reading coordinates it no longer holds.
