<!-- Copyright 2023--present rgpot developers -->

A `Neighbours::update` that throws leaves the object as a default-constructed
one with a new generation.
Before, the reference coordinates of the last build stayed beside bins and a
list the failed call had cleared.
A repeated call on a cell without a reciprocal basis then returned without the
error, and a later call with the last good coordinates reused the cleared state
and found no pairs.
A moved-from `Neighbours` is also left as a default-constructed one, so its next
update builds instead of reading the coordinates it no longer holds.
