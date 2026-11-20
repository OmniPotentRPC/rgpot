`SuttonChenPot` keys its stored-pair workspace on `Neighbours::generation()`
instead of the neighbour-list object's address, so a potential constructed at a
recycled address does not reuse another object's pair index or slice splits.
