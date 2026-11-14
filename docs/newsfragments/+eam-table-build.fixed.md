`-Dwith_eam_table=true` builds again.
`EAMTablePot.cc` included `rgpot/nlist/PairListCache.hpp`, which is not in
the tree; it now uses the neighbour layer in `rgpot/nlist/NeighbourList.hpp`.
