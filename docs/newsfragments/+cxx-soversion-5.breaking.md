librgpot's ELF soversion is 5. LJPot and MorsePot each hold a neighbour
list and a cached pair table, so their C++ layout is no longer the
soversion 4 layout. A caller built against soversion 4 does not load
librgpot.so.5. librgpot_mpi stays at soversion 3.
