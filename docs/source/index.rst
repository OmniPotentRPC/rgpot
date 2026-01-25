==============
``rgpot``
==============

High-performance chemical potential evaluation with RPC and persistent caching.

* Introduction
  ~rgpot~ provides a unified C++ interface for atomic potentials, featuring:
  - Distributed evaluation via **Cap'n Proto**.
  - Persistent disk caching via **RocksDB**.
  - Integration with the **eOn** client via ~eoncpotserv~.

* API Reference
  The following sections document the core C++ components.

* Potential Caching
  The ~PotentialCache~ class manages the persistence layer.

.. doxygenclass:: rgpot::cache::PotentialCache
   :project: rgpot
   :members:

* Base Structures
  The fundamental POD types used for data exchange.

.. doxygenstruct:: rgpot::ForceInput
   :project: rgpot
   :members:

.. doxygenstruct:: rgpot::ForceOut
   :project: rgpot
   :members:
