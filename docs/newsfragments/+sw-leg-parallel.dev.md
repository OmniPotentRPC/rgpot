With `with_eam_al_openmp`, each Stillinger-Weber leg factor is computed on its own pair. The stored values match the serial loop, so the forces do not depend on the thread count.
