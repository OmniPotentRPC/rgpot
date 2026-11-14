`with_eam_al_openmp` builds a threaded EAM-Al gather. It is off by default. With it on, one thread and cells below 32 atoms still call the serial routine directly.
