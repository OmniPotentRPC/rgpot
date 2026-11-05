`scripts/cpu_scaling.py` writes strong, one-cell and weak CPU scaling CSVs.
Each CSV header names the host, the compilers and the build options.
Every row is checked bit for bit against a 1-thread call.
It runs only when called and changes no library behaviour.
