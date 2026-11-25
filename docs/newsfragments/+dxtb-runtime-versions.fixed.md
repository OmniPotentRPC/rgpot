The dxtb worker checks the dxtb and tad-libcint versions of the Python it
runs in before serving a request. A wrong or missing package now fails
creation with a message naming the executable and what it lacks, instead of
reporting that the worker exited.
