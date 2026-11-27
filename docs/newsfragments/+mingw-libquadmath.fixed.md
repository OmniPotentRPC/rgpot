A MinGW link of the shared library follows static libgfortran with
libquadmath, so quadmath_snprintf is resolved. The MSYS2 CI job
installs the gcc-libs package that depends on that runtime.
