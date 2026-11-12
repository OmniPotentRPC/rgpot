The eonviz environment installs readcon 0.17.1 from PyPI.
That wheel's RoundTrip writer matches the vendored restart golden.
atoms_to_con writes one CON frame with the RoundTrip writer, precision omitted and canonical left off.
The UMA exporter reads CON files with atoms_from_con.
CI compares those writes with the byte-exact restart golden.
