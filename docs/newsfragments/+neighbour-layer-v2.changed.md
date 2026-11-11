Morse and ZBL take pairs from the shared neighbour layer. The separate
pair-list cache is gone. Stillinger-Weber and Tersoff use that layer for
every mode, including the default. One thread stays within 1e-12 of the
previous ordered walk. A streaming evaluation keeps a cell binning and a
per-atom scratch, and several threads walk those cells and build bonds
together.
