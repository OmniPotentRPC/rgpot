<!-- Copyright 2023--present rgpot developers -->

`meson setup -Dwith_rust_core=true` without the fast pair list builds on the
workspace Rust version, 1.88. That build used to ask for the 1.89 toolchain
the pair list needs, and stopped when rustup had none. The 1.89 toolchain is
now used only when the pair list is on. On macOS every Rust archive, with or
without the pair list, is still written with `/usr/bin/ar`.
