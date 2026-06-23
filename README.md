
# Table of Contents

1.  [RPC Generalized Potential Library](#about)
    1.  [Usage](#org83174f7)
        1.  [Optional potentials (v1.1+)](#orgeb831f2)
        2.  [Developing locally](#org840d394)
        3.  [Handling actions](#orgdcffcf4)
2.  [License](#org65d1864)


<a id="about"></a>

# RPC Generalized Potential Library

![img](https://raw.githubusercontent.com/OmniPotentRPC/rgpot/refs/heads/main/branding/logo/rgpot_logo.webp)

Originally designed for interfacing easily to potentials implemented [in
eOn](http://theory.cm.utexas.edu/eon/), but has grown to be more flexible.


<a id="org83174f7"></a>

## Usage

Setting up and running tests:

    pixi shell
    meson setup bbdir -Dwith_tests=True --buildtype="debug"
    meson test -C bbdir

`cmake` works as well.

    cmake -B build -DRGPOT_BUILD_TESTS=ON -DRGPOT_BUILD_EXAMPLES=ON
    cmake --build build
    ctest --test-dir build

For building only the client, with CXX17 and CapnProto only:

    cmake -B build_client\
        -DRGPOT_RPC_CLIENT_ONLY=ON\
        -DRGPOT_BUILD_TESTS=ON
    cmake --build build_client
    # needs a server instance, so in another terminal
    # say from the meson.build
    ./bbdir/CppCore/rgpot/rpc/potserv 12345 LJ
    # now test
    ctest --test-dir build_client/ --output-on-failure


<a id="orgeb831f2"></a>

### Optional potentials (v1.1+)

Feature-gated backends stay off by default. Enable them with Meson options and
matching pixi environments:

<table border="2" cellspacing="0" cellpadding="6" rules="groups" frame="hsides">


<colgroup>
<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />

<col  class="org-left" />
</colgroup>
<thead>
<tr>
<th scope="col" class="org-left">Backend</th>
<th scope="col" class="org-left">Meson flag</th>
<th scope="col" class="org-left">pixi env</th>
<th scope="col" class="org-left">Notes</th>
</tr>
</thead>
<tbody>
<tr>
<td class="org-left"><code>XTBPot</code></td>
<td class="org-left"><code>-Dwith_xtb=true</code></td>
<td class="org-left"><code>xtbbld</code> / <code>tbbld</code></td>
<td class="org-left">GFNFF, GFN0/1/2-xTB via xtb</td>
</tr>

<tr>
<td class="org-left"><code>TBLitePot</code></td>
<td class="org-left"><code>-Dwith_tblite=true</code></td>
<td class="org-left"><code>tblitebld</code> / <code>tbbld</code></td>
<td class="org-left">GFN1, GFN2, IPEA1 via tblite</td>
</tr>

<tr>
<td class="org-left"><code>MetatomicPot</code></td>
<td class="org-left"><code>-Dwith_metatomic=true</code></td>
<td class="org-left"><code>metatomicbld</code> (linux-64)</td>
<td class="org-left">TorchScript models via metatomic/PyTorch + vesin</td>
</tr>
</tbody>
</table>

    # Tight-binding (xtb + tblite)
    pixi shell -e tbbld
    meson setup bbdir -Dwith_tests=true -Dwith_xtb=true -Dwith_tblite=true
    meson compile -C bbdir && meson test -C bbdir
    
    # Metatomic ML potentials (direct TorchScript load, no Python at eval time)
    pixi shell -e metatomicbld
    meson setup bbdir -Dwith_tests=true -Dwith_metatomic=true
    meson compile -C bbdir && meson test -C bbdir
    
    # Serve a metatomic model over RPC (path after the colon)
    ./bbdir/CppCore/rgpot/rpc/potserv 12345 Metatomic:CppCore/tests/data/lj38/lennard-jones.pt

Native units are **eV** / **Angstrom**. `rgpot/units.hpp` provides CODATA 2018
constants and a runtime expression parser for compound units. RPC clients may
set `lengthUnit` / `energyUnit` on `ForceInput`; the server negotiates
conversion at the boundary (defaults remain Angstrom / eV).


<a id="org840d394"></a>

### Developing locally

Hooks use [prek](https://prek.j178.dev) (`prek.toml`), not the legacy Python `pre-commit` CLI.
CI runs `prek run -a` on every PR (`.github/workflows/prek.yml`). Locally:

    prek install                 # one-time git shim install
    prek run -a --config prek.toml
    # or via pixi:
    pixi r prek-install && pixi r prek


<a id="orgdcffcf4"></a>

### Handling actions

To keep the build matrix setup manageable, we use Nickel.

    pixi r gen-gha


<a id="org65d1864"></a>

# License

MIT, however note that some of the potentials are adapted from eOn which is
under a BSD 3-Clause License. The unit expression parser in `units.cc` is
derived from metatomic-torch (BSD-3-Clause, metatensor developers).

