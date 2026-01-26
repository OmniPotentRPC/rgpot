
# Table of Contents

1.  [About](#about)
    1.  [Usage](#orgf05d8d3)
        1.  [Developing locally](#orgd2c8aa2)
        2.  [Handling actions](#orgc8b950e)
2.  [License](#orga1e3f52)


<a id="about"></a>

# About

![img](https://raw.githubusercontent.com/OmniPotentRPC/rgpot/refs/heads/main/branding/logo/rgpot_logo.webp)

Originally designed for interfacing easily to potentials implemented [in
eOn](http://theory.cm.utexas.edu/eon/), but has grown to be more flexible.


<a id="orgf05d8d3"></a>

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


<a id="orgd2c8aa2"></a>

### Developing locally

A `pre-commit` job is setup on CI to enforce consistent styles, so it is best to
set it up locally as well (using [pipx](https://pypa.github.io/pipx) for isolation):

    # Run before commiting
    uvx pre-commit run --all-files


<a id="orgc8b950e"></a>

### Handling actions

To keep the build matrix setup manageable, we use Nickel.

    pixi r gen-gha


<a id="orga1e3f52"></a>

# License

MIT, however note that some of the potentials are adapted from eOn which is
under a BSD 3-Clause License.

