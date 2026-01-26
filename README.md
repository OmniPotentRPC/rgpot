
# Table of Contents

1.  [RPC Generalized Potential Library](#about)
    1.  [Usage](#org266167a)
        1.  [Developing locally](#org2b2a584)
        2.  [Handling actions](#org974eb42)
2.  [License](#org826bc28)


<a id="about"></a>

# RPC Generalized Potential Library

![img](https://raw.githubusercontent.com/OmniPotentRPC/rgpot/refs/heads/main/branding/logo/rgpot_logo.webp)

Originally designed for interfacing easily to potentials implemented [in
eOn](http://theory.cm.utexas.edu/eon/), but has grown to be more flexible.


<a id="org266167a"></a>

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


<a id="org2b2a584"></a>

### Developing locally

A `pre-commit` job is setup on CI to enforce consistent styles, so it is best to
set it up locally as well (using [uvx](https://docs.astral.sh/uv/guides/tools/) for isolation):

    # Run before commiting
    uvx pre-commit run --all-files


<a id="org974eb42"></a>

### Handling actions

To keep the build matrix setup manageable, we use Nickel.

    pixi r gen-gha


<a id="org826bc28"></a>

# License

MIT, however note that some of the potentials are adapted from eOn which is
under a BSD 3-Clause License.

