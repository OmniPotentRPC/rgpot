# Build Matrix workflow generator (Nickel → .github/workflows/build.yml).
#
# Regenerate:  pixi r -e cigen gen-gha
# Format:      pixi r -e cigen nickel format ci/gha/workflow.ncl
#
# Keep this file in lockstep with potctl CI helpers:
#   - potctl ci darwin-env  (macOS host clang/SDK + pixi dep paths)
#   - potctl ci preflight   (via ensure-potctl composite action)
#   - setup-ci-tools / ensure-potctl (portable musl Linux + fat macOS potctl)
let PIXi_VERSION = "v0.62.1" in
let CHECKOUT = "actions/checkout@v6" in
let SETUP_PIXI = "prefix-dev/setup-pixi@v0.9.4" in
let CACHE = "actions/cache@v5" in
let INSTALL_ACTION = "taiki-e/install-action@v2" in

# --- Small step builders ----------------------------------------------------
let run_pixi = fun script =>
  {
    shell = "pixi run bash -e {0}",
    run = script,
  }
in

let checkout_full = {
  uses = CHECKOUT,
  with = { submodules = "recursive", "fetch-depth" = 0 },
}
in

let setup_pixi = fun envs =>
  {
    uses = SETUP_PIXI,
    with = {
      "pixi-version" = PIXi_VERSION,
      cache = true,
      environments = envs,
      "activate-environment" = "devbld",
    },
  }
in

# potctl once per OS (Linux musl / macOS lipo fat); consumers restore via ensure-potctl.
let ensure_potctl = {
  name = "Ensure potctl (OS-matched portable/fat artifact)",
  uses = "./.github/actions/ensure-potctl",
  with = { "artifact-prefix" = "ci-tools-${{ github.sha }}" },
}
in

let ccache_step = {
  name = "Restore Compiler Cache",
  uses = CACHE,
  with = {
    path = ".ccache",
    key = "ccache-${{ runner.os }}-${{ matrix.sys }}-${{ github.sha }}",
    "restore-keys" = "ccache-${{ runner.os }}-${{ matrix.sys }}-",
  },
}
in

let ccache_env = {
  CCACHE_DIR = "${{ github.workspace }}/.ccache",
  CCACHE_COMPILERCHECK = "content",
  CCACHE_MAXSIZE = "500M",
}
in

# Shared prologue for matrix meson/cmake: SDKROOT on macOS, then potctl darwin-env.
let darwin_env_prelude = m%"
  # macOS/pixi host clang/SDK via potctl (unit-tested); Linux is a no-op eval.
  if [ "${{ runner.os }}" = "macOS" ]; then
    export SDKROOT="$(xcrun --show-sdk-path)"
  fi
  eval "$(potctl ci darwin-env)"
"%
in

let meson_build_test =
  run_pixi m%"
  %{darwin_env_prelude}
  meson setup bbdir \
    ${POTCTL_MESON_EXTRA:-} \
    -Dwith_tests=True \
    -Dwith_examples=False \
    -Dwith_xtensor=False \
    -Dwith_eigen=False \
    -Dwith_rpc=${{ matrix.rpc }} \
    -Dwith_cache=${{ matrix.cache }} \
    -Dpure_lib=False
  meson compile -C bbdir
  meson test -C bbdir --print-errorlogs
  ccache -s
"%
  & {
    name = "Build & Test (Meson)",
    "if" = "${{ matrix.sys == 'meson' }}",
  }
in

let cmake_build_test =
  run_pixi m%"
  CMAKE_EXTRA=""
  %{darwin_env_prelude}
  if [ "${{ runner.os }}" = "macOS" ]; then
    CMAKE_EXTRA="-DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DCMAKE_OSX_SYSROOT=${SDKROOT}"
    if [ -n "${CONDA_PREFIX:-}" ]; then
      CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_PREFIX_PATH=${CONDA_PREFIX} -DCMAKE_BUILD_RPATH=${CONDA_PREFIX}/lib -DCMAKE_INSTALL_RPATH=${CONDA_PREFIX}/lib"
    fi
    if [ -n "${FC:-}" ]; then
      CMAKE_EXTRA="${CMAKE_EXTRA} -DCMAKE_Fortran_COMPILER=${FC}"
    fi
  fi
  cmake -S . -B build \
    -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
    -DCMAKE_C_COMPILER_LAUNCHER=ccache \
    -DCMAKE_BUILD_TYPE=Release \
    -DRGPOT_BUILD_TESTS=True \
    -DRGPOT_BUILD_EXAMPLES=False \
    -DRGPOT_WITH_XTENSOR=OFF \
    -DRGPOT_WITH_EIGEN=OFF \
    -DRGPOT_WITH_RPC=${{ matrix.rpc }} \
    -DRGPOT_WITH_CACHE=${{ matrix.cache }} \
    -DRGPOT_PURE_LIB=OFF \
    $CMAKE_EXTRA
  cmake --build build -j
  ctest --test-dir build --output-on-failure
  ccache -s
"%
  & {
    name = "Build & Test (CMake)",
    "if" = "${{ matrix.sys == 'cmake' }}",
  }
in

let rpc_integration =
  run_pixi m%"
  pixi r -e rpctest \
  python tests/rpc_integ.py --server-bin ${{ matrix.sys == 'meson' && 'bbdir/CppCore/rgpot/rpc/potserv' || 'build/potserv' }}
"%
  & {
    name = "RPC Integration Test",
    "if" = "${{ matrix.rpc == 'true' && matrix.sys == 'meson' && matrix.os == 'ubuntu-latest' }}",
  }
in

# --- Jobs -------------------------------------------------------------------
let job_ci_tools = {
  name = "ci-tools (potctl / ${{ matrix.os }})",
  "runs-on" = "${{ matrix.os }}",
  strategy = {
    "fail-fast" = false,
    matrix = { os = ["ubuntu-latest", "macos-latest"] },
  },
  steps = [
    { uses = CHECKOUT },
    {
      name = "Stage portable/fat potctl",
      uses = "./.github/actions/setup-ci-tools",
      with = {
        "upload-artifact" = "true",
        "artifact-name" = "ci-tools-${{ github.sha }}-${{ runner.os }}",
      },
    },
  ],
}
in

let job_build_and_test = {
  needs = ["ci-tools"],
  name = "${{ matrix.sys }} / ${{ matrix.os }} [RPC:${{ matrix.rpc }} CACHE:${{ matrix.cache }}]",
  "runs-on" = "${{ matrix.os }}",
  env = ccache_env,
  strategy = {
    "fail-fast" = false,
    matrix = {
      os = ["ubuntu-latest", "macos-latest"],
      sys = ["meson", "cmake"],
      rpc = ["true", "false"],
      cache = ["true", "false"],
    },
  },
  steps =
    [
      checkout_full,
      ensure_potctl,
      setup_pixi "devbld rpctest",
      ccache_step,
    ]
    @ [meson_build_test, cmake_build_test, rpc_integration],
}
in

let job_rust_tests = {
  needs = ["ci-tools"],
  name = "Rust Core Tests (${{ matrix.os }})",
  "runs-on" = "${{ matrix.os }}",
  strategy = {
    "fail-fast" = false,
    matrix = { os = ["ubuntu-latest", "macos-latest"] },
  },
  steps = [
    checkout_full,
    ensure_potctl,
    setup_pixi "devbld",
    {
      name = "Install cargo-nextest",
      uses = INSTALL_ACTION,
      with = { tool = "cargo-nextest" },
    },
    {
      name = "potctl unit tests",
      run = "cargo test -p potctl",
    },
    run_pixi m%"
      cargo nextest run -p rgpot-core
      cargo nextest run -p rgpot-core --all-features
    "%
    & { name = "Run Rust tests (nextest)" },
    run_pixi m%"
      cargo test --doc -p rgpot-core
    "%
    & { name = "Run Rust doc-tests" },
  ],
}
in

let job_client_bridge_stress = {
  needs = ["ci-tools"],
  name = "Client Bridge Stress Test (Linux)",
  "runs-on" = "ubuntu-latest",
  env = {
    CCACHE_DIR = "${{ github.workspace }}/.ccache",
    CCACHE_COMPILERCHECK = "content",
  },
  steps = [
    checkout_full,
    ensure_potctl,
    setup_pixi "devbld rpctest",
    ccache_step,
    run_pixi m%"
      echo "--- Building Server (Heavy) ---"
      meson setup bbdir_server \
            -Dwith_rpc=true \
            -Dwith_tests=false \
            -Dwith_examples=false
      meson compile -C bbdir_server
    "%,
    run_pixi m%"
      echo "--- Building Client Bridge (Light) ---"
      cmake -S . -B build_client \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_BUILD_TYPE=Release \
            -DRGPOT_RPC_CLIENT_ONLY=ON \
            -DRGPOT_BUILD_TESTS=ON
      cmake --build build_client -j
    "%,
    run_pixi m%"
      echo "--- Starting Server ---"
      # Start server in background on port 12345 using LJ potential
      ./bbdir_server/CppCore/rgpot/rpc/potserv 12345 LJ &
      SERVER_PID=$!

      # Wait for server to be ready
      sleep 2

      echo "--- Running Client Stress Tests ---"
      # This runs BridgeStressTest, which connects to localhost:12345
      ctest --test-dir build_client --output-on-failure

      echo "--- Teardown ---"
      kill $SERVER_PID
    "%,
  ],
}
in

# --- Workflow root ----------------------------------------------------------
{
  name = "Build Matrix",

  on = {
    push = { branches = ["main"] },
    pull_request = { branches = ["main"] },
  },

  concurrency = {
    group = "${{ github.workflow }}-${{ github.ref }}",
    "cancel-in-progress" = true,
  },

  jobs = {
    # potctl once per OS: Linux musl (portable) + macOS fat (arm64+x86_64 lipo).
    # Consumer jobs restore ci-tools-<sha>-${{ runner.os }} (not a Linux ELF on Darwin).
    "ci-tools" = job_ci_tools,
    build_and_test = job_build_and_test,
    rust_tests = job_rust_tests,
    client_bridge_stress = job_client_bridge_stress,
  },
}
