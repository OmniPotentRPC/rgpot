# Generated target: .github/workflows/build.yml
let S = import "lib/steps.ncl" in
let P = S.pins in

let meson_build_test = S.run_pixi m%"
  %{S.darwin_env_prelude}
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

let cmake_build_test = S.run_pixi m%"
  CMAKE_EXTRA=""
  %{S.darwin_env_prelude}
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

let rpc_integration = S.run_pixi m%"
  pixi r -e rpctest \
  python tests/rpc_integ.py --server-bin ${{ matrix.sys == 'meson' && 'bbdir/CppCore/rgpot/rpc/potserv' || 'build/potserv' }}
"%
& {
  name = "RPC Integration Test",
  "if" = "${{ matrix.rpc == 'true' && matrix.sys == 'meson' && matrix.os == 'ubuntu-latest' }}",
}
in

{
  name = "Build Matrix",
  on = S.on_main_pr,
  concurrency = {
    group = "${{ github.workflow }}-${{ github.ref }}",
    "cancel-in-progress" = true,
  },
  jobs = {
    "ci-tools" = S.job_ci_tools_os_matrix [P.ubuntu, P.macos],
    build_and_test = {
      needs = ["ci-tools"],
      name = "${{ matrix.sys }} / ${{ matrix.os }} [RPC:${{ matrix.rpc }} CACHE:${{ matrix.cache }}]",
      "runs-on" = "${{ matrix.os }}",
      env = S.ccache_env,
      strategy = {
        "fail-fast" = false,
        matrix = {
          os = [P.ubuntu, P.macos],
          sys = ["meson", "cmake"],
          rpc = ["true", "false"],
          cache = ["true", "false"],
        },
      },
      steps =
        [S.checkout_full, S.ensure_potctl, S.setup_pixi "devbld rpctest" "devbld", S.ccache_step_matrix_sys]
        @ [meson_build_test, cmake_build_test, rpc_integration],
    },
    rust_tests = {
      needs = ["ci-tools"],
      name = "Rust Core Tests (${{ matrix.os }})",
      "runs-on" = "${{ matrix.os }}",
      strategy = {
        "fail-fast" = false,
        matrix = { os = [P.ubuntu, P.macos] },
      },
      steps = [
        S.checkout_full,
        S.ensure_potctl,
        S.setup_pixi "devbld" "devbld",
        {
          name = "Install cargo-nextest",
          uses = P.install_action,
          with = { tool = "cargo-nextest" },
        },
        { name = "potctl unit tests", run = "cargo test -p potctl" },
        S.run_pixi m%"
          cargo nextest run -p rgpot-core
          cargo nextest run -p rgpot-core --all-features
        "%
        & { name = "Run Rust tests (nextest)" },
        S.run_pixi m%"
          cargo test --doc -p rgpot-core
        "%
        & { name = "Run Rust doc-tests" },
      ],
    },
    client_bridge_stress = {
      needs = ["ci-tools"],
      name = "Client Bridge Stress Test (Linux)",
      "runs-on" = P.ubuntu,
      env = {
        CCACHE_DIR = "${{ github.workspace }}/.ccache",
        CCACHE_COMPILERCHECK = "content",
      },
      steps = [
        S.checkout_full,
        S.ensure_potctl,
        S.setup_pixi "devbld rpctest" "devbld",
        S.ccache_step_matrix_sys,
        S.run_pixi m%"
          echo "--- Building Server (Heavy) ---"
          meson setup bbdir_server \
                -Dwith_rpc=true \
                -Dwith_tests=false \
                -Dwith_examples=false
          meson compile -C bbdir_server
        "%,
        S.run_pixi m%"
          echo "--- Building Client Bridge (Light) ---"
          cmake -S . -B build_client \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
                -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_BUILD_TYPE=Release \
                -DRGPOT_RPC_CLIENT_ONLY=ON \
                -DRGPOT_BUILD_TESTS=ON
          cmake --build build_client -j
        "%,
        S.run_pixi m%"
          echo "--- Starting Server ---"
          ./bbdir_server/CppCore/rgpot/rpc/potserv 12345 LJ &
          SERVER_PID=$!
          sleep 2
          echo "--- Running Client Stress Tests ---"
          ctest --test-dir build_client --output-on-failure
          echo "--- Teardown ---"
          kill $SERVER_PID
        "%,
      ],
    },
  },
}
