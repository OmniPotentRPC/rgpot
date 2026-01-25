let run_script = fun script =>
  {
    shell = "pixi run bash -e {0}",
    run = script
  }
in

# Generate dynamic server path for integration tests
let get_server_bin = "${{ matrix.sys == 'meson' && 'bbdir/CppCore/rgpot/rpc/potserv' || 'build/potserv' }}"
in

# --- Flag Generation Logic ---

# Generate the Meson configuration command
let meson_conf = m%"
  meson setup bbdir \
    -Dwith_tests=True \
    -Dwith_examples=False \
    -Dwith_xtensor=False \
    -Dwith_eigen=False \
    -Dwith_rpc=${{ matrix.rpc }} \
    -Dwith_cache=${{ matrix.cache }} \
    -Dpure_lib=False
"% in

# Generate the CMake configuration command
let cmake_conf = m%"
  cmake -S . -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DPOTLIB_BUILD_TESTS=True \
    -DPOTLIB_BUILD_EXAMPLES=False \
    -DPOTLIB_WITH_XTENSOR=OFF \
    -DPOTLIB_WITH_EIGEN=OFF \
    -DPOTLIB_WITH_RPC=${{ matrix.rpc }} \
    -DPOTLIB_WITH_CACHE=${{ matrix.cache }} \
    -DPOTLIB_PURE_LIB=OFF
"% in

# --- Steps Definitions ---

let build_steps = [
  # 1. Meson Build & Test
  # '&' merges the script definition with the name/if metadata
  run_script m%"
    %{meson_conf}
    meson compile -C bbdir
    meson test -C bbdir --print-errorlogs
  "% & {
    name = "Build & Test (Meson)",
    "if" = "${{ matrix.sys == 'meson' }}"
  },

  # 2. CMake Build & Test
  run_script m%"
    %{cmake_conf}
    cmake --build build -j
    ctest --test-dir build --output-on-failure
  "% & {
    name = "Build & Test (CMake)",
    "if" = "${{ matrix.sys == 'cmake' }}"
  }
]
in

# Integration test only runs if RPC was enabled
let integration_step = [
  run_script m%"
    uv pip install numpy pycapnp
    uv run python tests/rpc_integ.py --server-bin %{get_server_bin}
  "% & {
    name = "RPC Integration Test",
    "if" = "${{ matrix.rpc == 'true' }}"
  }
]
in

# --- Main Workflow ---

{
  name = "Build Matrix",

  # Trigger logic
  on = {
    push = { branches = ["main"] },
    pull_request = { branches = ["main"] },
  },

  # Concurrency Group
  # This cancels previous runs of the same PR/branch when a new commit is pushed.
  concurrency = {
    group = "${{ github.workflow }}-${{ github.ref }}",
    cancel-in-progress = true,
  },

  jobs = {
    # Main Matrix Build
    build_and_test = {
      # Dynamic name for easy UI scanning
      name = "${{ matrix.sys }} [RPC:${{ matrix.rpc }} CACHE:${{ matrix.cache }}]",
      runs-on = "${{ matrix.os }}",

      strategy = {
        fail-fast = false,
        matrix = {
          os = ["ubuntu-latest", "macos-latest"],
          sys = ["meson", "cmake"],
          # 2 systems * 2^2 options = 4 Jobs
          rpc     = ["true", "false"],
          cache   = ["true", "false"],
          # Tip: If the jobs are too many, use 'exclude' here to skip invalid combos.
          # e.g. exclude = [{ pure = "true", rpc = "true" }]
        }
      },

      steps = [
        {
          uses = "actions/checkout@v4",
          with = { submodules = "recursive", fetch-depth = 0 }
        },
        {
          uses = "prefix-dev/setup-pixi@v0.8.1",
          with = { pixi-version = "v0.62.1", cache = true }
        },
        {
          uses = "astral-sh/setup-uv@v7",
          with = { enable-cache = true, activate-environment = true }
        },

        # Inject the Build Steps
      ] @ build_steps
        # Inject the Integration Test (conditional on RPC)
        @ integration_step
    },

    # Specialized "Client Bridge" Stress Test
    # This job builds the Server (Heavy) and Client (Light) separately to verify interoperability.
    client_bridge_stress = {
      name = "Client Bridge Stress Test (Linux)",
      runs-on = "ubuntu-latest",
      steps = [
        {
          uses = "actions/checkout@v4",
          with = { submodules = "recursive", fetch-depth = 0 }
        },
        {
          uses = "prefix-dev/setup-pixi@v0.8.1",
          with = { pixi-version = "v0.62.1", cache = true }
        },

        run_script m%"
          echo "--- Building Server (Heavy) ---"
          meson setup bbdir_server \
                -Dwith_rpc=true \
                -Dwith_tests=false \
                -Dwith_examples=false
          meson compile -C bbdir_server
        "%,

        run_script m%"
          echo "--- Building Client Bridge (Light) ---"
          cmake -S . -B build_client \
                -DCMAKE_BUILD_TYPE=Release \
                -DPOTLIB_RPC_CLIENT_ONLY=ON \
                -DPOTLIB_BUILD_TESTS=ON
          cmake --build build_client -j
        "%,

        run_script m%"
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
        "%
      ]
    }
  }
}
