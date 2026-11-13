# Workflow-private home for asv. The machine file is
# $HOME/.asv-machine.json under benchflow/out/home, not the account home.
# Capture toolchain locations before HOME changes.
if [ -z "${CARGO_HOME:-}" ]; then
    export CARGO_HOME="${HOME}/.cargo"
fi
if [ -z "${RUSTUP_HOME:-}" ]; then
    export RUSTUP_HOME="${HOME}/.rustup"
fi
BENCH_HOME="${PWD}/benchflow/out/home"
mkdir -p "${BENCH_HOME}"
export HOME="${BENCH_HOME}"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
