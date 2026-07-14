# Audit/reference mirror of orchestrator prek job (not exported as live workflow).
# Live: .github/workflows/ci-orchestrator.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "prek",
  on = S.on_main_pr_any,
  concurrency = S.concurrency_ref "prek",
  jobs = {
    prek = {
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_depth 0,
        {
          name = "Install environment",
          uses = P.setup_pixi,
          with = {
            "pixi-version" = P.pixi_version,
            cache = true,
            "cache-write" = "${{ github.event_name == 'push' && github.ref_name == 'main' }}",
            "activate-environment" = true,
          },
        },
        {
          name = "Install prek",
          run = m%"
            if ! command -v prek >/dev/null 2>&1; then
              curl --proto '=https' --tlsv1.2 -LsSf \
                https://github.com/j178/prek/releases/latest/download/prek-installer.sh | sh
              echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
            fi
            prek --version
          "%,
        },
        {
          name = "Install system hook deps",
          run = m%"
            sudo apt-get update -qq
            sudo apt-get install -y -qq cppcheck clang-format-16 || \
              sudo apt-get install -y -qq cppcheck clang-format
            # Pin v0.7.8: latest tag can ship without assets (404 on installer).
            SNAPPER_VER="${SNAPPER_VER:-0.7.8}"
            if ! command -v snapper >/dev/null 2>&1 && ! command -v snapper-fmt >/dev/null 2>&1; then
              curl -fsSL -o /tmp/snapper-fmt.tar.xz \
                "https://github.com/TurtleTech-ehf/snapper/releases/download/v${SNAPPER_VER}/snapper-fmt-x86_64-unknown-linux-gnu.tar.xz"
              mkdir -p "$HOME/.local/bin"
              tar -xJf /tmp/snapper-fmt.tar.xz -C /tmp
              BIN=$(find /tmp -maxdepth 3 -type f -name snapper-fmt 2>/dev/null | head -1)
              install -m 755 "$BIN" "$HOME/.local/bin/snapper-fmt"
              ln -sf "$HOME/.local/bin/snapper-fmt" "$HOME/.local/bin/snapper"
              echo "$HOME/.local/bin" >> "$GITHUB_PATH"
            fi
            if command -v snapper-fmt >/dev/null 2>&1 && ! command -v snapper >/dev/null 2>&1; then
              ln -sf "$(command -v snapper-fmt)" "$HOME/.cargo/bin/snapper"
              echo "$HOME/.cargo/bin" >> "$GITHUB_PATH"
            fi
          "%,
        },
        S.run_pixi m%"
          if command -v snapper-fmt >/dev/null 2>&1 && ! command -v snapper >/dev/null 2>&1; then
            mkdir -p "$HOME/.local/bin"
            ln -sf "$(command -v snapper-fmt)" "$HOME/.local/bin/snapper"
            export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
          fi
          prek validate-config prek.toml
          prek run -a --config prek.toml
        "%
        & { name = "Run prek on all files" },
        {
          name = "Audit for large files",
          uses = P.large_file_auditor,
          with = { "file-size-threshold" = "1M" },
        },
      ],
    },
  },
}
