# Audit/reference mirror of orchestrator docs_snapper/docs_lychee (not exported as live workflow).
# Live: .github/workflows/ci-orchestrator.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "Docs quality (snapper + lychee)",
  on = {
    push = { branches = ["main"] },
    pull_request = {},
    schedule = [{ cron = "15 7 * * 1" }],
  },
  concurrency = S.concurrency_ref "docs-quality",
  permissions = { contents = "read" },
  jobs = {
    snapper = {
      name = "snapper semantic line breaks",
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        {
          name = "Install snapper-fmt",
          run = m%"
            # Pin v0.7.8: latest tag can ship without assets (404 on installer).
            SNAPPER_VER="${SNAPPER_VER:-0.7.8}"
            curl -fsSL -o /tmp/snapper-fmt.tar.xz \
              "https://github.com/TurtleTech-ehf/snapper/releases/download/v${SNAPPER_VER}/snapper-fmt-x86_64-unknown-linux-gnu.tar.xz"
            mkdir -p "$HOME/.local/bin"
            tar -xJf /tmp/snapper-fmt.tar.xz -C /tmp
            BIN=$(find /tmp -maxdepth 3 -type f -name snapper-fmt 2>/dev/null | head -1)
            install -m 755 "$BIN" "$HOME/.local/bin/snapper-fmt"
            ln -sf "$HOME/.local/bin/snapper-fmt" "$HOME/.local/bin/snapper"
            echo "$HOME/.local/bin" >> "$GITHUB_PATH"
            export PATH="$HOME/.local/bin:$PATH"
            snapper --version || snapper-fmt --version
          "%,
        },
        {
          name = "Check prose formatting",
          run = m%"
            set -euo pipefail
            SNAPPER_BIN="$(command -v snapper || command -v snapper-fmt)"
            mapfile -t files < <(find docs/orgmode -type f -name '*.org' ! -path '*/.pixi/*' | sort)
            if [ "${#files[@]}" -eq 0 ]; then
              echo "No org prose files found"
              exit 0
            fi
            printf 'Checking %s org file(s)\n' "${#files[@]}"
            if ! out=$("$SNAPPER_BIN" --check --config .snapper.toml --format org "${files[@]}" 2>&1); then
              printf '%s\n' "$out"
              exit 1
            fi
            printf '%s\n' "$out"
            if printf '%s\n' "$out" | grep -q 'would reformat:'; then
              echo "::error::snapper would reformat prose; run \`pixi r snapper-fix\` and commit"
              exit 1
            fi
          "%,
        },
      ],
    },
    lychee = {
      name = "lychee link check",
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_minimal,
        {
          name = "Restore lychee cache",
          uses = P.cache,
          with = {
            path = ".lycheecache",
            key = "lychee-${{ runner.os }}-${{ hashFiles('.lychee.toml', 'README.md', 'docs/**/*.org', 'docs/**/*.md') }}",
            "restore-keys" = "lychee-${{ runner.os }}-",
          },
        },
        {
          name = "Link check",
          uses = P.lychee,
          with = {
            args = m%"
              --config .lychee.toml
              --cache
              --max-cache-age 2d
              --cache-exclude-status 429,500,502,503,504
              --no-progress
              README.md
              CHANGELOG.md
              CODE_OF_CONDUCT.md
              readme_src.org
              docs/orgmode/**/*.org
              rgpot-core/README.md
              rgpot-core/README.org
            "%,
            fail = true,
          },
          env = { GITHUB_TOKEN = "${{ secrets.GITHUB_TOKEN }}" },
        },
      ],
    },
  },
}
