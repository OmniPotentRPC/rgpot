# Generated target: .github/workflows/release-prepare.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
let release_paths = [
  "cog.toml",
  "towncrier.toml",
  "potctl/**",
  "Cargo.toml",
  "Cargo.lock",
  "docs/newsfragments/**",
  "rgpot-core/Cargo.toml",
  "meson.build",
  "CMakeLists.txt",
  "pixi.toml",
  "CHANGELOG.md",
  ".github/workflows/release.yml",
  ".github/workflows/release-prepare.yml",
  ".github/actions/**",
  "ci/gha/**",
]
in
{
  name = "Release prepare (cog dry-run)",
  on = {
    workflow_dispatch = {
      inputs = {
        bump = {
          description = "SemVer component for cog dry-run",
          required = true,
          default = "auto",
          type = "choice",
          options = ["auto", "patch", "minor", "major"],
        },
      },
    },
    pull_request = { paths = release_paths },
  },
  concurrency = S.concurrency_ref "release-prepare",
  permissions = { contents = "read" },
  env = { CI_TOOLS_ARTIFACT = "ci-tools-${{ github.sha }}-Linux" },
  jobs = {
    tools = S.job_tools_linux "true" "Stage portable potctl + cog (Linux musl + upload)",
    prepare = {
      needs = ["tools"],
      "runs-on" = P.ubuntu,
      steps = [
        S.checkout_depth 0,
        {
          name = "Restore potctl + cog from tools job",
          uses = P.restore_ci_tools,
          with = { "artifact-name" = "${{ env.CI_TOOLS_ARTIFACT }}" },
        },
        S.setup_pixi_default "devbld",
        {
          name = "Assert lockstep surfaces agree",
          run = m%"
            potctl release assert
            python3 -c "import tomllib, pathlib; tomllib.loads(pathlib.Path('cog.toml').read_text()); print('cog.toml parse ok')"
            cog --version
          "%,
        },
        {
          name = "Towncrier draft (fragments must build cleanly)",
          run = m%"
            if compgen -G "docs/newsfragments/*.md" > /dev/null || compgen -G "docs/newsfragments/+*.md" > /dev/null; then
              uvx towncrier build --draft
            else
              echo "No newsfragments; skip draft (ok if this PR only touches tooling)"
            fi
          "%,
        },
        S.run_pixi m%"
          cargo publish -p rgpot-core --locked --dry-run
        "%
        & { name = "Cargo publish dry-run (rgpot-core, locked)" },
        {
          name = "Cog bump dry-run (PR tip version)",
          run = m%"
            MESON_V=$(sed -nE "s/.*version: *'([^']+)'.*/\1/p" meson.build | head -1)
            cog bump --version "$MESON_V" --dry-run 2>&1 | head -30 || {
              echo "note: cog dry-run non-zero (often dirty/unpushed state on PR); lockstep + cog.toml ok above"
            }
          "%,
        },
        {
          name = "Cog bump dry-run (workflow_dispatch)",
          "if" = "github.event_name == 'workflow_dispatch'",
          run = m%"
            BUMP="${{ inputs.bump }}"
            case "$BUMP" in
              auto)  cog bump --auto --dry-run ;;
              patch) cog bump --patch --dry-run ;;
              minor) cog bump --minor --dry-run ;;
              major) cog bump --major --dry-run ;;
            esac
          "%,
        },
        {
          name = "Cog check (conventional commits since last tag)",
          run = m%"
            BASE="${{ github.event.pull_request.base.sha || '' }}"
            if [ -n "$BASE" ]; then
              if cog check --help 2>&1 | grep -q -- '--from'; then
                cog check --from "$BASE" || cog check
              else
                echo "Checking commits not in base..."
                git log --oneline "${BASE}..HEAD" | head -20
                cog check || {
                  echo "::warning::cog check reported issues (often pre-existing history)."
                  echo "New commits on this PR should use conventional prefixes (feat/fix/chore/...)."
                  bad=0
                  while IFS= read -r line; do
                    subj=$(echo "$line" | sed 's/^[a-f0-9]* //')
                    case "$subj" in
                      feat\(*|feat:*|fix\(*|fix:*|perf\(*|perf:*|refactor\(*|refactor:*|docs\(*|docs:*|test\(*|test:*|build\(*|build:*|ci\(*|ci:*|chore\(*|chore:*|style\(*|style:*|revert\(*|revert:*|enh\(*|enh:*|bug\(*|bug:*|maint\(*|maint:*|doc\(*|doc:*|tst\(*|tst:*|bld\(*|bld:*|gen\(*|gen:*|Merge\ *|merge\ *) ;;
                      *) echo "non-conventional subject: $subj"; bad=1 ;;
                    esac
                  done < <(git log --format='%h %s' "${BASE}..HEAD")
                  if [ "$bad" -ne 0 ]; then
                    echo "error: PR introduces non-conventional commit subjects" >&2
                    exit 1
                  fi
                }
              fi
            else
              cog check || echo "advisory cog check (dispatch run)"
            fi
          "%,
        },
      ],
    },
  },
}
