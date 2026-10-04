# Generated target: .github/workflows/ci_docs.yml
let S = import "lib/steps.ncl" in
let P = S.pins in
{
  name = "Build and Deploy Documentation",
  concurrency = { group = "pages", "cancel-in-progress" = true },
  on = {
    push = { branches = ["main"], tags = ["*"] },
    pull_request = {},
    schedule = [{ cron = "0 8 * * 1" }],
  },
  permissions = { contents = "read" },
  jobs = {
    build_docs = {
      name = "Build documentation",
      "runs-on" = P.ubuntu,
      steps = [
        {
          name = "Checkout repository",
          uses = P.checkout,
          with = { "fetch-depth" = 0 },
        },
        S.setup_pixi_cache_write_main "docs",
        S.run_pixi m%"
          pixi r docbld
        "%
        & { name = "Generate Docs" },
        {
          name = "Upload artifact",
          uses = P.upload_artifact_v6,
          with = { name = "documentation", path = "docs/build" },
        },
      ],
    },
    deploy_docs = {
      name = "Deploy documentation",
      needs = ["build_docs"],
      "if" = "github.event_name == 'push'",
      "runs-on" = P.ubuntu,
      permissions = { contents = "write" },
      steps = [
        { uses = P.checkout, with = { "persist-credentials" = false } },
        {
          name = "Download documentation",
          uses = P.download_artifact_v6,
          with = { name = "documentation", path = "docs/build" },
        },
        {
          name = "Deploy",
          uses = P.gh_pages,
          with = {
            github_token = "${{ secrets.GITHUB_TOKEN }}",
            publish_dir = "docs/build",
            cname = "rgpot.rgoswami.me",
          },
        },
      ],
    },
  },
}
