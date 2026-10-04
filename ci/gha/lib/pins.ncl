# Single place to bump action / pixi / tool pins for all generated workflows.
# Import via steps.ncl (`S.pins`) or `import "pins.ncl"` from modules in ci/gha/lib/.
# Workflow modules should not hardcode action refs; compose builders from steps.ncl.
{
  # --- toolchains / env managers ---
  pixi_version = "v0.70.2",

  # --- GitHub Actions (third-party / first-party) ---
  checkout = "actions/checkout@v6",
  setup_pixi = "prefix-dev/setup-pixi@v0.9.4",
  cache = "actions/cache@v5",
  install_action = "taiki-e/install-action@v2",
  rust_toolchain = "dtolnay/rust-toolchain@stable",
  rust_cache = "Swatinem/rust-cache@v2",
  upload_artifact_v4 = "actions/upload-artifact@v4",
  upload_artifact_v6 = "actions/upload-artifact@v6",
  download_artifact = "actions/download-artifact@v4",
  download_artifact_v6 = "actions/download-artifact@v6",
  gh_release = "softprops/action-gh-release@v2",
  gh_pages = "peaceiris/actions-gh-pages@v4",
  lychee = "lycheeverse/lychee-action@v2",
  doc_previewer = "HaoZeke/doc-previewer@v0.0.1",

  # --- in-repo composite actions (paths, not version pins) ---
  setup_ci_tools = "./.github/actions/setup-ci-tools",
  ensure_potctl = "./.github/actions/ensure-potctl",
  restore_ci_tools = "./.github/actions/restore-ci-tools",

  # --- common runners ---
  ubuntu = "ubuntu-latest",
  macos = "macos-latest",
  # windows-latest ships VS 18, where ilammy/msvc-dev-cmd fails (invalid
  # params on 18, "VS not found" when pinned to 2022). eOn's Windows leg hit
  # the same wall.
  windows = "windows-2022",
  msvc_dev_cmd = "ilammy/msvc-dev-cmd@v1",

  # --- meson machine files (repo-relative; also present in `git archive` trees) ---
  no_fortran_native_file = "ci/gha/machines/no-fortran.ini",
  win_flang_msvc_native_file = "ci/gha/machines/win-flang-msvc.ini",
}
