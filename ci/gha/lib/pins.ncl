# Single place to bump action / pixi / tool pins for all generated workflows.
# Import via steps.ncl (`S.pins`) or `import "pins.ncl"` from modules in ci/gha/lib/.
# Workflow modules should not hardcode action refs; compose builders from steps.ncl.
{
  # --- toolchains / env managers ---
  pixi_version = "v0.70.2",

  # --- GitHub Actions (third-party / first-party) ---
  checkout = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1", # v7.0.1
  setup_pixi = "prefix-dev/setup-pixi@d3f436a425481402e6a95a1d1fc10331c708cd9e", # v0.10.2
  cache = "actions/cache@55cc8345863c7cc4c66a329aec7e433d2d1c52a9", # v6.1.0
  install_action = "taiki-e/install-action@861a07ce7084f55488e375df125cdc99bba60eb7", # v2
  rust_toolchain = "dtolnay/rust-toolchain@89b12181fb390509a0842a86cc55eeb8eb928c1d", # stable
  rust_cache = "Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6", # v2
  upload_artifact_v4 = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", # v7.0.1
  upload_artifact_v6 = "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a", # v7.0.1
  download_artifact = "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", # v8.0.1
  download_artifact_v6 = "actions/download-artifact@3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c", # v8.0.1
  gh_release = "softprops/action-gh-release@efb35369e0ad2afab669f228072c1b0d510eae64", # v3.0.3
  gh_pages = "peaceiris/actions-gh-pages@84c30a85c19949d7eee79c4ff27748b70285e453", # v4
  lychee = "lycheeverse/lychee-action@e7477775783ea5526144ba13e8db5eec57747ce8", # v2
  doc_previewer = "HaoZeke/doc-previewer@0e1094735b4f7ee9a2c60a18a34d7a7e5bc6bb63", # v0.0.1

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
  msvc_dev_cmd = "ilammy/msvc-dev-cmd@0b201ec74fa43914dc39ae48a89fd1d8cb592756", # v1

  # --- meson machine files (repo-relative; also present in `git archive` trees) ---
  no_fortran_native_file = "ci/gha/machines/no-fortran.ini",
  win_flang_msvc_native_file = "ci/gha/machines/win-flang-msvc.ini",
}
