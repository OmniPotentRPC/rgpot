# Single place to bump action / pixi / tool pins for all generated workflows.
# Import via steps.ncl (`S.pins`) or `import "pins.ncl"` from modules in ci/gha/lib/.
# Workflow modules should not hardcode action refs; compose builders from steps.ncl.
{
  # --- toolchains / env managers ---
  pixi_version = "v0.70.2",

  # --- GitHub Actions (third-party / first-party) ---
  checkout = "actions/checkout@d23441a48e516b6c34aea4fa41551a30e30af803", # v6
  setup_pixi = "prefix-dev/setup-pixi@a0af7a228712d6121d37aba47adf55c1332c9c2e", # v0.9.4
  cache = "actions/cache@caa296126883cff596d87d8935842f9db880ef25", # v5
  install_action = "taiki-e/install-action@861a07ce7084f55488e375df125cdc99bba60eb7", # v2
  rust_toolchain = "dtolnay/rust-toolchain@89b12181fb390509a0842a86cc55eeb8eb928c1d", # stable
  rust_cache = "Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6", # v2
  upload_artifact_v4 = "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02", # v4
  upload_artifact_v6 = "actions/upload-artifact@b7c566a772e6b6bfb58ed0dc250532a479d7789f", # v6
  download_artifact = "actions/download-artifact@d3f86a106a0bac45b974a628896c90dbdf5c8093", # v4
  download_artifact_v6 = "actions/download-artifact@018cc2cf5baa6db3ef3c5f8a56943fffe632ef53", # v6
  gh_release = "softprops/action-gh-release@3bb12739c298aeb8a4eeaf626c5b8d85266b0e65", # v2
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
