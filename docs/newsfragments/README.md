# Towncrier news fragments

User-facing changes land here **before** a release is cut. On `cog bump`,
`uvx towncrier build` folds these into **`CHANGELOG.md`** (canonical user
changelog; see also `docs/orgmode/changelog.org`, which only includes that file
for Sphinx) and removes the fragment files.

## Discipline

- **One fragment per user-visible change** (not one per commit).
- Prefer issue-linked names (`42.added.md`) when there is a tracking issue; use
  `+slug.fixed.md` for internal/no-issue fixes.
- Write for **users/integrators**, not implementers (implementation detail goes
  in the PR body or `dev` type only when relevant to contributors).
- **Required on any pull request that runs the orchestrator workflow.** The
  `CI (orchestrator)` workflow (`.github/workflows/ci-orchestrator.yml`) lists
  `docs/**`, `**/*.md`, `newsfragments/**` and `LICENSE` under
  `pull_request` `paths-ignore`. When every file a pull request changes
  matches one of these patterns, the workflow does not run, so neither the
  `towncrier` job nor the fragment check runs. A pull request that changes
  any other file (code, build files, or CI files such as the workflows) runs
  the `towncrier` job, which runs `towncrier check` against the pull request
  base and fails when the pull request adds no fragment. `towncrier check`
  passes without a fragment when the pull request changes `CHANGELOG.md`. Use
  the `dev` type when the change does not alter library, CLI or crate
  behaviour.

## Fragment text

A fragment holds only the text of the entry. The template
`ci/release-notes.md` writes the bullet and the heading, so:

- Do not start a fragment with a list marker. Continuation lines need no
  indent.
- Do not put a copyright or SPDX line, or a comment block (`<!-- ... -->`),
  in a fragment. Neither belongs in the notes.

`python scripts/check_news_fragments.py` (or `pixi r news-fragments-check`)
fails on a fragment whose text, after leading whitespace, starts with `- ` or
`* `, on a fragment with a line that starts with `copyright `, `(c) `, `©`,
`spdx-` (ignoring case), `[//]:` or `<!--`, and on a fragment with no text. The
`prek` job in the same workflow runs it. Add `--fix` to strip these from the
files. The template strips the same lines and a leading marker when it builds
the notes.

Preview: `uvx towncrier build --draft` or `pixi r towncrier-draft` if defined.

## Naming

```
<issue-or-slug>.<type>.md
```

| `<type>` directory suffix | Section in CHANGELOG |
|---------------------------|----------------------|
| `breaking` | Breaking |
| `security` | Security |
| `removed` | Removed |
| `deprecated` | Deprecated |
| `added` | Added |
| `dev` | Developer |
| `changed` | Changed |
| `fixed` | Fixed |
| `misc` | Miscellaneous |

These are the types `towncrier.toml` defines. `towncrier check` fails on a
fragment whose type is not in this list, such as `style`.
`scripts/check_news_fragments.py` does not check the type.

Examples:

```bash
uvx towncrier create --content "MetatomicPot loads TorchScript models directly." 37.added.md
uvx towncrier create --content "Vesin 0.5+ device struct compatibility." +vesin05.fixed.md
```

After merge to `main`, maintainers cut the release with `cog bump` (see
`docs/orgmode/contributing/developer/release.org`). Do **not** hand-edit a
`## [X.Y.Z]` block above the towncrier marker in `CHANGELOG.md`.
