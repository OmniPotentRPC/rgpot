# Towncrier news fragments

User-facing changes land here **before** a release is cut. On `cog bump`,
`uvx towncrier build` folds these into `CHANGELOG.md` and removes the files.

## Naming

```
<issue-or-slug>.<type>.md
```

| `<type>` directory suffix | Section in CHANGELOG |
|---------------------------|----------------------|
| `security` | Security |
| `removed` | Removed |
| `deprecated` | Deprecated |
| `added` | Added |
| `dev` | Developer |
| `changed` | Changed |
| `fixed` | Fixed |
| `misc` | Miscellaneous |

Examples:

```bash
uvx towncrier create --content "MetatomicPot loads TorchScript models directly." 37.added.md
uvx towncrier create --content "Vesin 0.5+ device struct compatibility." +vesin05.fixed.md
```

Fragments are required on PRs that change shipped behavior (CI: `towncrier check`
against the PR base). Pure docs/ci/chore commits may omit them.
