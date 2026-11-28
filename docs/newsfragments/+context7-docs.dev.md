The `Context7 docs` workflow runs on each push to `develop`. It exports
every Org document to Markdown with `scripts/org_to_md.sh`, adds the
authored Markdown and a `context7.json`, and force-pushes that tree as one
parentless commit to the orphan branch `context7-docs`, which Context7 can
index because it does not read Org.
