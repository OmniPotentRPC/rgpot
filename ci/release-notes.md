{#-
Release-notes template for towncrier (see towncrier.toml).

It renders like the stock Markdown template, and it first normalizes
each fragment so that the notes never carry:

  * HTML comment blocks (<!-- ... -->) and `[//]: #` comment lines,
  * copyright and SPDX header lines,
  * a leading list marker ("- " or "* "), which would otherwise render
    as a doubled bullet.

scripts/check_news_fragments.py enforces the same rules on the files.
-#}
{%- macro heading(level) -%}
    {{- "#" * ((header_prefix | default("#")) | length + level - 1) }}
{%- endmacro -%}

{%- macro clean(text) -%}
    {%- set ns = namespace(raw=[], kept=[], comment=false, text="") -%}
    {#- towncrier indents every line after the first by two spaces. -#}
    {%- for line in text.split("\n") -%}
        {%- if not loop.first and line.startswith("  ") -%}
            {%- set _ = ns.raw.append(line[2:]) -%}
        {%- else -%}
            {%- set _ = ns.raw.append(line) -%}
        {%- endif -%}
    {%- endfor -%}
    {%- for line in ns.raw -%}
        {%- set low = line.strip().lower() -%}
        {%- if ns.comment -%}
            {%- if "-->" in line -%}
                {%- set ns.comment = false -%}
            {%- endif -%}
        {%- elif low.startswith("<!--") -%}
            {%- if "-->" not in low[4:] -%}
                {%- set ns.comment = true -%}
            {%- endif -%}
        {%- elif low.startswith("[//]:")
                 or low.startswith("copyright ")
                 or low.startswith("(c) ")
                 or low.startswith("\u00a9")
                 or low.startswith("spdx-") -%}
        {%- elif low or ns.kept -%}
            {%- set _ = ns.kept.append(line.rstrip()) -%}
        {%- endif -%}
    {%- endfor -%}
    {%- for _ in ns.kept -%}
        {%- if ns.kept and not ns.kept[-1] -%}
            {%- set _ = ns.kept.pop() -%}
        {%- endif -%}
    {%- endfor -%}
    {%- for _ in range(3) -%}
        {%- if ns.kept and (ns.kept[0].startswith("- ") or ns.kept[0].startswith("* ")) -%}
            {%- set head = ns.kept[0][2:].lstrip() -%}
            {%- set tail = [] -%}
            {%- for line in ns.kept[1:] -%}
                {%- set _ = tail.append(line[2:] if line.startswith("  ") else line) -%}
            {%- endfor -%}
            {%- set ns.kept = [head] + tail -%}
        {%- endif -%}
    {%- endfor -%}
    {%- for line in ns.kept -%}
        {%- if loop.first -%}
            {{- line -}}
        {%- elif line -%}
            {{- "\n  " ~ line -}}
        {%- else -%}
            {{- "\n" -}}
        {%- endif -%}
    {%- endfor -%}
{%- endmacro -%}

{%- set newline = "\n" -%}

{%- if render_title %}
    {%- if versiondata.name %}
        {{- heading(1) ~ " " ~ versiondata.name ~ " " ~ versiondata.version ~ " (" ~ versiondata.date ~ ")" ~ newline }}
    {%- else %}
        {{- heading(1) ~ " " ~ versiondata.version ~ " (" ~ versiondata.date ~ ")" ~ newline }}
    {%- endif %}
{%- endif %}
{{- newline }}

{%- for section, _ in sections.items() %}
    {%- if section %}
        {{- newline }}
        {{- heading(2) ~ " " ~ section ~ newline }}
        {{- newline }}
    {%- endif %}

    {%- if sections[section] %}
        {%- for category, val in definitions.items() if category in sections[section] %}
            {{- heading(3 if section else 2) ~ " " ~ definitions[category]["name"] ~ newline }}
            {{- newline }}

            {%- for raw, values in sections[section][category].items() %}
                {%- set text = clean(raw) %}
                {%- set issue_pks = [] %}
                {%- for v_issue in values %}
                    {%- set _ = issue_pks.append(v_issue.split(": ", 1)[0]) %}
                {%- endfor %}
                {%- set issues_list = issue_pks | join(", ") %}
                {%- set text_has_sublist = (("\n  - " in text) or ("\n  * " in text)) %}

                {%- if not text and issues_list %}
                    {{- "- " ~ issues_list ~ newline }}
                {%- elif text and issues_list %}
                    {%- if text_has_sublist %}
                        {{- "- " ~ text ~ newline ~ newline ~ "  (" ~ issues_list ~ ")" ~ newline }}
                    {%- else %}
                        {{- "- " ~ text ~ " (" ~ issues_list ~ ")" ~ newline }}
                    {%- endif %}
                {%- elif text %}
                    {{- "- " ~ text ~ newline }}
                {%- endif %}
            {%- endfor %}

            {{- newline }}

            {%- if issues_by_category[section][category] and "]: " in issues_by_category[section][category][0] %}
                {%- for issue in issues_by_category[section][category] %}
                    {{- issue ~ newline }}
                {%- endfor %}
                {{- newline }}
            {%- endif %}

            {%- if sections[section][category]|length == 0 %}
                {{- newline }}
                {{- "No significant changes." ~ newline * 2 }}
            {%- endif %}
        {%- endfor %}
    {%- else %}
        {{- "No significant changes." ~ newline * 2 }}
    {%- endif %}
{%- endfor %}
{{- newline -}}
