The long-range bench logs now record the checkout head, tree, dirty flag and
compiler of the build instead of `unknown`, and every ladder line and report
table carries the OpenMP team that ran as `active`. `report_from_log.py` and
`tip_base_panel.py` read ladder lines by key and keep the DSF rows, and a test
reads ladder lines with and without the `active` field.
