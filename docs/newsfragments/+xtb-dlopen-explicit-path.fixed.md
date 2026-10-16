``XTBDlopen`` no longer loads a different ``libxtb_engine`` when
``engine_path`` is set and that file cannot be opened. Environment variables
and the bare soname are used only when ``engine_path`` is empty.
