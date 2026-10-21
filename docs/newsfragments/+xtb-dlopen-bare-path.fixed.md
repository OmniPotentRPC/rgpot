``XTBDlopen`` rejects a slash-less ``engine_path``. A bare soname would be
opened through the loader search path and could substitute a different
``libxtb_engine``. Pass an absolute or relative path. When that path does
not open, the error names it and the loader diagnostic instead of
environment variables that are ignored.
