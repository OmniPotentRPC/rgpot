The MinGW Wine job skips AotiPublicationStressTest, which replaces a
file that is still open. Wine is detected only from the ntdll export
wine_get_version. A directory clock Wine will not move no longer fails
the stale-directory sweep, and path checks accept two spellings of one
directory.
