`librgpot` takes soversion 10 in `kCxxSoversion`, because the C++ layout stamp
now covers `GuptaConfig`, `GuptaPot` and `PotType::Gupta`. Soversion 9 shipped
in 3.8.0 and keeps its recorded layout hash. Programs linked against
`librgpot.so.9` need a rebuild against `librgpot.so.10`. The C ABI stamp and
the engine ABI are unchanged.
