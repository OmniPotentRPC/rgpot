`Neighbours` keeps nothrow default construction and guards the 64-bit
generation counter against wraparound at `UINT64_MAX`, throwing
`std::overflow_error` instead of returning the `0` sentinel.
