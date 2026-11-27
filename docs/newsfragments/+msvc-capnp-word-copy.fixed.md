A Windows build with MSVC can construct a potential from a flat
Cap'n Proto message again. The message words are copied as bytes into
an aligned buffer, and the message itself is unchanged.
