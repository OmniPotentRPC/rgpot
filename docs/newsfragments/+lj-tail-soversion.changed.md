The `librgpot` soversion moves from 9 to 10, because `LJConfig` gains
`tail_correction` and `LJConfig` and `LJPot` grow. Programs built against
`librgpot.so.9` need a rebuild to load `librgpot.so.10`.
