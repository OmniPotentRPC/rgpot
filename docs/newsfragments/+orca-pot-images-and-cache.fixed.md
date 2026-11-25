`ORCAPot` now checks a padded box over every lattice vector short enough to
matter, so a skewed cell in which a translated image of an atom is closer than
the atom itself is refused. With `ORCAConfig::restart` on, the result cache is
not used, so a cached call can no longer leave the orbitals of an earlier call
as the guess of the next one. Every call runs the executable with symlinks
resolved, the file whose size, modification time, change time, device, inode and
content hash the cache key holds, and throws `ORCAError` once that file or the
symlink pointing to it has changed: before running it when the change is seen
first, or instead of returning the result when it is seen only after the run.
