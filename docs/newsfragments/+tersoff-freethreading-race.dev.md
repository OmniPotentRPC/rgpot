The FortranStaticAudit test fails when a Fortran kernel archive holds
procedure-local static data, which every thread would share. gfortran
emits such a static for the hidden length of a deferred-length function
result inside a concatenation, even under -frecursive. That static
truncated Tersoff error messages and corrupted the heap when several
free-threaded Python callers failed at once. New native and Python tests
run the Tersoff stress path from many threads at once, interleaved with
simultaneous zero-volume failures, and require results identical to one
thread.
