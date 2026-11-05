On a cell narrower than three cutoffs, `rgpot_fast_pairs` answers a
short buffer from n(n-1)/2 and fills the caller's buffer directly. For
the 10k-atom bridge cell, it no longer builds a 400 MB pair list just to
report the count. A short-buffer return also clears the thread-local
error message.
