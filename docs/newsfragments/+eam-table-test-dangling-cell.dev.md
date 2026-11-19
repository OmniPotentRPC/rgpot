<!-- Copyright 2023--present rgpot developers -->

The tabulated EAM C ABI tests no longer hand a dangling cell pointer to
`rgpot_force_input_create`: the test input now borrows its arrays for the
lifetime of the C view, which fixes two release-mode failures, and the
`c_api::potential` unit tests keep their input arrays on the heap.
