// MIT License
// Copyright 2023--present rgpot developers

/// Fixed-order dot product. The sum walks indices 0..n, so the result
/// does not depend on a thread count. Panic is caught at the ABI edge.
#[no_mangle]
pub extern "C" fn rust_core_demo_dot(a: *const f64, b: *const f64, n: usize) -> f64 {
    match std::panic::catch_unwind(|| unsafe {
        if a.is_null() || b.is_null() {
            return f64::NAN;
        }
        let left = std::slice::from_raw_parts(a, n);
        let right = std::slice::from_raw_parts(b, n);
        let mut acc = 0.0f64;
        for i in 0..n {
            acc += left[i] * right[i];
        }
        acc
    }) {
        Ok(value) => value,
        Err(_) => f64::NAN,
    }
}
