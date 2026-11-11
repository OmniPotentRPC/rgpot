// MIT License
// Copyright 2023--present rgpot developers
//
// Hand-written C ABI. The request is initialised in C so the periodic
// flags and the struct size match the header. rustc links the rgpot
// neighbour objects; there is no bindgen step.

use std::os::raw::{c_char, c_int};
use std::process::exit;

#[repr(C)]
struct Request {
    bytes: u32,
    abi: u32,
    cutoff: f64,
    skin: f64,
    shape: i32,
    shell: i32,
    cut: i32,
    periodic: [i32; 3],
    want_shifts: i32,
    want_vectors: i32,
    want_distances: i32,
}

#[repr(C)]
struct Soa {
    bytes: u32,
    first: *mut i32,
    second: *mut i32,
    shift: *mut i32,
    vector: *mut f64,
    distance: *mut f64,
    mask: *mut u8,
    cap: usize,
}

extern "C" {
    fn rgpot_nlist_request_init(request: *mut Request);
    fn rgpot_nlist_build(
        positions: *const f64,
        n: usize,
        box9: *const f64,
        request: *const Request,
        out: *mut Soa,
        count: *mut usize,
        needed: *mut usize,
    ) -> i32;
    fn rgpot_nlist_strerror(code: i32) -> *const c_char;
}

fn fail(message: &str) -> ! {
    eprintln!("{message}");
    exit(1);
}

fn main() {
    let r = 2.0_f64.powf(1.0 / 6.0);
    let pos = [0.0, 0.0, 0.0, r, 0.0, 0.0];
    let box9 = [40.0, 0.0, 0.0, 0.0, 40.0, 0.0, 0.0, 0.0, 40.0];
    let mut request = unsafe { std::mem::zeroed::<Request>() };
    unsafe { rgpot_nlist_request_init(&mut request) };
    request.cutoff = 2.5;
    request.want_shifts = 0;
    request.want_vectors = 0;
    let mut needed = 0usize;
    let mut count = 0usize;
    let rc = unsafe {
        rgpot_nlist_build(
            pos.as_ptr(),
            2,
            box9.as_ptr(),
            &request,
            std::ptr::null_mut(),
            &mut count,
            &mut needed,
        )
    };
    if rc != 0 || needed != 1 {
        let text = unsafe { std::ffi::CStr::from_ptr(rgpot_nlist_strerror(rc)) };
        fail(&format!("probe {rc} {}: {text:?}", needed));
    }
    let mut first = 0i32;
    let mut second = 0i32;
    let mut distance = 0.0f64;
    let mut soa = Soa {
        bytes: std::mem::size_of::<Soa>() as u32,
        first: &mut first,
        second: &mut second,
        shift: std::ptr::null_mut(),
        vector: std::ptr::null_mut(),
        distance: &mut distance,
        mask: std::ptr::null_mut(),
        cap: 1,
    };
    let rc = unsafe {
        rgpot_nlist_build(
            pos.as_ptr(),
            2,
            box9.as_ptr(),
            &request,
            &mut soa,
            &mut count,
            &mut needed,
        )
    };
    if rc != 0 || count != 1 || first != 0 || second != 1 || (distance - r).abs() > 1e-12 {
        let text = unsafe { std::ffi::CStr::from_ptr(rgpot_nlist_strerror(rc)) };
        fail(&format!("fill {rc} {text:?} d={distance}"));
    }
    let _ = c_int::from(rc);
}
