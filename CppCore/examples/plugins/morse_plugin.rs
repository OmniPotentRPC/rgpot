// rgpot plugin: Morse potential in Rust.
//
// Demonstrates implementing an rgpot plugin in Rust using #[no_mangle]
// and extern "C" functions.
//
// E = D_e * sum_{i<j} (1 - exp(-alpha*(r_ij - r_e)))^2
// F = -dE/dr (analytic)
//
// Build:
//   Create a Cargo project with crate-type = ["cdylib"], then:
//   cargo build --release
//   cp target/release/libmorse_plugin.so $RGPOT_PLUGIN_PATH/
//
// Cargo.toml:
//   [package]
//   name = "morse_plugin"
//   version = "0.1.0"
//   edition = "2021"
//
//   [lib]
//   crate-type = ["cdylib"]
//
// This file is a self-contained example. Copy it to src/lib.rs in
// a Cargo project to build.

use std::ffi::{c_char, c_double, c_int, c_void};
use std::slice;

// Mirror the rgpot plugin ABI types (must match include/rgpot/plugin.h)

const RGPOT_PLUGIN_API_VERSION: u32 = 1;
const RGPOT_PLUGIN_OK: i32 = 0;
// const RGPOT_PLUGIN_ERROR: i32 = 2;

#[repr(C)]
pub struct RgpotPluginInput {
    n_atoms: usize,
    positions: *const c_double,
    atomic_numbers: *const c_int,
    cell: *const c_double,
}

#[repr(C)]
pub struct RgpotPluginOutput {
    energy: c_double,
    variance: c_double,
    forces: *mut c_double,
}

#[repr(C)]
pub struct RgpotPluginDescriptor {
    descriptor_size: usize,
    api_version: u32,
    name: *const c_char,
    version: *const c_char,
    length_unit: *const c_char,
    energy_unit: *const c_char,
    create: Option<unsafe extern "C" fn(*const c_char, *mut *mut c_void) -> i32>,
    destroy: Option<unsafe extern "C" fn(*mut c_void)>,
    calculate:
        Option<unsafe extern "C" fn(*mut c_void, *const RgpotPluginInput, *mut RgpotPluginOutput) -> i32>,
    capabilities: Option<unsafe extern "C" fn(*const c_void) -> u32>,
    last_error: Option<unsafe extern "C" fn(*const c_void) -> *const c_char>,
}

// Morse potential parameters
struct MorsePotential {
    d_e: f64,    // well depth (eV)
    alpha: f64,  // width parameter (1/angstrom)
    r_e: f64,    // equilibrium distance (angstrom)
    cutoff: f64, // cutoff radius (angstrom)
}

impl MorsePotential {
    fn calculate(&self, n: usize, pos: &[f64], forces: &mut [f64]) -> f64 {
        let mut energy = 0.0;
        forces.iter_mut().for_each(|f| *f = 0.0);

        for i in 0..n {
            for j in (i + 1)..n {
                let dx = pos[3 * i] - pos[3 * j];
                let dy = pos[3 * i + 1] - pos[3 * j + 1];
                let dz = pos[3 * i + 2] - pos[3 * j + 2];
                let r = (dx * dx + dy * dy + dz * dz).sqrt();

                if r > self.cutoff || r < 1e-10 {
                    continue;
                }

                let exp_term = (-self.alpha * (r - self.r_e)).exp();
                let one_minus_exp = 1.0 - exp_term;

                // Energy
                energy += self.d_e * one_minus_exp * one_minus_exp;

                // Force magnitude: dE/dr = 2 * D_e * alpha * (1-exp) * exp
                let de_dr = 2.0 * self.d_e * self.alpha * one_minus_exp * exp_term;
                let f_over_r = de_dr / r;

                // F = -dE/dr * r_hat
                forces[3 * i] -= f_over_r * dx;
                forces[3 * i + 1] -= f_over_r * dy;
                forces[3 * i + 2] -= f_over_r * dz;
                forces[3 * j] += f_over_r * dx;
                forces[3 * j + 1] += f_over_r * dy;
                forces[3 * j + 2] += f_over_r * dz;
            }
        }
        energy
    }
}

// Plugin ABI implementation

unsafe extern "C" fn morse_create(
    _config_json: *const c_char,
    out: *mut *mut c_void,
) -> i32 {
    let pot = Box::new(MorsePotential {
        d_e: 1.0,
        alpha: 1.0,
        r_e: 1.0,
        cutoff: 10.0,
    });
    *out = Box::into_raw(pot) as *mut c_void;
    RGPOT_PLUGIN_OK
}

unsafe extern "C" fn morse_destroy(instance: *mut c_void) {
    if !instance.is_null() {
        drop(Box::from_raw(instance as *mut MorsePotential));
    }
}

unsafe extern "C" fn morse_calculate(
    instance: *mut c_void,
    input: *const RgpotPluginInput,
    output: *mut RgpotPluginOutput,
) -> i32 {
    let pot = &*(instance as *const MorsePotential);
    let inp = &*input;
    let out = &mut *output;
    let n = inp.n_atoms;

    let positions = slice::from_raw_parts(inp.positions, n * 3);
    let forces = slice::from_raw_parts_mut(out.forces, n * 3);

    out.energy = pot.calculate(n, positions, forces);
    out.variance = 0.0;
    RGPOT_PLUGIN_OK
}

// Static descriptor

static MORSE_NAME: &[u8] = b"morse\0";
static MORSE_VERSION: &[u8] = b"0.1.0\0";

#[no_mangle]
pub static MORSE_DESCRIPTOR: RgpotPluginDescriptor = RgpotPluginDescriptor {
    descriptor_size: std::mem::size_of::<RgpotPluginDescriptor>(),
    api_version: RGPOT_PLUGIN_API_VERSION,
    name: MORSE_NAME.as_ptr() as *const c_char,
    version: MORSE_VERSION.as_ptr() as *const c_char,
    length_unit: std::ptr::null(), // angstrom (native)
    energy_unit: std::ptr::null(), // eV (native)
    create: Some(morse_create),
    destroy: Some(morse_destroy),
    calculate: Some(morse_calculate),
    capabilities: None,
    last_error: None,
};

#[no_mangle]
pub extern "C" fn rgpot_plugin_init() -> *const RgpotPluginDescriptor {
    &MORSE_DESCRIPTOR
}
