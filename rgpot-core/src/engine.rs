// MIT License
// Copyright 2023--present rgpot developers

//! Native loader for the generic engine C ABI of `rgpot/engine_c_abi.h`.
//!
//! An engine shared library (`libcpmd_engine.so`, `libuma_engine.so`, ...)
//! exports one symbol set. Which backend runs is which library is loaded, and
//! the backend's configuration is the opaque message the library's create
//! call reads. [`EnginePotential::load`] resolves that symbol set, checks both
//! [`ENGINE_ABI_VERSION`] and [`RgpotEngineAbiStamp::HOST`] against the
//! library's ABI identity, asks `rgpot_engine_available`, when the library
//! exports it, whether the backend can create instances, and keeps one engine
//! instance.
//! [`EnginePotential::force`] is one geometry in Angstrom and one answer in eV
//! and eV/Angstrom, the units the header fixes.
//!
//! Return codes follow the header: zero is success, a positive code is an
//! evaluation that failed at this input ([`EngineError::Recoverable`]), and a
//! negative code leaves the instance unusable ([`EngineError::Fatal`], or
//! [`EngineError::Lock`] for [`ENGINE_ERR_LOCK`]) and poisons the handle so
//! every later call returns the same error without re-entering native
//! evaluation or query functions. A host callback
//! or transform that fails during a call is considered before the code the
//! engine returns. A fatal failure from either side wins over a recoverable
//! one, and the first fatal failure, with its message, is the one latched.
//! Dropping a poisoned handle still calls `rgpot_engine_destroy` once to free
//! the `RgpotEnginePot` allocation, as required by `engine_c_abi.h`. The message of
//! a failed call is the engine's `rgpot_engine_last_error` when it exports one.
//!
//! [`EnginePotential::into_potential`] wraps the instance as a
//! [`PotentialImpl`], the handle the rest of the crate evaluates through, so
//! an engine library takes part in everything that takes a potential.
//!
//! [`EngineSession`] owns callbacks and transform calls around one
//! [`EnginePotential`]. Loading either type is unsafe because a dynamic
//! library's initializers and exported signatures cannot be verified by Rust.
//! Once loaded, session evaluation is safe: callback state is owned for the
//! native registration lifetime, and transform invocations are serialized
//! even when an engine parallelizes a batch.
//!
//! Dropping the handle destroys the engine instance. The library stays
//! resident: [`EnginePotential::load`] keeps one process-wide registry entry
//! per file, keyed by device and inode, that holds the `dlopen` handle and the
//! resolved [`EngineSymbols`]. The library is opened with `RTLD_NODELETE` and
//! the loader never calls `dlclose` on it. A backend may own threads, OpenMP
//! or CUDA runtimes, MPI state, TLS destructors or `atexit` handlers that call
//! into the library after its last instance is gone, and a tensor runtime
//! crashes if `dlclose` runs its static teardown. Loading the same file again,
//! through any path, reuses the entry without another `dlopen`.
//!
//! The registry refuses two loads it cannot key safely. A path whose file was
//! replaced after it was loaded names a new inode while the dynamic linker
//! would hand back the old mapping, and a bare library name is resolved by the
//! loader's search path rather than the filesystem. Both are
//! [`EngineError::Load`].
//!
//! A plugin that exports `rgpot_engine_unload_policy` returning
//! [`ENGINE_UNLOAD_SAFE`], with an ABI stamp of minor
//! [`ENGINE_UNLOAD_POLICY_MINOR`] or later, is opened without `RTLD_NODELETE`
//! and can be unloaded with the `unsafe` [`EnginePotential::unload`]. It stays
//! resident until then. Every other plugin is refused.

use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_long, c_longlong, c_uint, c_ushort, c_void, CString};
use std::fmt;
use std::mem::ManuallyDrop;
use std::num::NonZeroU32;
use std::path::{Path, PathBuf};
use std::ptr::NonNull;
use std::sync::{Arc, Condvar, LazyLock, Mutex, MutexGuard, OnceLock};
use std::thread::ThreadId;

use libloading::Library;

use crate::potential::PotentialImpl;
use crate::status::{catch_unwind, rgpot_status_t, set_last_error};
use crate::tensor::{copy_f64_tensor, copy_i32_tensor, rgpot_tensor_owned_cpu_f64_2d};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// cbindgen:ignore
/// Engine code meaning acquiring an engine or handle lock failed (`RGPOT_ENGINE_ERR_LOCK`).
pub const ENGINE_ERR_LOCK: c_int = -2;

/// The ABI major this loader speaks, `RGPOT_ENGINE_ABI_VERSION` of the header.
pub const ENGINE_ABI_VERSION: c_int = 1;
/// The ABI minor this loader advertises, `RGPOT_ENGINE_ABI_MINOR` of the header.
pub const ENGINE_ABI_MINOR: u16 = 12;
/// The struct layout revision this loader expects, `RGPOT_ENGINE_ABI_LAYOUT_REVISION` of the header.
pub const ENGINE_ABI_LAYOUT_REVISION: u16 = 1;
/// Engine code meaning an argument is invalid.
pub const ENGINE_ERR_INVALID: c_int = 1;
/// Engine code meaning the backend does not support the requested operation.
pub const ENGINE_ERR_UNSUPPORTED: c_int = 3;
/// `RGPOT_ENGINE_UNLOAD_RESIDENT`: the library must stay loaded.
pub const ENGINE_UNLOAD_RESIDENT: c_int = 0;
/// `RGPOT_ENGINE_UNLOAD_SAFE`: the library may be unloaded after its last
/// instance is destroyed.
pub const ENGINE_UNLOAD_SAFE: c_int = 1;
/// The first ABI minor that defines `rgpot_engine_unload_policy`.
pub const ENGINE_UNLOAD_POLICY_MINOR: u16 = 7;

const ERROR_CAPACITY: usize = 1024;
const REENTRANCY_END: usize =
    std::mem::offset_of!(RgpotEngineCaps, reentrancy) + std::mem::size_of::<c_int>();
/// `RgpotEngineCaps.reentrancy` values (0 shared instance, 1 per instance)
/// under which distinct handles may run at the same time. Any other value, 2
/// (process serial) included, serializes calls across handles.
const REENTRANCY_PARALLEL: std::ops::RangeInclusive<i32> = 0..=1;

/// Status recorded when a host callback or transform panics.
const PANIC_STATUS: EngineStatus = EngineStatus::from_code(-1);
/// Status recorded when a host callback hands back something the ABI cannot carry.
const INVALID_STATUS: EngineStatus = EngineStatus::from_code(ENGINE_ERR_INVALID);

/// Three-valued status of an engine ABI call.
///
/// The payload of each failure variant is the magnitude of the C return code,
/// so a recoverable failure is always positive and a fatal one is always
/// negative on the wire, and no failure can encode as zero. Magnitudes beyond
/// the range of `c_int` saturate to `c_int::MAX` and `c_int::MIN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EngineStatus {
    /// Return code zero.
    Success,
    /// A positive return code of this magnitude: the call failed at this input.
    RecoverableInputFailure(NonZeroU32),
    /// A negative return code of this magnitude: the instance is unusable.
    FatalInstanceFailure(NonZeroU32),
}

impl EngineStatus {
    /// A recoverable failure with the positive `code`, or `None` when `code`
    /// is not positive.
    pub const fn recoverable(code: c_int) -> Option<Self> {
        if code > 0 {
            Some(Self::from_code(code))
        } else {
            None
        }
    }

    /// A fatal failure with the negative `code`, or `None` when `code` is not
    /// negative.
    pub const fn fatal(code: c_int) -> Option<Self> {
        if code < 0 {
            Some(Self::from_code(code))
        } else {
            None
        }
    }

    /// Classify an engine return code without inspecting its error text.
    pub const fn from_code(code: c_int) -> Self {
        let magnitude = code.unsigned_abs();
        match NonZeroU32::new(magnitude) {
            None => Self::Success,
            Some(magnitude) if code > 0 => Self::RecoverableInputFailure(magnitude),
            Some(magnitude) => Self::FatalInstanceFailure(magnitude),
        }
    }

    /// The C ABI return code. Its sign is the class of the status, so
    /// `from_code(status.code())` is the same status for every status
    /// [`Self::from_code`] produces. A failure built directly with a magnitude
    /// beyond the range of `c_int` saturates to `c_int::MAX` or `c_int::MIN`,
    /// so that code is not the exact magnitude and does not round-trip.
    pub const fn code(self) -> c_int {
        match self {
            Self::Success => 0,
            Self::RecoverableInputFailure(magnitude) => {
                if magnitude.get() > c_int::MAX as u32 {
                    c_int::MAX
                } else {
                    magnitude.get() as c_int
                }
            }
            Self::FatalInstanceFailure(magnitude) => {
                if magnitude.get() > c_int::MIN.unsigned_abs() {
                    c_int::MIN
                } else {
                    (magnitude.get() as c_int).wrapping_neg()
                }
            }
        }
    }

    /// The same status with a failure's code guaranteed nonzero: a success
    /// becomes the recoverable [`ENGINE_ERR_INVALID`].
    pub const fn or_invalid(self) -> Self {
        match self {
            Self::Success => Self::from_code(ENGINE_ERR_INVALID),
            failure => failure,
        }
    }

    const fn is_fatal(self) -> bool {
        matches!(self, Self::FatalInstanceFailure(_))
    }
}

/// A typed failure returned by a host transform or oracle callback.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineFailure {
    pub status: EngineStatus,
    pub message: String,
}

/// Keep the failure that decides the call: the first fatal one if any
/// failure is fatal, otherwise the first one.
fn keep_decisive(slot: &mut Option<EngineFailure>, failure: EngineFailure) {
    match slot {
        Some(kept) if kept.status.is_fatal() || !failure.status.is_fatal() => {}
        _ => *slot = Some(failure),
    }
}

/// A host-side coordinate transform applied to the engine's scratch geometry.
pub trait CoordinateTransform: Send {
    fn transform(
        &mut self,
        positions: &mut [f64],
        box_matrix: &mut [f64; 9],
    ) -> Result<(), EngineFailure>;
}

/// Magic number of `RgpotCoupledWellConfigV1` and `RgpotCoupledWellConfigV2`.
pub const COUPLED_WELL_CONFIG_MAGIC: u32 = 0x4357_5054;
/// Version of the 48-byte coupled-well create message.
pub const COUPLED_WELL_CONFIG_VERSION_V1: u32 = 1;
/// Version of [`CoupledWellConfig`], the version-2 create message.
pub const COUPLED_WELL_CONFIG_VERSION: u32 = 2;
/// Byte size of `RgpotCoupledWellConfigV1`.
pub const COUPLED_WELL_CONFIG_V1_SIZE: u32 = 48;
/// Minimum message size of the version-2 coupled-well create message.
pub const COUPLED_WELL_CONFIG_MIN_SIZE: u32 = 56;
/// Atomic number of [`CoupledWellConfig::default`].
pub const COUPLED_WELL_DEFAULT_ATOMIC_NUMBER: i32 = 1;
/// Barrier energy `V0` of [`CoupledWellConfig::default`], in eV.
pub const COUPLED_WELL_DEFAULT_BARRIER_ENERGY: f64 = 1.0;
/// Well position `x0` of [`CoupledWellConfig::default`], in Angstrom.
pub const COUPLED_WELL_DEFAULT_WELL_POSITION: f64 = 1.0;
/// Transverse spring `k` of [`CoupledWellConfig::default`], in eV/Angstrom^2.
pub const COUPLED_WELL_DEFAULT_TRANSVERSE_SPRING: f64 = 1.0;
/// Coupling `alpha` of [`CoupledWellConfig::default`], in 1/Angstrom.
pub const COUPLED_WELL_DEFAULT_COUPLING: f64 = 0.0;
/// Magic number of `RgpotPotlibcConfigV1` and `RgpotPotlibcConfigV2`.
pub const POTLIBC_CONFIG_MAGIC: u32 = 0x504f_544c;
/// Version of the 144-byte potlibc create message.
pub const POTLIBC_CONFIG_VERSION_V1: u32 = 1;
/// Version of [`PotlibcConfig`], the version-2 create message.
pub const POTLIBC_CONFIG_VERSION: u32 = 2;
/// Byte size of `RgpotPotlibcConfigV1`.
pub const POTLIBC_CONFIG_V1_SIZE: u32 = 144;
/// Minimum message size of the version-2 potlibc create message.
pub const POTLIBC_CONFIG_MIN_SIZE: u32 = 152;
/// Capacity of the potlibc surface identifier, including its terminator.
pub const POTLIBC_SURFACE_ID_CAPACITY: usize = 128;

const _: () = assert!(COUPLED_WELL_CONFIG_V1_SIZE == 48);
const _: () = assert!(COUPLED_WELL_CONFIG_VERSION_V1 == 1);
const _: () = assert!(COUPLED_WELL_CONFIG_VERSION == 2);
const _: () = assert!(POTLIBC_CONFIG_V1_SIZE == 144);
const _: () = assert!(POTLIBC_CONFIG_VERSION_V1 == 1);
const _: () = assert!(POTLIBC_CONFIG_VERSION == 2);

/// Parameters for `libpotlibc_engine`.
///
/// Build one with [`PotlibcConfig::new`], or start from
/// [`PotlibcConfig::default`] and chain [`PotlibcConfig::with_surface_id`] and
/// [`PotlibcConfig::with_atom_count`]. The default has no surface id and is
/// refused by [`PotlibcConfig::validate`] and by the engine until one is set.
///
/// ```
/// use rgpot_core::engine::PotlibcConfig;
///
/// let config = PotlibcConfig::default()
///     .with_surface_id("vbo-beta1")
///     .unwrap()
///     .with_atom_count(4);
/// assert!(config.validate().is_ok());
/// assert!(PotlibcConfig::default().validate().is_err());
/// ```
///
/// The struct is `#[non_exhaustive]`: other crates build it with these methods.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PotlibcConfig {
    /// Configuration identity.
    pub magic: u32,
    /// Message size in bytes (minimum [`POTLIBC_CONFIG_MIN_SIZE`]).
    pub size: u32,
    /// Configuration layout version.
    pub version: u32,
    /// Must be zero.
    pub reserved: u32,
    /// Zero for a fixed-size surface, else the exact atom count of a
    /// variable-size surface.
    pub atom_count: u64,
    /// NUL-terminated canonical catalog identifier.
    pub surface_id: [u8; POTLIBC_SURFACE_ID_CAPACITY],
}

const _: () = assert!(std::mem::size_of::<PotlibcConfig>() == POTLIBC_CONFIG_MIN_SIZE as usize);
const _: () = assert!(std::mem::offset_of!(PotlibcConfig, magic) == 0);
const _: () = assert!(std::mem::offset_of!(PotlibcConfig, size) == 4);
const _: () = assert!(std::mem::offset_of!(PotlibcConfig, version) == 8);
const _: () = assert!(std::mem::offset_of!(PotlibcConfig, reserved) == 12);
const _: () = assert!(std::mem::offset_of!(PotlibcConfig, atom_count) == 16);
const _: () = assert!(std::mem::offset_of!(PotlibcConfig, surface_id) == 24);

impl Default for PotlibcConfig {
    fn default() -> Self {
        Self {
            magic: POTLIBC_CONFIG_MAGIC,
            size: POTLIBC_CONFIG_MIN_SIZE,
            version: POTLIBC_CONFIG_VERSION,
            reserved: 0,
            atom_count: 0,
            surface_id: [0; POTLIBC_SURFACE_ID_CAPACITY],
        }
    }
}

impl PotlibcConfig {
    /// Construct a version-two parameter message.
    pub fn new(surface_id: &str, atom_count: u64) -> Result<Self, EngineError> {
        Self::default()
            .with_atom_count(atom_count)
            .with_surface_id(surface_id)
    }

    /// Set the exact atom count of a variable-size surface; zero selects a
    /// fixed-size surface.
    #[must_use]
    pub fn with_atom_count(mut self, atom_count: u64) -> Self {
        self.atom_count = atom_count;
        self
    }

    /// Set the canonical catalog identifier. It must be non-empty, shorter
    /// than [`POTLIBC_SURFACE_ID_CAPACITY`] bytes and free of NUL.
    pub fn with_surface_id(mut self, surface_id: &str) -> Result<Self, EngineError> {
        let bytes = surface_id.as_bytes();
        if bytes.is_empty() || bytes.len() >= POTLIBC_SURFACE_ID_CAPACITY {
            return Err(EngineError::Invalid(
                "potlibc surface id is empty or too long".into(),
            ));
        }
        if bytes.contains(&0) {
            return Err(EngineError::Invalid(
                "potlibc surface id contains NUL".into(),
            ));
        }
        self.surface_id = [0; POTLIBC_SURFACE_ID_CAPACITY];
        self.surface_id[..bytes.len()].copy_from_slice(bytes);
        Ok(self)
    }

    /// Check what `rgpot_engine_create` checks of the message itself: magic,
    /// size, version, reserved field, and a NUL-terminated, non-empty surface
    /// id. Whether the surface exists and takes this atom count is the
    /// engine's to decide.
    pub fn validate(&self) -> Result<(), EngineError> {
        if self.magic != POTLIBC_CONFIG_MAGIC {
            return Err(EngineError::Invalid(
                "potlibc config magic, version, or reserved field".into(),
            ));
        }
        if self.version != POTLIBC_CONFIG_VERSION {
            return Err(EngineError::Invalid(
                "potlibc config version is unknown".into(),
            ));
        }
        if self.reserved != 0 {
            return Err(EngineError::Invalid(
                "potlibc config magic, version, or reserved field".into(),
            ));
        }
        if self.size < POTLIBC_CONFIG_MIN_SIZE {
            return Err(EngineError::Invalid(
                "potlibc config size is smaller than PotlibcConfig".into(),
            ));
        }
        if self.surface_id[0] == 0 || !self.surface_id.contains(&0) {
            return Err(EngineError::Invalid(
                "potlibc config needs a non-empty NUL-terminated surface id".into(),
            ));
        }
        Ok(())
    }

    /// Native bytes consumed by `rgpot_engine_create`.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `PotlibcConfig` is `repr(C)` plain data whose size equals the
        // sum of its fields (asserted above), so it has no padding and every
        // byte is initialised.
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// Other crates cannot write a [`CoupledWellConfig`] struct literal.
///
/// ```compile_fail,E0639
/// use rgpot_core::engine::CoupledWellConfig;
///
/// let _ = CoupledWellConfig {
///     magic: 0,
///     size: 0,
///     version: 0,
///     reserved0: 0,
///     atomic_number: 1,
///     reserved1: 0,
///     barrier_energy: 1.0,
///     well_position: 1.0,
///     transverse_spring: 1.0,
///     coupling: 0.0,
/// };
/// ```
#[cfg(doctest)]
struct CoupledWellConfigIsNonExhaustive;

/// Parameters for `libcoupled_well_engine`.
///
/// The engine evaluates one atom and uses its x and y coordinates:
/// `V0 ((x / x0)^2 - 1)^2 + k/2 (y - alpha x^2)^2`.
/// Values use the engine ABI units, eV and Angstrom.
///
/// [`CoupledWellConfig::default`] is a valid hydrogen atom in a unit double
/// well with no coupling; change it with the `with_*` builders:
///
/// ```
/// use rgpot_core::engine::CoupledWellConfig;
///
/// let config = CoupledWellConfig::default()
///     .with_barrier_energy(0.5)
///     .with_well_position(1.5)
///     .with_coupling(0.2);
/// assert!(config.validate().is_ok());
/// ```
///
/// The struct is `#[non_exhaustive]`, so other crates cannot write a struct
/// literal and a new field is not a breaking change.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct CoupledWellConfig {
    /// Configuration identity.
    pub magic: u32,
    /// Message size in bytes (minimum [`COUPLED_WELL_CONFIG_MIN_SIZE`]).
    pub size: u32,
    /// Configuration layout version.
    pub version: u32,
    /// Must be zero.
    pub reserved0: u32,
    /// Atomic number accepted by the engine.
    pub atomic_number: i32,
    /// Must be zero.
    pub reserved1: u32,
    /// Quartic barrier energy `V0` in eV.
    pub barrier_energy: f64,
    /// Position `x0` of either minimum along x in Angstrom.
    pub well_position: f64,
    /// Transverse spring constant `k` in eV/Angstrom squared.
    pub transverse_spring: f64,
    /// Coupling coefficient `alpha` in inverse Angstrom.
    pub coupling: f64,
}

const _: () =
    assert!(std::mem::size_of::<CoupledWellConfig>() == COUPLED_WELL_CONFIG_MIN_SIZE as usize);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, magic) == 0);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, size) == 4);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, version) == 8);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, reserved0) == 12);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, atomic_number) == 16);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, reserved1) == 20);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, barrier_energy) == 24);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, well_position) == 32);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, transverse_spring) == 40);
const _: () = assert!(std::mem::offset_of!(CoupledWellConfig, coupling) == 48);

impl Default for CoupledWellConfig {
    fn default() -> Self {
        Self {
            magic: COUPLED_WELL_CONFIG_MAGIC,
            size: COUPLED_WELL_CONFIG_MIN_SIZE,
            version: COUPLED_WELL_CONFIG_VERSION,
            reserved0: 0,
            atomic_number: COUPLED_WELL_DEFAULT_ATOMIC_NUMBER,
            reserved1: 0,
            barrier_energy: COUPLED_WELL_DEFAULT_BARRIER_ENERGY,
            well_position: COUPLED_WELL_DEFAULT_WELL_POSITION,
            transverse_spring: COUPLED_WELL_DEFAULT_TRANSVERSE_SPRING,
            coupling: COUPLED_WELL_DEFAULT_COUPLING,
        }
    }
}

impl CoupledWellConfig {
    /// Construct the version-two parameter message.
    pub fn new(
        atomic_number: i32,
        barrier_energy: f64,
        well_position: f64,
        transverse_spring: f64,
        coupling: f64,
    ) -> Self {
        Self::default()
            .with_atomic_number(atomic_number)
            .with_barrier_energy(barrier_energy)
            .with_well_position(well_position)
            .with_transverse_spring(transverse_spring)
            .with_coupling(coupling)
    }

    /// Set the one atomic number the engine accepts; it must be positive.
    #[must_use]
    pub fn with_atomic_number(mut self, atomic_number: i32) -> Self {
        self.atomic_number = atomic_number;
        self
    }

    /// Set the barrier energy `V0` in eV; it must be finite and positive.
    #[must_use]
    pub fn with_barrier_energy(mut self, barrier_energy: f64) -> Self {
        self.barrier_energy = barrier_energy;
        self
    }

    /// Set the well position `x0` in Angstrom; it must be finite and positive.
    #[must_use]
    pub fn with_well_position(mut self, well_position: f64) -> Self {
        self.well_position = well_position;
        self
    }

    /// Set the transverse spring `k` in eV/Angstrom squared; it must be finite
    /// and positive.
    #[must_use]
    pub fn with_transverse_spring(mut self, transverse_spring: f64) -> Self {
        self.transverse_spring = transverse_spring;
        self
    }

    /// Set the coupling `alpha` in inverse Angstrom; it must be finite.
    #[must_use]
    pub fn with_coupling(mut self, coupling: f64) -> Self {
        self.coupling = coupling;
        self
    }

    /// Check what `rgpot_engine_create` checks, so a bad parameter is reported
    /// before the library is opened. The engine repeats these checks.
    pub fn validate(&self) -> Result<(), EngineError> {
        let invalid = |why: &str| Err(EngineError::Invalid(format!("coupled-well {why}")));
        if self.magic != COUPLED_WELL_CONFIG_MAGIC {
            return invalid("config magic, version, or reserved fields");
        }
        if self.version != COUPLED_WELL_CONFIG_VERSION {
            return invalid("config version is unknown");
        }
        if self.reserved0 != 0 || self.reserved1 != 0 {
            return invalid("config magic, version, or reserved fields");
        }
        if self.size < COUPLED_WELL_CONFIG_MIN_SIZE {
            return invalid("config size is smaller than CoupledWellConfig");
        }
        if self.atomic_number <= 0 {
            return invalid("atomic_number must be positive");
        }
        let positive = |value: f64| value.is_finite() && value > 0.0;
        if !positive(self.barrier_energy) {
            return invalid("barrier_energy must be finite and positive");
        }
        if !positive(self.well_position) {
            return invalid("well_position must be finite and positive");
        }
        if !positive(self.transverse_spring) {
            return invalid("transverse_spring must be finite and positive");
        }
        if !self.coupling.is_finite() {
            return invalid("coupling must be finite");
        }
        Ok(())
    }

    /// Native bytes consumed by `rgpot_engine_create`.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: `CoupledWellConfig` is `repr(C)` plain data whose size equals
        // the sum of its fields (asserted above), so it has no padding and
        // every byte is initialised.
        unsafe {
            std::slice::from_raw_parts(
                std::ptr::from_ref(self).cast::<u8>(),
                std::mem::size_of::<Self>(),
            )
        }
    }
}

/// Opaque engine instance, `RgpotEnginePot` of the header.
#[repr(C)]
pub struct RgpotEnginePot {
    _private: [u8; 0],
}

/// Coordinate transform of the header. This loader never passes one.
pub type CoordTransformFn = unsafe extern "C" fn(
    user: *mut c_void,
    n_atoms: c_long,
    positions: *mut f64,
    box_: *mut f64,
) -> c_int;

/// Compatibility identity of a loaded engine, `RgpotEngineAbiStamp` of the header.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgpotEngineAbiStamp {
    /// Incompatible ABI version (`RGPOT_ENGINE_ABI_VERSION`).
    pub abi_major: c_ushort,
    /// Additive compatible minor version (`RGPOT_ENGINE_ABI_MINOR`).
    pub abi_minor: c_ushort,
    /// Struct size and field-order revision (`RGPOT_ENGINE_ABI_LAYOUT_REVISION`).
    pub layout_revision: c_ushort,
}

impl RgpotEngineAbiStamp {
    /// The ABI stamp this loader was built against.
    pub const HOST: Self = Self {
        abi_major: ENGINE_ABI_VERSION as c_ushort,
        abi_minor: ENGINE_ABI_MINOR,
        layout_revision: ENGINE_ABI_LAYOUT_REVISION,
    };

    /// Compatibility rule for a loaded plugin stamp (`self`) against `host`:
    /// equal major, equal layout revision, and a plugin minor at most `host`'s,
    /// so plugins from earlier minors whose newer optional exports are absent
    /// still load while plugins from a newer minor than `host` are rejected.
    pub const fn serves(self, host: Self) -> bool {
        self.abi_major == host.abi_major
            && self.layout_revision == host.layout_revision
            && self.abi_minor <= host.abi_minor
    }
}

impl fmt::Display for RgpotEngineAbiStamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}.{} (layout {})",
            self.abi_major, self.abi_minor, self.layout_revision
        )
    }
}

/// `rgpot_engine_abi_version`.
pub type AbiVersionFn = unsafe extern "C" fn() -> c_int;
/// `rgpot_engine_available`.
pub type AvailableFn = unsafe extern "C" fn() -> c_int;
/// `rgpot_engine_abi_stamp`.
pub type AbiStampFn = unsafe extern "C" fn() -> RgpotEngineAbiStamp;
/// `rgpot_engine_create`.
pub type CreateFn =
    unsafe extern "C" fn(*const c_void, usize, *mut c_char, usize) -> *mut RgpotEnginePot;
/// `rgpot_engine_destroy`.
pub type DestroyFn = unsafe extern "C" fn(*mut RgpotEnginePot);
/// `rgpot_engine_force`.
pub type ForceFn = unsafe extern "C" fn(
    *mut RgpotEnginePot,
    c_long,
    *const f64,
    *const c_int,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
    Option<CoordTransformFn>,
    *mut c_void,
) -> c_int;
/// `rgpot_engine_force_batch`.
pub type ForceBatchFn = unsafe extern "C" fn(
    *mut RgpotEnginePot,
    c_long,
    c_long,
    *const f64,
    *const c_int,
    *mut f64,
    *mut f64,
    *mut f64,
    *const f64,
    Option<CoordTransformFn>,
    *mut c_void,
) -> c_int;
/// `rgpot_engine_hvp`.
pub type HvpFn = unsafe extern "C" fn(
    *mut RgpotEnginePot,
    c_long,
    *const f64,
    *const c_int,
    *const f64,
    *mut f64,
    *const f64,
) -> c_int;
/// `RgpotEngineDerivativeReport` of `rgpot/engine_c_abi.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RgpotEngineDerivativeReport {
    /// Nonzero when `operations` and `scale` are meaningful.
    pub tracked: c_int,
    /// Arithmetic operations of the derivative evaluation.
    pub operations: c_longlong,
    /// Largest magnitude in the units of the jet.
    pub scale: f64,
}
/// `rgpot_engine_taylor_jet`.
pub type TaylorJetFn = unsafe extern "C" fn(
    *mut RgpotEnginePot,
    c_long,
    c_long,
    *const f64,
    *const c_int,
    *const f64,
    c_long,
    *mut f64,
    *mut RgpotEngineDerivativeReport,
    *const f64,
) -> c_int;
/// `rgpot_engine_caps`.
pub type CapsFn = unsafe extern "C" fn(*const RgpotEnginePot, *mut RgpotEngineCaps) -> c_int;
/// `rgpot_engine_last_error`.
pub type LastErrorFn = unsafe extern "C" fn(*const RgpotEnginePot, *mut c_char, usize) -> usize;
/// `rgpot_engine_last_components`.
pub type LastComponentsFn = unsafe extern "C" fn(
    *const RgpotEnginePot,
    *mut RgpotEngineComponent,
    *mut f64,
    c_long,
) -> c_long;
/// `rgpot_engine_component_name`.
pub type ComponentNameFn =
    unsafe extern "C" fn(*const RgpotEnginePot, c_long, *mut c_char, usize) -> usize;
/// `rgpot_engine_unload_policy`.
pub type UnloadPolicyFn = unsafe extern "C" fn() -> c_int;

/// Raw `RgpotEngineComponent` from `rgpot/engine_c_abi.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RgpotEngineComponent {
    /// Weight applied to the term in the composite sum.
    pub weight: f64,
    /// The term's own energy in eV before its weight.
    pub energy: f64,
    /// The term's own stress in eV/Angstrom^3 before its weight.
    pub stress: [f64; 9],
    /// Nonzero when the term is enabled.
    pub enabled: c_int,
    /// Nonzero when the term was evaluated on the last force call.
    pub evaluated: c_int,
    /// Nonzero when `stress` holds the term's stress.
    pub has_stress: c_int,
}

/// `rgpot_engine_model_state`.
pub type ModelStateFn = unsafe extern "C" fn(*const RgpotEnginePot) -> u64;
/// `rgpot_engine_set_callbacks`.
pub type SetCallbacksFn =
    unsafe extern "C" fn(*mut RgpotEnginePot, *const RgpotEngineCallbacks) -> c_int;
/// `rgpot_engine_hint`.
pub type HintFn =
    unsafe extern "C" fn(*mut RgpotEnginePot, *const c_void, usize, *mut c_char, usize) -> c_int;

/// Why a learning engine acquired an oracle point.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireReason {
    Variance,
    Certify,
    Batch,
    Reject,
    Unknown(c_int),
}

impl From<c_int> for AcquireReason {
    fn from(value: c_int) -> Self {
        match value {
            1 => Self::Variance,
            2 => Self::Certify,
            3 => Self::Batch,
            4 => Self::Reject,
            other => Self::Unknown(other),
        }
    }
}

/// Host callbacks owned by an [`EngineSession`].
pub trait EngineCallbacks: Send {
    fn oracle(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineFailure>;

    fn on_acquire(&mut self, _reason: AcquireReason) {}

    fn on_model_update(&mut self, _model_state: u64) {}
}

/// Raw callback table copied by `rgpot_engine_set_callbacks`.
#[repr(C)]
pub struct RgpotEngineCallbacks {
    ctx: *mut c_void,
    oracle: Option<
        unsafe extern "C" fn(
            *mut c_void,
            c_long,
            *const f64,
            *const c_int,
            *mut f64,
            *mut f64,
            *const f64,
        ) -> c_int,
    >,
    on_acquire: Option<unsafe extern "C" fn(*mut c_void, c_int)>,
    on_model_update: Option<unsafe extern "C" fn(*mut c_void, u64)>,
}

/// Raw `RgpotEngineCaps` from `rgpot/engine_c_abi.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RgpotEngineCaps {
    /// Struct size in bytes, set by the host before `rgpot_engine_caps`.
    pub size: c_uint,
    /// Reentrancy class: 0 shared instance, 1 per instance, 2 process serial.
    pub reentrancy: c_int,
    /// Nonzero when multi-image methods should clone one instance per image.
    pub per_image_instances: c_int,
    /// Nonzero when `rgpot_engine_force_batch` is served natively.
    pub batched: c_int,
    /// Nonzero when periodic boundary conditions are supported.
    pub periodic: c_int,
    /// Nonzero when cell stress is available.
    pub stress: c_int,
    /// Nonzero when every rank of a calculator group enters each call.
    pub group_collective: c_int,
    /// Nonzero when orbital selection is supported.
    pub orbital_selection: c_int,
    /// Nonzero when orbital copying is supported.
    pub orbital_copy: c_int,
    /// Nonzero when analytic HVP (`rgpot_engine_hvp`) is supported.
    pub analytic_hvp: c_int,
    /// Highest order of `rgpot_engine_taylor_jet`; 0 when it is not served.
    pub jet_order: c_int,
}

/// Capabilities advertised by an engine instance via `rgpot_engine_caps`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct EngineCaps {
    /// Reentrancy class: 0 shared instance, 1 per instance, 2 process serial.
    pub reentrancy: i32,
    /// Whether multi-image methods should clone one instance per image.
    pub per_image_instances: bool,
    /// Whether `rgpot_engine_force_batch` is served natively.
    pub batched: bool,
    /// Whether periodic boundary conditions are supported.
    pub periodic: bool,
    /// Whether cell stress is available.
    pub stress: bool,
    /// Whether every rank of a calculator group enters each call.
    pub group_collective: bool,
    /// Whether analytic HVP (`rgpot_engine_hvp`) is supported.
    pub analytic_hvp: bool,
    /// Highest order of `rgpot_engine_taylor_jet`; 0 when it is not served.
    pub jet_order: usize,
}

/// Roundoff report of one derivative call: `operations` times the machine
/// epsilon times `scale` is the engine's first-order estimate (not a rigorous
/// bound) of the roundoff of the returned derivatives. `scale` is in the units
/// of the jet. For a batch, `operations` is the sum over the systems and
/// `scale` the largest.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DerivativeReport {
    /// Arithmetic operations of the evaluation.
    pub operations: u64,
    /// Largest magnitude among the gradient and the returned orders.
    pub scale: f64,
}

/// A uniform batch of jet requests: `n_systems` systems of one composition.
#[derive(Debug, Clone, Copy)]
pub struct JetBatchRequest<'a> {
    /// Atomic numbers, one per atom, shared by every system.
    pub atomic_numbers: &'a [i32],
    /// Positions in Angstrom, `n_systems` blocks of `3 * n_atoms`.
    pub positions: &'a [f64],
    /// Directions in Angstrom, laid out like `positions`.
    pub directions: &'a [f64],
    /// Row-major 3 by 3 cells in Angstrom, `n_systems` blocks of 9.
    pub boxes: &'a [f64],
}

/// Directional derivatives of the gradient along a line, from
/// `rgpot_engine_taylor_jet`, for a uniform batch. Derivative `k` of system
/// `s` is the `k`-th derivative with respect to the line parameter of the
/// gradient at the request positions of that system, in eV/Angstrom for a
/// direction in Angstrom (every derivative of the gradient is a force, and the
/// Angstrom of the line parameter is absorbed by the direction). Derivative 1
/// is the Hessian-vector product. These are derivatives, not Taylor
/// coefficients.
#[derive(Debug, Clone, PartialEq)]
pub struct GradientJet {
    order: usize,
    n_atoms: usize,
    n_systems: usize,
    derivatives: Vec<f64>,
    report: Option<DerivativeReport>,
}

impl GradientJet {
    /// Highest derivative order held.
    pub fn order(&self) -> usize {
        self.order
    }

    /// Number of systems in the batch.
    pub fn n_systems(&self) -> usize {
        self.n_systems
    }

    /// The `k`-th derivative of every system, `n_systems * 3 * n_atoms`
    /// entries in system order, for `k` in `1..=order`.
    pub fn derivative(&self, k: usize) -> Option<&[f64]> {
        if k == 0 || k > self.order {
            return None;
        }
        let block = self.n_systems * 3 * self.n_atoms;
        self.derivatives.get((k - 1) * block..k * block)
    }

    /// The `k`-th derivative of system `system`, `3 * n_atoms` entries.
    pub fn system_derivative(&self, k: usize, system: usize) -> Option<&[f64]> {
        if system >= self.n_systems {
            return None;
        }
        let width = 3 * self.n_atoms;
        self.derivative(k)?
            .get(system * width..(system + 1) * width)
    }

    /// The roundoff report of the call, `None` when the engine does not count
    /// operations.
    pub fn report(&self) -> Option<DerivativeReport> {
        self.report
    }
}

/// The symbols of one engine library. `abi_version`, `create`, `destroy` and
/// `force` are required by the loader; the optional entries are probed and may
/// be absent. `available` is optional so that a plugin that does not export
/// it still loads; an absent `available` means the engine is available.
///
/// The struct is `#[non_exhaustive]`, so a symbol added to the ABI later does
/// not break callers. Build one with [`EngineSymbols::new`] and the `with_*`
/// methods, one per optional symbol.
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct EngineSymbols {
    /// Required symbol `rgpot_engine_abi_version`.
    pub abi_version: AbiVersionFn,
    /// Optional symbol `rgpot_engine_available`. When absent the engine is
    /// treated as available.
    pub available: Option<AvailableFn>,
    /// Optional symbol `rgpot_engine_abi_stamp`.
    pub abi_stamp: Option<AbiStampFn>,
    /// Required symbol `rgpot_engine_create`.
    pub create: CreateFn,
    /// Required symbol `rgpot_engine_destroy`.
    pub destroy: DestroyFn,
    /// Required symbol `rgpot_engine_force`.
    pub force: ForceFn,
    /// Optional symbol `rgpot_engine_force_batch`.
    pub force_batch: Option<ForceBatchFn>,
    /// Optional symbol `rgpot_engine_hvp`.
    pub hvp: Option<HvpFn>,
    /// Optional symbol `rgpot_engine_taylor_jet`.
    pub taylor_jet: Option<TaylorJetFn>,
    /// Optional symbol `rgpot_engine_caps`.
    pub caps: Option<CapsFn>,
    /// Optional symbol `rgpot_engine_last_error`.
    pub last_error: Option<LastErrorFn>,
    /// Optional symbol `rgpot_engine_last_components`.
    pub last_components: Option<LastComponentsFn>,
    /// Optional symbol `rgpot_engine_component_name`.
    pub component_name: Option<ComponentNameFn>,
    /// Optional symbol `rgpot_engine_model_state`.
    pub model_state: Option<ModelStateFn>,
    /// Optional symbol `rgpot_engine_set_callbacks`.
    pub set_callbacks: Option<SetCallbacksFn>,
    /// Optional symbol `rgpot_engine_hint`.
    pub hint: Option<HintFn>,
}

impl EngineSymbols {
    /// A symbol table with the four required symbols and every optional one
    /// absent.
    #[must_use]
    pub fn new(
        abi_version: AbiVersionFn,
        create: CreateFn,
        destroy: DestroyFn,
        force: ForceFn,
    ) -> Self {
        Self {
            abi_version,
            available: None,
            abi_stamp: None,
            create,
            destroy,
            force,
            force_batch: None,
            hvp: None,
            taylor_jet: None,
            caps: None,
            last_error: None,
            last_components: None,
            component_name: None,
            model_state: None,
            set_callbacks: None,
            hint: None,
        }
    }

    /// Provide the optional symbol `rgpot_engine_available`.
    #[must_use]
    pub fn with_available(mut self, available: AvailableFn) -> Self {
        self.available = Some(available);
        self
    }

    /// Provide the optional symbol `rgpot_engine_abi_stamp`.
    #[must_use]
    pub fn with_abi_stamp(mut self, abi_stamp: AbiStampFn) -> Self {
        self.abi_stamp = Some(abi_stamp);
        self
    }

    /// Provide the optional symbol `rgpot_engine_force_batch`.
    #[must_use]
    pub fn with_force_batch(mut self, force_batch: ForceBatchFn) -> Self {
        self.force_batch = Some(force_batch);
        self
    }

    /// Provide the optional symbol `rgpot_engine_hvp`.
    #[must_use]
    pub fn with_hvp(mut self, hvp: HvpFn) -> Self {
        self.hvp = Some(hvp);
        self
    }

    /// Provide the optional symbol `rgpot_engine_taylor_jet`.
    #[must_use]
    pub fn with_taylor_jet(mut self, taylor_jet: TaylorJetFn) -> Self {
        self.taylor_jet = Some(taylor_jet);
        self
    }

    /// Provide the optional symbol `rgpot_engine_caps`.
    #[must_use]
    pub fn with_caps(mut self, caps: CapsFn) -> Self {
        self.caps = Some(caps);
        self
    }

    /// Provide the optional symbol `rgpot_engine_last_error`.
    #[must_use]
    pub fn with_last_error(mut self, last_error: LastErrorFn) -> Self {
        self.last_error = Some(last_error);
        self
    }

    /// Provide the optional symbol `rgpot_engine_last_components`.
    #[must_use]
    pub fn with_last_components(mut self, last_components: LastComponentsFn) -> Self {
        self.last_components = Some(last_components);
        self
    }

    /// Provide the optional symbol `rgpot_engine_component_name`.
    #[must_use]
    pub fn with_component_name(mut self, component_name: ComponentNameFn) -> Self {
        self.component_name = Some(component_name);
        self
    }

    /// Provide the optional symbol `rgpot_engine_model_state`.
    #[must_use]
    pub fn with_model_state(mut self, model_state: ModelStateFn) -> Self {
        self.model_state = Some(model_state);
        self
    }

    /// Provide the optional symbol `rgpot_engine_set_callbacks`.
    #[must_use]
    pub fn with_set_callbacks(mut self, set_callbacks: SetCallbacksFn) -> Self {
        self.set_callbacks = Some(set_callbacks);
        self
    }

    /// Provide the optional symbol `rgpot_engine_hint`.
    #[must_use]
    pub fn with_hint(mut self, hint: HintFn) -> Self {
        self.hint = Some(hint);
        self
    }
}

/// Why an engine could not be loaded or an evaluation failed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineError {
    /// The shared library did not open.
    Load(String),
    /// A symbol the header requires is not exported or has a null address.
    Missing(String),
    /// The library speaks another ABI major.
    Abi {
        /// The ABI major expected by this loader.
        expected: i32,
        /// The ABI major returned by `rgpot_engine_abi_version`.
        found: i32,
    },
    /// The library's ABI stamp does not serve this loader's stamp.
    AbiStamp {
        /// The ABI stamp this loader was built against.
        expected: RgpotEngineAbiStamp,
        /// The ABI stamp reported (or implied) by the loaded engine.
        found: RgpotEngineAbiStamp,
    },
    /// An exported `rgpot_engine_available` reported that the engine cannot
    /// create instances.
    Unavailable,
    /// `rgpot_engine_create` returned no instance; the text is its error buffer.
    Create(String),
    /// The request does not fit the ABI: a length or a non-finite answer.
    Invalid(String),
    /// The requested optional operation is unavailable.
    Unsupported {
        /// Optional engine symbol for the operation.
        operation: &'static str,
        /// Why the operation is unavailable.
        message: String,
    },
    /// A positive code: the evaluation failed at this input.
    Recoverable {
        /// Positive error code returned by the engine.
        code: i32,
        /// Error message from `rgpot_engine_last_error`.
        message: String,
    },
    /// A negative code: the instance is unusable and the handle is poisoned;
    /// every subsequent call returns this error without calling into the engine.
    Fatal {
        /// Negative code of the first fatal failure, from the engine or from a
        /// host callback or transform.
        code: i32,
        /// Message of that failure: the host's, or the engine's error text.
        message: String,
    },
    /// Acquiring an engine or handle lock failed (`RGPOT_ENGINE_ERR_LOCK`); the
    /// instance is unusable and the handle is poisoned like [`EngineError::Fatal`].
    Lock {
        /// Engine symbol whose lock acquisition failed.
        operation: &'static str,
        /// Error message from `rgpot_engine_last_error`.
        message: String,
    },
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Load(why) => write!(f, "engine library did not load: {why}"),
            Self::Missing(name) => write!(f, "engine library does not export {name}"),
            Self::Abi { expected, found } => {
                write!(f, "engine ABI major {found}, this loader speaks {expected}")
            }
            Self::AbiStamp { expected, found } => {
                write!(
                    f,
                    "engine ABI stamp {found} cannot serve host stamp {expected}"
                )
            }
            Self::Unavailable => {
                write!(
                    f,
                    "rgpot_engine_available reported that the engine is unavailable"
                )
            }
            Self::Create(why) => write!(f, "rgpot_engine_create failed: {why}"),
            Self::Invalid(why) => write!(f, "invalid engine request: {why}"),
            Self::Unsupported { operation, message } => {
                write!(f, "engine does not support {operation}: {message}")
            }
            Self::Recoverable { code, message } => {
                write!(f, "engine evaluation failed with code {code}: {message}")
            }
            Self::Fatal { code, message } => {
                write!(f, "engine instance is unusable, code {code}: {message}")
            }
            Self::Lock { operation, message } => {
                write!(f, "engine lock failed in {operation}: {message}")
            }
        }
    }
}

impl std::error::Error for EngineError {}

impl EngineError {
    /// The engine ABI status when this error came from an engine call.
    pub const fn engine_status(&self) -> Option<EngineStatus> {
        match self {
            Self::Recoverable { code, .. } => EngineStatus::recoverable(*code),
            Self::Fatal { code, .. } => EngineStatus::fatal(*code),
            Self::Lock { .. } => EngineStatus::fatal(ENGINE_ERR_LOCK),
            _ => None,
        }
    }

    /// Map this failure onto the rgpot-core C API status code.
    pub fn status(&self) -> rgpot_status_t {
        match self {
            Self::Invalid(_) => rgpot_status_t::RGPOT_INVALID_PARAMETER,
            _ => rgpot_status_t::RGPOT_INTERNAL_ERROR,
        }
    }
}

/// One geometry. Positions in Angstrom, row-major `[n_atoms, 3]`; the cell is
/// the row-major 3 by 3 matrix in Angstrom.
#[derive(Debug, Clone, Copy)]
pub struct EngineRequest<'a> {
    /// Atomic coordinates in Angstrom, row-major `[n_atoms, 3]`.
    pub positions: &'a [f64],
    /// Atomic numbers, one per atom.
    pub atomic_numbers: &'a [i32],
    /// Row-major 3 by 3 lattice vectors in Angstrom.
    pub box_matrix: &'a [f64; 9],
}

/// Energy in eV and forces in eV/Angstrom.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineEvaluation {
    /// Total potential energy in eV.
    pub energy: f64,
    /// Cartesian atomic forces in eV/Angstrom, row-major `[n_atoms, 3]`.
    pub forces: Vec<f64>,
}

/// The error a nonzero engine return code stands for, raised by `operation`.
fn native_error(code: c_int, operation: &'static str, message: String) -> EngineError {
    let message = if message.is_empty() {
        format!("the engine left no message; code {code}")
    } else {
        message
    };
    if code > 0 {
        EngineError::Recoverable { code, message }
    } else {
        unusable_instance(code, operation, message)
    }
}

/// The error a host callback or transform failure stands for.
fn host_error(failure: EngineFailure) -> EngineError {
    let status = failure.status;
    let code = status.code();
    let message = described(failure);
    match status {
        EngineStatus::Success => EngineError::Invalid(message),
        EngineStatus::RecoverableInputFailure(_) => EngineError::Recoverable { code, message },
        EngineStatus::FatalInstanceFailure(_) => EngineError::Fatal { code, message },
    }
}

/// The message of a host failure, never empty.
fn described(failure: EngineFailure) -> String {
    let code = failure.status.code();
    match (failure.status, failure.message.is_empty()) {
        (EngineStatus::Success, true) => "a host failure was reported with success status".into(),
        (EngineStatus::Success, false) => format!(
            "a host failure was reported with success status: {}",
            failure.message
        ),
        (_, true) => format!("a host callback failed with code {code} and gave no message"),
        (_, false) => failure.message,
    }
}

/// The decisive failure the host reported during one native call, from the
/// transform and then the registered callbacks, including any refused state
/// the engine kept registered.
fn host_failure(
    transform: Option<&TransformCall<'_>>,
    callbacks: HostCallbacks<'_>,
) -> Option<EngineFailure> {
    let mut decisive = transform.and_then(TransformCall::take_failure);
    if let Some(failure) = callbacks.take_failure() {
        keep_decisive(&mut decisive, failure);
    }
    decisive
}

struct TransformState<'a> {
    transform: &'a mut dyn CoordinateTransform,
    failure: Option<EngineFailure>,
}

struct TransformCall<'a> {
    state: Mutex<TransformState<'a>>,
}

impl TransformCall<'_> {
    fn take_failure(&self) -> Option<EngineFailure> {
        match self.state.lock() {
            Ok(mut state) => state.failure.take(),
            Err(mut poisoned) => poisoned.get_mut().failure.take(),
        }
    }
}

unsafe extern "C" fn transform_trampoline(
    user: *mut c_void,
    n_atoms: c_long,
    positions: *mut f64,
    box_matrix: *mut f64,
) -> c_int {
    if user.is_null() || n_atoms <= 0 || positions.is_null() || box_matrix.is_null() {
        return ENGINE_ERR_INVALID;
    }
    let call = unsafe { &*user.cast::<TransformCall<'_>>() };
    let count = match usize::try_from(n_atoms) {
        Ok(count) => count,
        Err(_) => return -1,
    };
    let Some(position_len) = count.checked_mul(3) else {
        return -1;
    };
    let positions = unsafe { std::slice::from_raw_parts_mut(positions, position_len) };
    let box_matrix = unsafe { &mut *box_matrix.cast::<[f64; 9]>() };
    let mut state = match call.state.lock() {
        Ok(state) => state,
        Err(poisoned) => poisoned.into_inner(),
    };
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        state.transform.transform(positions, box_matrix)
    })) {
        Ok(Ok(())) => 0,
        Ok(Err(failure)) => {
            let code = match failure.status {
                EngineStatus::Success => ENGINE_ERR_INVALID,
                status => status.code(),
            };
            keep_decisive(&mut state.failure, failure);
            code
        }
        Err(_) => {
            keep_decisive(
                &mut state.failure,
                EngineFailure {
                    status: PANIC_STATUS,
                    message: "coordinate transform panicked".into(),
                },
            );
            -1
        }
    }
}

fn transform_parts(
    transform: &mut Option<TransformCall<'_>>,
) -> (Option<CoordTransformFn>, *mut c_void) {
    match transform {
        Some(call) => (
            Some(transform_trampoline),
            (call as *const TransformCall<'_>).cast_mut().cast(),
        ),
        None => (None, std::ptr::null_mut()),
    }
}

/// Geometry results obtained from one attached rgpot result cache.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EngineCacheCounts {
    /// Geometry results evaluated by the kernel.
    pub computed: u64,
    /// Geometry results read from the cache.
    pub served: u64,
}

type SetCache = unsafe extern "C" fn(*mut RgpotEnginePot, *const c_char) -> c_int;
type CacheCounts = unsafe extern "C" fn(*mut RgpotEnginePot, *mut u64, *mut u64) -> c_int;

/// Encode the schema's BuiltinParams message for an engine's default potential.
#[cfg(feature = "schema")]
pub fn builtin_params(name: &str) -> Vec<u8> {
    let mut message = capnp::message::Builder::new_default();
    message
        .init_root::<crate::Potentials_capnp::builtin_params::Builder>()
        .set_name(name);
    capnp::serialize::write_message_to_words(&message)
}

/// One engine instance behind the generic engine C ABI.
pub struct EnginePotential {
    set_cache: Option<SetCache>,
    cache_counts: Option<CacheCounts>,
    symbols: EngineSymbols,
    stamp: RgpotEngineAbiStamp,
    handle: NonNull<RgpotEnginePot>,
    fatal: OnceLock<EngineError>,
    process_lock: Option<Arc<Mutex<()>>>,
    caps: OnceLock<EngineCaps>,
    library: Option<Arc<Resident>>,
}

/// The file a plugin was loaded from.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum FileIdentity {
    #[cfg(unix)]
    Inode { dev: u64, ino: u64 },
    #[cfg(not(unix))]
    Path(PathBuf),
}

/// One loaded plugin file. The handle is closed only by
/// [`EnginePotential::unload`], and only when `unload_refusal` is `None`.
struct Resident {
    library: ManuallyDrop<Library>,
    _pin: Option<ManuallyDrop<Library>>,
    symbols: EngineSymbols,
    unload_refusal: Option<String>,
}

enum RegistryEntry {
    Loading { owner: ThreadId },
    Ready(Arc<Resident>),
}

#[derive(Default)]
struct Registry {
    by_file: HashMap<FileIdentity, RegistryEntry>,
    by_path: HashMap<PathBuf, FileIdentity>,
}

static REGISTRY: LazyLock<(Mutex<Registry>, Condvar)> =
    LazyLock::new(|| (Mutex::default(), Condvar::new()));

fn registry() -> MutexGuard<'static, Registry> {
    REGISTRY
        .0
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn canonical_path(path: &Path) -> Result<PathBuf, EngineError> {
    if path
        .parent()
        .is_none_or(|parent| parent.as_os_str().is_empty())
    {
        return Err(EngineError::Load(format!(
            "{}: a bare library name is resolved by the dynamic loader's search path, \
             so its file cannot be identified before it is opened; pass a path to the file",
            path.display()
        )));
    }
    std::fs::canonicalize(path)
        .map_err(|why| EngineError::Load(format!("{}: {why}", path.display())))
}

#[cfg(unix)]
fn file_identity(canonical: &Path) -> Result<FileIdentity, EngineError> {
    use std::os::unix::fs::MetadataExt;
    let meta = std::fs::metadata(canonical)
        .map_err(|why| EngineError::Load(format!("{}: {why}", canonical.display())))?;
    Ok(FileIdentity::Inode {
        dev: meta.dev(),
        ino: meta.ino(),
    })
}

#[cfg(not(unix))]
fn file_identity(canonical: &Path) -> Result<FileIdentity, EngineError> {
    Ok(FileIdentity::Path(canonical.to_path_buf()))
}

#[cfg(unix)]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
    unsafe { UnixLibrary::open(Some(path), RTLD_LAZY | RTLD_LOCAL) }.map(Library::from)
}

#[cfg(not(unix))]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    unsafe { Library::new(path) }
}

/// Marks the already open library at `path` `RTLD_NODELETE`. The returned
/// handle is never closed.
#[cfg(unix)]
unsafe fn pin_library(path: &Path) -> Result<Option<Library>, libloading::Error> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_LAZY, RTLD_LOCAL};
    let flags = RTLD_LAZY | RTLD_LOCAL | libc::RTLD_NOLOAD | libc::RTLD_NODELETE;
    unsafe { UnixLibrary::open(Some(path), flags) }.map(|pin| Some(Library::from(pin)))
}

#[cfg(not(unix))]
unsafe fn pin_library(_path: &Path) -> Result<Option<Library>, libloading::Error> {
    Ok(None)
}

/// Why the library may not be unloaded, or `None` when it declares itself
/// unload-safe through `rgpot_engine_unload_policy`.
unsafe fn unload_refusal(library: &Library, symbols: &EngineSymbols) -> Option<String> {
    let Some(policy) =
        (unsafe { optional::<UnloadPolicyFn>(library, "rgpot_engine_unload_policy") })
    else {
        return Some("rgpot_engine_unload_policy is not exported".into());
    };
    let major = unsafe { (symbols.abi_version)() };
    let stamp = symbols.abi_stamp.map(|abi_stamp| unsafe { abi_stamp() });
    match stamp {
        Some(stamp)
            if major == ENGINE_ABI_VERSION
                && stamp.serves(RgpotEngineAbiStamp::HOST)
                && stamp.abi_minor >= ENGINE_UNLOAD_POLICY_MINOR => {}
        _ => {
            return Some(format!(
                "the plugin's ABI stamp does not serve this loader at minor \
                 {ENGINE_UNLOAD_POLICY_MINOR} or later, so rgpot_engine_unload_policy is ignored"
            ))
        }
    }
    match unsafe { policy() } {
        ENGINE_UNLOAD_SAFE => None,
        code => Some(format!(
            "rgpot_engine_unload_policy returned {code}, not RGPOT_ENGINE_UNLOAD_SAFE"
        )),
    }
}

fn replaced(canonical: &Path) -> EngineError {
    EngineError::Load(format!(
        "{}: the file was replaced after it was loaded, and the loaded plugin stays \
         resident; load the new file from another path or restart the process",
        canonical.display()
    ))
}

/// Create an engine from the registry entry for `path`, opening it on first use.
///
/// A loading entry makes concurrent callers wait without holding the mutex
/// across `dlopen`. The completed library is published only after its symbols,
/// ABI stamp and first engine instance have all been validated.
unsafe fn load_resident(path: &Path, config: &[u8]) -> Result<EnginePotential, EngineError> {
    let canonical = canonical_path(path)?;
    let identity = file_identity(&canonical)?;
    let owner = std::thread::current().id();

    let cached = loop {
        let mut registry = registry();
        if registry
            .by_path
            .get(&canonical)
            .is_some_and(|known| *known != identity)
        {
            return Err(replaced(&canonical));
        }
        match registry.by_file.get(&identity) {
            Some(RegistryEntry::Ready(resident)) => {
                let resident = Arc::clone(resident);
                registry.by_path.insert(canonical.clone(), identity.clone());
                break Some(resident);
            }
            Some(RegistryEntry::Loading {
                owner: loading_owner,
            }) if *loading_owner == owner => {
                return Err(EngineError::Load(format!(
                    "{}: the plugin initializer recursively loaded the same file",
                    path.display()
                )));
            }
            Some(RegistryEntry::Loading { .. }) => {
                drop(
                    REGISTRY
                        .1
                        .wait(registry)
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                );
            }
            None => {
                registry
                    .by_file
                    .insert(identity.clone(), RegistryEntry::Loading { owner });
                registry.by_path.insert(canonical.clone(), identity.clone());
                break None;
            }
        }
    };

    if let Some(resident) = cached {
        let mut engine = unsafe { EnginePotential::from_symbols(resident.symbols, config)? };
        unsafe { engine.attach(resident) };
        return Ok(engine);
    }

    // From here until `publish`, dropping the claim removes the loading entry
    // and wakes the waiting threads, on an error return and on an unwind alike.
    let claim = LoadClaim {
        canonical,
        identity,
        published: false,
    };
    let (resident, mut engine) = unsafe { open_unpublished(path, &claim, config)? };
    let resident = Arc::new(resident);
    claim.publish(Arc::clone(&resident));
    unsafe { engine.attach(resident) };
    Ok(engine)
}

/// The loading entry a thread owns for one plugin file.
///
/// Another thread that wants the same file waits while the entry exists. A
/// claim dropped before [`Self::publish`] removes the entry and wakes those
/// threads, so a load that fails or panics never leaves them blocked.
struct LoadClaim {
    canonical: PathBuf,
    identity: FileIdentity,
    published: bool,
}

impl LoadClaim {
    /// Replace the loading entry with the finished plugin and wake the waiters.
    fn publish(mut self, resident: Arc<Resident>) {
        let mut registry = registry();
        registry
            .by_file
            .insert(self.identity.clone(), RegistryEntry::Ready(resident));
        registry
            .by_path
            .insert(self.canonical.clone(), self.identity.clone());
        self.published = true;
        REGISTRY.1.notify_all();
    }
}

impl Drop for LoadClaim {
    fn drop(&mut self) {
        if self.published {
            return;
        }
        let mut registry = registry();
        registry.by_file.remove(&self.identity);
        registry.by_path.retain(|_, known| *known != self.identity);
        REGISTRY.1.notify_all();
    }
}

/// Open the plugin, validate it and create its first engine, without touching
/// the registry.
///
/// Every handle is an owned [`Library`] until the last check passes, so each
/// early return and each unwind closes it. The locals drop in reverse
/// declaration order: the engine is destroyed while the library it calls into
/// is still open. Only the success path moves the handles into [`Resident`],
/// which never closes them.
unsafe fn open_unpublished(
    path: &Path,
    claim: &LoadClaim,
    config: &[u8],
) -> Result<(Resident, EnginePotential), EngineError> {
    let library = unsafe { open_library(&claim.canonical) }
        .map_err(|why| EngineError::Load(format!("{}: {why}", path.display())))?;
    let symbols = unsafe { resolve(&library)? };
    let engine = unsafe { EnginePotential::from_symbols(symbols, config)? };
    if file_identity(&claim.canonical)? != claim.identity {
        return Err(replaced(&claim.canonical));
    }
    let unload_refusal = unsafe { unload_refusal(&library, &symbols) };
    let pin = match unload_refusal {
        None => None,
        Some(_) => unsafe { pin_library(&claim.canonical) }.map_err(|why| {
            EngineError::Load(format!(
                "{}: RTLD_NODELETE could not be set: {why}",
                path.display()
            ))
        })?,
    };
    Ok((
        Resident {
            library: ManuallyDrop::new(library),
            _pin: pin.map(ManuallyDrop::new),
            symbols,
            unload_refusal,
        },
        engine,
    ))
}

// The instance is used by one caller at a time through `&mut self`. The
// header's reentrancy classes govern concurrent handles, which this type
// does not offer.
unsafe impl Send for EnginePotential {}

impl EnginePotential {
    /// Open the engine library at `path` and create an instance from the
    /// backend's configuration message.
    ///
    /// A library is published to the resident registry only after its required
    /// symbols and ABI identity validate and its first instance is created.
    /// Failed first attempts are closed and removed from the loading table. A
    /// later load of a published file reuses it. `path` must name the file: a
    /// bare library name, or a path whose file was replaced since it was
    /// loaded, is [`EngineError::Load`].
    ///
    /// # Safety
    ///
    /// The library must implement the engine C ABI with the signatures of
    /// this module, and `config` must be the message its create call reads.
    pub unsafe fn load(path: &Path, config: &[u8]) -> Result<Self, EngineError> {
        unsafe { load_resident(path, config) }
    }

    /// Keep `resident` open for this instance and resolve the optional cache
    /// exports it provides.
    ///
    /// # Safety
    ///
    /// `resident` must be the library whose symbols created this instance.
    unsafe fn attach(&mut self, resident: Arc<Resident>) {
        self.set_cache = unsafe { optional(&resident.library, "rgpot_engine_set_cache") };
        self.cache_counts = unsafe { optional(&resident.library, "rgpot_engine_cache_counts") };
        self.library = Some(resident);
    }

    /// Unload the plugin file at `path`: remove its registry entry and
    /// `dlclose` it.
    ///
    /// Only a plugin that exports `rgpot_engine_unload_policy` returning
    /// [`ENGINE_UNLOAD_SAFE`], with an ABI stamp of minor
    /// [`ENGINE_UNLOAD_POLICY_MINOR`] or later, can be unloaded. Any other
    /// plugin stays resident and the call returns [`EngineError::Unsupported`]
    /// for `rgpot_engine_unload_policy`. A plugin that was not loaded through
    /// [`Self::load`], or that still has a live [`EnginePotential`] or a
    /// [`PotentialImpl`] built from one, is [`EngineError::Invalid`]. A path
    /// whose file was replaced since it was loaded is [`EngineError::Load`].
    /// A later [`Self::load`] of the file opens it again.
    ///
    /// # Safety
    ///
    /// The plugin's declaration must be true: once this returns, no thread,
    /// TLS destructor, `atexit` handler or callback of the library may run,
    /// and no function pointer taken from it, including a copy of
    /// [`Self::symbols`], may be called.
    pub unsafe fn unload(path: &Path) -> Result<(), EngineError> {
        let canonical = canonical_path(path)?;
        let identity = file_identity(&canonical)?;
        let library = {
            let mut registry = registry();
            match registry.by_path.get(&canonical) {
                Some(known) if *known != identity => return Err(replaced(&canonical)),
                Some(_) => {}
                None => {
                    return Err(EngineError::Invalid(format!(
                        "{}: this plugin was not loaded",
                        canonical.display()
                    )))
                }
            }
            let resident = match registry.by_file.remove(&identity) {
                Some(RegistryEntry::Ready(resident)) => resident,
                Some(loading @ RegistryEntry::Loading { .. }) => {
                    registry.by_file.insert(identity, loading);
                    return Err(EngineError::Invalid(format!(
                        "{}: this plugin is still loading",
                        canonical.display()
                    )));
                }
                None => {
                    return Err(EngineError::Invalid(format!(
                        "{}: this plugin was not loaded",
                        canonical.display()
                    )))
                }
            };
            if let Some(why) = &resident.unload_refusal {
                let message = format!("{}: {why}; the plugin stays loaded", canonical.display());
                registry
                    .by_file
                    .insert(identity, RegistryEntry::Ready(resident));
                return Err(EngineError::Unsupported {
                    operation: "rgpot_engine_unload_policy",
                    message,
                });
            }
            let resident = match Arc::try_unwrap(resident) {
                Ok(resident) => resident,
                Err(shared) => {
                    let live = Arc::strong_count(&shared) - 1;
                    registry
                        .by_file
                        .insert(identity, RegistryEntry::Ready(shared));
                    return Err(EngineError::Invalid(format!(
                        "{}: {live} engine instance(s) still use this plugin",
                        canonical.display()
                    )));
                }
            };
            registry.by_path.retain(|_, known| *known != identity);
            ManuallyDrop::into_inner(resident.library)
        };
        // The registry lock is released: dlclose runs the library's destructors
        // and takes the dynamic loader's lock, and either may load an engine.
        library
            .close()
            .map_err(|why| EngineError::Load(format!("{}: {why}", canonical.display())))
    }

    /// Create an instance from symbols already in hand. [`Self::load`] is
    /// this after resolving them from a library.
    ///
    /// # Safety
    ///
    /// The symbols must honour the engine C ABI for as long as the instance
    /// lives, and `config` must be the message `symbols.create` reads.
    pub unsafe fn from_symbols(symbols: EngineSymbols, config: &[u8]) -> Result<Self, EngineError> {
        let found = unsafe { (symbols.abi_version)() };
        if found != ENGINE_ABI_VERSION {
            return Err(EngineError::Abi {
                expected: ENGINE_ABI_VERSION,
                found,
            });
        }
        // `rgpot_engine_abi_stamp` is optional; per `engine_c_abi.h`, an engine
        // without it is minor 0, layout 1.
        let stamp = match symbols.abi_stamp {
            Some(abi_stamp) => unsafe { abi_stamp() },
            None => RgpotEngineAbiStamp {
                abi_major: ENGINE_ABI_VERSION as c_ushort,
                abi_minor: 0,
                layout_revision: 1,
            },
        };
        if !stamp.serves(RgpotEngineAbiStamp::HOST) {
            return Err(EngineError::AbiStamp {
                expected: RgpotEngineAbiStamp::HOST,
                found: stamp,
            });
        }
        if let Some(available) = symbols.available {
            if unsafe { available() } == 0 {
                return Err(EngineError::Unavailable);
            }
        }
        // In-tree engines cast `config` to `capnp::word`, which requires
        // 8-byte alignment even when the caller passes an unaligned `&[u8]`.
        let mut aligned = vec![0_u64; config.len().div_ceil(8)];
        for (word, chunk) in aligned.iter_mut().zip(config.chunks(8)) {
            let mut bytes = [0_u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            *word = u64::from_ne_bytes(bytes);
        }
        let mut errbuf = vec![0 as c_char; ERROR_CAPACITY];
        let raw = unsafe {
            (symbols.create)(
                aligned.as_ptr().cast(),
                config.len(),
                errbuf.as_mut_ptr(),
                errbuf.len(),
            )
        };
        match NonNull::new(raw) {
            Some(handle) => {
                let process_lock = match probe_process_serial(&symbols, handle) {
                    Ok(lock) => lock,
                    Err(error) => {
                        unsafe { (symbols.destroy)(handle.as_ptr()) };
                        return Err(error);
                    }
                };
                Ok(Self {
                    set_cache: None,
                    cache_counts: None,
                    symbols,
                    stamp,
                    handle,
                    fatal: OnceLock::new(),
                    process_lock,
                    caps: OnceLock::new(),
                    library: None,
                })
            }
            None => Err(EngineError::Create(text_of(&errbuf))),
        }
    }

    /// The compatibility identity reported (or implied) by the loaded engine.
    pub fn abi_stamp(&self) -> RgpotEngineAbiStamp {
        self.stamp
    }

    /// The resolved symbol table of the loaded engine.
    pub fn symbols(&self) -> &EngineSymbols {
        &self.symbols
    }

    /// Energy and forces of one geometry.
    pub fn force(&mut self, request: &EngineRequest<'_>) -> Result<EngineEvaluation, EngineError> {
        self.force_with_transform(request, None)
    }

    /// Energy and forces of one geometry after an optional host transform.
    pub fn force_with_transform(
        &mut self,
        request: &EngineRequest<'_>,
        transform: Option<&mut dyn CoordinateTransform>,
    ) -> Result<EngineEvaluation, EngineError> {
        self.force_with_host(request, transform, HostCallbacks::NONE)
    }

    fn force_with_host(
        &mut self,
        request: &EngineRequest<'_>,
        transform: Option<&mut dyn CoordinateTransform>,
        callbacks: HostCallbacks<'_>,
    ) -> Result<EngineEvaluation, EngineError> {
        self.check_alive()?;
        let n_atoms = request.atomic_numbers.len();
        check_lengths(n_atoms, request.positions.len())?;
        let mut forces = vec![0.0; 3 * n_atoms];
        let mut energy = 0.0;
        let mut transform = transform.map(|transform| TransformCall {
            state: Mutex::new(TransformState {
                transform,
                failure: None,
            }),
        });
        let (transform_fn, transform_user) = transform_parts(&mut transform);
        let code = {
            let _serial = self.lock_process("rgpot_engine_force")?;
            unsafe {
                (self.symbols.force)(
                    self.handle.as_ptr(),
                    n_atoms as c_long,
                    request.positions.as_ptr(),
                    request.atomic_numbers.as_ptr(),
                    forces.as_mut_ptr(),
                    &mut energy,
                    std::ptr::null_mut(),
                    request.box_matrix.as_ptr(),
                    transform_fn,
                    transform_user,
                )
            }
        };
        let failure = host_failure(transform.as_ref(), callbacks);
        self.settle(code, "rgpot_engine_force", failure)?;
        finite(energy, &forces)?;
        Ok(EngineEvaluation { energy, forces })
    }

    /// `n_systems` geometries that share one composition, in one engine call.
    /// `positions` holds `3 * n_atoms` values per system and `boxes` nine.
    /// A library without `rgpot_engine_force_batch` is
    /// [`EngineError::Unsupported`]; the caller decides whether to loop.
    pub fn force_batch(
        &mut self,
        atomic_numbers: &[i32],
        positions: &[f64],
        boxes: &[f64],
    ) -> Result<Vec<EngineEvaluation>, EngineError> {
        self.force_batch_with_transform(atomic_numbers, positions, boxes, None)
    }

    /// Batched energy and forces after applying an optional transform to each
    /// engine-owned scratch geometry.
    pub fn force_batch_with_transform(
        &mut self,
        atomic_numbers: &[i32],
        positions: &[f64],
        boxes: &[f64],
        transform: Option<&mut dyn CoordinateTransform>,
    ) -> Result<Vec<EngineEvaluation>, EngineError> {
        self.force_batch_with_host(
            atomic_numbers,
            positions,
            boxes,
            transform,
            HostCallbacks::NONE,
        )
    }

    fn force_batch_with_host(
        &mut self,
        atomic_numbers: &[i32],
        positions: &[f64],
        boxes: &[f64],
        transform: Option<&mut dyn CoordinateTransform>,
        callbacks: HostCallbacks<'_>,
    ) -> Result<Vec<EngineEvaluation>, EngineError> {
        self.check_alive()?;
        let batch = self
            .symbols
            .force_batch
            .ok_or_else(|| unsupported_symbol("rgpot_engine_force_batch"))?;
        let n_atoms = atomic_numbers.len();
        if n_atoms == 0 || positions.is_empty() || !positions.len().is_multiple_of(3 * n_atoms) {
            return Err(EngineError::Invalid(format!(
                "{} coordinates are not a whole number of {n_atoms}-atom systems",
                positions.len()
            )));
        }
        let n_systems = positions.len() / (3 * n_atoms);
        if boxes.len() != 9 * n_systems {
            return Err(EngineError::Invalid(format!(
                "{} cell values for {n_systems} systems; expected {}",
                boxes.len(),
                9 * n_systems
            )));
        }
        let mut forces = vec![0.0; positions.len()];
        let mut energies = vec![0.0; n_systems];
        let mut transform = transform.map(|transform| TransformCall {
            state: Mutex::new(TransformState {
                transform,
                failure: None,
            }),
        });
        let (transform_fn, transform_user) = transform_parts(&mut transform);
        let code = {
            let _serial = self.lock_process("rgpot_engine_force_batch")?;
            unsafe {
                batch(
                    self.handle.as_ptr(),
                    n_systems as c_long,
                    n_atoms as c_long,
                    positions.as_ptr(),
                    atomic_numbers.as_ptr(),
                    forces.as_mut_ptr(),
                    energies.as_mut_ptr(),
                    std::ptr::null_mut(),
                    boxes.as_ptr(),
                    transform_fn,
                    transform_user,
                )
            }
        };
        let failure = host_failure(transform.as_ref(), callbacks);
        self.settle(code, "rgpot_engine_force_batch", failure)?;
        let mut out = Vec::with_capacity(n_systems);
        for (energy, block) in energies.iter().zip(forces.chunks_exact(3 * n_atoms)) {
            finite(*energy, block)?;
            out.push(EngineEvaluation {
                energy: *energy,
                forces: block.to_vec(),
            });
        }
        Ok(out)
    }

    /// The engine's capabilities from `rgpot_engine_caps`, queried once per
    /// instance. A caps struct shorter than this host's (an older engine
    /// minor) leaves the missing trailing fields at zero, so `jet_order` of a
    /// minor-4 engine is 0.
    pub fn caps(&self) -> Result<EngineCaps, EngineError> {
        self.check_alive()?;
        if let Some(caps) = self.caps.get() {
            return Ok(*caps);
        }
        let caps_fn = self
            .symbols
            .caps
            .ok_or_else(|| unsupported_symbol("rgpot_engine_caps"))?;
        let mut raw = RgpotEngineCaps {
            size: std::mem::size_of::<RgpotEngineCaps>() as c_uint,
            ..Default::default()
        };
        let code = {
            let _serial = self.lock_process("rgpot_engine_caps")?;
            unsafe { caps_fn(self.handle.as_ptr(), &mut raw) }
        };
        self.settle(code, "rgpot_engine_caps", None)?;
        let written = raw.size as usize;
        let jet_end =
            std::mem::offset_of!(RgpotEngineCaps, jet_order) + std::mem::size_of::<c_int>();
        let served_jet = if written >= jet_end { raw.jet_order } else { 0 };
        let jet_order = usize::try_from(served_jet).map_err(|_| {
            EngineError::Invalid(format!(
                "the engine reports a negative jet order {served_jet}"
            ))
        })?;
        let caps = EngineCaps {
            reentrancy: raw.reentrancy,
            per_image_instances: raw.per_image_instances != 0,
            batched: raw.batched != 0,
            periodic: raw.periodic != 0,
            stress: raw.stress != 0,
            group_collective: raw.group_collective != 0,
            analytic_hvp: raw.analytic_hvp != 0,
            jet_order,
        };
        Ok(*self.caps.get_or_init(|| caps))
    }

    /// Analytic Hessian-vector product `H(positions) * direction` in eV/Angstrom^2
    /// times `direction`, in internal units (Angstrom).
    ///
    /// Returns [`EngineError::Unsupported`] when the library omits
    /// `rgpot_engine_hvp`, when `rgpot_engine_caps` reports `analytic_hvp == 0`,
    /// or when the call returns `ENGINE_ERR_UNSUPPORTED` (3). No finite-difference
    /// fallback is attempted.
    pub fn hvp(
        &mut self,
        request: &EngineRequest<'_>,
        direction: &[f64],
    ) -> Result<Vec<f64>, EngineError> {
        self.check_alive()?;
        let hvp_fn = self
            .symbols
            .hvp
            .ok_or_else(|| unsupported_symbol("rgpot_engine_hvp"))?;
        if self.symbols.caps.is_none() {
            return Err(EngineError::Unsupported {
                operation: "rgpot_engine_hvp",
                message: "rgpot_engine_caps is not exported".into(),
            });
        }
        if !self.caps()?.analytic_hvp {
            return Err(EngineError::Unsupported {
                operation: "rgpot_engine_hvp",
                message: "analytic_hvp is not advertised".into(),
            });
        }
        let n_atoms = request.atomic_numbers.len();
        check_lengths(n_atoms, request.positions.len())?;
        if direction.len() != 3 * n_atoms {
            return Err(EngineError::Invalid(format!(
                "{} direction entries for {n_atoms} atoms; expected {}",
                direction.len(),
                3 * n_atoms
            )));
        }
        if let Some(i) = request.positions.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::Invalid(format!(
                "the engine position at element {i} is not finite"
            )));
        }
        if let Some(i) = direction.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::Invalid(format!(
                "the engine direction at element {i} is not finite"
            )));
        }
        let mut hv_out = vec![0.0; 3 * n_atoms];
        let code = {
            let _serial = self.lock_process("rgpot_engine_hvp")?;
            unsafe {
                hvp_fn(
                    self.handle.as_ptr(),
                    n_atoms as c_long,
                    request.positions.as_ptr(),
                    request.atomic_numbers.as_ptr(),
                    direction.as_ptr(),
                    hv_out.as_mut_ptr(),
                    request.box_matrix.as_ptr(),
                )
            }
        };
        if code == ENGINE_ERR_UNSUPPORTED {
            return Err(EngineError::Unsupported {
                operation: "rgpot_engine_hvp",
                message: self.last_error_text(code),
            });
        }
        self.settle(code, "rgpot_engine_hvp", None)?;
        finite_product(&hv_out)?;
        Ok(hv_out)
    }

    /// Attach a persistent result cache or detach it with `None`.
    ///
    /// The directory is dedicated to one model and its parameters.
    /// The engine receives the path as a narrow C string.
    /// On Unix the bytes of the path pass through unchanged.
    /// On Windows the engine opens the path as UTF-8, so a path that is not
    /// valid Unicode is refused with [`EngineError::Invalid`] instead of
    /// being altered on the way.
    pub fn set_cache(&mut self, path: Option<&Path>) -> Result<(), EngineError> {
        self.check_alive()?;
        let operation = "rgpot_engine_set_cache";
        let set = self
            .set_cache
            .ok_or_else(|| unsupported_symbol(operation))?;
        let path = path.map(cache_path).transpose()?;
        let code = unsafe {
            set(
                self.handle.as_ptr(),
                path.as_ref().map_or(std::ptr::null(), |s| s.as_ptr()),
            )
        };
        self.cache_status(operation, code)
    }

    /// Read the engine's kernel-computed and cache-served geometry counts.
    /// An unattached cache or an unavailable query returns [`EngineError::Unsupported`].
    pub fn cache_counts(&self) -> Result<EngineCacheCounts, EngineError> {
        self.check_alive()?;
        let operation = "rgpot_engine_cache_counts";
        let read = self
            .cache_counts
            .ok_or_else(|| unsupported_symbol(operation))?;
        let mut counts = EngineCacheCounts::default();
        let code = unsafe {
            read(
                self.handle.as_ptr(),
                &mut counts.computed,
                &mut counts.served,
            )
        };
        self.cache_status(operation, code)?;
        Ok(counts)
    }

    fn cache_status(&self, operation: &'static str, code: c_int) -> Result<(), EngineError> {
        if code == ENGINE_ERR_UNSUPPORTED {
            return Err(EngineError::Unsupported {
                operation,
                message: self.last_error_text(code),
            });
        }
        self.settle(code, operation, None)
    }

    /// Per-term record of the last successful force call on a composite engine
    /// via `rgpot_engine_last_components`. Returns [`EngineError::Unsupported`]
    /// when the plugin does not export `rgpot_engine_last_components`.
    pub fn last_components(&self) -> Result<Vec<RgpotEngineComponent>, EngineError> {
        self.check_alive()?;
        let last_components = self
            .symbols
            .last_components
            .ok_or_else(|| unsupported_symbol("rgpot_engine_last_components"))?;
        let total = unsafe {
            last_components(
                self.handle.as_ptr(),
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if total < 0 {
            return Err(EngineError::Invalid(format!(
                "rgpot_engine_last_components returned negative count {total}"
            )));
        }
        if total == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![RgpotEngineComponent::default(); total as usize];
        let written = unsafe {
            last_components(
                self.handle.as_ptr(),
                out.as_mut_ptr(),
                std::ptr::null_mut(),
                total,
            )
        };
        if written < 0 {
            return Err(EngineError::Invalid(format!(
                "rgpot_engine_last_components returned negative count {written}"
            )));
        }
        out.truncate((written as usize).min(out.len()));
        Ok(out)
    }

    /// Name of term `index` of a composite engine via
    /// `rgpot_engine_component_name`. Returns [`EngineError::Unsupported`] when
    /// the plugin does not export `rgpot_engine_component_name`.
    pub fn component_name(&self, index: usize) -> Result<String, EngineError> {
        self.check_alive()?;
        let component_name = self
            .symbols
            .component_name
            .ok_or_else(|| unsupported_symbol("rgpot_engine_component_name"))?;
        let index = c_long::try_from(index).map_err(|_| {
            EngineError::Invalid(format!("component index {index} does not fit c_long"))
        })?;
        let len = unsafe { component_name(self.handle.as_ptr(), index, std::ptr::null_mut(), 0) };
        if len == 0 {
            return Ok(String::new());
        }
        let mut buf = vec![0 as c_char; len + 1];
        unsafe { component_name(self.handle.as_ptr(), index, buf.as_mut_ptr(), buf.len()) };
        Ok(text_of(&buf))
    }

    /// Directional derivatives of the gradient up to `order` along the
    /// directions (Angstrom) of a uniform batch, in one engine call, with the
    /// engine's roundoff report when it tracks one.
    ///
    /// Returns [`EngineError::Unsupported`] when the library omits
    /// `rgpot_engine_taylor_jet` or `rgpot_engine_caps` (an older ABI minor),
    /// when `jet_order` is below `order`, or when the call returns
    /// `ENGINE_ERR_UNSUPPORTED` (3). No finite-difference fallback is
    /// attempted. A non-finite position, direction or box, and an order of
    /// zero, are [`EngineError::Invalid`] before the engine is called. A
    /// non-finite or negative report scale is [`EngineError::Invalid`] as
    /// well.
    pub fn taylor_jet(
        &mut self,
        request: &JetBatchRequest<'_>,
        order: usize,
    ) -> Result<GradientJet, EngineError> {
        self.taylor_jet_with_host(request, order, HostCallbacks::NONE)
    }

    fn taylor_jet_with_host(
        &mut self,
        request: &JetBatchRequest<'_>,
        order: usize,
        callbacks: HostCallbacks<'_>,
    ) -> Result<GradientJet, EngineError> {
        self.check_alive()?;
        let jet_fn = self
            .symbols
            .taylor_jet
            .ok_or_else(|| unsupported_symbol("rgpot_engine_taylor_jet"))?;
        if self.symbols.caps.is_none() {
            return Err(EngineError::Unsupported {
                operation: "rgpot_engine_taylor_jet",
                message: "rgpot_engine_caps is not exported".into(),
            });
        }
        if order == 0 {
            return Err(EngineError::Invalid("a jet needs order at least 1".into()));
        }
        let served = self.caps()?.jet_order;
        if served < order {
            return Err(EngineError::Unsupported {
                operation: "rgpot_engine_taylor_jet",
                message: format!("jet order {order} requested; the engine serves {served}"),
            });
        }
        let n_atoms = request.atomic_numbers.len();
        if n_atoms == 0
            || request.positions.is_empty()
            || !request.positions.len().is_multiple_of(3 * n_atoms)
        {
            return Err(EngineError::Invalid(format!(
                "{} coordinates are not a whole number of {n_atoms}-atom systems",
                request.positions.len()
            )));
        }
        let n_systems = request.positions.len() / (3 * n_atoms);
        if request.directions.len() != request.positions.len() {
            return Err(EngineError::Invalid(format!(
                "{} direction entries for {} coordinates",
                request.directions.len(),
                request.positions.len()
            )));
        }
        if request.boxes.len() != 9 * n_systems {
            return Err(EngineError::Invalid(format!(
                "{} cell values for {n_systems} systems; expected {}",
                request.boxes.len(),
                9 * n_systems
            )));
        }
        if let Some(i) = request.positions.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::Invalid(format!(
                "the engine position at element {i} is not finite"
            )));
        }
        if let Some(i) = request.directions.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::Invalid(format!(
                "the engine direction at element {i} is not finite"
            )));
        }
        if let Some(i) = request.boxes.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::Invalid(format!(
                "the engine box at element {i} is not finite"
            )));
        }
        let jet_len = order.checked_mul(request.positions.len()).ok_or_else(|| {
            EngineError::Invalid(format!(
                "a jet of order {order} for {} coordinates does not fit in memory",
                request.positions.len()
            ))
        })?;
        let mut derivatives = vec![0.0; jet_len];
        let mut raw = RgpotEngineDerivativeReport::default();
        let code = {
            let _serial = self.lock_process("rgpot_engine_taylor_jet")?;
            unsafe {
                jet_fn(
                    self.handle.as_ptr(),
                    n_systems as c_long,
                    n_atoms as c_long,
                    request.positions.as_ptr(),
                    request.atomic_numbers.as_ptr(),
                    request.directions.as_ptr(),
                    order as c_long,
                    derivatives.as_mut_ptr(),
                    &mut raw,
                    request.boxes.as_ptr(),
                )
            }
        };
        let failure = host_failure(None, callbacks);
        if code == ENGINE_ERR_UNSUPPORTED && failure.is_none() {
            return Err(EngineError::Unsupported {
                operation: "rgpot_engine_taylor_jet",
                message: self.last_error_text(code),
            });
        }
        self.settle(code, "rgpot_engine_taylor_jet", failure)?;
        if let Some(i) = derivatives.iter().position(|v| !v.is_finite()) {
            return Err(EngineError::Invalid(format!(
                "the engine jet at element {i} is not finite"
            )));
        }
        let report = if raw.tracked != 0 {
            let operations = u64::try_from(raw.operations).map_err(|_| {
                EngineError::Invalid(format!(
                    "the engine reports a negative operation count {}",
                    raw.operations
                ))
            })?;
            if !raw.scale.is_finite() {
                return Err(EngineError::Invalid(
                    "the engine jet scale is not finite".into(),
                ));
            }
            if raw.scale < 0.0 {
                return Err(EngineError::Invalid(
                    "the engine jet scale is negative".into(),
                ));
            }
            Some(DerivativeReport {
                operations,
                scale: raw.scale,
            })
        } else {
            None
        };
        Ok(GradientJet {
            order,
            n_atoms,
            n_systems,
            derivatives,
            report,
        })
    }

    /// The instance as a [`PotentialImpl`]. The handle owns the engine and
    /// destroys it on drop, and holds the same resident registry entry as
    /// [`Self::load`]. Calls through the handle are serialized. When
    /// both the optional capability and HVP symbols are present and
    /// `analytic_hvp` is advertised, the handle also receives an analytic
    /// Hessian-vector callback.
    pub fn into_potential(self) -> PotentialImpl {
        let analytic_hvp =
            self.symbols.hvp.is_some() && matches!(self.caps(), Ok(caps) if caps.analytic_hvp);
        let owned = Box::into_raw(Box::new(Mutex::new(self))).cast::<c_void>();
        let potential = PotentialImpl::new(engine_callback, owned, Some(drop_engine));
        if analytic_hvp {
            potential
                .as_handle()
                .set_hessian_vector_callback(Some(engine_hessian_vector_callback));
        }
        potential
    }

    /// Take the library-wide lock of a process-serial engine, if it has one.
    /// A poisoned lock means a thread panicked inside native code that this
    /// handle shares, so the handle is latched unusable with
    /// [`EngineError::Lock`] like any other fatal failure and every later call
    /// returns that same error.
    fn lock_process(
        &self,
        operation: &'static str,
    ) -> Result<Option<MutexGuard<'_, ()>>, EngineError> {
        match &self.process_lock {
            Some(lock) => match lock.lock() {
                Ok(guard) => Ok(Some(guard)),
                Err(_) => Err(self.poison(EngineError::Lock {
                    operation,
                    message: "process-serial engine lock is poisoned".into(),
                })),
            },
            None => Ok(None),
        }
    }

    fn check_alive(&self) -> Result<(), EngineError> {
        match self.fatal.get() {
            Some(error) => Err(error.clone()),
            None => Ok(()),
        }
    }

    /// [`Self::settle_with`] with the message from `rgpot_engine_last_error`.
    fn settle(
        &self,
        code: c_int,
        operation: &'static str,
        host: Option<EngineFailure>,
    ) -> Result<(), EngineError> {
        self.settle_with(code, operation, host, || self.last_error_text(code))
    }

    /// Classify one native call that returned `code`.
    ///
    /// `host` is the decisive failure a host callback or transform reported
    /// during the call. It happened before the engine returned, so it is
    /// considered first. A fatal failure from either source wins over a
    /// recoverable one, and the first fatal failure poisons the handle: the
    /// error returned here and by every later call is that one, message
    /// included.
    fn settle_with(
        &self,
        code: c_int,
        operation: &'static str,
        host: Option<EngineFailure>,
        native_message: impl FnOnce() -> String,
    ) -> Result<(), EngineError> {
        let native = (code != 0).then(|| native_error(code, operation, native_message()));
        let mut decisive = None;
        for error in host.map(host_error).into_iter().chain(native) {
            if matches!(error, EngineError::Fatal { .. } | EngineError::Lock { .. }) {
                return Err(self.poison(error));
            }
            decisive.get_or_insert(error);
        }
        decisive.map_or(Ok(()), Err)
    }

    /// Latch `error` unless a fatal failure is already latched, and return the
    /// latched error.
    fn poison(&self, error: EngineError) -> EngineError {
        self.fatal.get_or_init(|| error).clone()
    }

    fn last_error_text(&self, code: c_int) -> String {
        engine_message(&self.symbols, self.handle, code)
    }
}

fn engine_message(symbols: &EngineSymbols, handle: NonNull<RgpotEnginePot>, code: c_int) -> String {
    let Some(last_error) = symbols.last_error else {
        return format!("no rgpot_engine_last_error; code {code}");
    };
    let mut buffer = vec![0 as c_char; ERROR_CAPACITY];
    let length = unsafe { last_error(handle.as_ptr(), buffer.as_mut_ptr(), buffer.len()) };
    if length == 0 {
        return format!("the engine left no message; code {code}");
    }
    text_of(&buffer)
}

fn unusable_instance(code: c_int, operation: &'static str, message: String) -> EngineError {
    if code == ENGINE_ERR_LOCK {
        EngineError::Lock { operation, message }
    } else {
        EngineError::Fatal { code, message }
    }
}

struct CallbackState {
    callbacks: Mutex<Box<dyn EngineCallbacks>>,
    failure: Mutex<Option<EngineFailure>>,
}

impl CallbackState {
    fn slot(&self) -> MutexGuard<'_, Option<EngineFailure>> {
        self.failure
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn record(&self, failure: EngineFailure) {
        keep_decisive(&mut self.slot(), failure);
    }

    fn take_failure(&self) -> Option<EngineFailure> {
        self.slot().take()
    }

    /// Take the recorded failure only when it is fatal. A recoverable one
    /// stays recorded for the next native call to report.
    fn take_fatal(&self) -> Option<EngineFailure> {
        let mut slot = self.slot();
        if slot
            .as_ref()
            .is_some_and(|failure| failure.status.is_fatal())
        {
            slot.take()
        } else {
            None
        }
    }
}

/// Every callback state an engine may still call: the registered one and the
/// refused ones an older engine kept.
#[derive(Clone, Copy)]
struct HostCallbacks<'a> {
    active: Option<&'a CallbackState>,
    refused: &'a [Box<CallbackState>],
}

impl<'a> HostCallbacks<'a> {
    const NONE: Self = Self {
        active: None,
        refused: &[],
    };

    fn new(active: &'a Option<Box<CallbackState>>, refused: &'a RefusedCallbacks) -> Self {
        Self {
            active: active.as_deref(),
            refused: &refused.0,
        }
    }

    fn states(self) -> impl Iterator<Item = &'a CallbackState> {
        self.active
            .into_iter()
            .chain(self.refused.iter().map(|state| &**state))
    }

    /// The decisive recorded failure, drained from every state so none is
    /// left to surface in a later call.
    fn take_failure(self) -> Option<EngineFailure> {
        let mut decisive = None;
        for failure in self.states().filter_map(CallbackState::take_failure) {
            keep_decisive(&mut decisive, failure);
        }
        decisive
    }

    /// The decisive recorded fatal failure. Recoverable ones stay recorded.
    fn take_fatal(self) -> Option<EngineFailure> {
        let mut decisive = None;
        for failure in self.states().filter_map(CallbackState::take_fatal) {
            keep_decisive(&mut decisive, failure);
        }
        decisive
    }
}

unsafe extern "C" fn oracle_trampoline(
    ctx: *mut c_void,
    n_atoms: c_long,
    positions: *const f64,
    atomic_numbers: *const c_int,
    forces: *mut f64,
    energy: *mut f64,
    box_matrix: *const f64,
) -> c_int {
    if ctx.is_null()
        || n_atoms <= 0
        || positions.is_null()
        || atomic_numbers.is_null()
        || forces.is_null()
        || energy.is_null()
        || box_matrix.is_null()
    {
        return ENGINE_ERR_INVALID;
    }
    let state = unsafe { &*ctx.cast::<CallbackState>() };
    let count = match usize::try_from(n_atoms) {
        Ok(count) => count,
        Err(_) => return -1,
    };
    let Some(position_len) = count.checked_mul(3) else {
        return -1;
    };
    let request = EngineRequest {
        positions: unsafe { std::slice::from_raw_parts(positions, position_len) },
        atomic_numbers: unsafe { std::slice::from_raw_parts(atomic_numbers, count) },
        box_matrix: unsafe { &*box_matrix.cast::<[f64; 9]>() },
    };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut callbacks = match state.callbacks.lock() {
            Ok(callbacks) => callbacks,
            Err(poisoned) => poisoned.into_inner(),
        };
        callbacks.oracle(&request)
    }));
    let failure = match result {
        Ok(Ok(answer)) if answer.forces.len() != position_len => EngineFailure {
            status: INVALID_STATUS,
            message: "oracle returned the wrong force length".into(),
        },
        Ok(Ok(answer)) => match non_finite_output(answer.energy, &answer.forces) {
            Some(message) => EngineFailure {
                status: INVALID_STATUS,
                message: format!("oracle {message}"),
            },
            None => {
                unsafe {
                    std::ptr::copy_nonoverlapping(answer.forces.as_ptr(), forces, position_len);
                    *energy = answer.energy;
                }
                return 0;
            }
        },
        Ok(Err(failure)) => EngineFailure {
            status: failure.status.or_invalid(),
            message: failure.message,
        },
        Err(_) => EngineFailure {
            status: PANIC_STATUS,
            message: "oracle callback panicked".into(),
        },
    };
    let code = failure.status.code();
    state.record(failure);
    code
}

unsafe extern "C" fn acquire_trampoline(ctx: *mut c_void, reason: c_int) {
    if ctx.is_null() {
        return;
    }
    let state = unsafe { &*ctx.cast::<CallbackState>() };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut callbacks = match state.callbacks.lock() {
            Ok(callbacks) => callbacks,
            Err(poisoned) => poisoned.into_inner(),
        };
        callbacks.on_acquire(reason.into());
    }));
    if result.is_err() {
        state.record(EngineFailure {
            status: PANIC_STATUS,
            message: "acquisition callback panicked".into(),
        });
    }
}

unsafe extern "C" fn model_update_trampoline(ctx: *mut c_void, model_state: u64) {
    if ctx.is_null() {
        return;
    }
    let state = unsafe { &*ctx.cast::<CallbackState>() };
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut callbacks = match state.callbacks.lock() {
            Ok(callbacks) => callbacks,
            Err(poisoned) => poisoned.into_inner(),
        };
        callbacks.on_model_update(model_state);
    }));
    if result.is_err() {
        state.record(EngineFailure {
            status: PANIC_STATUS,
            message: "model-update callback panicked".into(),
        });
    }
}

/// Callback states an engine refused, kept until the engine is destroyed.
///
/// A refused `rgpot_engine_set_callbacks` call must retain nothing, but a
/// 1.5 engine written before the header said so may have kept the context it
/// rejected. Freeing it would leave that engine holding a dangling pointer, so
/// the session keeps each refused state alive until `rgpot_engine_destroy` has
/// run. A destructor that panics is contained here: it cannot unwind out of
/// the session's own drop.
#[derive(Default)]
// The boxes are the point: an engine holds the address of each state.
#[allow(clippy::vec_box)]
struct RefusedCallbacks(Vec<Box<CallbackState>>);

impl Drop for RefusedCallbacks {
    fn drop(&mut self) {
        for state in self.0.drain(..) {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || drop(state)));
        }
    }
}

/// Owning session over one version-checked engine instance.
///
/// The fields drop in order: the native engine is destroyed first, so no
/// native callback can reach a context after it is freed.
pub struct EngineSession {
    engine: EnginePotential,
    callbacks: Option<Box<CallbackState>>,
    refused: RefusedCallbacks,
}

impl EngineSession {
    /// Load a version-compatible engine and create one session.
    ///
    /// The engine is created before any callback can be registered, because
    /// `rgpot_engine_create` takes none. Register an oracle afterwards with
    /// [`Self::set_callbacks`]; a learning engine that needs one fails its
    /// evaluations until then.
    ///
    /// # Safety
    ///
    /// The library is trusted to run initializers safely and to implement the
    /// versioned `rgpot_engine_*` ABI exactly. The version and optional ABI
    /// stamp checks cannot prove that arbitrary machine code obeys that
    /// contract.
    pub unsafe fn load(path: &Path, config: &[u8]) -> Result<Self, EngineError> {
        let engine = unsafe { EnginePotential::load(path, config) }?;
        Ok(Self {
            engine,
            callbacks: None,
            refused: RefusedCallbacks::default(),
        })
    }

    /// Evaluate one geometry.
    pub fn force(
        &mut self,
        request: &EngineRequest<'_>,
        transform: Option<&mut dyn CoordinateTransform>,
    ) -> Result<EngineEvaluation, EngineError> {
        let answer = self.engine.force_with_host(
            request,
            transform,
            HostCallbacks::new(&self.callbacks, &self.refused),
        );
        self.finish(answer)
    }

    /// Evaluate a strided-uniform batch in one engine call.
    pub fn force_batch(
        &mut self,
        atomic_numbers: &[i32],
        positions: &[f64],
        boxes: &[f64],
        transform: Option<&mut dyn CoordinateTransform>,
    ) -> Result<Vec<EngineEvaluation>, EngineError> {
        let answer = self.engine.force_batch_with_host(
            atomic_numbers,
            positions,
            boxes,
            transform,
            HostCallbacks::new(&self.callbacks, &self.refused),
        );
        self.finish(answer)
    }

    /// Directional derivatives of the gradient up to `order` along the
    /// directions of a uniform batch, in one engine call.
    ///
    /// A failure a registered callback reports during the call is settled with
    /// the code the engine returns, exactly as for [`Self::force`]. An engine
    /// that serves no jet, or none of this order, is
    /// [`EngineError::Unsupported`]; nothing is estimated by finite
    /// differences instead.
    pub fn taylor_jet(
        &mut self,
        request: &JetBatchRequest<'_>,
        order: usize,
    ) -> Result<GradientJet, EngineError> {
        let answer = self.engine.taylor_jet_with_host(
            request,
            order,
            HostCallbacks::new(&self.callbacks, &self.refused),
        );
        self.finish(answer)
    }

    /// Register owned learning-engine callbacks, or clear them with `None`.
    ///
    /// `None` is represented by a null callback-table pointer. If native
    /// replacement fails, the previous owned callbacks remain registered and
    /// the refused ones are kept alive until the engine is destroyed, in case
    /// an older engine retained them. A failure recorded by a refused state
    /// that the engine still calls is reported like one from the registered
    /// callbacks.
    /// A failure the previous callbacks recorded and no call has reported yet
    /// is reported here, since a successful replacement drops them. So is a
    /// panic in their destructor, which poisons the session like any other
    /// callback panic.
    pub fn set_callbacks(
        &mut self,
        callbacks: Option<Box<dyn EngineCallbacks>>,
    ) -> Result<(), EngineError> {
        self.engine.check_alive()?;
        let set_callbacks = self
            .engine
            .symbols
            .set_callbacks
            .ok_or_else(|| unsupported_symbol("rgpot_engine_set_callbacks"))?;
        let mut next = callbacks.map(|callbacks| {
            Box::new(CallbackState {
                callbacks: Mutex::new(callbacks),
                failure: Mutex::new(None),
            })
        });
        let raw = match &mut next {
            Some(state) => RgpotEngineCallbacks {
                ctx: (&mut **state as *mut CallbackState).cast(),
                oracle: Some(oracle_trampoline),
                on_acquire: Some(acquire_trampoline),
                on_model_update: Some(model_update_trampoline),
            },
            None => RgpotEngineCallbacks {
                ctx: std::ptr::null_mut(),
                oracle: None,
                on_acquire: None,
                on_model_update: None,
            },
        };
        let raw_ptr = if next.is_some() {
            &raw
        } else {
            std::ptr::null()
        };
        let code = unsafe { set_callbacks(self.engine.handle.as_ptr(), raw_ptr) };
        let mut failure = HostCallbacks::new(&self.callbacks, &self.refused).take_failure();
        if let Some(replacement) = next.as_deref().and_then(CallbackState::take_failure) {
            keep_decisive(&mut failure, replacement);
        }
        if code == 0 {
            // Install the replacement before the old state is dropped, so a
            // destructor that panics cannot leave the engine holding a
            // context the session no longer owns.
            let previous = std::mem::replace(&mut self.callbacks, next);
            let dropped = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                drop(previous);
            }));
            if dropped.is_err() {
                keep_decisive(
                    &mut failure,
                    EngineFailure {
                        status: PANIC_STATUS,
                        message: "callback destructor panicked".into(),
                    },
                );
            }
        } else if let Some(refused) = next {
            self.refused.0.push(refused);
        }
        self.engine
            .settle(code, "rgpot_engine_set_callbacks", failure)
    }

    /// Attach a persistent result cache stored in the directory `path`.
    ///
    /// Give each model and parameter set its own directory.
    /// Attaching starts the counts at zero and replaces any cache already
    /// attached.
    /// A refused attachment leaves the previous cache in place.
    /// An engine that does not export `rgpot_engine_set_cache`, or a
    /// composite engine whose per-term records the cache cannot hold,
    /// returns [`EngineError::Unsupported`].
    pub fn attach_cache(&mut self, path: &Path) -> Result<(), EngineError> {
        let answer = self.engine.set_cache(Some(path));
        self.finish(answer)
    }

    /// Detach the result cache and close it.
    ///
    /// The stored entries stay on disk and serve a later attachment of the
    /// same directory.
    pub fn clear_cache(&mut self) -> Result<(), EngineError> {
        let answer = self.engine.set_cache(None);
        self.finish(answer)
    }

    /// Status of the attached cache: results the kernel computed and results
    /// the cache served since attachment.
    ///
    /// With no cache attached, or on an engine that does not export
    /// `rgpot_engine_cache_counts`, this returns [`EngineError::Unsupported`].
    pub fn cache_counts(&self) -> Result<EngineCacheCounts, EngineError> {
        self.finish(self.engine.cache_counts())
    }

    /// Monotone model state, or zero for a static engine.
    pub fn model_state(&self) -> Result<u64, EngineError> {
        self.engine.check_alive()?;
        Ok(match self.engine.symbols.model_state {
            Some(model_state) => unsafe { model_state(self.engine.handle.as_ptr()) },
            None => 0,
        })
    }

    /// Send a typed runtime hint message to a learning engine.
    ///
    /// The message of a failed hint is the text the engine wrote to the error
    /// buffer, or one naming the code when the engine wrote none.
    pub fn hint(&mut self, message: &[u8]) -> Result<(), EngineError> {
        let answer = self.send_hint(message);
        self.finish(answer)
    }

    fn send_hint(&mut self, message: &[u8]) -> Result<(), EngineError> {
        self.engine.check_alive()?;
        if message.is_empty() {
            return Err(EngineError::Invalid(
                "an engine hint cannot be empty".into(),
            ));
        }
        let hint = self
            .engine
            .symbols
            .hint
            .ok_or_else(|| unsupported_symbol("rgpot_engine_hint"))?;
        let mut aligned = vec![0_u64; message.len().div_ceil(8)];
        for (word, chunk) in aligned.iter_mut().zip(message.chunks(8)) {
            let mut bytes = [0_u8; 8];
            bytes[..chunk.len()].copy_from_slice(chunk);
            *word = u64::from_ne_bytes(bytes);
        }
        let mut error = vec![0 as c_char; ERROR_CAPACITY];
        let code = unsafe {
            hint(
                self.engine.handle.as_ptr(),
                aligned.as_ptr().cast(),
                message.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        let failure = host_failure(None, HostCallbacks::new(&self.callbacks, &self.refused));
        self.engine
            .settle_with(code, "rgpot_engine_hint", failure, || {
                let text = text_of(&error);
                if text.is_empty() {
                    format!("rgpot_engine_hint failed with code {code} and wrote no message")
                } else {
                    text
                }
            })
    }

    /// Report a fatal callback failure that no native call has settled, which
    /// happens when a call is refused before it reaches the engine. A fatal
    /// failure wins over the refusal; an error the handle has already latched
    /// is returned unchanged.
    fn finish<T>(&self, answer: Result<T, EngineError>) -> Result<T, EngineError> {
        if matches!(
            answer,
            Err(EngineError::Fatal { .. } | EngineError::Lock { .. })
        ) {
            return answer;
        }
        match HostCallbacks::new(&self.callbacks, &self.refused).take_fatal() {
            Some(failure) => {
                let code = failure.status.code();
                Err(self.engine.poison(EngineError::Fatal {
                    code,
                    message: described(failure),
                }))
            }
            None => answer,
        }
    }
}

/// Releases the native handle with `rgpot_engine_destroy`.
///
/// In `engine_c_abi.h`, a negative return code means the instance is unusable
/// for further evaluations, not that it has freed itself (`rgpot_engine_last_error`
/// still reads from `pot` afterwards, and every handle created by
/// `rgpot_engine_create` is freed by `rgpot_engine_destroy`). `Drop` therefore
/// still calls `rgpot_engine_destroy` once on a poisoned handle, serialized by
/// `&mut self`. The library stays resident afterwards.
impl Drop for EnginePotential {
    fn drop(&mut self) {
        let _serial = match &self.process_lock {
            Some(lock) => match lock.lock() {
                Ok(guard) => Some(guard),
                Err(poisoned) => Some(poisoned.into_inner()),
            },
            None => None,
        };
        unsafe { (self.symbols.destroy)(self.handle.as_ptr()) };
    }
}

fn process_serial_lock(key: usize) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<Mutex<HashMap<usize, Arc<Mutex<()>>>>> = OnceLock::new();
    let table = LOCKS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut map = match table.lock() {
        Ok(map) => map,
        Err(poisoned) => poisoned.into_inner(),
    };
    Arc::clone(map.entry(key).or_insert_with(|| Arc::new(Mutex::new(()))))
}

/// The library-wide lock of an engine whose `rgpot_engine_caps` reports
/// `reentrancy == 2`, keyed by the library's force entry point. An engine
/// that exports no caps is treated as reentrant. An engine that exports caps
/// is serialized unless the call succeeds and reports `reentrancy` 0 or 1,
/// so a positive code or a truncated answer never skips the lock. A negative
/// code means the instance is unusable and is returned as an error; the
/// caller must destroy the handle and not use it.
fn probe_process_serial(
    symbols: &EngineSymbols,
    handle: NonNull<RgpotEnginePot>,
) -> Result<Option<Arc<Mutex<()>>>, EngineError> {
    let Some(caps_fn) = symbols.caps else {
        return Ok(None);
    };
    let mut raw = RgpotEngineCaps {
        size: std::mem::size_of::<RgpotEngineCaps>() as c_uint,
        ..Default::default()
    };
    let code = unsafe { caps_fn(handle.as_ptr(), &mut raw) };
    if code < 0 {
        return Err(unusable_instance(
            code,
            "rgpot_engine_caps",
            engine_message(symbols, handle, code),
        ));
    }
    let reports_reentrant = code == 0
        && (raw.size as usize) >= REENTRANCY_END
        && REENTRANCY_PARALLEL.contains(&raw.reentrancy);
    if reports_reentrant {
        Ok(None)
    } else {
        Ok(Some(process_serial_lock(symbols.force as usize)))
    }
}

fn check_lengths(n_atoms: usize, n_positions: usize) -> Result<(), EngineError> {
    if n_atoms == 0 {
        return Err(EngineError::Invalid(
            "a geometry needs at least one atom".into(),
        ));
    }
    if n_positions != 3 * n_atoms {
        return Err(EngineError::Invalid(format!(
            "{n_positions} coordinates for {n_atoms} atoms; expected {}",
            3 * n_atoms
        )));
    }
    Ok(())
}

fn unsupported_symbol(operation: &'static str) -> EngineError {
    EngineError::Unsupported {
        operation,
        message: "symbol is not exported".into(),
    }
}

fn non_finite_output(energy: f64, forces: &[f64]) -> Option<String> {
    if !energy.is_finite() {
        return Some("energy is not finite".into());
    }
    forces
        .iter()
        .position(|value| !value.is_finite())
        .map(|index| format!("force at element {index} is not finite"))
}

fn finite(energy: f64, forces: &[f64]) -> Result<(), EngineError> {
    match non_finite_output(energy, forces) {
        Some(message) => Err(EngineError::Invalid(format!("the engine {message}"))),
        None => Ok(()),
    }
}

fn finite_product(product: &[f64]) -> Result<(), EngineError> {
    match product.iter().position(|value| !value.is_finite()) {
        Some(index) => Err(EngineError::Invalid(format!(
            "the engine Hessian-vector product at element {index} is not finite"
        ))),
        None => Ok(()),
    }
}

fn text_of(buffer: &[c_char]) -> String {
    let bytes: Vec<u8> = buffer
        .iter()
        .map(|value| *value as u8)
        .take_while(|value| *value != 0)
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

#[cfg(not(windows))]
fn cache_path(path: &Path) -> Result<CString, EngineError> {
    CString::new(path.as_os_str().as_encoded_bytes())
        .map_err(|_| EngineError::Invalid("cache path contains a null byte".into()))
}

#[cfg(windows)]
fn cache_path(path: &Path) -> Result<CString, EngineError> {
    let text = path.to_str().ok_or_else(|| {
        EngineError::Invalid("cache path must be valid Unicode on Windows".into())
    })?;
    CString::new(text).map_err(|_| EngineError::Invalid("cache path contains a null byte".into()))
}

unsafe fn optional<T: Copy>(library: &Library, name: &str) -> Option<T> {
    const { assert!(std::mem::size_of::<T>() == std::mem::size_of::<*const c_void>()) };
    let symbol = unsafe { library.get::<*const c_void>(format!("{name}\0").as_bytes()) }.ok()?;
    let raw = *symbol;
    if raw.is_null() {
        None
    } else {
        Some(unsafe { std::mem::transmute_copy::<*const c_void, T>(&raw) })
    }
}

unsafe fn resolve(library: &Library) -> Result<EngineSymbols, EngineError> {
    unsafe fn required<T: Copy>(library: &Library, name: &str) -> Result<T, EngineError> {
        const { assert!(std::mem::size_of::<T>() == std::mem::size_of::<*const c_void>()) };
        let symbol = unsafe { library.get::<*const c_void>(format!("{name}\0").as_bytes()) }
            .map_err(|_| EngineError::Missing(name.to_owned()))?;
        let raw = *symbol;
        if raw.is_null() {
            return Err(EngineError::Missing(name.to_owned()));
        }
        Ok(unsafe { std::mem::transmute_copy::<*const c_void, T>(&raw) })
    }
    Ok(EngineSymbols {
        abi_version: unsafe { required(library, "rgpot_engine_abi_version")? },
        available: unsafe { optional(library, "rgpot_engine_available") },
        abi_stamp: unsafe { optional(library, "rgpot_engine_abi_stamp") },
        create: unsafe { required(library, "rgpot_engine_create")? },
        destroy: unsafe { required(library, "rgpot_engine_destroy")? },
        force: unsafe { required(library, "rgpot_engine_force")? },
        force_batch: unsafe { optional(library, "rgpot_engine_force_batch") },
        hvp: unsafe { optional(library, "rgpot_engine_hvp") },
        taylor_jet: unsafe { optional(library, "rgpot_engine_taylor_jet") },
        caps: unsafe { optional(library, "rgpot_engine_caps") },
        last_error: unsafe { optional(library, "rgpot_engine_last_error") },
        last_components: unsafe { optional(library, "rgpot_engine_last_components") },
        component_name: unsafe { optional(library, "rgpot_engine_component_name") },
        model_state: unsafe { optional(library, "rgpot_engine_model_state") },
        set_callbacks: unsafe { optional(library, "rgpot_engine_set_callbacks") },
        hint: unsafe { optional(library, "rgpot_engine_hint") },
    })
}

unsafe extern "C" fn drop_engine(user_data: *mut c_void) {
    drop(unsafe { Box::from_raw(user_data.cast::<Mutex<EnginePotential>>()) });
}

fn lock_engine(user_data: *mut c_void) -> MutexGuard<'static, EnginePotential> {
    match unsafe { &*user_data.cast::<Mutex<EnginePotential>>() }.lock() {
        Ok(engine) => engine,
        Err(poisoned) => poisoned.into_inner(),
    }
}

unsafe extern "C" fn engine_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() || input.is_null() || output.is_null() {
            set_last_error("engine potential: null user data, input or output");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let mut engine = lock_engine(user_data);
        if let Err(why) = engine.check_alive() {
            set_last_error(&why.to_string());
            return why.status();
        }
        let input = unsafe { &*input };
        let Some(n_atoms) = (unsafe { input.n_atoms() }) else {
            set_last_error("engine potential: positions tensor has no shape");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let positions = copy_f64_tensor(input.positions, "positions", 3 * n_atoms, true);
        let numbers = copy_i32_tensor(input.atomic_numbers, "atomic_numbers", n_atoms);
        let cell = copy_f64_tensor(input.box_matrix, "box_matrix", 9, false);
        let (positions, numbers, cell) = match (positions, numbers, cell) {
            (Ok(positions), Ok(numbers), Ok(cell)) => (positions, numbers, cell),
            (Err(why), _, _) | (_, Err(why), _) | (_, _, Err(why)) => {
                set_last_error(&why);
                return rgpot_status_t::RGPOT_INVALID_PARAMETER;
            }
        };
        let mut matrix = [0.0; 9];
        matrix.copy_from_slice(&cell);
        let request = EngineRequest {
            positions: &positions,
            atomic_numbers: &numbers,
            box_matrix: &matrix,
        };
        match engine.force(&request) {
            Ok(answer) => {
                let forces = unsafe {
                    rgpot_tensor_owned_cpu_f64_2d(answer.forces.as_ptr(), n_atoms as i64, 3)
                };
                if forces.is_null() {
                    return rgpot_status_t::RGPOT_INTERNAL_ERROR;
                }
                unsafe {
                    (*output).forces = forces;
                    (*output).energy = answer.energy;
                    (*output).variance = 0.0;
                }
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(why) => {
                set_last_error(&why.to_string());
                why.status()
            }
        }
    }))
}

unsafe extern "C" fn engine_hessian_vector_callback(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    direction: *const f64,
    hv_out: *mut f64,
) -> rgpot_status_t {
    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() || input.is_null() || direction.is_null() || hv_out.is_null() {
            set_last_error("engine HVP: null user data, input, direction or output");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let mut engine = lock_engine(user_data);
        if let Err(why) = engine.check_alive() {
            set_last_error(&why.to_string());
            return why.status();
        }
        let input = unsafe { &*input };
        let Some(n_atoms) = (unsafe { input.n_atoms() }) else {
            set_last_error("engine HVP: positions tensor has no shape");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let Some(dim) = n_atoms.checked_mul(3) else {
            set_last_error("engine HVP: atom count overflows coordinate length");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let bytes = dim.saturating_mul(std::mem::size_of::<f64>());
        let (d_addr, o_addr) = (direction as usize, hv_out as usize);
        // The public HVP entry point rejects this first. Keep this check for
        // direct engine callbacks, which are also part of the C ABI surface.
        if dim > 0 && d_addr < o_addr.saturating_add(bytes) && o_addr < d_addr.saturating_add(bytes)
        {
            set_last_error("engine HVP: direction and hv_out overlap");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let positions = copy_f64_tensor(input.positions, "positions", dim, true);
        let numbers = copy_i32_tensor(input.atomic_numbers, "atomic_numbers", n_atoms);
        let cell = copy_f64_tensor(input.box_matrix, "box_matrix", 9, false);
        let (positions, numbers, cell) = match (positions, numbers, cell) {
            (Ok(positions), Ok(numbers), Ok(cell)) => (positions, numbers, cell),
            (Err(why), _, _) | (_, Err(why), _) | (_, _, Err(why)) => {
                set_last_error(&why);
                return rgpot_status_t::RGPOT_INVALID_PARAMETER;
            }
        };
        let mut matrix = [0.0; 9];
        matrix.copy_from_slice(&cell);
        let request = EngineRequest {
            positions: &positions,
            atomic_numbers: &numbers,
            box_matrix: &matrix,
        };
        let direction = unsafe { std::slice::from_raw_parts(direction, dim) };
        match engine.hvp(&request, direction) {
            Ok(product) => {
                unsafe { std::ptr::copy_nonoverlapping(product.as_ptr(), hv_out, product.len()) };
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(why) => {
                set_last_error(&why.to_string());
                why.status()
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    struct CountingTransform {
        calls: AtomicUsize,
    }

    impl CoordinateTransform for CountingTransform {
        fn transform(
            &mut self,
            _positions: &mut [f64],
            _box_matrix: &mut [f64; 9],
        ) -> Result<(), EngineFailure> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            std::thread::yield_now();
            Ok(())
        }
    }

    #[test]
    fn null_context_trampolines_are_rejected_or_noops() {
        assert_eq!(
            unsafe {
                transform_trampoline(
                    std::ptr::null_mut(),
                    1,
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            },
            ENGINE_ERR_INVALID
        );
        assert_eq!(
            unsafe {
                oracle_trampoline(
                    std::ptr::null_mut(),
                    1,
                    std::ptr::null(),
                    std::ptr::null(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                    std::ptr::null(),
                )
            },
            ENGINE_ERR_INVALID
        );
        unsafe {
            acquire_trampoline(std::ptr::null_mut(), 1);
            model_update_trampoline(std::ptr::null_mut(), 1);
        }
    }

    #[test]
    fn transform_trampoline_serializes_concurrent_batch_invocations() {
        const THREADS: usize = 8;
        let mut transform = CountingTransform {
            calls: AtomicUsize::new(0),
        };
        let call = TransformCall {
            state: Mutex::new(TransformState {
                transform: &mut transform,
                failure: None,
            }),
        };
        let user = (&call as *const TransformCall<'_>).addr();
        std::thread::scope(|scope| {
            for _ in 0..THREADS {
                scope.spawn(move || {
                    let mut positions = [0.0; 3];
                    let mut box_matrix = [0.0; 9];
                    let code = unsafe {
                        transform_trampoline(
                            user as *mut c_void,
                            1,
                            positions.as_mut_ptr(),
                            box_matrix.as_mut_ptr(),
                        )
                    };
                    assert_eq!(code, 0);
                });
            }
        });
        assert!(call.take_failure().is_none());
        drop(call);
        assert_eq!(transform.calls.load(Ordering::SeqCst), THREADS);
    }

    /// A load that panics after it claimed a file must not leave the loading
    /// entry behind, or every later load of the file waits for it forever.
    #[cfg(unix)]
    #[test]
    fn a_claim_dropped_by_a_panic_wakes_the_threads_waiting_on_the_file() {
        let file = std::env::temp_dir().join(format!("rgpot-claim-{}.so", std::process::id()));
        std::fs::write(&file, b"not a shared object").expect("write the stand-in plugin");
        let canonical = canonical_path(&file).expect("canonical path");
        let identity = file_identity(&canonical).expect("file identity");
        {
            let mut registry = registry();
            registry.by_file.insert(
                identity.clone(),
                RegistryEntry::Loading {
                    owner: std::thread::current().id(),
                },
            );
            registry.by_path.insert(canonical.clone(), identity.clone());
        }
        let claim = LoadClaim {
            canonical: canonical.clone(),
            identity: identity.clone(),
            published: false,
        };

        let (sender, receiver) = std::sync::mpsc::channel();
        let waiter_path = file.clone();
        let waiter = std::thread::spawn(move || {
            let result = unsafe { load_resident(&waiter_path, &[]) }.map(drop);
            sender.send(result).expect("report the waiter's result");
        });
        assert!(
            receiver.recv_timeout(Duration::from_millis(300)).is_err(),
            "a load of a file that is still loading must wait"
        );

        let unwound = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            let _claim = claim;
            panic!("the plugin initializer panicked");
        }));
        assert!(unwound.is_err());

        let result = receiver
            .recv_timeout(Duration::from_secs(10))
            .expect("the waiting thread was not woken by the unwound claim");
        assert!(matches!(result, Err(EngineError::Load(_))), "{result:?}");
        waiter.join().expect("join the waiting thread");
        {
            let registry = registry();
            assert!(!registry.by_file.contains_key(&identity));
            assert!(!registry.by_path.contains_key(&canonical));
        }
        std::fs::remove_file(&file).expect("remove the stand-in plugin");
    }

    static SERIAL_STUB_FORCE_CALLS: AtomicUsize = AtomicUsize::new(0);

    unsafe extern "C" fn serial_stub_abi_version() -> c_int {
        ENGINE_ABI_VERSION
    }

    unsafe extern "C" fn serial_stub_create(
        _config: *const c_void,
        _config_len: usize,
        _errbuf: *mut c_char,
        _errlen: usize,
    ) -> *mut RgpotEnginePot {
        NonNull::<RgpotEnginePot>::dangling().as_ptr()
    }

    unsafe extern "C" fn serial_stub_destroy(_pot: *mut RgpotEnginePot) {}

    #[allow(clippy::too_many_arguments)]
    unsafe extern "C" fn serial_stub_force(
        _pot: *mut RgpotEnginePot,
        n_atoms: c_long,
        _positions: *const f64,
        _atomic_numbers: *const c_int,
        forces: *mut f64,
        energy: *mut f64,
        _variance: *mut f64,
        _box: *const f64,
        _transform: Option<CoordTransformFn>,
        _user: *mut c_void,
    ) -> c_int {
        SERIAL_STUB_FORCE_CALLS.fetch_add(1, Ordering::SeqCst);
        unsafe {
            *energy = 0.0;
            std::slice::from_raw_parts_mut(forces, 3 * n_atoms as usize).fill(0.0);
        }
        0
    }

    unsafe extern "C" fn serial_stub_caps(
        _pot: *const RgpotEnginePot,
        out: *mut RgpotEngineCaps,
    ) -> c_int {
        unsafe {
            (*out).reentrancy = 2;
        }
        0
    }

    #[test]
    fn a_poisoned_process_lock_latches_the_handle_unusable() {
        let symbols = EngineSymbols::new(
            serial_stub_abi_version,
            serial_stub_create,
            serial_stub_destroy,
            serial_stub_force,
        )
        .with_caps(serial_stub_caps);
        let lock = process_serial_lock(symbols.force as usize);
        let mut engine =
            unsafe { EnginePotential::from_symbols(symbols, &[]) }.expect("the stub engine loads");
        let request = EngineRequest {
            positions: &[0.0; 3],
            atomic_numbers: &[1],
            box_matrix: &[0.0; 9],
        };

        engine.force(&request).expect("an unpoisoned lock serves");
        assert_eq!(SERIAL_STUB_FORCE_CALLS.load(Ordering::SeqCst), 1);

        let holder = Arc::clone(&lock);
        let panicked = std::thread::spawn(move || {
            let _guard = holder.lock().expect("take the process lock");
            panic!("a thread died inside the engine library");
        })
        .join();
        assert!(panicked.is_err());
        assert!(lock.is_poisoned());

        let first = engine.force(&request).unwrap_err();
        assert!(
            matches!(
                &first,
                EngineError::Lock {
                    operation: "rgpot_engine_force",
                    ..
                }
            ),
            "{first}"
        );
        assert_eq!(first.engine_status(), EngineStatus::fatal(ENGINE_ERR_LOCK));

        // Once the lock itself is healthy again the handle must stay latched,
        // and every operation reports the first failure instead of retrying.
        lock.clear_poison();
        assert_eq!(engine.force(&request).unwrap_err(), first);
        assert_eq!(engine.caps().unwrap_err(), first);
        assert_eq!(
            engine.force_batch(&[1], &[0.0; 3], &[0.0; 9]).unwrap_err(),
            first
        );
        assert_eq!(
            SERIAL_STUB_FORCE_CALLS.load(Ordering::SeqCst),
            1,
            "a latched handle must not re-enter native evaluation"
        );
    }
}
