//! Pair candidates from linkcell and minimum-image vectors from minimage.
//!
//! The C entry points are what the classical pots call. A failure is a
//! negative code plus a thread-local message. There is no second search
//! behind these functions.

use std::cell::RefCell;
use std::ffi::CString;
use std::os::raw::c_char;

use linkcell::{pairs_within, wrap_many, Cell};

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
}

fn fail(msg: &str) -> i32 {
    let text = msg.replace('\0', " ");
    let owned = CString::new(text).unwrap_or_else(|_| CString::new("fastnlist error").unwrap());
    LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(owned));
    -1
}

/// Thread-local message for the most recent negative return.
///
/// # Safety
///
/// The pointer is valid until the next fastnlist call on this thread.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_last_error() -> *const c_char {
    LAST_ERROR.with(|slot| match slot.borrow().as_ref() {
        Some(msg) => msg.as_ptr(),
        None => std::ptr::null(),
    })
}

fn triples<'a>(ptr: *const f64, n: usize) -> Result<&'a [[f64; 3]], i32> {
    if n == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(fail("null coordinate buffer"));
    }
    Ok(unsafe { std::slice::from_raw_parts(ptr as *const [f64; 3], n) })
}

fn cell_rows(cell9: *const f64) -> Result<[[f64; 3]; 3], i32> {
    if cell9.is_null() {
        return Err(fail("null cell"));
    }
    let s = unsafe { std::slice::from_raw_parts(cell9, 9) };
    Ok([
        [s[0], s[1], s[2]],
        [s[3], s[4], s[5]],
        [s[6], s[7], s[8]],
    ])
}

fn periodic3(periodic: *const i32) -> Result<[bool; 3], i32> {
    if periodic.is_null() {
        return Err(fail("null periodic flags"));
    }
    let s = unsafe { std::slice::from_raw_parts(periodic, 3) };
    Ok([s[0] != 0, s[1] != 0, s[2] != 0])
}

fn orthorhombic(cell: &[[f64; 3]; 3]) -> bool {
    cell[0][1] == 0.0
        && cell[0][2] == 0.0
        && cell[1][0] == 0.0
        && cell[1][2] == 0.0
        && cell[2][0] == 0.0
        && cell[2][1] == 0.0
}

/// Candidate pairs. See `fastnlist.h`.
///
/// # Safety
///
/// `xyz` is `n * 3` doubles. `cell9` is 9 doubles. `periodic` is 3 ints.
/// `out_count` is writable. On a fill, `out_ij` holds `2 * cap` ints.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_pairs(
    xyz: *const f64,
    n: usize,
    cell9: *const f64,
    periodic: *const i32,
    list_cutoff: f64,
    out_ij: *mut i32,
    cap: usize,
    out_count: *mut usize,
) -> i32 {
    if out_count.is_null() {
        return fail("null out_count");
    }
    let points = match triples(xyz, n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let cell = match cell_rows(cell9) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let periodic = match periodic3(periodic) {
        Ok(v) => v,
        Err(code) => return code,
    };
    match collect_pairs(points, &cell, periodic, list_cutoff) {
        Ok(pairs) => {
            unsafe { *out_count = pairs.len() };
            if pairs.len() > cap {
                return 1;
            }
            if pairs.len() > 0 && out_ij.is_null() {
                return fail("null out_ij");
            }
            for (k, (i, j)) in pairs.iter().enumerate() {
                unsafe {
                    *out_ij.add(2 * k) = *i;
                    *out_ij.add(2 * k + 1) = *j;
                }
            }
            clear_error();
            0
        }
        Err(code) => code,
    }
}

/// True when a linked-cell walk visits fewer images than one pass over
/// every unordered pair. A periodic axis narrower than three cutoffs has
/// one or two bins and a fat stencil. An open axis whose points all lie
/// inside the cutoff has nothing to reject.
fn linkcell_helps(
    widths: &[f64; 3],
    periodic: &[bool; 3],
    mins: &[f64; 3],
    maxs: &[f64; 3],
    list_cutoff: f64,
) -> bool {
    for k in 0..3 {
        if periodic[k] {
            if widths[k] < 3.0 * list_cutoff {
                return false;
            }
        } else if maxs[k] - mins[k] <= list_cutoff {
            return false;
        }
    }
    true
}

fn collect_pairs(
    xyz: &[[f64; 3]],
    cell: &[[f64; 3]; 3],
    periodic: [bool; 3],
    list_cutoff: f64,
) -> Result<Vec<(i32, i32)>, i32> {
    if xyz.len() < 2 || !(list_cutoff > 0.0) || !list_cutoff.is_finite() {
        return Ok(Vec::new());
    }
    if !orthorhombic(cell) {
        return Err(fail("pair list requires an orthorhombic cell"));
    }
    if xyz.len() > i32::MAX as usize {
        return Err(fail("atom count does not fit in i32"));
    }
    let mut widths = [cell[0][0], cell[1][1], cell[2][2]];
    let mut shift = [0.0; 3];
    let mut any_open = false;
    let mut mins = [f64::INFINITY; 3];
    let mut maxs = [f64::NEG_INFINITY; 3];
    for p in xyz {
        for k in 0..3 {
            mins[k] = mins[k].min(p[k]);
            maxs[k] = maxs[k].max(p[k]);
        }
    }
    for k in 0..3 {
        if periodic[k] {
            if !(widths[k] > 0.0) {
                return Err(fail("periodic axis has a non-positive width"));
            }
        } else {
            any_open = true;
            let extent = maxs[k] - mins[k];
            widths[k] = extent + 2.0 * list_cutoff + 1.0;
            if !(widths[k] > 0.0) || !widths[k].is_finite() {
                return Err(fail("open axis padding is not finite"));
            }
            shift[k] = list_cutoff + 0.5 - mins[k];
        }
    }
    let owned;
    let pts: &[[f64; 3]] = if any_open {
        let mut moved = xyz.to_vec();
        for p in &mut moved {
            for k in 0..3 {
                if !periodic[k] {
                    p[k] += shift[k];
                }
            }
        }
        owned = moved;
        &owned
    } else {
        xyz
    };
    // One bin, or a cutoff that reaches the neighbouring image, makes the
    // linked-cell stencil revisit every pair. Emitting i < j and letting
    // the caller fold with minimage is the shorter walk, at any atom
    // count. The 10k-atom bridge cell is narrower than three cutoffs;
    // asking linkcell for that stencil allocates tens of gigabytes.
    // Wider cells still go through linkcell.
    if !linkcell_helps(&widths, &periodic, &mins, &maxs, list_cutoff) {
        let mut pairs = Vec::with_capacity(xyz.len() * (xyz.len() - 1) / 2);
        for i in 0..xyz.len() {
            for j in (i + 1)..xyz.len() {
                pairs.push((i as i32, j as i32));
            }
        }
        return Ok(pairs);
    }
    let sim = match Cell::ortho(widths[0], widths[1], widths[2]) {
        Ok(c) => c,
        Err(err) => return Err(fail(&err.to_string())),
    };
    let rows = match pairs_within(pts, &sim, list_cutoff, None, None, true) {
        Ok(rows) => rows,
        Err(err) => return Err(fail(&err.to_string())),
    };
    let mut pairs = Vec::with_capacity(rows.len());
    for row in rows {
        if row.i == row.j {
            continue;
        }
        let (i, j) = if row.i < row.j {
            (row.i, row.j)
        } else {
            (row.j, row.i)
        };
        pairs.push((i as i32, j as i32));
    }
    pairs.sort_unstable();
    pairs.dedup();
    Ok(pairs)
}

pub struct SkinList {
    cutoff: f64,
    skin: f64,
    periodic: [bool; 3],
    cell: [[f64; 3]; 3],
    reference: Vec<[f64; 3]>,
    pairs: Vec<(i32, i32)>,
    ready: bool,
}

impl SkinList {
    fn new() -> Self {
        Self {
            cutoff: 0.0,
            skin: 0.0,
            periodic: [false; 3],
            cell: [[0.0; 3]; 3],
            reference: Vec::new(),
            pairs: Vec::new(),
            ready: false,
        }
    }

    fn reusable(
        &self,
        xyz: &[[f64; 3]],
        cell: &[[f64; 3]; 3],
        periodic: [bool; 3],
        cutoff: f64,
        skin: f64,
    ) -> bool {
        if !self.ready || !(skin > 0.0) || self.skin != skin || self.cutoff != cutoff {
            return false;
        }
        if self.periodic != periodic || self.cell != *cell || self.reference.len() != xyz.len() {
            return false;
        }
        // Strict `>`. A displacement of exactly half the skin still reuses.
        let limit2 = 0.25 * skin * skin;
        for (prev, next) in self.reference.iter().zip(xyz.iter()) {
            let dx = next[0] - prev[0];
            let dy = next[1] - prev[1];
            let dz = next[2] - prev[2];
            if dx * dx + dy * dy + dz * dz > limit2 {
                return false;
            }
        }
        true
    }

    fn pairs(
        &mut self,
        xyz: &[[f64; 3]],
        cell: &[[f64; 3]; 3],
        periodic: [bool; 3],
        cutoff: f64,
        skin: f64,
    ) -> Result<&[(i32, i32)], i32> {
        if !(cutoff > 0.0) || !cutoff.is_finite() || !(skin >= 0.0) || !skin.is_finite() {
            return Err(fail("cutoff and skin must be finite and the cutoff positive"));
        }
        if self.reusable(xyz, cell, periodic, cutoff, skin) {
            return Ok(&self.pairs);
        }
        let reach = cutoff + skin;
        let pairs = collect_pairs(xyz, cell, periodic, reach)?;
        self.cutoff = cutoff;
        self.skin = skin;
        self.periodic = periodic;
        self.cell = *cell;
        self.reference = xyz.to_vec();
        self.pairs = pairs;
        self.ready = true;
        Ok(&self.pairs)
    }
}

/// Allocate an empty Verlet shell.
#[no_mangle]
pub extern "C" fn rgpot_fast_skin_new() -> *mut SkinList {
    Box::into_raw(Box::new(SkinList::new()))
}

/// # Safety
///
/// `list` is null or a pointer returned by `rgpot_fast_skin_new` and not
/// yet freed.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_skin_free(list: *mut SkinList) {
    if !list.is_null() {
        drop(unsafe { Box::from_raw(list) });
    }
}

/// Verlet pair shell. See `fastnlist.h`.
///
/// # Safety
///
/// `list` came from `rgpot_fast_skin_new`. The buffers follow
/// `rgpot_fast_pairs`: `xyz` is `n * 3` doubles, `cell9` is 9 doubles,
/// `periodic` is 3 ints, and a fill writes `2 * count` ints.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_skin_pairs(
    list: *mut SkinList,
    xyz: *const f64,
    n: usize,
    cell9: *const f64,
    periodic: *const i32,
    cutoff: f64,
    skin: f64,
    out_ij: *mut i32,
    cap: usize,
    out_count: *mut usize,
) -> i32 {
    if list.is_null() {
        return fail("null skin list");
    }
    if out_count.is_null() {
        return fail("null out_count");
    }
    let points = match triples(xyz, n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let cell = match cell_rows(cell9) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let periodic = match periodic3(periodic) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let list = unsafe { &mut *list };
    match list.pairs(points, &cell, periodic, cutoff, skin) {
        Ok(pairs) => {
            unsafe { *out_count = pairs.len() };
            if pairs.len() > cap {
                return 1;
            }
            if !pairs.is_empty() && out_ij.is_null() {
                return fail("null out_ij");
            }
            for (k, (i, j)) in pairs.iter().enumerate() {
                unsafe {
                    *out_ij.add(2 * k) = *i;
                    *out_ij.add(2 * k + 1) = *j;
                }
            }
            clear_error();
            0
        }
        Err(code) => code,
    }
}

fn wrap_into(cell: &Cell, diffs: &[[f64; 3]], out: &mut [[f64; 3]]) -> Result<(), i32> {
    wrap_many(cell, diffs, out).map_err(|err| fail(&err.to_string()))
}

/// Full-cell minimum-image wrap. See `fastnlist.h`.
///
/// # Safety
///
/// `cell9` is 9 doubles. `diffs` and `out` are `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_wrap(
    cell9: *const f64,
    diffs: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    let cell = match cell_rows(cell9) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let diffs = match triples(diffs, n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    if n > 0 && out.is_null() {
        return fail("null wrap output");
    }
    let sim = match Cell::from_vesin(cell) {
        Ok(c) => c,
        Err(err) => return fail(&err.to_string()),
    };
    let out = if n == 0 {
        &mut [][..]
    } else {
        unsafe { std::slice::from_raw_parts_mut(out as *mut [f64; 3], n) }
    };
    match wrap_into(&sim, diffs, out) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(code) => code,
    }
}

/// Orthorhombic wrap that skips open axes. See `fastnlist.h`.
///
/// # Safety
///
/// `cell9` is 9 doubles. `periodic` is 3 ints. `diffs` and `out` are
/// `n * 3` doubles.
#[no_mangle]
pub unsafe extern "C" fn rgpot_fast_wrap_axes(
    cell9: *const f64,
    periodic: *const i32,
    diffs: *const f64,
    n: usize,
    out: *mut f64,
) -> i32 {
    let cell = match cell_rows(cell9) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let periodic = match periodic3(periodic) {
        Ok(v) => v,
        Err(code) => return code,
    };
    let diffs = match triples(diffs, n) {
        Ok(v) => v,
        Err(code) => return code,
    };
    if n > 0 && out.is_null() {
        return fail("null wrap output");
    }
    if !orthorhombic(&cell) {
        return fail("axis wrap requires an orthorhombic cell");
    }
    // An open axis gets a width no difference in this library can cross,
    // so the orthorhombic wrap leaves that component alone.
    let mut widths = [cell[0][0], cell[1][1], cell[2][2]];
    for k in 0..3 {
        if periodic[k] {
            if !(widths[k] > 0.0) {
                return fail("periodic axis has a non-positive width");
            }
        } else {
            widths[k] = 1.0e16;
        }
    }
    let sim = match Cell::ortho(widths[0], widths[1], widths[2]) {
        Ok(c) => c,
        Err(err) => return fail(&err.to_string()),
    };
    let out = if n == 0 {
        &mut [][..]
    } else {
        unsafe { std::slice::from_raw_parts_mut(out as *mut [f64; 3], n) }
    };
    match wrap_into(&sim, diffs, out) {
        Ok(()) => {
            clear_error();
            0
        }
        Err(code) => code,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_away(t: f64) -> f64 {
        (t + 0.5_f64.copysign(t)) as i64 as f64
    }

    fn scalar_pairs(xyz: &[[f64; 3]], box_w: [f64; 3], periodic: [bool; 3], cutoff: f64) -> Vec<(i32, i32)> {
        let cut2 = cutoff * cutoff;
        let mut out = Vec::new();
        for i in 0..xyz.len() {
            for j in (i + 1)..xyz.len() {
                let mut d = [
                    xyz[i][0] - xyz[j][0],
                    xyz[i][1] - xyz[j][1],
                    xyz[i][2] - xyz[j][2],
                ];
                for k in 0..3 {
                    if periodic[k] {
                        let inv = 1.0 / box_w[k];
                        d[k] -= box_w[k] * round_away(d[k] * inv);
                    }
                }
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 <= cut2 {
                    out.push((i as i32, j as i32));
                }
            }
        }
        out
    }

    #[test]
    fn periodic_slab_matches_the_scalar_fold() {
        let mut xyz = Vec::new();
        for i in 0..12 {
            let t = i as f64;
            xyz.push([
                (t * 1.7).rem_euclid(18.0),
                (t * 2.3).rem_euclid(18.0),
                3.0 + (t * 0.4).sin(),
            ]);
        }
        // One pair sits across the x boundary.
        xyz.push([0.2, 9.0, 4.0]);
        xyz.push([17.7, 9.1, 4.2]);
        let cell = [[18.0, 0.0, 0.0], [0.0, 18.0, 0.0], [0.0, 0.0, 30.0]];
        let periodic = [true, true, true];
        let cutoff = 6.0;
        let got = collect_pairs(&xyz, &cell, periodic, cutoff + 1.0).unwrap();
        let expect = scalar_pairs(&xyz, [18.0, 18.0, 30.0], periodic, cutoff + 1.0);
        // The box is only a couple of cutoffs wide, so the candidate
        // buffer is every unordered pair. The scalar set is the in-cutoff
        // subset the caller keeps.
        assert!(expect.iter().all(|pair| got.contains(pair)));
        assert_eq!(got.len(), xyz.len() * (xyz.len() - 1) / 2);
    }

    #[test]
    fn narrow_cell_above_the_old_cap_is_every_pair() {
        // Just past the old 4096 guard. A 10 A box with cutoff 15 A is the
        // bridge stress cell; linkcell must not see it.
        let n = 4100;
        let mut xyz = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f64;
            xyz.push([
                (t * 0.37).rem_euclid(10.0),
                (t * 0.53).rem_euclid(10.0),
                (t * 0.71).rem_euclid(10.0),
            ]);
        }
        let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let got = collect_pairs(&xyz, &cell, [true, true, true], 15.0).unwrap();
        assert_eq!(got.len(), n * (n - 1) / 2);
        assert_eq!(got.first().copied(), Some((0, 1)));
        assert_eq!(got.last().copied(), Some((n as i32 - 2, n as i32 - 1)));
    }

    #[test]
    fn wide_cell_partial_periodicity_matches_all_pairs() {
        let mut xyz = Vec::new();
        for i in 0..16 {
            let t = i as f64;
            xyz.push([
                (t * 2.5).rem_euclid(48.0),
                (t * 1.7).rem_euclid(36.0),
                (t * 0.9).rem_euclid(40.0),
            ]);
        }
        // On the faces, including a pair that meets across x.
        xyz.push([0.0, 10.0, 12.0]);
        xyz.push([47.8, 10.2, 12.1]);
        xyz.push([20.0, 0.0, 8.0]);
        xyz.push([20.4, 35.9, 8.2]);
        let cell = [[48.0, 0.0, 0.0], [0.0, 36.0, 0.0], [0.0, 0.0, 40.0]];
        let widths = [48.0, 36.0, 40.0];
        // Cutoff well under a third of every side, so the linked-cell
        // stencil is the one that runs. One and two axes are periodic.
        let masks = [
            [true, false, false],
            [false, true, false],
            [false, false, true],
            [true, true, false],
            [true, false, true],
            [false, true, true],
        ];
        let cutoff = 6.0;
        for periodic in masks {
            let got = collect_pairs(&xyz, &cell, periodic, cutoff).unwrap();
            let expect = scalar_pairs(&xyz, widths, periodic, cutoff);
            assert_eq!(got, expect, "{periodic:?}");
            assert!(got.len() < xyz.len() * (xyz.len() - 1) / 2);
        }
    }

    #[test]
    fn cutoff_near_half_a_side_matches_all_pairs() {
        // Half of 10 is 5. A cutoff of 4.9 is too wide for the linked-cell
        // stencil, so the list is the all-pairs buffer filtered by the
        // caller. Faces are included.
        let xyz = vec![
            [0.0, 1.0, 2.0],
            [9.8, 1.1, 2.2],
            [5.0, 0.0, 4.0],
            [5.2, 9.9, 4.1],
            [2.0, 3.0, 0.0],
            [2.1, 3.2, 9.7],
            [4.0, 6.0, 5.0],
        ];
        let cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let widths = [10.0, 10.0, 10.0];
        let cutoff = 4.9;
        for periodic in [
            [true, false, false],
            [true, true, false],
            [true, true, true],
        ] {
            let got = collect_pairs(&xyz, &cell, periodic, cutoff).unwrap();
            let expect = scalar_pairs(&xyz, widths, periodic, cutoff);
            assert!(expect.iter().all(|pair| got.contains(pair)), "{periodic:?}");
        }
    }

    #[test]
    fn wide_cell_returns_only_in_cutoff_pairs() {
        let mut xyz = Vec::new();
        for i in 0..20 {
            let t = i as f64;
            xyz.push([t * 1.6, (t * 0.7).sin() * 2.0 + 10.0, 5.0]);
        }
        let cell = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]];
        let periodic = [true, true, true];
        let cutoff = 6.0;
        let got = collect_pairs(&xyz, &cell, periodic, cutoff).unwrap();
        let expect = scalar_pairs(&xyz, [40.0, 40.0, 40.0], periodic, cutoff);
        assert_eq!(got, expect);
        assert!(got.len() < xyz.len() * (xyz.len() - 1) / 2);
    }

    #[test]
    fn open_cluster_matches_raw_distances() {
        let xyz = vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 8.0],
        ];
        let cell = [[1.0e6, 0.0, 0.0], [0.0, 1.0e6, 0.0], [0.0, 0.0, 1.0e6]];
        let got = collect_pairs(&xyz, &cell, [false, false, false], 3.5).unwrap();
        let expect = scalar_pairs(&xyz, [1.0e6, 1.0e6, 1.0e6], [false, false, false], 3.5);
        assert_eq!(expect, vec![(0, 1), (0, 2), (1, 2)]);
        assert!(expect.iter().all(|pair| got.contains(pair)));
    }

    #[test]
    fn skin_reuses_inside_half_and_rebuilds_outside() {
        let mut xyz = Vec::new();
        for i in 0..24 {
            let t = i as f64;
            xyz.push([
                (t * 1.3).rem_euclid(40.0),
                (t * 2.1).rem_euclid(40.0),
                (t * 0.7).rem_euclid(40.0),
            ]);
        }
        xyz.push([0.2, 10.0, 10.0]);
        xyz.push([39.6, 10.2, 10.1]);
        let cell = [[40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 0.0, 40.0]];
        let periodic = [true, true, true];
        let cutoff = 5.0;
        let skin = 1.0;
        let mut list = SkinList::new();
        let first = list
            .pairs(&xyz, &cell, periodic, cutoff, skin)
            .unwrap()
            .to_vec();
        let shell = collect_pairs(&xyz, &cell, periodic, cutoff + skin).unwrap();
        assert_eq!(first, shell);
        let mut inside = xyz.clone();
        for p in &mut inside {
            p[0] += 0.4;
        }
        let reused = list
            .pairs(&inside, &cell, periodic, cutoff, skin)
            .unwrap()
            .to_vec();
        assert_eq!(reused, first);
        let cut2 = cutoff * cutoff;
        for i in 0..inside.len() {
            for j in (i + 1)..inside.len() {
                let mut d = [
                    inside[i][0] - inside[j][0],
                    inside[i][1] - inside[j][1],
                    inside[i][2] - inside[j][2],
                ];
                for k in 0..3 {
                    let inv = 1.0 / 40.0;
                    d[k] -= 40.0 * round_away(d[k] * inv);
                }
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 <= cut2 {
                    assert!(reused.contains(&(i as i32, j as i32)));
                }
            }
        }
        let mut outside = xyz.clone();
        outside[0][0] += 0.6;
        let rebuilt = list
            .pairs(&outside, &cell, periodic, cutoff, skin)
            .unwrap()
            .to_vec();
        let fresh = collect_pairs(&outside, &cell, periodic, cutoff + skin).unwrap();
        assert_eq!(rebuilt, fresh);
    }

    #[test]
    fn zero_skin_rebuilds_when_coordinates_change() {
        let cell = [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]];
        let xyz = vec![[1.0, 2.0, 3.0], [4.0, 2.0, 3.0], [20.0, 10.0, 8.0]];
        let mut nudged = xyz.clone();
        nudged[0][0] += 1.0e-6;
        let mut list = SkinList::new();
        let periodic = [true, true, true];
        let first = list.pairs(&xyz, &cell, periodic, 6.0, 0.0).unwrap().to_vec();
        assert_eq!(first, collect_pairs(&xyz, &cell, periodic, 6.0).unwrap());
        let second = list
            .pairs(&nudged, &cell, periodic, 6.0, 0.0)
            .unwrap()
            .to_vec();
        assert_eq!(
            second,
            collect_pairs(&nudged, &cell, periodic, 6.0).unwrap()
        );
    }

    #[test]
    fn wrap_matches_round_away_off_the_tie() {
        let cell = [[10.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 14.0]];
        let diffs = [[9.2, -0.4, 3.0], [-6.0, 7.0, -8.0], [0.0, 0.0, 0.0]];
        let mut out = [[0.0; 3]; 3];
        let sim = Cell::from_vesin(cell).unwrap();
        wrap_many(&sim, &diffs, &mut out).unwrap();
        for (d, w) in diffs.iter().zip(out.iter()) {
            let widths = [10.0, 12.0, 14.0];
            for k in 0..3 {
                let folded = d[k] - widths[k] * round_away(d[k] / widths[k]);
                assert!((w[k] - folded).abs() < 1e-12, "{} vs {}", w[k], folded);
            }
        }
    }
}
