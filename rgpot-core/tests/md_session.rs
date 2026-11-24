// MIT License
// Copyright 2023--present rgpot developers

//! `CAbiMdSession::open_engine` against the C plugin in
//! `tests/fixtures/md_session_engine.c`.

#![cfg(unix)]

use std::path::{Path, PathBuf};

use rgpot_core::md_session::{
    CAbiMdSession, MdNlistBuffers, MdNlistPairs, MdNlistRequest, MdSession,
};

fn build_fixture() -> PathBuf {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/md_session_engine.c");
    let include = Path::new(env!("CARGO_MANIFEST_DIR")).join("../CppCore");
    let suffix = if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!("libmd_session_engine.{suffix}"));
    let target = env!("RGPOT_BUILD_TARGET");
    let compiler = match cc::Build::new()
        .target(target)
        .host(target)
        .opt_level(0)
        .try_get_compiler()
    {
        Ok(compiler) => compiler,
        Err(why) => panic!(
            "cannot compile the MD session fixture: no C compiler available for target {target} ({why})"
        ),
    };
    let mut command = compiler.to_command();
    if cfg!(target_os = "macos") {
        command.arg("-dynamiclib");
    } else {
        command.args(["-shared", "-fPIC"]);
    }
    command
        .arg("-I")
        .arg(include)
        .arg("-o")
        .arg(&out)
        .arg(&source)
        .arg("-lm");
    let status = match command.status() {
        Ok(status) => status,
        Err(why) => panic!("spawn C compiler {:?}: {why}", compiler.path()),
    };
    assert!(
        status.success(),
        "the MD session fixture did not compile: {status}"
    );
    out
}

fn open(path: &Path, mode: &str) -> Result<CAbiMdSession, String> {
    unsafe { CAbiMdSession::open_engine(path, &[], Some(mode)) }.map_err(|err| err.to_string())
}

fn refusal(path: &Path, mode: &str) -> String {
    match open(path, mode) {
        Ok(_) => panic!("the {mode} fixture opened"),
        Err(why) => why,
    }
}

/// The cases share the fixture's destroy counter, so they run in one test.
#[test]
fn open_engine_validates_the_vtable_and_releases_refused_sessions() {
    let path = build_fixture();
    let library = unsafe { libloading::Library::new(&path) }.expect("load the fixture");
    let destroyed = unsafe {
        library
            .get::<unsafe extern "C" fn() -> i32>(b"md_fixture_destroyed\0")
            .expect("md_fixture_destroyed")
    };
    let count = || unsafe { destroyed() };

    let mut session = open(&path, "ok").expect("a complete vtable opens");
    assert_eq!(session.dof(), 6);
    assert_eq!(session.masses().unwrap(), vec![1.0, 1.0]);
    session
        .set_positions(&[0.0, 0.0, 0.0, 3.0, 4.0, 0.0])
        .unwrap();
    session
        .set_velocities(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        .unwrap();
    session.step_nve(0.5, 2).unwrap();
    let state = session.state().unwrap();
    assert_eq!(state.positions, vec![1.0, 0.0, 0.0, 3.0, 4.0, 0.0]);
    assert_eq!(state.kinetic_energy, 0.5);

    let request = MdNlistRequest {
        want_distances: 1,
        ..MdNlistRequest::new()
    };
    let (mut first, mut second, mut distance) = ([-1; 2], [-1; 2], [0.0; 2]);
    let buffers = MdNlistBuffers::new(&mut first, &mut second).with_distance(&mut distance);
    assert_eq!(
        session.get_neighbors(&request, Some(buffers)).unwrap(),
        (1, 1)
    );
    assert_eq!((first[0], second[0]), (0, 1));
    assert_eq!(distance[0], (2.0f64 * 2.0 + 4.0 * 4.0).sqrt());
    assert!(session
        .set_neighbors(&MdNlistRequest::new(), &MdNlistPairs::new(&[0], &[1]))
        .unwrap_err()
        .to_string()
        .contains("does not accept supplied neighbors"));
    assert!(session
        .step_langevin(
            &Default::default(),
            &[0.0; 6],
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[],
            &[],
            1
        )
        .unwrap_err()
        .to_string()
        .contains("no Langevin dynamics"));
    let before = count();
    drop(session);
    assert_eq!(count(), before + 1, "dropping the session calls destroy");

    let before = count();
    let short = refusal(&path, "short_size");
    assert_eq!(short, "MD session vtable struct_size is too small");
    assert_eq!(
        count(),
        before,
        "a destroy outside struct_size is not called"
    );

    let before = count();
    let null = refusal(&path, "null_callback");
    assert_eq!(null, "MD session operation table has a null get_state");
    assert_eq!(count(), before + 1, "a refused vtable releases its context");

    let before = count();
    let empty = refusal(&path, "zero_atoms");
    assert_eq!(empty, "MD session requires at least one atom");
    assert_eq!(
        count(),
        before + 1,
        "a refused session releases its context"
    );

    let before = count();
    let refused = refusal(&path, "refuse");
    assert_eq!(refused, "fixture refused the session");
    assert_eq!(count(), before, "a failed create returns no context");
}
