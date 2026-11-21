use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use rgpot_core::Potentials_capnp::potential_config;

#[test]
fn morse_arm_round_trips() {
    let mut message = Builder::new_default();
    {
        let config = message.init_root::<potential_config::Builder>();
        let mut morse = config.init_morse();
        morse.set_de(1.25);
        morse.set_a(1.5);
        morse.set_re(2.75);
        morse.set_cutoff(8.0);
    }
    let encoded = serialize::write_message_to_words(&message);

    let mut bytes = encoded.as_slice();
    let reader = serialize::read_message_from_flat_slice(&mut bytes, ReaderOptions::new())
        .expect("read PotentialConfig");
    let config = reader
        .get_root::<potential_config::Reader>()
        .expect("PotentialConfig root");
    let morse = match config.which().expect("known PotentialConfig arm") {
        potential_config::Which::Morse(morse) => morse.expect("morse arm"),
        _ => panic!("PotentialConfig decoded to another arm"),
    };
    assert_eq!(morse.get_de(), 1.25);
    assert_eq!(morse.get_a(), 1.5);
    assert_eq!(morse.get_re(), 2.75);
    assert_eq!(morse.get_cutoff(), 8.0);
}

#[test]
fn morse_arm_defaults_are_the_platinum_parameters() {
    let mut message = Builder::new_default();
    message
        .init_root::<potential_config::Builder>()
        .init_morse();
    let config = message
        .get_root_as_reader::<potential_config::Reader>()
        .expect("PotentialConfig root");
    let potential_config::Which::Morse(morse) = config.which().expect("known arm") else {
        panic!("PotentialConfig decoded to another arm");
    };
    let morse = morse.expect("morse arm");
    assert_eq!(morse.get_de(), 0.7102);
    assert_eq!(morse.get_a(), 1.6047);
    assert_eq!(morse.get_re(), 2.897);
    assert_eq!(morse.get_cutoff(), 9.5);
}

#[test]
fn lj_arm_round_trips_tail_correction_and_validates_config() {
    use rgpot_core::lj::{LjConfig, LjConfigError};

    let default_cfg = LjConfig::default();
    assert_eq!(default_cfg.u0, 1.0);
    assert_eq!(default_cfg.psi, 1.0);
    assert_eq!(default_cfg.cutoff, 15.0);
    assert_eq!(default_cfg.switch_width, 0.0);
    assert!(!default_cfg.tail_correction);
    assert_eq!(default_cfg.validate(), Ok(()));

    let cfg = LjConfig::default()
        .with_u0(0.85)
        .with_psi(2.5)
        .with_cutoff(9.0)
        .with_switch_width(1.5)
        .with_tail_correction(true);
    assert_eq!(cfg.validate(), Ok(()));

    let mut message = Builder::new_default();
    cfg.write_potential_config(message.init_root::<potential_config::Builder>())
        .expect("write PotentialConfig.lj");
    let encoded = serialize::write_message_to_words(&message);

    let mut bytes = encoded.as_slice();
    let reader = serialize::read_message_from_flat_slice(&mut bytes, ReaderOptions::new())
        .expect("read PotentialConfig");
    let root = reader
        .get_root::<potential_config::Reader>()
        .expect("PotentialConfig root");
    let decoded = LjConfig::from_potential_config(root).expect("decode LjConfig");
    assert_eq!(decoded.u0, 0.85);
    assert_eq!(decoded.psi, 2.5);
    assert_eq!(decoded.cutoff, 9.0);
    assert!(decoded.tail_correction);

    // Rejects unsupported switch_width == cutoff when tail_correction is enabled.
    let bad_window = LjConfig::default()
        .with_cutoff(5.0)
        .with_switch_width(5.0)
        .with_tail_correction(true);
    assert_eq!(
        bad_window.validate(),
        Err(LjConfigError::SwitchCoversWholeCutoff)
    );
    let mut bad_msg = Builder::new_default();
    assert_eq!(
        bad_window.write_potential_config(bad_msg.init_root::<potential_config::Builder>()),
        Err(LjConfigError::SwitchCoversWholeCutoff)
    );

    // Rejects non-positive cutoff on decode.
    let mut invalid_wire = Builder::new_default();
    {
        let mut lj = invalid_wire
            .init_root::<potential_config::Builder>()
            .init_lj();
        lj.set_cutoff(0.0);
        lj.set_tail_correction(true);
    }
    let root_bad = invalid_wire
        .get_root_as_reader::<potential_config::Reader>()
        .expect("reader");
    assert_eq!(
        LjConfig::from_potential_config(root_bad),
        Err(LjConfigError::InvalidParameters)
    );
}

/// potentials-schema's arm ordinals, shared with gpridge and every other
/// consumer. A renumbered arm would decode as another backend on the wire.
const UPSTREAM_ARMS: [(&str, u16); 17] = [
    ("none", 0),
    ("nwchem", 1),
    ("cpmd", 2),
    ("metatomic", 4),
    ("lammps", 5),
    ("xtb", 6),
    ("tblite", 7),
    ("uma", 8),
    ("lj", 9),
    ("morse", 10),
    ("zbl", 11),
    ("d3", 12),
    ("d4", 13),
    ("mopac", 14),
    ("skala", 15),
    ("expr", 16),
    ("cp2k", 17),
];

/// The arms rgpot adds in CppCore/rgpot/rpc/local, after the upstream ones.
const LOCAL_ARMS: [(&str, u16); 3] = [("composite", 18), ("d2", 19), ("d2cpmd", 20)];

#[test]
fn union_arms_keep_the_upstream_ordinals() {
    use capnp::introspect::{Introspect, TypeVariant};
    use capnp::schema_capnp::field;

    let TypeVariant::Struct(raw) = potential_config::Owned::introspect().which() else {
        panic!("PotentialConfig is not a struct");
    };
    let schema = capnp::schema::StructSchema::new(raw);
    let arms = schema.get_union_fields().expect("PotentialConfig union");
    let mut seen = Vec::new();
    for arm in arms {
        let proto = arm.get_proto();
        let name = proto.get_name().expect("name").to_str().expect("utf-8");
        let ordinal = match proto.get_ordinal().which().expect("ordinal") {
            field::ordinal::Explicit(ordinal) => ordinal,
            field::ordinal::Implicit(()) => panic!("{name} ordinal must be explicit"),
        };
        seen.push((name.to_owned(), ordinal));
    }
    let want: Vec<(String, u16)> = UPSTREAM_ARMS
        .iter()
        .chain(LOCAL_ARMS.iter())
        .map(|(name, ordinal)| ((*name).to_owned(), *ordinal))
        .collect();
    assert_eq!(seen, want);
}
