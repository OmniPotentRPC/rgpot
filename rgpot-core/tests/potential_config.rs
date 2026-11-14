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
fn morse_is_union_ordinal_six() {
    use capnp::introspect::{Introspect, TypeVariant};
    use capnp::schema_capnp::field;

    let TypeVariant::Struct(raw) = potential_config::Owned::introspect().which() else {
        panic!("PotentialConfig is not a struct");
    };
    let schema = capnp::schema::StructSchema::new(raw);
    let morse = schema.get_field_by_name("morse").expect("morse field");
    let proto = morse.get_proto();
    match proto.get_ordinal().which().expect("ordinal") {
        field::ordinal::Explicit(ordinal) => assert_eq!(ordinal, 6),
        field::ordinal::Implicit(()) => panic!("morse ordinal must be explicit"),
    }
}
