use super::*;

#[test]
fn test_read() {
    let reader = Reader::new(
        "data/002scattered.training_examples.tfrecord",
        Compression::None,
        &["label", "image/encoded", "image/shape"],
    );
    assert!(reader.is_ok());
}
