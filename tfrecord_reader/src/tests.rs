use super::*;

#[test]
fn test_read() {
    let reader = Reader::new(
        // "data/002scattered.training_examples.tfrecord",
        "/data/Runs/training_example/002scattered.training_examples.tfrecord",
        Compression::None,
        &["label", "image/encoded", "image/shape"],
    );
    assert!(reader.is_ok());

    for r in reader.unwrap() {
        let r = r.unwrap();
        eprintln!("{r:?}");
        break;
    }
}
