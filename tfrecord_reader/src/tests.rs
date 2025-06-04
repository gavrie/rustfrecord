use anyhow::{anyhow, Result};

use super::*;

#[test]
fn test_read() -> Result<()> {
    let reader = Reader::new(
        "../tf_example/sample_images.tfrecord",
        Compression::None,
        &["label", "image/encoded", "image/shape"],
    )?;

    for record in reader {
        let record = record?;

        // First, check if the record contains the expected features
        let Array::Bytes(ref label) = record["label"] else {
            return Err(anyhow!("Expected 'label' to be of type Bytes"));
        };

        let Array::I64(ref shape) = record["image/shape"] else {
            return Err(anyhow!("Expected 'image/shape' to be of type I64"));
        };

        let Array::Bytes(ref encoded) = record["image/encoded"] else {
            return Err(anyhow!("Expected 'image/encoded' to be of type Bytes"));
        };

        // Then, extract the values from the features
        let [label] = label.as_slice() else {
            return Err(anyhow!("Expected 'label' to have exactly 1 element"));
        };
        let label = str::from_utf8(label.as_slice())?;

        let [width, height, depth] = shape[..] else {
            return Err(anyhow!("Expected 'image/shape' to have exactly 3 elements"));
        };

        let [encoded] = encoded.as_slice() else {
            return Err(anyhow!("Expected 'image/encoded' to have exactly 1 element"));
        };

        // Finally, print the values
        eprintln!("Label: {label}");
        eprintln!("Shape: {width}x{height}x{depth}");
        eprintln!("Encoded: {} bytes", encoded.len());
        eprintln!();
    }

    Ok(())
}
