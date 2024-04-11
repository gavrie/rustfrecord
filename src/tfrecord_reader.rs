use std::{fs, io::Read, path::Path};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use tfrecord::{ExampleIter, RecordReaderConfig};

pub fn tfrecord_reader(filename: &str, compressed: bool) -> Result<()> {
    let path = Path::new(filename);

    let conf = RecordReaderConfig {
        check_integrity: false,
    };

    let example_iter = fs::File::open(path)
        .with_context(|| format!("failed to open {path:?}"))
        .map(|r| -> Box<dyn Read> {
            if compressed {
                Box::new(GzDecoder::new(r))
            } else {
                Box::new(r)
            }
        })
        .map(|r| ExampleIter::from_reader(r, conf))?;

    // Comment from example.proto:
    //
    // An Example is a mostly-normalized data format for storing data for training and inference.
    // It contains a key-value store (features); where each key (string) maps to a Feature message
    // (which is one of packed BytesList, FloatList, or Int64List).
    //
    for example in example_iter {
        let hm = example?.into_hash_map();
        
        let hm = hm.into_iter().map(|(name, feature)| {
            println!("Feature: {name}: {feature:?}");
        }).collect();

        // break; // FIXME
        return Ok(hm);
    }

    Ok(())
}
