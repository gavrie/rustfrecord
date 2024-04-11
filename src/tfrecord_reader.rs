use std::{collections::HashMap, fs, io::Read, path::Path};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use tch::Tensor;
use tfrecord::{ExampleIter, Feature, FeatureKind, RecordReaderConfig};

pub fn tfrecord_reader(filename: &str, compressed: bool) -> Result<HashMap<String, Tensor>> {
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
        let hm = example?
            .into_iter()
            .map(|(name, feature)| {
                // println!("Feature: {name}"); //: {feature:?}");
                // let tensor = Tensor::new();
                let tensor = match feature.into_kinds() {
                    Some(FeatureKind::F32(value)) => Tensor::from_slice(&value),
                    Some(FeatureKind::I64(value)) => Tensor::from_slice(&value),
                    Some(FeatureKind::Bytes(value)) => Tensor::from_slice2(&value),
                    None => Tensor::new(),
                };
                (name, tensor)
            })
            .collect();

        return Ok(hm); // FIXME
    }

    unreachable!()
}
