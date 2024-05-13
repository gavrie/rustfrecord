use std::{
    collections::{HashMap, HashSet},
    fs,
    io::Read,
    path::Path,
};

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use tfrecord::{Example, ExampleIter, FeatureKind, RecordReaderConfig};
pub use tfrecord::Error;

#[cfg(test)]
mod tests;

pub enum Compression {
    None,
    Gzip,
}

pub enum Array {
    Bytes(Vec<Vec<u8>>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    None,
}

pub struct Reader {
    example_iter: ExampleIter<Box<dyn Read + Send>>,
    features: HashSet<String>,
}

impl Reader {
    pub fn new(
        filename: &str,
        compression: Compression,
        features: &[impl AsRef<str>],
    ) -> Result<Self> {
        let path = Path::new(filename);

        let conf = RecordReaderConfig {
            check_integrity: false,
        };

        let file = fs::File::open(path).with_context(|| format!("failed to open {path:?}"))?;

        let reader: Box<dyn Read + Send> = match compression {
            Compression::Gzip => Box::new(GzDecoder::new(file)),
            Compression::None => Box::new(file),
        };

        let example_iter = ExampleIter::from_reader(reader, conf);
        let features = features.iter().map(|s| s.as_ref().to_string()).collect();

        Ok(Self {
            example_iter,
            features,
        })
    }
}

impl Iterator for Reader {
    // Iterate over Examples.
    //
    // Comment from example.proto:
    //
    // An Example is a mostly-normalized data format for storing data for training and inference.
    // It contains a key-value store (features); where each key (string) maps to a Feature message
    // (which is one of packed BytesList, FloatList, or Int64List).

    type Item = tfrecord::Result<HashMap<String, Array>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.example_iter
            .next()
            .map(|e| e.map(|e| example_to_hashmap(e, &self.features)))
    }
}

fn example_to_hashmap(example: Example, features: &HashSet<String>) -> HashMap<String, Array> {
    example
        .into_iter()
        .filter(|(name, _)| features.is_empty() || features.contains(name))
        .map(|(name, feature)| {
            let array = match feature.into_kinds() {
                Some(FeatureKind::F32(value)) => Array::F32(value),
                Some(FeatureKind::I64(value)) => Array::I64(value),
                Some(FeatureKind::Bytes(value)) => Array::Bytes(value),
                None => Array::None,
            };
            (name, array)
        })
        .collect()
}
