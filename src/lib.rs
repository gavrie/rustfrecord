use std::collections::HashMap;

use pyo3::exceptions::{PyOSError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};
use tch::Tensor;

mod pyo3_tch;
mod tfrecord_reader;

#[pyclass]
struct Reader {
    inner: tfrecord_reader::Reader,
}

#[pyfunction]
fn new(filename: &str, compressed: bool) -> PyResult<Reader> {
    tfrecord_reader::Reader::new(filename, compressed)
        .map(|r| Reader { inner: r })
        .map_err(|e| PyErr::new::<PyOSError, _>(format!("{e:?}")))
}

// fn next(filename: &str, compressed: bool) -> PyResult<HashMap<String, PyTensor>> {
// .map(|hm| hm.into_iter().map(|(k, v)| (k, PyTensor(v))).collect())

#[pymodule]
fn rustfrecord(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.py().import_bound("torch")?;
    m.add_function(wrap_pyfunction!(new, m)?)?;
    Ok(())
}
