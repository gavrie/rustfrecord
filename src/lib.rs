use std::collections::HashMap;

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};
use tch::Tensor;

mod pyo3_tch;
mod tfrecord_reader;

#[pyfunction]
fn tfrecord_open(filename: &str, compressed: bool) -> PyResult<HashMap<String, PyTensor>> {
    tfrecord_reader::tfrecord_reader(filename, compressed)
        .map(|hm| hm.into_iter().map(|(k, v)| (k, PyTensor(v))).collect())
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("{e:?}")))
}

#[pymodule]
fn rustfrecord(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.py().import_bound("torch")?;
    m.add_function(wrap_pyfunction!(tfrecord_open, m)?)?;
    Ok(())
}
