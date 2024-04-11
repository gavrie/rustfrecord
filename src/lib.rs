use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3_tch::{wrap_tch_err, PyTensor};

mod pyo3_tch;
mod tfrecord_reader;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

#[pyfunction]
fn add_one(tensor: PyTensor) -> PyResult<PyTensor> {
    let tensor = tensor.f_add_scalar(1.0).map_err(wrap_tch_err)?;
    Ok(PyTensor(tensor))
}

#[pyfunction]
fn tfrecord_open(filename: &str, compressed: bool) -> PyResult<()> {
    tfrecord_reader::tfrecord_reader(filename, compressed)
        .map_err(|e| PyErr::new::<PyValueError, _>(format!("{e:?}")))?;
    Ok(())
}

/// A Python module implemented in Rust.
#[pymodule]
fn rustfrecord(py: Python, m: &PyModule) -> PyResult<()> {
    py.import("torch")?;
    m.add_function(wrap_pyfunction!(add_one, m)?)?;
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(tfrecord_open, m)?)?;
    Ok(())
}
