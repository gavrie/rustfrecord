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

#[pymethods]
impl Reader {
    #[new]
    fn new(filename: &str, compressed: bool) -> PyResult<Self> {
        tfrecord_reader::Reader::new(filename, compressed)
            .map(|r| Reader { inner: r })
            .map_err(|e| PyOSError::new_err(format!("{e:?}")))
    }

    // TODO: Implement Python iterator protocol (or maybe do that in a Python wrapper?)
    fn next(&mut self) -> PyResult<Option<HashMap<String, PyTensor>>> {
        let wrap_pytensor =
            |hm: HashMap<_, _>| hm.into_iter().map(|(k, v)| (k, PyTensor(v))).collect();

        self.inner
            .next()
            .map(|r| {
                r.map(wrap_pytensor)
                    .map_err(|e| PyValueError::new_err(format!("{e:?}")))
            })
            .transpose()
    }
}

#[pymodule]
fn rustfrecord(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.py().import_bound("torch")?;
    m.add_class::<Reader>()?;
    Ok(())
}
