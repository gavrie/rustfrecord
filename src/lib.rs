use std::collections::HashMap;

use pyo3::exceptions::{PyOSError, PyValueError};
use pyo3::prelude::*;

use numpy::{pyarray_bound, IntoPyArray};

#[pyclass]
struct Reader {
    filename: String,
    inner: tfrecord_reader::Reader,
}

#[pymethods]
impl Reader {
    #[new]
    fn new(filename: &str, compressed: bool, features: Option<Vec<String>>) -> PyResult<Self> {
        let features = features.unwrap_or_default();

        let compression = match compressed {
            true => tfrecord_reader::Compression::Gzip,
            false => tfrecord_reader::Compression::None,
        };

        tfrecord_reader::Reader::new(filename, compression, &features)
            .map(|r| Reader {
                filename: filename.to_owned(),
                inner: r,
            })
            .map_err(|e| PyOSError::new_err(format!("{filename}: {e:?}")))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> PyResult<Option<HashMap<String, PyObject>>> {
        let py = slf.py();

        let wrap_pytensor = |hm: HashMap<_, _>| {
            hm.into_iter()
                .map(|(k, v)| {
                    use tfrecord_reader::Array;

                    let value = match v {
                        Array::Bytes(mut v) => v.pop().unwrap().into_pyarray_bound(py).into_py(py),
                        Array::F32(v) => v.into_pyarray_bound(py).into_py(py),
                        Array::I64(v) => v.into_pyarray_bound(py).into_py(py),
                        Array::None => pyarray_bound![py, [0]].into_py(py),
                    };

                    (k, value)
                })
                .collect()
        };

        slf.inner
            .next()
            .map(|r| {
                r.map(wrap_pytensor).map_err(|e| {
                    use tfrecord_reader::Error;
                    let filename = &slf.filename;
                    match e {
                        Error::IoError(_) | Error::UnexpectedEof => {
                            PyOSError::new_err(format!("{filename}: {e:?}"))
                        }
                        _ => PyValueError::new_err(format!("{filename}: {e:?}")),
                    }
                })
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
