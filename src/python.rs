//! Python bindings for Taguchi.
//!
//! This module exposes the core functionality of the library to Python
//! using PyO3. Enable the `python` feature to use this.

use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::oa::OA;
use crate::OABuilder;

/// Python wrapper for OAParams
#[pyclass(name = "OAParams")]
#[derive(Clone)]
pub struct PyOAParams {
    /// Number of runs
    #[pyo3(get)]
    pub runs: usize,
    /// Number of factors
    #[pyo3(get)]
    pub factors: usize,
    /// Strength of the array
    #[pyo3(get)]
    pub strength: u32,
}

/// Python wrapper for OA
#[pyclass(name = "OA")]
pub struct PyOA {
    inner: OA,
}

#[pymethods]
impl PyOA {
    /// Get the number of runs.
    #[getter]
    fn runs(&self) -> usize {
        self.inner.runs()
    }

    /// Get the number of factors.
    #[getter]
    fn factors(&self) -> usize {
        self.inner.factors()
    }

    /// Get the strength.
    #[getter]
    fn strength(&self) -> u32 {
        self.inner.strength()
    }

    /// Get the data as a list of lists.
    fn data(&self, py: Python<'_>) -> PyResult<PyObject> {
        let data = self.inner.data();
        let rows = data.nrows();
        let cols = data.ncols();

        let list = PyList::empty(py);
        for i in 0..rows {
            let row_list = PyList::empty(py);
            for j in 0..cols {
                row_list.append(data[[i, j]])?;
            }
            list.append(row_list)?;
        }
        Ok(list.into())
    }

    /// Check balance.
    fn is_balanced(&self) -> bool {
        let report = self.inner.balance_report();
        report.factor_balance.iter().all(|&b| b)
    }
}

/// Construct an orthogonal array.
#[pyfunction]
#[pyo3(signature = (levels, factors, strength=2))]
fn construct(levels: u32, factors: usize, strength: u32) -> PyResult<PyOA> {
    let oa = OABuilder::new()
        .levels(levels)
        .factors(factors)
        .strength(strength)
        .build()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyOA { inner: oa })
}

/// Construct a mixed-level orthogonal array.
#[pyfunction]
#[pyo3(signature = (levels, strength=2))]
fn construct_mixed(levels: Vec<u32>, strength: u32) -> PyResult<PyOA> {
    let oa = OABuilder::new()
        .mixed_levels(levels)
        .strength(strength)
        .build()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;

    Ok(PyOA { inner: oa })
}

/// The Taguchi Python module.
#[pymodule]
fn taguchi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyOAParams>()?;
    m.add_class::<PyOA>()?;
    m.add_function(wrap_pyfunction!(construct, m)?)?;
    m.add_function(wrap_pyfunction!(construct_mixed, m)?)?;
    Ok(())
}
