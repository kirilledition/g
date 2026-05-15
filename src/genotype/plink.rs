//! PLINK reader integration points.

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PlinkBedAdapterPlan {
    pub bed_reader_crate_blocked: bool,
    pub reason: String,
}

pub fn bed_reader_dependency_status() -> PlinkBedAdapterPlan {
    PlinkBedAdapterPlan {
        bed_reader_crate_blocked: true,
        reason: "bed-reader v1.0.6 depends on numpy/pyo3 0.22 and currently conflicts with this extension's pyo3 0.28 dependency graph.".to_string(),
    }
}
