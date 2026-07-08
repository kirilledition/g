#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PreflightReportPayload {
    pub sample_count: i64,
    pub covariate_count: i64,
    pub chromosome_count: i64,
    pub warning_messages: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SingleTraitPreflightShapePayload {
    pub sample_count: i64,
    pub covariate_count: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiTraitPreflightShapePayload {
    pub trait_count: i64,
    pub sample_count: i64,
    pub covariate_count: i64,
}
