use serde::Serialize;

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct QueueBackpressureKey {
    pub queue_name: String,
    pub operation_name: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct QueueBackpressureAccumulator {
    pub observation_count: i64,
    pub max_depth: i64,
    pub max_capacity: i64,
    pub total_elapsed_seconds: f64,
    pub total_blocked_seconds: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct QueueBackpressureSnapshot {
    pub queue_name: String,
    pub operation_name: String,
    pub observation_count: i64,
    pub max_depth: i64,
    pub max_capacity: i64,
    pub total_elapsed_seconds: f64,
    pub total_blocked_seconds: f64,
}
