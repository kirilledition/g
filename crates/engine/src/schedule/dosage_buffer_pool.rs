use std::collections::BTreeSet;

use super::normalize_callback_queue_wait_timeout_seconds;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageBufferReusePlan {
    pub requires_slice: bool,
    pub slice_dimensions: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DosageBufferAcquireAttemptPlan {
    pub should_take_free_buffer: bool,
    pub should_allocate: bool,
    pub should_wait: bool,
    pub wait_timeout_seconds: f64,
    pub free_buffer_count: usize,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageBufferRegisterAttemptPlan {
    pub should_register: bool,
    pub has_registration_error: bool,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageBufferReturnAttemptPlan {
    pub should_return: bool,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DosageBufferDiscardAttemptPlan {
    pub should_discard: bool,
    pub allocated_count: usize,
    pub buffer_limit: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageBufferPoolObservationPlan {
    pub operation_name: String,
    pub blocked: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DosageBufferPoolState {
    buffer_limit: usize,
    buffer_identifiers: BTreeSet<usize>,
}

impl DosageBufferPoolState {
    #[must_use]
    pub fn new(buffer_limit: usize) -> Self {
        Self { buffer_limit, buffer_identifiers: BTreeSet::new() }
    }

    #[must_use]
    pub const fn buffer_limit(&self) -> usize {
        self.buffer_limit
    }

    #[must_use]
    pub fn allocated_count(&self) -> usize {
        self.buffer_identifiers.len()
    }

    #[must_use]
    pub fn buffer_identifiers(&self) -> Vec<usize> {
        self.buffer_identifiers.iter().copied().collect()
    }

    #[must_use]
    pub fn has_available_slot(&self) -> bool {
        self.allocated_count() < self.buffer_limit
    }

    #[must_use]
    pub fn owns_buffer(&self, buffer_identifier: usize) -> bool {
        self.buffer_identifiers.contains(&buffer_identifier)
    }

    pub fn register_buffer(&mut self, buffer_identifier: usize) -> bool {
        if !self.has_available_slot() || self.owns_buffer(buffer_identifier) {
            return false;
        }
        self.buffer_identifiers.insert(buffer_identifier)
    }

    pub fn discard_buffer(&mut self, buffer_identifier: usize) -> bool {
        self.buffer_identifiers.remove(&buffer_identifier)
    }
}

pub(super) fn plan_dosage_buffer_acquire_attempt(
    buffer_pool_state: &DosageBufferPoolState,
    free_buffer_count: usize,
    wait_timeout_seconds: f64,
) -> DosageBufferAcquireAttemptPlan {
    let should_take_free_buffer = free_buffer_count > 0;
    let should_allocate = !should_take_free_buffer && buffer_pool_state.has_available_slot();
    let normalized_wait_timeout_seconds = if should_take_free_buffer || should_allocate {
        0.0
    } else {
        normalize_callback_queue_wait_timeout_seconds(wait_timeout_seconds)
    };
    DosageBufferAcquireAttemptPlan {
        should_take_free_buffer,
        should_allocate,
        should_wait: !should_take_free_buffer && !should_allocate && normalized_wait_timeout_seconds > 0.0,
        wait_timeout_seconds: normalized_wait_timeout_seconds,
        free_buffer_count,
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

pub(super) fn plan_dosage_buffer_register_attempt(
    buffer_pool_state: &mut DosageBufferPoolState,
    buffer_identifier: usize,
) -> DosageBufferRegisterAttemptPlan {
    let registered_buffer = buffer_pool_state.register_buffer(buffer_identifier);
    DosageBufferRegisterAttemptPlan {
        should_register: registered_buffer,
        has_registration_error: !registered_buffer,
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

pub(super) fn plan_dosage_buffer_return_attempt(
    buffer_pool_state: &DosageBufferPoolState,
    buffer_identifier: usize,
) -> DosageBufferReturnAttemptPlan {
    DosageBufferReturnAttemptPlan {
        should_return: buffer_pool_state.owns_buffer(buffer_identifier),
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

pub(super) fn plan_dosage_buffer_discard_attempt(
    buffer_pool_state: &mut DosageBufferPoolState,
    buffer_identifier: usize,
) -> DosageBufferDiscardAttemptPlan {
    let discarded_buffer = buffer_pool_state.discard_buffer(buffer_identifier);
    DosageBufferDiscardAttemptPlan {
        should_discard: discarded_buffer,
        allocated_count: buffer_pool_state.allocated_count(),
        buffer_limit: buffer_pool_state.buffer_limit(),
    }
}

#[must_use]
pub fn plan_dosage_buffer_pool_observation(operation_name: &str, blocked: bool) -> DosageBufferPoolObservationPlan {
    DosageBufferPoolObservationPlan { operation_name: operation_name.to_string(), blocked }
}

#[must_use]
pub fn plan_dosage_buffer_reuse(buffered_shape: &[usize], expected_shape: &[usize]) -> Option<DosageBufferReusePlan> {
    if buffered_shape.len() != expected_shape.len() {
        return None;
    }
    if buffered_shape
        .iter()
        .zip(expected_shape)
        .any(|(buffered_dimension, expected_dimension)| buffered_dimension < expected_dimension)
    {
        return None;
    }
    Some(DosageBufferReusePlan {
        requires_slice: buffered_shape != expected_shape,
        slice_dimensions: expected_shape.to_vec(),
    })
}
