//! Native BGEN delivery for the coarse JAX association backend.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::sync::Arc;

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use g_engine::AssociationBackend;

use super::backend::PyJaxBackend;
use super::{errors, output};

/// Runtime controls and output state shared by single- and multi-trait delivery.
pub(crate) struct AssociationDeliverySettings {
    pub(crate) writer_sessions: Vec<Arc<output::OutputWriterSession>>,
    pub(crate) committed_chunk_identifier_sets: Vec<BTreeSet<usize>>,
    pub(crate) null_logistic_nonconvergence_policy: String,
    pub(crate) staging_depth: usize,
    pub(crate) result_in_flight_limit: usize,
    pub(crate) output_statistic_dtype: g_plan::FloatingPointDtype,
    pub(crate) sample_indices: Vec<usize>,
    pub(crate) use_packed8: bool,
}

/// Owned inputs for one single-trait delivery.
pub(crate) struct SingleAssociationDeliveryRequest {
    pub(crate) aligned_sample_data: g_input::AlignedSampleData,
    pub(crate) prediction_source: g_input::PredictionSource,
    pub(crate) settings: AssociationDeliverySettings,
}

/// Owned inputs for one multi-trait delivery.
pub(crate) struct MultiAssociationDeliveryRequest {
    pub(crate) aligned_sample_data: g_input::MultiAlignedSampleData,
    pub(crate) prediction_source: g_input::MultiPredictionSource,
    pub(crate) settings: AssociationDeliverySettings,
}

/// Owned inputs for one grouped-union delivery.
pub(crate) struct GroupedUnionAssociationDeliveryRequest {
    pub(crate) groups: Vec<MultiAssociationDeliveryRequest>,
    pub(crate) union_sample_indices: Vec<usize>,
}

/// One native delivery request independent of phenotype cardinality.
pub(crate) enum AssociationDeliveryRequest {
    Single(SingleAssociationDeliveryRequest),
    Multi(MultiAssociationDeliveryRequest),
}

enum AssociationGroupInput {
    Single { aligned_sample_data: g_input::AlignedSampleData, prediction_source: g_input::PredictionSource },
    Multi { aligned_sample_data: g_input::MultiAlignedSampleData, prediction_source: g_input::MultiPredictionSource },
}

impl AssociationGroupInput {
    fn sample_indices(&self) -> &[usize] {
        match self {
            Self::Single { aligned_sample_data, .. } => &aligned_sample_data.sample_indices,
            Self::Multi { aligned_sample_data, .. } => &aligned_sample_data.sample_indices,
        }
    }

    fn phenotype_names(&self) -> &[String] {
        match self {
            Self::Single { aligned_sample_data, .. } => std::slice::from_ref(&aligned_sample_data.phenotype_name),
            Self::Multi { aligned_sample_data, .. } => &aligned_sample_data.phenotype_names,
        }
    }

    fn trait_count(&self) -> usize {
        match self {
            Self::Single { .. } => 1,
            Self::Multi { aligned_sample_data, .. } => aligned_sample_data.phenotype_row_count,
        }
    }

    fn group_preparation_input(&self) -> g_engine::GroupPreparationInput<'_> {
        match self {
            Self::Single { aligned_sample_data, .. } => g_engine::GroupPreparationInput {
                phenotypes: g_engine::TraitMajorPhenotypeMatrixView {
                    values: &aligned_sample_data.phenotype_vector,
                    trait_count: 1,
                    sample_count: aligned_sample_data.sample_indices.len(),
                },
                covariates: g_engine::SampleMajorCovariateMatrixView {
                    values: &aligned_sample_data.covariate_matrix_values,
                    sample_count: aligned_sample_data.covariate_row_count,
                    covariate_count: aligned_sample_data.covariate_column_count,
                },
            },
            Self::Multi { aligned_sample_data, .. } => g_engine::GroupPreparationInput {
                phenotypes: g_engine::TraitMajorPhenotypeMatrixView {
                    values: &aligned_sample_data.phenotype_matrix_values,
                    trait_count: aligned_sample_data.phenotype_row_count,
                    sample_count: aligned_sample_data.phenotype_column_count,
                },
                covariates: g_engine::SampleMajorCovariateMatrixView {
                    values: &aligned_sample_data.covariate_matrix_values,
                    sample_count: aligned_sample_data.covariate_row_count,
                    covariate_count: aligned_sample_data.covariate_column_count,
                },
            },
        }
    }

    fn chromosome_predictions<'input>(&'input self, chromosome: &str) -> PyResult<ChromosomePredictionValues<'input>> {
        match self {
            Self::Single { prediction_source, aligned_sample_data } => {
                let values = prediction_source
                    .chromosome_predictions(chromosome)
                    .map_err(|error| errors::convert_prediction_error("chromosome_predictions", &error))?;
                Ok(ChromosomePredictionValues {
                    values: Cow::Borrowed(values),
                    trait_count: 1,
                    sample_count: aligned_sample_data.sample_indices.len(),
                })
            }
            Self::Multi { prediction_source, .. } => {
                let matrix = prediction_source
                    .chromosome_prediction_matrix(chromosome)
                    .map_err(|error| errors::convert_prediction_error("chromosome_prediction_matrix", &error))?;
                Ok(ChromosomePredictionValues {
                    values: Cow::Owned(matrix.prediction_values),
                    trait_count: matrix.trait_count,
                    sample_count: matrix.sample_count,
                })
            }
        }
    }

    fn uses_scalar_null_convergence(&self) -> bool {
        matches!(self, Self::Single { .. })
    }
}

struct ChromosomePredictionValues<'values> {
    values: Cow<'values, [f32]>,
    trait_count: usize,
    sample_count: usize,
}

impl ChromosomePredictionValues<'_> {
    fn preparation_input(&self) -> g_engine::ChromosomePreparationInput<'_> {
        g_engine::ChromosomePreparationInput {
            predictions: g_engine::TraitMajorPredictionMatrixView {
                values: self.values.as_ref(),
                trait_count: self.trait_count,
                sample_count: self.sample_count,
            },
        }
    }
}

struct DeliveryExecution {
    group_input: AssociationGroupInput,
    settings: AssociationDeliverySettings,
}

impl From<AssociationDeliveryRequest> for DeliveryExecution {
    fn from(request: AssociationDeliveryRequest) -> Self {
        match request {
            AssociationDeliveryRequest::Single(request) => Self {
                group_input: AssociationGroupInput::Single {
                    aligned_sample_data: request.aligned_sample_data,
                    prediction_source: request.prediction_source,
                },
                settings: request.settings,
            },
            AssociationDeliveryRequest::Multi(request) => Self {
                group_input: AssociationGroupInput::Multi {
                    aligned_sample_data: request.aligned_sample_data,
                    prediction_source: request.prediction_source,
                },
                settings: request.settings,
            },
        }
    }
}

struct GroupedUnionGroupRuntime {
    execution: DeliveryExecution,
    group_state: Py<PyAny>,
    sample_positions: Vec<usize>,
    pipeline: Option<g_engine::AssociationBatchPipeline<PyJaxBackend>>,
    buffer_pool: GenotypeBufferPool,
}

#[derive(Default)]
struct GenotypeBufferPool {
    dosage_buffers: Vec<Vec<f32>>,
    packed8_buffers: Vec<Vec<u8>>,
}

impl GenotypeBufferPool {
    fn acquire(&mut self, value_count: usize, use_packed8: bool) -> PyResult<g_engine::OwnedGenotypeBuffer> {
        if use_packed8 {
            let packed_value_count = value_count
                .checked_mul(2)
                .ok_or_else(|| PyValueError::new_err("Packed8 genotype buffer size overflowed usize."))?;
            let values = take_matching_buffer(&mut self.packed8_buffers, packed_value_count)
                .unwrap_or_else(|| vec![0_u8; packed_value_count]);
            return Ok(g_engine::OwnedGenotypeBuffer::Packed8(values));
        }
        let values =
            take_matching_buffer(&mut self.dosage_buffers, value_count).unwrap_or_else(|| vec![0.0_f32; value_count]);
        Ok(g_engine::OwnedGenotypeBuffer::Dosage(values))
    }

    fn release(&mut self, buffer: g_engine::OwnedGenotypeBuffer) {
        match buffer {
            g_engine::OwnedGenotypeBuffer::Dosage(values) => self.dosage_buffers.push(values),
            g_engine::OwnedGenotypeBuffer::Packed8(values) => self.packed8_buffers.push(values),
        }
    }
}

/// Decode, compute, materialize, and write every uncommitted chunk in one group.
///
/// Writer sessions remain open for the caller to finish or abort.
pub(crate) fn run_association_delivery(
    py: Python<'_>,
    engine: &g_engine::Regenie2RunEngineCore,
    backend: Arc<PyJaxBackend>,
    request: AssociationDeliveryRequest,
) -> PyResult<usize> {
    let execution = request.into();
    py.detach(move || run_association_delivery_detached(engine, &backend, &execution))
}

/// Decode each union-sample chunk once and deliver projected dosage batches to every group.
///
/// Writer sessions remain open for the caller to finish or abort.
pub(crate) fn run_grouped_union_association_delivery(
    py: Python<'_>,
    engine: &g_engine::Regenie2RunEngineCore,
    backend: Arc<PyJaxBackend>,
    request: GroupedUnionAssociationDeliveryRequest,
) -> PyResult<usize> {
    py.detach(move || run_grouped_union_association_delivery_detached(engine, &backend, request))
}

fn run_association_delivery_detached(
    engine: &g_engine::Regenie2RunEngineCore,
    backend: &Arc<PyJaxBackend>,
    execution: &DeliveryExecution,
) -> PyResult<usize> {
    validate_execution(execution)?;
    engine
        .reader()
        .prepare_sample_selection(&execution.settings.sample_indices)
        .map_err(|error| errors::convert_bgen_error("prepare_sample_selection", error))?;

    let delivery_result = run_prepared_association_delivery(engine, backend, execution);
    let clear_result = engine
        .reader()
        .clear_prepared_sample_selection()
        .map_err(|error| errors::convert_bgen_error("clear_prepared_sample_selection", error));
    match (delivery_result, clear_result) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
    }
}

fn run_grouped_union_association_delivery_detached(
    engine: &g_engine::Regenie2RunEngineCore,
    backend: &Arc<PyJaxBackend>,
    request: GroupedUnionAssociationDeliveryRequest,
) -> PyResult<usize> {
    validate_grouped_union_request(&request)?;
    engine
        .reader()
        .prepare_sample_selection(&request.union_sample_indices)
        .map_err(|error| errors::convert_bgen_error("prepare_sample_selection", error))?;

    let delivery_result = run_prepared_grouped_union_association_delivery(engine, backend, request);
    let clear_result = engine
        .reader()
        .clear_prepared_sample_selection()
        .map_err(|error| errors::convert_bgen_error("clear_prepared_sample_selection", error));
    match (delivery_result, clear_result) {
        (Err(error), _) | (Ok(_), Err(error)) => Err(error),
        (Ok(processed_chunk_count), Ok(())) => Ok(processed_chunk_count),
    }
}

fn run_prepared_association_delivery(
    engine: &g_engine::Regenie2RunEngineCore,
    backend: &Arc<PyJaxBackend>,
    execution: &DeliveryExecution,
) -> PyResult<usize> {
    let group_state = backend
        .prepare_group(execution.group_input.group_preparation_input())
        .map_err(|error| errors::convert_backend_error("prepare_group", &error))?;
    let committed_chunk_identifiers =
        g_engine::intersect_committed_chunk_identifier_sets(&execution.settings.committed_chunk_identifier_sets);
    let chunk_specs = engine
        .plan_chunks(&committed_chunk_identifiers)
        .map_err(|error| errors::convert_genotype_error("plan_chunks", error))?;
    let mut buffer_pool = GenotypeBufferPool::default();
    let mut current_chromosome = None;
    let mut pipeline: Option<g_engine::AssociationBatchPipeline<PyJaxBackend>> = None;
    let mut processed_chunk_count = 0_usize;

    for chunk_spec in chunk_specs {
        Python::attach(crate::binding::runtime::check_process_signals)?;
        let variant_count = chunk_spec.variant_stop_index.saturating_sub(chunk_spec.variant_start_index);
        let metadata = engine
            .reader()
            .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
            .map_err(|error| errors::convert_bgen_error("variant_metadata_slice", error))?;
        let chromosome = homogeneous_chromosome(&metadata, variant_count)?;
        if current_chromosome.as_deref() != Some(chromosome.as_str()) {
            if let Some(mut previous_pipeline) = pipeline.take() {
                finish_and_drain_pipeline(&mut previous_pipeline, &execution.settings, &mut buffer_pool)?;
            }
            let predictions = execution.group_input.chromosome_predictions(&chromosome)?;
            let prepared_chromosome = backend
                .prepare_chromosome(&group_state, predictions.preparation_input())
                .map_err(|error| errors::convert_backend_error("prepare_chromosome", &error))?;
            enforce_null_logistic_policy(
                &chromosome,
                prepared_chromosome.null_model_diagnostics.as_ref(),
                &execution.group_input,
                &execution.settings.null_logistic_nonconvergence_policy,
            )?;
            pipeline = Some(
                g_engine::AssociationBatchPipeline::new(
                    Arc::clone(backend),
                    prepared_chromosome.state,
                    execution.settings.staging_depth,
                    execution.settings.result_in_flight_limit,
                )
                .map_err(|error| errors::convert_scheduler_error("start", &error))?,
            );
            current_chromosome = Some(chromosome);
        }

        let active_trait_indices = active_trait_indices(
            execution.settings.writer_sessions.len(),
            chunk_spec.variant_start_index,
            &execution.settings.committed_chunk_identifier_sets,
        )?;
        if active_trait_indices.is_empty() {
            return Err(PyRuntimeError::new_err(
                "Native delivery planned a chunk already committed by every output writer.",
            ));
        }
        let sample_count = execution.settings.sample_indices.len();
        let genotype_value_count = variant_count
            .checked_mul(sample_count)
            .ok_or_else(|| PyValueError::new_err("Genotype batch dimensions overflowed usize."))?;
        let mut genotype_buffer = buffer_pool.acquire(genotype_value_count, execution.settings.use_packed8)?;
        let statistics = decode_genotype_buffer(
            engine.reader(),
            chunk_spec.variant_start_index,
            chunk_spec.variant_stop_index,
            &mut genotype_buffer,
        )?;
        let scheduled_batch = g_engine::ScheduledAssociationBatch {
            variant_start_index: chunk_spec.variant_start_index,
            variant_count,
            sample_count,
            metadata,
            statistics,
            genotype_buffer,
            active_trait_indices,
            output_statistic_dtype: execution.settings.output_statistic_dtype,
        };
        let active_pipeline = pipeline
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Native association pipeline was not initialized."))?;
        if let Err(error) = active_pipeline.submit(scheduled_batch) {
            if let Some(active_pipeline) = pipeline.as_mut() {
                let _ = active_pipeline.abort();
            }
            return Err(errors::convert_scheduler_error("submit", &error));
        }
        processed_chunk_count += 1;
        drain_available_batches(active_pipeline, &execution.settings, &mut buffer_pool)?;
    }

    if let Some(mut final_pipeline) = pipeline {
        finish_and_drain_pipeline(&mut final_pipeline, &execution.settings, &mut buffer_pool)?;
    }
    Python::attach(crate::binding::runtime::check_process_signals)?;
    Ok(processed_chunk_count)
}

fn run_prepared_grouped_union_association_delivery(
    engine: &g_engine::Regenie2RunEngineCore,
    backend: &Arc<PyJaxBackend>,
    request: GroupedUnionAssociationDeliveryRequest,
) -> PyResult<usize> {
    let GroupedUnionAssociationDeliveryRequest { groups, union_sample_indices } = request;
    let committed_chunk_identifier_sets = groups
        .iter()
        .flat_map(|group| group.settings.committed_chunk_identifier_sets.iter().cloned())
        .collect::<Vec<_>>();
    let shared_committed_chunk_identifiers =
        g_engine::intersect_committed_chunk_identifier_sets(&committed_chunk_identifier_sets);
    let chunk_specs = engine
        .plan_chunks(&shared_committed_chunk_identifiers)
        .map_err(|error| errors::convert_genotype_error("plan_chunks", error))?;
    let mut group_runtimes = prepare_grouped_union_runtimes(backend, groups, &union_sample_indices)?;
    let union_sample_count = union_sample_indices.len();
    let mut union_dosage_buffers = Vec::new();
    let mut current_chromosome = None;
    let mut processed_chunk_count = 0_usize;

    for chunk_spec in chunk_specs {
        Python::attach(crate::binding::runtime::check_process_signals)?;
        let variant_count = chunk_spec.variant_stop_index.saturating_sub(chunk_spec.variant_start_index);
        let metadata = engine
            .reader()
            .variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
            .map_err(|error| errors::convert_bgen_error("variant_metadata_slice", error))?;
        let chromosome = homogeneous_chromosome(&metadata, variant_count)?;
        if current_chromosome.as_deref() != Some(chromosome.as_str()) {
            finish_grouped_union_pipelines(&mut group_runtimes)?;
            start_grouped_union_chromosome_pipelines(backend, &mut group_runtimes, &chromosome)?;
            current_chromosome = Some(chromosome);
        }

        let union_value_count = variant_count
            .checked_mul(union_sample_count)
            .ok_or_else(|| PyValueError::new_err("Union genotype batch dimensions overflowed usize."))?;
        let mut union_dosages = take_matching_buffer(&mut union_dosage_buffers, union_value_count)
            .unwrap_or_else(|| vec![0.0_f32; union_value_count]);
        engine
            .reader()
            .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                chunk_spec.variant_start_index,
                chunk_spec.variant_stop_index,
                g_genotype::OutputBufferAddress::from_mut_ptr(union_dosages.as_mut_ptr()),
                g_genotype::OutputValueCount::new(union_dosages.len()),
            )
            .map_err(|error| {
                errors::convert_bgen_error("read_preprocessed_variant_major_dosage_f32_into_address_prepared", error)
            })?;

        let submission_result = submit_grouped_union_chunk(
            &mut group_runtimes,
            &union_dosages,
            union_sample_count,
            variant_count,
            chunk_spec.variant_start_index,
            &metadata,
        );
        union_dosage_buffers.push(union_dosages);
        if let Err(error) = submission_result {
            abort_grouped_union_pipelines(&mut group_runtimes);
            return Err(error);
        }
        processed_chunk_count += 1;
        if let Err(error) = drain_grouped_union_pipelines(&mut group_runtimes) {
            abort_grouped_union_pipelines(&mut group_runtimes);
            return Err(error);
        }
    }

    finish_grouped_union_pipelines(&mut group_runtimes)?;
    Python::attach(crate::binding::runtime::check_process_signals)?;
    Ok(processed_chunk_count)
}

fn prepare_grouped_union_runtimes(
    backend: &PyJaxBackend,
    groups: Vec<MultiAssociationDeliveryRequest>,
    union_sample_indices: &[usize],
) -> PyResult<Vec<GroupedUnionGroupRuntime>> {
    let mut group_runtimes = Vec::with_capacity(groups.len());
    for group in groups {
        let sample_positions =
            g_input::build_group_sample_position_array(union_sample_indices, &group.aligned_sample_data.sample_indices)
                .map_err(|error| PyValueError::new_err(error.to_string()))?
                .into_iter()
                .map(|position| {
                    usize::try_from(position)
                        .map_err(|_| PyValueError::new_err("Grouped-union sample position must be nonnegative."))
                })
                .collect::<PyResult<Vec<_>>>()?;
        let execution = DeliveryExecution {
            group_input: AssociationGroupInput::Multi {
                aligned_sample_data: group.aligned_sample_data,
                prediction_source: group.prediction_source,
            },
            settings: group.settings,
        };
        let group_state = backend
            .prepare_group(execution.group_input.group_preparation_input())
            .map_err(|error| errors::convert_backend_error("prepare_group", &error))?;
        group_runtimes.push(GroupedUnionGroupRuntime {
            execution,
            group_state,
            sample_positions,
            pipeline: None,
            buffer_pool: GenotypeBufferPool::default(),
        });
    }
    Ok(group_runtimes)
}

fn start_grouped_union_chromosome_pipelines(
    backend: &Arc<PyJaxBackend>,
    group_runtimes: &mut [GroupedUnionGroupRuntime],
    chromosome: &str,
) -> PyResult<()> {
    for group_index in 0..group_runtimes.len() {
        let start_result = (|| {
            let group_runtime = &mut group_runtimes[group_index];
            let predictions = group_runtime.execution.group_input.chromosome_predictions(chromosome)?;
            let prepared_chromosome = backend
                .prepare_chromosome(&group_runtime.group_state, predictions.preparation_input())
                .map_err(|error| errors::convert_backend_error("prepare_chromosome", &error))?;
            enforce_null_logistic_policy(
                chromosome,
                prepared_chromosome.null_model_diagnostics.as_ref(),
                &group_runtime.execution.group_input,
                &group_runtime.execution.settings.null_logistic_nonconvergence_policy,
            )?;
            g_engine::AssociationBatchPipeline::new(
                Arc::clone(backend),
                prepared_chromosome.state,
                group_runtime.execution.settings.staging_depth,
                group_runtime.execution.settings.result_in_flight_limit,
            )
            .map_err(|error| errors::convert_scheduler_error("start", &error))
        })();
        match start_result {
            Ok(pipeline) => group_runtimes[group_index].pipeline = Some(pipeline),
            Err(error) => {
                abort_grouped_union_pipelines(group_runtimes);
                return Err(error);
            }
        }
    }
    Ok(())
}

fn submit_grouped_union_chunk(
    group_runtimes: &mut [GroupedUnionGroupRuntime],
    union_dosages: &[f32],
    union_sample_count: usize,
    variant_count: usize,
    variant_start_index: usize,
    metadata: &g_genotype::VariantMetadataColumns,
) -> PyResult<()> {
    for group_runtime in group_runtimes {
        let settings = &group_runtime.execution.settings;
        let active_trait_indices = active_trait_indices(
            settings.writer_sessions.len(),
            variant_start_index,
            &settings.committed_chunk_identifier_sets,
        )?;
        if active_trait_indices.is_empty() {
            continue;
        }
        let sample_count = group_runtime.sample_positions.len();
        let genotype_value_count = variant_count
            .checked_mul(sample_count)
            .ok_or_else(|| PyValueError::new_err("Projected genotype batch dimensions overflowed usize."))?;
        let genotype_buffer = group_runtime.buffer_pool.acquire(genotype_value_count, false)?;
        let g_engine::OwnedGenotypeBuffer::Dosage(mut group_dosages) = genotype_buffer else {
            return Err(PyRuntimeError::new_err(
                "Grouped-union delivery unexpectedly acquired a packed8 genotype buffer.",
            ));
        };
        project_variant_major_dosages(
            union_dosages,
            union_sample_count,
            variant_count,
            &group_runtime.sample_positions,
            &mut group_dosages,
        )?;
        let statistics = g_genotype::summarize_variant_major_dosage_matrix(&group_dosages, sample_count, variant_count)
            .map_err(|error| errors::convert_genotype_error("summarize_grouped_union_dosages", error))?;
        let scheduled_batch = g_engine::ScheduledAssociationBatch {
            variant_start_index,
            variant_count,
            sample_count,
            metadata: metadata.clone(),
            statistics,
            genotype_buffer: g_engine::OwnedGenotypeBuffer::Dosage(group_dosages),
            active_trait_indices,
            output_statistic_dtype: settings.output_statistic_dtype,
        };
        let pipeline = group_runtime
            .pipeline
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Grouped-union association pipeline was not initialized."))?;
        pipeline.submit(scheduled_batch).map_err(|error| errors::convert_scheduler_error("submit", &error))?;
    }
    Ok(())
}

fn project_variant_major_dosages(
    union_dosages: &[f32],
    union_sample_count: usize,
    variant_count: usize,
    group_sample_positions: &[usize],
    group_dosages: &mut [f32],
) -> PyResult<()> {
    let expected_union_value_count = variant_count
        .checked_mul(union_sample_count)
        .ok_or_else(|| PyValueError::new_err("Union genotype dimensions overflowed usize."))?;
    if union_dosages.len() != expected_union_value_count {
        return Err(PyValueError::new_err(format!(
            "Union genotype buffer contains {} values, expected {expected_union_value_count}.",
            union_dosages.len()
        )));
    }
    let expected_group_value_count = variant_count
        .checked_mul(group_sample_positions.len())
        .ok_or_else(|| PyValueError::new_err("Projected genotype dimensions overflowed usize."))?;
    if group_dosages.len() != expected_group_value_count {
        return Err(PyValueError::new_err(format!(
            "Projected genotype buffer contains {} values, expected {expected_group_value_count}.",
            group_dosages.len()
        )));
    }
    if let Some(invalid_position) = group_sample_positions.iter().find(|position| **position >= union_sample_count) {
        return Err(PyValueError::new_err(format!(
            "Grouped-union sample position {invalid_position} is out of range for {union_sample_count} samples."
        )));
    }
    for (union_row, group_row) in
        union_dosages.chunks_exact(union_sample_count).zip(group_dosages.chunks_exact_mut(group_sample_positions.len()))
    {
        for (output_value, sample_position) in group_row.iter_mut().zip(group_sample_positions) {
            *output_value = union_row[*sample_position];
        }
    }
    Ok(())
}

fn drain_grouped_union_pipelines(group_runtimes: &mut [GroupedUnionGroupRuntime]) -> PyResult<()> {
    for group_runtime in group_runtimes {
        let GroupedUnionGroupRuntime { execution, pipeline, buffer_pool, .. } = group_runtime;
        let pipeline = pipeline
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("Grouped-union association pipeline was not initialized."))?;
        drain_available_batches(pipeline, &execution.settings, buffer_pool)?;
    }
    Ok(())
}

fn finish_grouped_union_pipelines(group_runtimes: &mut [GroupedUnionGroupRuntime]) -> PyResult<()> {
    for group_index in 0..group_runtimes.len() {
        let finish_result = {
            let GroupedUnionGroupRuntime { execution, pipeline, buffer_pool, .. } = &mut group_runtimes[group_index];
            match pipeline.take() {
                Some(mut pipeline) => finish_and_drain_pipeline(&mut pipeline, &execution.settings, buffer_pool),
                None => Ok(()),
            }
        };
        if let Err(error) = finish_result {
            abort_grouped_union_pipelines(group_runtimes);
            return Err(error);
        }
    }
    Ok(())
}

fn abort_grouped_union_pipelines(group_runtimes: &mut [GroupedUnionGroupRuntime]) {
    for group_runtime in group_runtimes {
        if let Some(pipeline) = group_runtime.pipeline.as_mut() {
            let _ = pipeline.abort();
        }
    }
}

fn validate_grouped_union_request(request: &GroupedUnionAssociationDeliveryRequest) -> PyResult<()> {
    if request.groups.is_empty() {
        return Err(PyValueError::new_err("Grouped-union association delivery requires at least one phenotype group."));
    }
    if request.union_sample_indices.is_empty() {
        return Err(PyValueError::new_err("Grouped-union association delivery requires at least one union sample."));
    }
    let sample_indices_by_group =
        request.groups.iter().map(|group| group.aligned_sample_data.sample_indices.clone()).collect::<Vec<_>>();
    let expected_union_sample_indices = g_input::build_union_sample_indices(&sample_indices_by_group);
    if request.union_sample_indices != expected_union_sample_indices {
        return Err(PyValueError::new_err(
            "Grouped-union sample indices do not match the ordered union of group sample indices.",
        ));
    }
    for group in &request.groups {
        validate_delivery_settings(
            group.aligned_sample_data.phenotype_row_count,
            &group.aligned_sample_data.sample_indices,
            &group.settings,
        )?;
        if group.settings.use_packed8 {
            return Err(PyValueError::new_err("Grouped-union association delivery supports dosage genotypes only."));
        }
    }
    Ok(())
}

fn validate_execution(execution: &DeliveryExecution) -> PyResult<()> {
    validate_delivery_settings(
        execution.group_input.trait_count(),
        execution.group_input.sample_indices(),
        &execution.settings,
    )
}

fn validate_delivery_settings(
    trait_count: usize,
    aligned_sample_indices: &[usize],
    settings: &AssociationDeliverySettings,
) -> PyResult<()> {
    if aligned_sample_indices.is_empty() {
        return Err(PyValueError::new_err("Native association delivery requires at least one aligned sample."));
    }
    let writer_session_count = settings.writer_sessions.len();
    if writer_session_count == 0 {
        return Err(PyValueError::new_err("Native association delivery requires at least one output writer."));
    }
    if writer_session_count != trait_count {
        return Err(PyValueError::new_err(format!(
            "Output writer count {writer_session_count} does not match phenotype trait count {trait_count}.",
        )));
    }
    if settings.committed_chunk_identifier_sets.len() != writer_session_count {
        return Err(PyValueError::new_err(format!(
            "Committed chunk set count {} does not match output writer count {writer_session_count}.",
            settings.committed_chunk_identifier_sets.len()
        )));
    }
    if settings.sample_indices != aligned_sample_indices {
        return Err(PyValueError::new_err("Delivery sample indices do not match the aligned sample data indices."));
    }
    if settings.staging_depth == 0 {
        return Err(PyValueError::new_err("Association staging depth must be positive."));
    }
    if settings.result_in_flight_limit == 0 {
        return Err(PyValueError::new_err("Association result in-flight limit must be positive."));
    }
    if !matches!(settings.null_logistic_nonconvergence_policy.as_str(), "fail" | "warn") {
        return Err(PyValueError::new_err(format!(
            "Unsupported null logistic nonconvergence policy: {}",
            settings.null_logistic_nonconvergence_policy
        )));
    }
    Ok(())
}

fn homogeneous_chromosome(metadata: &g_genotype::VariantMetadataColumns, variant_count: usize) -> PyResult<String> {
    if variant_count == 0 {
        return Err(PyValueError::new_err("Native delivery received an empty BGEN chunk."));
    }
    if metadata.chromosome.len() != variant_count {
        return Err(PyValueError::new_err(format!(
            "Chromosome metadata contains {} values for a {variant_count}-variant chunk.",
            metadata.chromosome.len()
        )));
    }
    let chromosome = metadata
        .chromosome
        .first()
        .ok_or_else(|| PyValueError::new_err("Native delivery chunk has no chromosome metadata."))?;
    if metadata.chromosome.iter().any(|value| value != chromosome) {
        return Err(PyValueError::new_err("Native delivery received a chunk spanning multiple chromosomes."));
    }
    Ok(chromosome.clone())
}

fn active_trait_indices(
    writer_session_count: usize,
    chunk_identifier: usize,
    committed_chunk_identifier_sets: &[BTreeSet<usize>],
) -> PyResult<Vec<usize>> {
    g_engine::plan_multi_trait_chunk_write(writer_session_count, chunk_identifier, committed_chunk_identifier_sets)
        .map(|plan| plan.active_trait_indices)
        .map_err(|error| errors::convert_schedule_error(&error))
}

fn decode_genotype_buffer(
    reader: &g_genotype::BgenReaderCore,
    variant_start_index: usize,
    variant_stop_index: usize,
    buffer: &mut g_engine::OwnedGenotypeBuffer,
) -> PyResult<g_genotype::ChunkStats> {
    match buffer {
        g_engine::OwnedGenotypeBuffer::Dosage(values) => reader
            .read_preprocessed_variant_major_dosage_f32_into_address_prepared(
                variant_start_index,
                variant_stop_index,
                g_genotype::OutputBufferAddress::from_mut_ptr(values.as_mut_ptr()),
                g_genotype::OutputValueCount::new(values.len()),
            )
            .map_err(|error| {
                errors::convert_bgen_error("read_preprocessed_variant_major_dosage_f32_into_address_prepared", error)
            }),
        g_engine::OwnedGenotypeBuffer::Packed8(values) => reader
            .read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared(
                variant_start_index,
                variant_stop_index,
                g_genotype::OutputBufferAddress::from_mut_ptr(values.as_mut_ptr()),
                g_genotype::OutputValueCount::new(values.len()),
            )
            .map_err(|error| {
                errors::convert_bgen_error(
                    "read_preprocessed_variant_major_packed8_probability_pairs_into_address_prepared",
                    error,
                )
            }),
    }
}

fn drain_available_batches(
    pipeline: &g_engine::AssociationBatchPipeline<PyJaxBackend>,
    settings: &AssociationDeliverySettings,
    buffer_pool: &mut GenotypeBufferPool,
) -> PyResult<()> {
    loop {
        let completed_batch =
            pipeline.try_receive().map_err(|error| errors::convert_scheduler_error("receive", &error))?;
        let Some(completed_batch) = completed_batch else {
            return Ok(());
        };
        write_completed_batch(completed_batch, settings, buffer_pool)?;
    }
}

fn finish_and_drain_pipeline(
    pipeline: &mut g_engine::AssociationBatchPipeline<PyJaxBackend>,
    settings: &AssociationDeliverySettings,
    buffer_pool: &mut GenotypeBufferPool,
) -> PyResult<()> {
    pipeline.finish().map_err(|error| errors::convert_scheduler_error("finish", &error))?;
    loop {
        let completed_batch = pipeline.receive().map_err(|error| errors::convert_scheduler_error("receive", &error))?;
        let Some(completed_batch) = completed_batch else {
            return Ok(());
        };
        write_completed_batch(completed_batch, settings, buffer_pool)?;
    }
}

fn write_completed_batch(
    completed_batch: g_engine::CompletedAssociationBatch,
    settings: &AssociationDeliverySettings,
    buffer_pool: &mut GenotypeBufferPool,
) -> PyResult<()> {
    let active_trait_indices = active_trait_indices(
        settings.writer_sessions.len(),
        completed_batch.variant_start_index,
        &settings.committed_chunk_identifier_sets,
    )?;
    output::write_host_association_batch(
        &settings.writer_sessions,
        &active_trait_indices,
        completed_batch.variant_start_index,
        &completed_batch.metadata,
        &completed_batch.statistics,
        &completed_batch.result,
    )?;
    buffer_pool.release(completed_batch.genotype_buffer);
    Ok(())
}

fn enforce_null_logistic_policy(
    chromosome: &str,
    diagnostics: Option<&g_engine::NullModelDiagnostics>,
    group_input: &AssociationGroupInput,
    policy: &str,
) -> PyResult<()> {
    let Some(diagnostics) = diagnostics else {
        return Ok(());
    };
    let scalar_convergence = group_input.uses_scalar_null_convergence();
    let phenotype_names = (!scalar_convergence).then(|| group_input.phenotype_names());
    let plan = g_engine::plan_null_logistic_nonconvergence(
        chromosome,
        &diagnostics.logistic_converged,
        scalar_convergence,
        phenotype_names,
        policy,
    )
    .map_err(|error| errors::convert_null_logistic_policy_error(&error))?;
    match plan.action {
        g_engine::NullLogisticNonconvergenceAction::Continue => Ok(()),
        g_engine::NullLogisticNonconvergenceAction::Fail => Err(PyRuntimeError::new_err(
            plan.message.unwrap_or_else(|| "Binary null logistic model did not converge.".to_string()),
        )),
        g_engine::NullLogisticNonconvergenceAction::Warn => {
            let warning_message = plan.warning_message.unwrap_or_else(|| {
                "Binary null logistic model did not converge; continuing under warning policy.".to_string()
            });
            tracing::warn!(
                target: "g.python",
                g_event = "null_logistic_nonconvergence",
                chromosome = chromosome,
                nonconverged_count = plan.nonconverged_count,
                total_fit_count = plan.total_fit_count,
                "{warning_message}"
            );
            Ok(())
        }
    }
}

fn take_matching_buffer<Buffer>(buffers: &mut Vec<Vec<Buffer>>, value_count: usize) -> Option<Vec<Buffer>> {
    let buffer_index = buffers.iter().position(|values| values.len() == value_count)?;
    Some(buffers.swap_remove(buffer_index))
}
