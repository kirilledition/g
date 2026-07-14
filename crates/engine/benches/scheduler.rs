// The benchmark includes production-private modules so it exercises the exact
// scheduler implementation without widening the engine's public API.
#![allow(dead_code)]

use std::convert::Infallible;
use std::sync::Arc;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};

#[path = "../src/association_scheduler.rs"]
mod association_scheduler;
#[path = "../src/backend.rs"]
pub mod backend;
#[path = "../src/output_schedule.rs"]
mod output_schedule;

use association_scheduler::{AssociationBatchPipeline, ScheduledAssociationBatch};
use backend::{AssociationBackend, GenotypeBatchInput, GroupPreparationInput, PreparedChromosome};
use g_genotype::{ChunkComputeStatistics, ChunkStats, DecodedGenotypeBatch, OwnedGenotypeBuffer};
use g_genotype_contracts::{
    ChunkOutputStatistics, NullableFloat32Column, VariantMetadataColumns, VariantMetadataStore,
};
use g_output::{NativeVariantMetadataHandle, Regenie2StatisticBatch};
use output_schedule::ActiveTraitSelection;

const BATCH_COUNT: usize = 512;
const SAMPLE_COUNT: usize = 8;
const VARIANT_COUNT: usize = 8;

struct MockBackend;

impl AssociationBackend for MockBackend {
    type GroupState = ();
    type ChromosomeState = ();
    type TransferredInput = GenotypeBatchInput;
    type DeviceResult = usize;
    type Error = Infallible;

    fn prepare_group(&self, _input: GroupPreparationInput) -> Result<Self::GroupState, Self::Error> {
        Ok(())
    }

    fn prepare_chromosome(
        &self,
        _group: &Self::GroupState,
        _predictions: g_input::ChromosomePredictionMatrix,
    ) -> Result<PreparedChromosome<Self::ChromosomeState>, Self::Error> {
        Ok(PreparedChromosome { state: (), null_logistic_converged: None })
    }

    fn transfer_batch(&self, input: GenotypeBatchInput) -> Result<Self::TransferredInput, Self::Error> {
        Ok(input)
    }

    fn compute_batch(
        &self,
        _chromosome: &Self::ChromosomeState,
        input: Self::TransferredInput,
    ) -> Result<Self::DeviceResult, Self::Error> {
        std::hint::black_box(input);
        Ok(VARIANT_COUNT)
    }

    fn materialize_batch(
        &self,
        result: Self::DeviceResult,
        _active_trait_indices: Option<&[usize]>,
        logical_variant_count: usize,
    ) -> Result<Regenie2StatisticBatch, Self::Error> {
        std::hint::black_box(result);
        Ok(Regenie2StatisticBatch {
            trait_count: 1,
            variant_count: logical_variant_count,
            beta: Vec::new(),
            standard_error: Vec::new(),
            chi_squared: Vec::new(),
            log10_p_value: Vec::new(),
            correction_code: None,
        })
    }
}

fn build_metadata() -> VariantMetadataColumns {
    let dictionary: Box<[Arc<str>]> = [Arc::from("22"), Arc::from("A"), Arc::from("G")].into();
    let chromosome_codes = vec![0_u32; VARIANT_COUNT].into_boxed_slice();
    let variant_identifier_text = "v".repeat(VARIANT_COUNT).into_boxed_str();
    let variant_identifier_offsets = (0..=VARIANT_COUNT)
        .map(|index| u32::try_from(index).expect("benchmark variant index fits u32"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let positions = (0..VARIANT_COUNT)
        .map(|index| i64::try_from(index).expect("benchmark variant index fits i64"))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let allele_one_codes = vec![1_u32; VARIANT_COUNT].into_boxed_slice();
    let allele_two_codes = vec![2_u32; VARIANT_COUNT].into_boxed_slice();
    let store = Arc::new(VariantMetadataStore::from_parts(
        dictionary,
        chromosome_codes,
        variant_identifier_text,
        variant_identifier_offsets,
        positions,
        allele_one_codes,
        allele_two_codes,
    ));
    VariantMetadataColumns::new(store, 0..VARIANT_COUNT)
}

fn build_batch(metadata: &VariantMetadataColumns, batch_index: usize) -> ScheduledAssociationBatch {
    let genotype_value_count = VARIANT_COUNT * SAMPLE_COUNT * 2;
    ScheduledAssociationBatch {
        decoded: DecodedGenotypeBatch {
            variant_start_index: batch_index * VARIANT_COUNT,
            logical_variant_count: VARIANT_COUNT,
            compute_variant_count: VARIANT_COUNT,
            sample_count: SAMPLE_COUNT,
            genotypes: OwnedGenotypeBuffer::Packed8(vec![0_u8; genotype_value_count].into()),
            statistics: ChunkStats {
                output: ChunkOutputStatistics {
                    allele_one_frequency: vec![0.0; VARIANT_COUNT],
                    observation_count: vec![0; VARIANT_COUNT],
                    info_score: NullableFloat32Column {
                        values: vec![0.0; VARIANT_COUNT],
                        validity_bytes: vec![0; VARIANT_COUNT.div_ceil(8)],
                    },
                },
                compute: ChunkComputeStatistics {
                    genotype_mean: vec![0.0; VARIANT_COUNT],
                    imputed_dosage_square_sum: None,
                    sparse_candidate_mask: Some(vec![false; VARIANT_COUNT]),
                },
            },
        },
        metadata: NativeVariantMetadataHandle::try_new(metadata).expect("benchmark metadata is valid"),
        active_trait_selection: ActiveTraitSelection::All,
    }
}

fn build_batches(metadata: &VariantMetadataColumns) -> Vec<ScheduledAssociationBatch> {
    (0..BATCH_COUNT).map(|batch_index| build_batch(metadata, batch_index)).collect()
}

fn run_pipeline(batches: Vec<ScheduledAssociationBatch>) -> usize {
    let mut pipeline = AssociationBatchPipeline::new(Arc::new(MockBackend)).expect("benchmark pipeline starts");
    pipeline.prepare_chromosome(()).expect("benchmark chromosome is prepared");
    let mut completed_batch_count = 0_usize;
    for batch in batches {
        let mut pending_batch = batch;
        loop {
            match pipeline.try_submit(pending_batch).expect("benchmark batch is submitted") {
                None => break,
                Some(returned_batch) => {
                    std::hint::black_box(pipeline.receive().expect("benchmark batch completes"));
                    completed_batch_count += 1;
                    pending_batch = returned_batch;
                }
            }
        }
        while let Some(completed_batch) = pipeline.try_receive().expect("benchmark completion check succeeds") {
            std::hint::black_box(completed_batch);
            completed_batch_count += 1;
        }
    }
    while !pipeline.is_drained() {
        std::hint::black_box(pipeline.receive().expect("benchmark batch drains"));
        completed_batch_count += 1;
    }
    pipeline.release_chromosome().expect("benchmark chromosome is released");
    pipeline.close_submission();
    pipeline.join().expect("benchmark workers join");
    completed_batch_count
}

fn benchmark_scheduler(criterion: &mut Criterion) {
    let metadata = build_metadata();
    let mut group = criterion.benchmark_group("association_scheduler");
    group.throughput(Throughput::Elements(u64::try_from(BATCH_COUNT).expect("benchmark batch count fits u64")));
    group.bench_function("packed8_noop_roundtrip", |bencher| {
        bencher.iter_batched(
            || build_batches(&metadata),
            |batches| {
                assert_eq!(run_pipeline(batches), BATCH_COUNT);
            },
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

criterion_group!(benches, benchmark_scheduler);
criterion_main!(benches);
