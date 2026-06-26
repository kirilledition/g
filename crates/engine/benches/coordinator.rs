use std::hint;

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use g_engine::{EngineCoordinator, EngineRunInput, FakeBackend, GenotypeBatchView, PredictionView, PreparedGroupInput};

fn build_run_input() -> EngineRunInput<'static> {
    EngineRunInput::new(
        PreparedGroupInput::new("binary".to_string(), 4),
        "chr22",
        PredictionView::new("chr22", 2048),
        GenotypeBatchView::new("chr22", 512, 4096),
    )
}

fn benchmark_single_batch_coordinator(criterion: &mut Criterion) {
    let run_input = build_run_input();
    criterion.bench_function("engine_single_batch_fake_backend", |bencher| {
        bencher.iter_batched(
            || (EngineCoordinator::new(FakeBackend::succeed()), run_input.clone()),
            |(mut coordinator, input)| {
                hint::black_box(
                    coordinator
                        .run_single_batch(hint::black_box(&input))
                        .expect("fake-backend coordinator run should complete"),
                );
            },
            BatchSize::SmallInput,
        );
    });
}

criterion_group!(benches, benchmark_single_batch_coordinator);
criterion_main!(benches);
