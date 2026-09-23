use std::fs::{File, FileTimes};
use std::io::{BufWriter, Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use super::alignment::LocoSampleAlignment;
use super::cache::LocoFileIndexCache;
use super::loco::{index_loco_file, read_loco_chromosome_predictions_into};
use super::{PredictionError, PredictionLocoPath, PredictionSource, PredictionSourceLoader};
use crate::test_support::TemporaryDirectory;

const HEADER: &str = "FID_IID family-a_person-a family-b_person-b\n";

fn source_for_paths(paths: &[PredictionLocoPath]) -> PredictionSource {
    let phenotype_names = paths.iter().map(|path| path.phenotype_name.to_string()).collect::<Vec<_>>();
    let mut loader = PredictionSourceLoader::new(paths, &phenotype_names).expect("prediction catalog should load");
    loader
        .load(
            &(0..paths.len()).collect::<Vec<_>>(),
            &["family-a".to_string(), "family-b".to_string()],
            &["person-a".to_string(), "person-b".to_string()],
            &[1, 0],
        )
        .expect("prediction source should align reordered samples")
}

fn digest_file(path: &Path) -> [u8; 32] {
    let mut file = File::open(path).expect("independent fingerprint reader should open");
    let mut digest = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let byte_count = file.read(&mut buffer).expect("independent fingerprint reader should read");
        if byte_count == 0 {
            break;
        }
        digest.update(&buffer[..byte_count]);
    }
    digest.finalize().into()
}

#[test]
fn indexed_fingerprints_hash_exact_file_bytes_and_preserve_deferred_reads() {
    let directory = TemporaryDirectory::new("exact-indexed-fingerprint");
    let variants = [
        format!("{HEADER}2 2 3\n1 0 1\n"),
        format!("{HEADER}\n \t\n2 2 3\n\n1 0 1\n\n"),
        "FID_IID family-a_person-a family-b_person-b\r\n \t\r\n2 2 3\r\n\r\n1 0 1".to_string(),
        "FID_IID family-a_person-a family-b_person-b\r\n2 2 3\r\n1 0 1\r\n\r\n".to_string(),
    ];
    for (variant_index, contents) in variants.iter().enumerate() {
        let path = directory.write(&format!("variant-{variant_index}.loco"), contents);
        let indexed = index_loco_file(&path).expect("valid physical LOCO rows should index");
        let snapshot = indexed.file_index.file_fingerprint(&path).expect("unchanged index should yield a snapshot");
        assert_eq!(snapshot.path(), path);
        assert_eq!(*snapshot.content_sha256(), digest_file(&path));
        assert_eq!(snapshot.metadata().len(), u64::try_from(contents.len()).expect("small fixture size"));
        assert_eq!(snapshot.metadata().ino(), path.metadata().expect("fixture metadata should load").ino());
        assert_eq!(indexed.file_index.sample_count, 2);
        let mut predictions = Vec::new();
        read_loco_chromosome_predictions_into(
            &indexed.file_index,
            "1",
            &LocoSampleAlignment::Identity,
            &mut predictions,
        )
        .expect("fingerprinting must preserve deferred row offsets");
        read_loco_chromosome_predictions_into(
            &indexed.file_index,
            "2",
            &LocoSampleAlignment::Identity,
            &mut predictions,
        )
        .expect("out-of-order chromosome rows should remain addressable");
        assert_eq!(
            predictions.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
            [0, 1.0_f32.to_bits(), 2.0_f32.to_bits(), 3.0_f32.to_bits()]
        );
    }
}

#[test]
fn physical_row_order_and_blank_lines_do_not_change_alignment_fingerprints() {
    let directory = TemporaryDirectory::new("independent-alignment-fingerprint");
    let first_path = directory.write("ordered.loco", &format!("{HEADER}1 0 1\n2 2 3\n"));
    let second_path = directory.write("reordered.loco", &format!("{HEADER}\n2 2 3\n \t\n1 0 1\n\n"));
    let first_index = index_loco_file(&first_path).expect("ordered predictions should index");
    let second_index = index_loco_file(&second_path).expect("reordered predictions should index");
    assert_eq!(first_index.file_index.header_digest, second_index.file_index.header_digest);
    assert_eq!(first_index.file_index.source_digest, second_index.file_index.source_digest);
    let first_paths = [PredictionLocoPath { phenotype_name: Arc::from("trait"), loco_file_path: first_path }];
    let second_paths = [PredictionLocoPath { phenotype_name: Arc::from("trait"), loco_file_path: second_path }];
    let mut first_source = source_for_paths(&first_paths);
    let mut second_source = source_for_paths(&second_paths);
    let original_alignment_digest = first_source.alignment_source_digest();
    let first_fingerprints = first_source.indexed_file_fingerprints().expect("ordered source should verify");
    let second_fingerprints = second_source.indexed_file_fingerprints().expect("reordered source should verify");
    assert_ne!(first_fingerprints[0].content_sha256(), second_fingerprints[0].content_sha256());
    assert_eq!(original_alignment_digest, first_source.alignment_source_digest());
    assert_eq!(original_alignment_digest, second_source.alignment_source_digest());
    for source in [&mut first_source, &mut second_source] {
        source.plan_uses(&[Arc::from("1"), Arc::from("1")]).expect("repeated chromosome uses should plan");
        for _ in 0..2 {
            let matrix = source.take_chromosome_prediction_matrix("1").expect("deferred reordered samples should load");
            assert_eq!(matrix.trait_count, 1);
            assert_eq!(matrix.sample_count, 2);
            assert_eq!(
                matrix.prediction_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                [1.0_f32.to_bits(), 0]
            );
        }
    }
}

#[test]
fn indexed_fingerprint_rejects_in_place_edit_with_restored_size_and_mtime() {
    let directory = TemporaryDirectory::new("fingerprint-in-place-change");
    let path = directory.write("trait.loco", &format!("{HEADER}1 0 1\n"));
    let indexed = index_loco_file(&path).expect("original prediction source should index");
    let original_metadata = path.metadata().expect("original metadata should load");
    // Cross the timestamp boundary on filesystems with one-second ctime precision.
    std::thread::sleep(Duration::from_millis(1_100));
    std::fs::write(&path, format!("{HEADER}1 1 0\n")).expect("same-size in-place mutation should write");
    File::options()
        .write(true)
        .open(&path)
        .expect("changed file should open")
        .set_times(FileTimes::new().set_modified(original_metadata.modified().expect("original mtime should load")))
        .expect("original modification time should restore");
    let changed_metadata = path.metadata().expect("changed metadata should load");
    assert_ne!(original_metadata.ctime(), changed_metadata.ctime(), "mutation must have a distinct ctime");
    assert_eq!(original_metadata.len(), changed_metadata.len());
    assert_eq!(
        original_metadata.modified().expect("original mtime should load"),
        changed_metadata.modified().expect("restored mtime should load")
    );
    assert_eq!(original_metadata.ino(), changed_metadata.ino());
    assert!(matches!(indexed.file_index.file_fingerprint(&path), Err(PredictionError::IndexedLocoFileChanged { .. })));
}

#[test]
fn indexed_fingerprint_rejects_replacement_even_with_identical_bytes_and_mtime() {
    let directory = TemporaryDirectory::new("fingerprint-replacement");
    let contents = format!("{HEADER}1 0 1\n");
    let path = directory.write("trait.loco", &contents);
    let indexed = index_loco_file(&path).expect("original prediction source should index");
    let original_metadata = path.metadata().expect("original metadata should load");
    let replacement = directory.write("replacement.loco", &contents);
    File::options()
        .write(true)
        .open(&replacement)
        .expect("replacement should open")
        .set_times(FileTimes::new().set_modified(original_metadata.modified().expect("original mtime should load")))
        .expect("replacement should retain original modification time");
    std::fs::rename(replacement, &path).expect("replacement should replace indexed path");
    assert!(matches!(indexed.file_index.file_fingerprint(&path), Err(PredictionError::IndexedLocoFileChanged { .. })));
}

#[test]
fn shared_source_aliases_keep_trait_order_and_reject_a_retargeted_alias() {
    let directory = TemporaryDirectory::new("fingerprint-trait-aliases");
    let contents = format!("{HEADER}1 0 1\n");
    let source_path = directory.write("source.loco", &contents);
    let other_path = directory.write("other.loco", &contents);
    let alias_path = directory.path().join("alias.loco");
    std::os::unix::fs::symlink(&source_path, &alias_path).expect("trait alias should be created");
    let mut cache = LocoFileIndexCache::default();
    let first_index = cache.index(&source_path).expect("canonical source should index");
    let alias_index = cache.index(&alias_path).expect("alias should reuse indexed source");
    assert!(Arc::ptr_eq(&first_index.file_index, &alias_index.file_index));
    let paths = [
        PredictionLocoPath { phenotype_name: Arc::from("alias-trait"), loco_file_path: alias_path.clone() },
        PredictionLocoPath { phenotype_name: Arc::from("source-trait"), loco_file_path: source_path.clone() },
    ];
    let mut source = source_for_paths(&paths);
    let snapshots = source.indexed_file_fingerprints().expect("both configured aliases should verify");
    assert_eq!(snapshots.len(), 2);
    assert_eq!(snapshots[0].path(), alias_path);
    assert_eq!(snapshots[1].path(), source_path);
    assert_eq!(snapshots[0].content_sha256(), snapshots[1].content_sha256());
    assert_eq!(snapshots[0].metadata().ino(), snapshots[1].metadata().ino());
    source.plan_uses(&[Arc::from("1")]).expect("shared predictions should plan");
    let matrix = source.take_chromosome_prediction_matrix("1").expect("shared trait predictions should materialize");
    assert_eq!(matrix.trait_count, 2);
    assert_eq!(
        matrix.prediction_values.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
        [1.0_f32.to_bits(), 0, 1.0_f32.to_bits(), 0]
    );
    std::fs::remove_file(&alias_path).expect("original alias should unlink");
    std::os::unix::fs::symlink(&other_path, &alias_path).expect("alias should point to a different source");
    assert!(first_index.file_index.file_fingerprint(&source_path).is_ok());
    assert!(matches!(
        alias_index.file_index.file_fingerprint(&alias_path),
        Err(PredictionError::IndexedLocoFileChanged { .. })
    ));
    assert!(matches!(source.indexed_file_fingerprints(), Err(PredictionError::IndexedLocoFileChanged { .. })));
}

fn write_scaled_predictions(path: &Path, sample_count: usize) {
    let mut writer = BufWriter::new(File::create(path).expect("scaled benchmark fixture should create"));
    write!(writer, "FID_IID").expect("scaled header marker should write");
    for sample_index in 0..sample_count {
        write!(writer, " family-{sample_index}_person-{sample_index}").expect("scaled sample identifier should write");
    }
    writeln!(writer).expect("scaled header newline should write");
    let prediction_row = " 0.123456789".repeat(sample_count);
    for chromosome in 1..=23 {
        writeln!(writer, "{chromosome}{prediction_row}").expect("scaled chromosome row should write");
    }
    writer.flush().expect("scaled benchmark fixture should flush before measurement");
}

fn benchmark_index_snapshot(path: &Path, label: &str, repetitions: usize, report: &mut impl Write) {
    let expected_digest = digest_file(path);
    let byte_count = path.metadata().expect("benchmark source metadata should load").len();
    for repetition in 0..repetitions {
        let index_start = Instant::now();
        let indexed =
            std::hint::black_box(index_loco_file(std::hint::black_box(path)).expect("benchmark source indexes"));
        let index_seconds = index_start.elapsed().as_secs_f64();
        let verification_start = Instant::now();
        let snapshot = indexed.file_index.file_fingerprint(path).expect("benchmark indexed snapshot verifies");
        let verification_seconds = verification_start.elapsed().as_secs_f64();
        assert_eq!(*snapshot.content_sha256(), expected_digest);
        let second_pass_start = Instant::now();
        let second_pass_digest = std::hint::black_box(digest_file(path));
        let second_pass_seconds = second_pass_start.elapsed().as_secs_f64();
        assert_eq!(second_pass_digest, expected_digest);
        writeln!(
            report,
            "{label},{repetition},{byte_count},{index_seconds:.9},{verification_seconds:.9},{second_pass_seconds:.9}"
        )
        .expect("benchmark timing row should write");
    }
}

#[test]
#[ignore = "Run explicitly on a CPU compute node; writes large synthetic LOCO files and timing artifacts."]
fn benchmark_loco_index_and_verified_snapshot() {
    let artifact_directory = std::env::var_os("GWAS_ENGINE_LOCO_FINGERPRINT_RESULTS").map_or_else(
        || Path::new(env!("CARGO_MANIFEST_DIR")).join("../../results/loco-fingerprint-microbenchmark"),
        std::path::PathBuf::from,
    );
    std::fs::create_dir_all(&artifact_directory).expect("benchmark artifact directory should create");
    let repetitions = std::env::var("GWAS_ENGINE_LOCO_FINGERPRINT_REPETITIONS")
        .map_or(5, |value| value.parse::<usize>().expect("benchmark repetitions must be a positive integer"));
    assert!(repetitions > 0, "at least one benchmark repetition is required");
    let mut report = BufWriter::new(
        File::create(artifact_directory.join("optimized-stages.csv")).expect("benchmark stage report should create"),
    );
    writeln!(report, "fixture,repetition,bytes,index_seconds,snapshot_seconds,incremental_second_read_hash_seconds")
        .expect("benchmark report header should write");
    for sample_count in [100_000, 500_000] {
        let label = format!("synthetic-{sample_count}-23-chromosomes");
        let path = artifact_directory.join(format!("{label}.loco"));
        write_scaled_predictions(&path, sample_count);
        benchmark_index_snapshot(&path, &label, repetitions, &mut report);
    }
    if let Some(real_path) = std::env::var_os("GWAS_ENGINE_LOCO_FINGERPRINT_REAL_PATH") {
        benchmark_index_snapshot(Path::new(&real_path), "real-loco", repetitions, &mut report);
    }
    report.flush().expect("benchmark stage report should flush");
}
