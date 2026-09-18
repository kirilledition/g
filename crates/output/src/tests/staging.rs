use std::fs::File;
use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Float32Array, StringArray, UInt8Array};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde_json::Value;

use super::{
    PRIMARY_PHENOTYPE, TestDirectory, header, initialize_manager, metadata_store, planned_run_directories,
    read_manifest, run_plan, single_chunk_plan, test_chunk, test_inputs,
};
use crate::{OutputManager, resume, write_regenie2_multi_trait_chunk_f32, writer};

fn chunk_write_batch(chunk_ranges: &[Range<usize>]) -> writer::RegenieStep2ChunkWriteBatch {
    let first_range = chunk_ranges.first().expect("test batch contains chunks");
    let last_range = chunk_ranges.last().expect("test batch contains chunks");
    let store = metadata_store(last_range.end);
    writer::RegenieStep2ChunkWriteBatch {
        chunk_file_name: writer::build_part_file_name(
            i64::try_from(first_range.start).expect("test first index fits int64"),
            i64::try_from(last_range.start).expect("test last index fits int64"),
        ),
        chunks: chunk_ranges
            .iter()
            .map(|chunk_range| {
                let chunk = test_chunk(&store, chunk_range.clone(), 1);
                writer::RegenieStep2ChunkJob {
                    chunk_handle: chunk.handle,
                    beta: Arc::new(Float32Array::from(chunk.statistics.beta)),
                    se: Arc::new(Float32Array::from(chunk.statistics.standard_error)),
                    chisq: Arc::new(Float32Array::from(chunk.statistics.chi_squared)),
                    log10p: Arc::new(Float32Array::from(chunk.statistics.log10_p_value)),
                    correction_code: chunk
                        .statistics
                        .correction_code
                        .map(|values| Arc::new(UInt8Array::from(values)) as arrow::array::ArrayRef),
                }
            })
            .collect(),
    }
}

/// Read every file visible under Arrow's default hidden-prefix discovery rules.
fn read_dataset_identifiers(parts_directory: &Path) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let mut part_paths = Vec::new();
    for directory_entry in std::fs::read_dir(parts_directory)? {
        let directory_entry = directory_entry?;
        let file_name = directory_entry.file_name();
        let file_name = file_name.to_string_lossy();
        if !file_name.starts_with('.') && !file_name.starts_with('_') && directory_entry.file_type()?.is_file() {
            part_paths.push(directory_entry.path());
        }
    }
    part_paths.sort();
    let mut identifiers = Vec::new();
    for part_path in part_paths {
        let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(part_path)?)?.build()?;
        for batch in reader {
            let batch = batch?;
            let identifier_column = batch
                .column_by_name("ID")
                .expect("variant identifier column exists")
                .as_any()
                .downcast_ref::<StringArray>()
                .expect("variant identifiers are strings");
            identifiers.extend((0..batch.num_rows()).map(|index| identifier_column.value(index).to_string()));
        }
    }
    Ok(identifiers)
}

#[test]
fn failed_part_publication_leaves_hidden_staging_outside_dataset_discovery() {
    let directory = TestDirectory::new("hidden-staging");
    let chunk_ranges = (0..3).map(|index| index..index + 1).collect::<Vec<_>>();
    let batch = chunk_write_batch(&chunk_ranges);
    let final_path = directory.path.join(&batch.chunk_file_name);
    let staging_path = directory.path.join(format!(".{}.tmp", batch.chunk_file_name));
    let legacy_path = directory.path.join(format!("{}.tmp", batch.chunk_file_name));
    std::fs::create_dir(&final_path).expect("directory prevents atomic publication");

    writer::write_regenie_step2_chunk_job(&directory.path, batch, false)
        .err()
        .expect("publication fails after staging file closes");

    assert!(staging_path.is_file());
    assert!(!legacy_path.exists());
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(&staging_path).expect("staging file opens"))
        .expect("staging file contains a complete unpublished Parquet part");
    assert_eq!(reader.metadata().file_metadata().num_rows(), 3);
    std::fs::remove_dir(final_path).expect("publication blocker is removed");
    directory.write(".part_000000003.parquet.tmp", b"incomplete Parquet stream");
    assert!(read_dataset_identifiers(&directory.path).expect("hidden staging is ignored").is_empty());
}

#[test]
fn resume_removes_crashed_tail_before_regrouping_chunks() {
    for hidden_staging in [false, true] {
        for complete_staging in [false, true] {
            let directory = TestDirectory::new("staging-regroup");
            let phenotype_names = [PRIMARY_PHENOTYPE];
            let inputs = test_inputs(&directory, &phenotype_names);
            let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
            let run_directory = planned_run_directories(&plan).remove(0);
            let planned_ranges = (0..8).map(|index| index..index + 1).collect::<Vec<_>>();
            let manager = initialize_manager(plan, &inputs, &phenotype_names, &planned_ranges);
            manager.abort().expect("initial manager closes with no committed chunks");
            let parts_directory = run_directory.join("parts");
            let batch = chunk_write_batch(&planned_ranges[..3]);
            let part_path = parts_directory.join(&batch.chunk_file_name);
            let prefix = if hidden_staging { "." } else { "" };
            let staging_path = parts_directory.join(format!("{prefix}{}.tmp", batch.chunk_file_name));
            writer::write_regenie_step2_chunk_job(&parts_directory, batch, false).expect("tail fixture writes");
            std::fs::rename(&part_path, &staging_path).expect("crash leaves only unpublished tail staging");
            if !complete_staging {
                std::fs::write(&staging_path, b"PAR1 interrupted before footer").expect("tail fixture is truncated");
            }
            let prior_dataset = read_dataset_identifiers(&parts_directory);
            if hidden_staging {
                assert!(prior_dataset.expect("hidden staging is ignored").is_empty());
            } else if complete_staging {
                assert_eq!(prior_dataset.expect("legacy staging is accidentally discoverable"), ["rs0", "rs1", "rs2"]);
            } else {
                assert!(prior_dataset.is_err(), "incomplete legacy staging breaks dataset discovery");
            }

            let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
            let manager = initialize_manager(resume_plan, &inputs, &phenotype_names, &planned_ranges);
            assert!(!staging_path.exists(), "validated resume removes abandoned staging");
            let delivery = manager
                .delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()])
                .expect("resumed delivery is available");
            assert!(delivery.committed_chunk_identifier_sets[0].is_empty());
            let store = metadata_store(8);
            for chunk_range in planned_ranges {
                let chunk = test_chunk(&store, chunk_range, 1);
                write_regenie2_multi_trait_chunk_f32(&delivery.writer_sessions, None, &chunk.handle, chunk.statistics)
                    .expect("regrouped chunk is accepted");
            }
            drop(delivery);
            manager.finish().expect("regrouped output completes");

            assert!(parts_directory.join("part_000000000_000000007.parquet").is_file());
            assert_eq!(
                read_dataset_identifiers(&parts_directory).expect("resumed dataset reads without stale tail"),
                (0..8).map(|index| format!("rs{index}")).collect::<Vec<_>>()
            );
            assert_eq!(
                read_manifest(&run_directory)["committed_chunks"].as_array().expect("commits are a list").len(),
                8
            );
        }
    }
}

#[test]
fn staging_cleanup_matches_only_canonical_engine_file_names() {
    let directory = TestDirectory::new("staging-names");
    let staging_names = [
        "part_000000000.parquet.tmp",
        "part_000000000_000000002.parquet.tmp",
        ".part_000000003.parquet.tmp",
        ".part_1000000000_1000000001.parquet.tmp",
    ];
    let unrelated_names = [
        "notes.parquet.tmp",
        "part_0.parquet.tmp",
        "part_000000000_000000000.parquet.tmp",
        "part_-00000001.parquet.tmp",
        "part_9223372036854775808.parquet.tmp",
        "part_000000000_000000001_000000002.parquet.tmp",
        "..part_000000000.parquet.tmp",
        "part_000000000.parquet",
    ];
    for file_name in staging_names.into_iter().chain(unrelated_names) {
        directory.write(file_name, b"preserved unless staging");
    }
    let nested_directory = directory.path.join("unrelated-directory");
    std::fs::create_dir(&nested_directory).expect("unrelated directory is created");
    let nested_staging = nested_directory.join(staging_names[0]);
    std::fs::write(&nested_staging, b"nested staging belongs to another run").expect("nested file is created");
    let symlink_path = directory.path.join("unrelated-symlink");
    std::os::unix::fs::symlink(&nested_staging, &symlink_path).expect("unrelated symlink is created");

    let mut staging_paths = resume::find_stale_chunk_staging_files(&directory.path).expect("staging scan succeeds");
    staging_paths.sort();
    let mut expected_paths = staging_names.map(|name| directory.path.join(name));
    expected_paths.sort();
    assert_eq!(staging_paths, expected_paths);
    resume::remove_stale_chunk_staging_files(&staging_paths).expect("staging cleanup succeeds");

    assert!(staging_paths.iter().all(|path| !path.exists()));
    for file_name in unrelated_names {
        assert_eq!(
            std::fs::read(directory.path.join(file_name)).expect("unrelated file remains"),
            b"preserved unless staging"
        );
    }
    assert_eq!(std::fs::read(&nested_staging).expect("nested file remains"), b"nested staging belongs to another run");
    assert!(std::fs::symlink_metadata(symlink_path).expect("unrelated symlink remains").is_symlink());
    assert!(
        resume::find_stale_chunk_staging_files(&directory.path.join("missing"))
            .expect("missing parts are valid")
            .is_empty()
    );
}

#[test]
fn incompatible_later_phenotype_preserves_earlier_staging() {
    let directory = TestDirectory::new("staging-rejected-resume");
    let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
    let inputs = test_inputs(&directory, &phenotype_names);
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let run_directories = planned_run_directories(&plan);
    let planned_ranges = single_chunk_plan(0..1);
    let manager = initialize_manager(plan, &inputs, &phenotype_names, &planned_ranges);
    manager.abort().expect("initial manager closes");
    let staging_path = run_directories[0].join("parts/part_000000000.parquet.tmp");
    let staging_bytes = b"unpublished staging must survive rejected resume";
    std::fs::write(&staging_path, staging_bytes).expect("legacy staging is written");
    let manifest_path = run_directories[1].join("run_manifest.json");
    let mut incompatible_manifest = read_manifest(&run_directories[1]);
    incompatible_manifest["schema_version"] = Value::from(1);
    let incompatible_bytes = serde_json::to_vec_pretty(&incompatible_manifest).expect("manifest serializes");
    std::fs::write(&manifest_path, &incompatible_bytes).expect("later phenotype is incompatible");
    let first_manifest_bytes =
        std::fs::read(run_directories[0].join("run_manifest.json")).expect("first manifest reads");
    let config_bytes = run_directories
        .iter()
        .map(|path| std::fs::read(path.join("effective_config.toml")).expect("configuration reads"))
        .collect::<Vec<_>>();

    let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
    let mut manager =
        OutputManager::open(resume_plan, "# rejected resume\n".to_string()).expect("resume manager opens");
    let headers = phenotype_names.iter().map(|name| header(name, &inputs, 1)).collect();
    let error =
        manager.initialize(headers, &planned_ranges, false).expect_err("later compatibility failure rejects resume");
    assert!(error.to_string().contains("schema_version"), "unexpected error: {error}");
    drop(manager);

    assert_eq!(std::fs::read(staging_path).expect("staging remains"), staging_bytes);
    assert_eq!(std::fs::read(manifest_path).expect("incompatible manifest remains"), incompatible_bytes);
    assert_eq!(
        std::fs::read(run_directories[0].join("run_manifest.json")).expect("first manifest remains"),
        first_manifest_bytes
    );
    for (run_directory, original_bytes) in run_directories.iter().zip(config_bytes) {
        assert_eq!(
            std::fs::read(run_directory.join("effective_config.toml")).expect("configuration remains"),
            original_bytes
        );
    }
}

#[test]
fn reserved_staging_symlinks_and_directories_reject_resume_without_cleanup() {
    for hidden_staging in [false, true] {
        for symlink_staging in [false, true] {
            let directory = TestDirectory::new("staging-reserved-path");
            let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
            let inputs = test_inputs(&directory, &phenotype_names);
            let plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
            let run_directories = planned_run_directories(&plan);
            let planned_ranges = single_chunk_plan(0..1);
            let manager = initialize_manager(plan, &inputs, &phenotype_names, &planned_ranges);
            manager.abort().expect("initial manager closes");
            let regular_staging = run_directories[0].join("parts/part_000000000.parquet.tmp");
            std::fs::write(&regular_staging, b"earlier staging").expect("earlier staging is written");
            let target_path = directory.write("external-target", b"external bytes must remain untouched");
            let prefix = if hidden_staging { "." } else { "" };
            let reserved_path = run_directories[1].join(format!("parts/{prefix}part_000000000.parquet.tmp"));
            if symlink_staging {
                std::os::unix::fs::symlink(&target_path, &reserved_path).expect("reserved staging symlink is created");
            } else {
                std::fs::create_dir(&reserved_path).expect("reserved staging directory is created");
            }
            let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
            let mut manager =
                OutputManager::open(resume_plan, "# rejected staging\n".to_string()).expect("manager opens");
            let headers = phenotype_names.iter().map(|name| header(name, &inputs, 1)).collect();
            let error =
                manager.initialize(headers, &planned_ranges, false).expect_err("nonregular staging rejects resume");
            assert!(error.to_string().contains("staging path must be a regular file"), "unexpected error: {error}");
            drop(manager);

            assert_eq!(std::fs::read(regular_staging).expect("earlier staging remains"), b"earlier staging");
            assert_eq!(
                std::fs::read(target_path).expect("external target remains"),
                b"external bytes must remain untouched"
            );
            let metadata = std::fs::symlink_metadata(reserved_path).expect("reserved entry remains");
            assert_eq!(metadata.is_symlink(), symlink_staging);
            assert_eq!(metadata.is_dir(), !symlink_staging);
        }
    }
}
