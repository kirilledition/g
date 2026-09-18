use std::path::PathBuf;
use std::sync::Arc;

use super::{
    PRIMARY_PHENOTYPE, TestDirectory, header, initialize_manager, metadata_store, planned_run_directories, run_plan,
    single_chunk_plan, test_chunk, test_inputs,
};
use crate::{OutputManager, write_regenie2_multi_trait_chunk_f32};

#[test]
fn output_directory_resolution_preserves_links_before_parent_components() {
    let directory = TestDirectory::new("physical-output-paths");
    let target = directory.path.join("target/nested");
    std::fs::create_dir_all(&target).expect("link target exists");
    std::os::unix::fs::symlink(&target, directory.path.join("link")).expect("ancestor link exists");
    for suffix in ["link/../result.run", "missing/../link/../result.run"] {
        assert_eq!(
            crate::paths::resolve_output_directory(&directory.path.join(suffix)).expect("output path resolves"),
            directory.path.canonicalize().expect("fixture canonicalizes").join("target/result.run")
        );
    }
    assert!(!directory.path.join("missing").exists());
    assert!(!directory.path.join("target/result.run").exists());
}

#[test]
fn output_directory_resolution_rejects_dangling_links_and_non_directories() {
    let directory = TestDirectory::new("invalid-output-paths");
    directory.write("file", b"sentinel");
    std::os::unix::fs::symlink("missing", directory.path.join("dangling")).expect("dangling link exists");
    for suffix in
        ["file", "file/child.run", "file/../child.run", "dangling", "dangling/child.run", "dangling/../child.run"]
    {
        assert!(crate::paths::resolve_output_directory(&directory.path.join(suffix)).is_err(), "accepted {suffix}");
    }
    assert_eq!(std::fs::read(directory.path.join("file")).expect("sentinel is readable"), b"sentinel");
    assert!(!directory.path.join("missing").exists());
    assert!(!directory.path.join("child.run").exists());
}

#[test]
fn manager_rejects_equal_and_nested_physical_run_directories_without_mutation() {
    for second_name in ["missing/../first.run", "first.run/nested.run", "link.run", "linked-parent/nested.run"] {
        let directory = TestDirectory::new("aliased-run-directories");
        let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
        let inputs = test_inputs(&directory, &phenotype_names);
        let mut plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
        let plan_value = Arc::get_mut(&mut plan).expect("test plan has one owner");
        plan_value.phenotype_runs[0].output_directory_name = "first.run".to_string();
        plan_value.phenotype_runs[1].output_directory_name = second_name.to_string();
        let output_root = PathBuf::from(&plan_value.output.output_run_root);
        let first_directory = output_root.join("first.run");
        std::fs::create_dir_all(&first_directory).expect("empty first run exists");
        std::os::unix::fs::symlink(&first_directory, output_root.join("link.run")).expect("run alias exists");
        std::os::unix::fs::symlink(&first_directory, output_root.join("linked-parent")).expect("ancestor alias exists");
        let error = OutputManager::open(plan, "# rejected physical alias\n".to_string())
            .err()
            .expect("equal or nested physical run paths must fail planning");
        assert!(error.to_string().contains("resolve to equal or nested output directories"), "{error}");
        assert_eq!(std::fs::read_dir(&first_directory).expect("first run is readable").count(), 0);
        assert!(!output_root.join("missing").exists());
    }
}

#[test]
fn manager_rejects_shared_or_nested_relocated_parts_before_resume_mutation() {
    for nested_parts in [false, true] {
        let directory = TestDirectory::new("aliased-parts-directories");
        let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
        let inputs = test_inputs(&directory, &phenotype_names);
        let initial_plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
        let run_directories = planned_run_directories(&initial_plan);
        let manager = initialize_manager(initial_plan, &inputs, &phenotype_names, &single_chunk_plan(0..1));
        manager.abort().expect("initial manager closes");
        let shared_parts = directory.path.join("relocated-parts");
        let second_parts = if nested_parts { shared_parts.join("nested") } else { shared_parts.clone() };
        std::fs::create_dir_all(&second_parts).expect("relocated directories exist");
        let original_manifests = run_directories
            .iter()
            .map(|run_directory| std::fs::read(run_directory.join("run_manifest.json")).expect("manifest is readable"))
            .collect::<Vec<_>>();
        let original_configurations = run_directories
            .iter()
            .map(|run_directory| {
                std::fs::read(run_directory.join("effective_config.toml")).expect("configuration is readable")
            })
            .collect::<Vec<_>>();
        for (run_directory, parts_target) in run_directories.iter().zip([&shared_parts, &second_parts]) {
            std::fs::remove_dir(run_directory.join("parts")).expect("original empty parts directory is removed");
            std::os::unix::fs::symlink(parts_target, run_directory.join("parts")).expect("relocated parts link exists");
        }
        let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 2);
        let error = OutputManager::open(resume_plan, "# rejected shared parts\n".to_string())
            .err()
            .expect("equal or nested physical parts paths must fail planning");
        assert!(error.to_string().contains("resolve to equal or nested output directories"), "{error}");
        for (index, run_directory) in run_directories.iter().enumerate() {
            assert_eq!(std::fs::read(run_directory.join("run_manifest.json")).unwrap(), original_manifests[index]);
            assert_eq!(
                std::fs::read(run_directory.join("effective_config.toml")).unwrap(),
                original_configurations[index]
            );
        }
        assert_eq!(std::fs::read_dir(&second_parts).expect("relocated parts remain readable").count(), 0);
    }
}

#[test]
fn manager_rechecks_physical_parts_ownership_before_initialization() {
    let directory = TestDirectory::new("parts-alias-after-planning");
    let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
    let inputs = test_inputs(&directory, &phenotype_names);
    let plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
    let run_directories = planned_run_directories(&plan);
    let mut manager = OutputManager::open(plan, "# rejected changed paths\n".to_string()).expect("fresh manager plans");
    let shared_parts = directory.path.join("shared-parts");
    std::fs::create_dir(&shared_parts).expect("shared parts directory exists");
    for run_directory in &run_directories {
        std::fs::create_dir_all(run_directory).expect("run directory exists");
        std::os::unix::fs::symlink(&shared_parts, run_directory.join("parts")).expect("shared parts alias exists");
    }
    let error = manager
        .initialize(
            vec![header(PRIMARY_PHENOTYPE, &inputs, 1), header("trait_beta", &inputs, 1)],
            &single_chunk_plan(0..1),
            false,
        )
        .expect_err("new shared parts alias must fail initialization");
    assert!(error.to_string().contains("changed after planning"), "{error}");
    for run_directory in &run_directories {
        assert!(!run_directory.join("run_manifest.json").exists());
        assert!(!run_directory.join("effective_config.toml").exists());
    }
    assert_eq!(std::fs::read_dir(&shared_parts).expect("shared parts remain readable").count(), 0);
}

#[test]
fn manager_rejects_parts_aliasing_its_run_or_another_run() {
    for parts_target_is_other_run in [false, true] {
        let directory = TestDirectory::new("parts-run-alias");
        let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
        let inputs = test_inputs(&directory, &phenotype_names);
        let plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
        let run_directories = planned_run_directories(&plan);
        for run_directory in &run_directories {
            std::fs::create_dir_all(run_directory).expect("run directory exists");
        }
        let target_index = usize::from(parts_target_is_other_run);
        std::os::unix::fs::symlink(&run_directories[target_index], run_directories[0].join("parts"))
            .expect("parts aliases a run directory");
        let error = OutputManager::open(plan, "# rejected run-parts alias\n".to_string())
            .err()
            .expect("parts/run alias must fail planning");
        assert!(error.to_string().contains("Phenotype output"), "{error}");
        assert!(run_directories.iter().all(|run_directory| !run_directory.join("run_manifest.json").exists()));
    }
}

#[test]
fn manager_accepts_disjoint_relocated_parts_during_resume() {
    let directory = TestDirectory::new("disjoint-relocated-parts");
    let phenotype_names = [PRIMARY_PHENOTYPE, "trait_beta"];
    let inputs = test_inputs(&directory, &phenotype_names);
    let initial_plan = run_plan(&directory, &inputs, &phenotype_names, false, 2);
    let run_directories = planned_run_directories(&initial_plan);
    let planned_ranges = single_chunk_plan(0..1);
    let manager = initialize_manager(initial_plan, &inputs, &phenotype_names, &planned_ranges);
    manager.abort().expect("initial manager closes");
    let relocated_paths = [directory.path.join("relocated-alpha"), directory.path.join("relocated-beta")];
    for (run_directory, relocated_path) in run_directories.iter().zip(&relocated_paths) {
        let parts_directory = run_directory.join("parts");
        std::fs::rename(&parts_directory, relocated_path).expect("parts directory relocates");
        std::os::unix::fs::symlink(relocated_path, &parts_directory).expect("unique parts alias exists");
        std::fs::write(relocated_path.join("part_000000000.parquet.tmp"), b"abandoned staging")
            .expect("relocated staging fixture exists");
    }
    let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 2);
    let manager = initialize_manager(resume_plan, &inputs, &phenotype_names, &planned_ranges);
    manager.abort().expect("disjoint relocated parts resume");
    for (run_directory, relocated_path) in run_directories.iter().zip(&relocated_paths) {
        assert!(std::fs::symlink_metadata(run_directory.join("parts")).unwrap().is_symlink());
        assert_eq!(std::fs::read_dir(relocated_path).expect("relocated parts remain readable").count(), 0);
    }
}

#[test]
fn missing_parent_components_cannot_hide_populated_output_from_fresh_planning() {
    let directory = TestDirectory::new("populated-missing-parent-alias");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let initial_plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let manager = initialize_manager(initial_plan, &inputs, &phenotype_names, &single_chunk_plan(0..1));
    let delivery =
        manager.delivery_state_for_phenotypes(&[PRIMARY_PHENOTYPE.to_string()]).expect("initial delivery is available");
    let chunk = test_chunk(&metadata_store(1), 0..1, 1);
    write_regenie2_multi_trait_chunk_f32(&delivery.writer_sessions, None, &chunk.handle, chunk.statistics)
        .expect("initial chunk writes");
    drop(delivery);
    let completed = manager.finish().expect("initial output completes");
    let run_directory = &completed[0].run_directory;
    let artifact_names = ["run_manifest.json", "effective_config.toml", "parts/part_000000000.parquet"];
    let original_bytes = artifact_names
        .iter()
        .map(|name| std::fs::read(run_directory.join(name)).expect("original artifact is readable"))
        .collect::<Vec<_>>();
    let mut alias_plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    Arc::get_mut(&mut alias_plan).expect("test plan has one owner").output.output_run_root =
        directory.path.join("missing/../results").display().to_string();
    let error = OutputManager::open(alias_plan, "# must not replace existing output\n".to_string())
        .err()
        .expect("normalized populated output must reject fresh planning");
    assert!(error.to_string().contains("already exists and is not empty"), "{error}");
    assert!(!directory.path.join("missing").exists());
    for (artifact_name, original_bytes) in artifact_names.iter().zip(original_bytes) {
        assert_eq!(std::fs::read(run_directory.join(artifact_name)).unwrap(), original_bytes);
    }
}

#[test]
fn fresh_output_uses_resolved_paths_without_creating_cancelled_components() {
    let directory = TestDirectory::new("resolved-fresh-output");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let mut plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    Arc::get_mut(&mut plan).expect("test plan has one owner").output.output_run_root =
        directory.path.join("missing/../results").display().to_string();
    let manager = initialize_manager(plan, &inputs, &phenotype_names, &single_chunk_plan(0..1));
    let completed = manager.finish().expect("fresh resolved output completes");
    let expected_directory = directory
        .path
        .canonicalize()
        .expect("test directory canonicalizes")
        .join("results/phenotype_0000_trait_alpha.regenie2_binary.run");
    assert_eq!(completed[0].run_directory, expected_directory);
    assert_eq!(completed[0].parts_directory, expected_directory.join("parts"));
    assert!(expected_directory.join("run_manifest.json").is_file());
    assert!(!directory.path.join("missing").exists());
}

#[test]
fn resumed_output_rejects_retargeted_parts_links_before_cleanup_or_writes() {
    let directory = TestDirectory::new("retargeted-resume-parts");
    let phenotype_names = [PRIMARY_PHENOTYPE];
    let inputs = test_inputs(&directory, &phenotype_names);
    let initial_plan = run_plan(&directory, &inputs, &phenotype_names, false, 1);
    let run_directory = planned_run_directories(&initial_plan).remove(0);
    let planned_ranges = single_chunk_plan(0..1);
    let manager = initialize_manager(initial_plan, &inputs, &phenotype_names, &planned_ranges);
    manager.abort().expect("initial manager closes");
    let parts_path = run_directory.join("parts");
    let original_parts = directory.path.join("original-parts");
    let alternate_parts = directory.path.join("alternate-parts");
    std::fs::rename(&parts_path, &original_parts).expect("original parts relocate");
    std::fs::create_dir(&alternate_parts).expect("alternate parts exist");
    std::os::unix::fs::symlink(&original_parts, &parts_path).expect("original parts link exists");
    for parts_directory in [&original_parts, &alternate_parts] {
        std::fs::write(parts_directory.join("part_000000000.parquet.tmp"), b"preserve staging")
            .expect("staging fixture is written");
    }
    let manifest_bytes = std::fs::read(run_directory.join("run_manifest.json")).expect("manifest is readable");
    let configuration_bytes =
        std::fs::read(run_directory.join("effective_config.toml")).expect("configuration is readable");
    let resume_plan = run_plan(&directory, &inputs, &phenotype_names, true, 1);
    let mut manager = OutputManager::open(resume_plan, "# retargeted resume\n".to_string()).expect("resume plans");
    std::fs::remove_file(&parts_path).expect("original link is removed");
    std::os::unix::fs::symlink(&alternate_parts, &parts_path).expect("parts link retargets");
    let error = manager
        .initialize(vec![header(PRIMARY_PHENOTYPE, &inputs, 1)], &planned_ranges, false)
        .expect_err("retargeted parts must reject initialization");
    assert!(error.to_string().contains("changed after planning"), "{error}");
    drop(manager);
    assert_eq!(std::fs::read(run_directory.join("run_manifest.json")).unwrap(), manifest_bytes);
    assert_eq!(std::fs::read(run_directory.join("effective_config.toml")).unwrap(), configuration_bytes);
    for parts_directory in [&original_parts, &alternate_parts] {
        assert_eq!(std::fs::read(parts_directory.join("part_000000000.parquet.tmp")).unwrap(), b"preserve staging");
    }
}
