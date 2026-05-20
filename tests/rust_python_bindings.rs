use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use pyo3::ffi::c_str;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

struct FixtureDirectory {
    path: PathBuf,
}

impl FixtureDirectory {
    fn new() -> Self {
        let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("g-python-binding-tests-{}-{fixture_id}", std::process::id()));
        fs::create_dir_all(&path).expect("fixture directory should be created");
        Self { path }
    }

    fn write_file(&self, file_name: &str, contents: &str) -> PathBuf {
        let path = self.path.join(file_name);
        fs::write(&path, contents).expect("fixture file should be written");
        path
    }
}

impl Drop for FixtureDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn haplotypes_bgen_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/bgen/haplotypes.bgen")
}

fn python_site_packages_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/lib/python3.14/site-packages")
}

fn write_minimal_bgen_header(path: &Path) {
    let mut bytes = vec![0_u8; 24];
    bytes[0..4].copy_from_slice(&20_u32.to_le_bytes());
    bytes[4..8].copy_from_slice(&20_u32.to_le_bytes());
    bytes[8..12].copy_from_slice(&0_u32.to_le_bytes());
    bytes[12..16].copy_from_slice(&0_u32.to_le_bytes());
    bytes[16..20].copy_from_slice(b"bgen");
    bytes[20..24].copy_from_slice(&(2_u32 << 2).to_le_bytes());
    fs::write(path, bytes).expect("minimal BGEN fixture should be written");
}

#[test]
#[allow(clippy::too_many_lines)]
fn registered_python_module_exercises_core_bindings() -> PyResult<()> {
    Python::initialize();
    let fixture = FixtureDirectory::new();
    let phenotype_path = fixture
        .write_file("phenotypes.tsv", "FID\tIID\ttrait_a\ttrait_b\tcase\nF1\tI1\t1.0\t2.0\t1\nF2\tI2\t3.0\t4.0\t2\n");
    let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\nF2\tI2\t50\n");
    let sample_path = fixture.write_file("study.sample", "ID_1 ID_2 missing\n0 0 0\nF1 I1 0\nF2 I2 0\n");
    let engine_sample_path =
        fixture.write_file("study4.sample", "ID_1 ID_2 missing\n0 0 0\nF1 I1 0\nF2 I2 0\nF3 I3 0\nF4 I4 0\n");
    let first_loco_path = fixture.write_file("trait_a.loco", "FID_IID F2_I2 F1_I1\n22 0.2 0.1\n");
    let second_loco_path = fixture.write_file("trait_b.loco", "FID_IID F2_I2 F1_I1\n22 2.2 2.1\n");
    let prediction_list_path = fixture.write_file(
        "pred.list",
        &format!("trait_a {}\ntrait_b {}\n", first_loco_path.display(), second_loco_path.display()),
    );
    let no_sample_bgen_path = fixture.path.join("no-sample.bgen");
    write_minimal_bgen_header(&no_sample_bgen_path);
    let run_directory = fixture.path.join("run");
    let chunks_directory = run_directory.join("chunks");
    fs::create_dir_all(&chunks_directory).expect("chunk directory should be created");
    fs::write(run_directory.join("run_manifest.json"), "{}\n").expect("manifest should be initialized");

    Python::attach(|py| {
        let module = PyModule::new(py, "_core_test")?;
        _core::python::register_module(&module)?;

        let globals = PyDict::new(py);
        globals.set_item("_core", &module)?;
        globals.set_item("bgen_path", haplotypes_bgen_path().to_string_lossy().as_ref())?;
        globals.set_item("phenotype_path", phenotype_path.to_string_lossy().as_ref())?;
        globals.set_item("covariate_path", covariate_path.to_string_lossy().as_ref())?;
        globals.set_item("sample_path", sample_path.to_string_lossy().as_ref())?;
        globals.set_item("engine_sample_path", engine_sample_path.to_string_lossy().as_ref())?;
        globals.set_item("prediction_list_path", prediction_list_path.to_string_lossy().as_ref())?;
        globals.set_item("no_sample_bgen_path", no_sample_bgen_path.to_string_lossy().as_ref())?;
        globals.set_item("run_directory", run_directory.to_string_lossy().as_ref())?;
        globals.set_item("chunks_directory", chunks_directory.to_string_lossy().as_ref())?;
        globals.set_item("site_packages_path", python_site_packages_path().to_string_lossy().as_ref())?;

        py.run(
            c_str!(
                r#"
import sys
import os
sys.path.insert(0, site_packages_path)
import numpy as np

assert _core.hello_from_bin() == "Hello from g!"
chunks = _core.plan_genotype_chunks(12, 5, [0, 3, 9, 12], None, [5])
assert [(chunk.variant_start_index, chunk.variant_stop_index) for chunk in chunks] == [(0, 3), (3, 5), (9, 10), (10, 12)]

aligned = _core.align_sample_data(
    np.array([0, 1], dtype=np.int64),
    ["F1", "F2"],
    ["I1", "I2"],
    phenotype_path,
    "case",
    covariate_path,
    ["age"],
    True,
    "fid_iid",
)
assert aligned.sample_indices.tolist() == [0, 1]
assert aligned.family_identifiers == ["F1", "F2"]
assert aligned.individual_identifiers == ["I1", "I2"]
assert aligned.phenotype_name == "case"
assert aligned.phenotype_vector.tolist() == [0.0, 1.0]
assert aligned.covariate_names == ["intercept", "age"]
assert aligned.covariate_matrix.shape == (2, 2)
assert aligned.is_binary_trait is True

sample_file_aligned = _core.align_sample_data_from_sample_file(
    sample_path,
    2,
    phenotype_path,
    "trait_a",
    covariate_path,
    ["age"],
    False,
    "fid_iid",
)
assert sample_file_aligned.sample_indices.tolist() == [0, 1]
assert sample_file_aligned.phenotype_vector.tolist() == [1.0, 3.0]
sample_file_multi_aligned = _core.align_multi_sample_data_from_sample_file(
    sample_path,
    2,
    phenotype_path,
    ["trait_a", "trait_b"],
    None,
    None,
    False,
    "fid_iid",
)
assert sample_file_multi_aligned.phenotype_matrix.shape == (2, 2)

multi_aligned = _core.align_multi_sample_data(
    np.array([0, 1], dtype=np.int64),
    ["F1", "F2"],
    ["I1", "I2"],
    phenotype_path,
    ["trait_a", "trait_b"],
    None,
    None,
    False,
    "fid_iid",
)
assert multi_aligned.sample_indices.tolist() == [0, 1]
assert multi_aligned.phenotype_names == ["trait_a", "trait_b"]
assert multi_aligned.phenotype_matrix.shape == (2, 2)
assert multi_aligned.covariate_names == ["intercept"]
assert multi_aligned.covariate_matrix.shape == (2, 1)
assert multi_aligned.is_binary_trait is False

prediction_source = _core.RegeniePredictionSource.from_native_aligned_sample_data(
    prediction_list_path,
    "trait_a",
    aligned,
    "fid_iid",
)
assert np.allclose(prediction_source.get_chromosome_predictions("chr22"), [0.1, 0.2])
direct_prediction_source = _core.RegeniePredictionSource(
    prediction_list_path,
    "trait_a",
    ["F1", "F2"],
    ["I1", "I2"],
    "fid_iid",
)
assert np.allclose(direct_prediction_source.get_chromosome_predictions("22"), [0.1, 0.2])
multi_prediction_source = _core.MultiRegeniePredictionSource.from_native_multi_aligned_sample_data(
    prediction_list_path,
    multi_aligned,
    "fid_iid",
)
assert np.allclose(multi_prediction_source.get_chromosome_predictions("22"), [[0.1, 0.2], [2.1, 2.2]])
direct_multi_prediction_source = _core.MultiRegeniePredictionSource(
    prediction_list_path,
    ["trait_a", "trait_b"],
    ["F1", "F2"],
    ["I1", "I2"],
    "fid_iid",
)
assert np.allclose(direct_multi_prediction_source.get_chromosome_predictions("22"), [[0.1, 0.2], [2.1, 2.2]])
try:
    direct_prediction_source.get_chromosome_predictions("1")
    raise AssertionError("missing chromosome should fail")
except ValueError:
    pass
try:
    _core.align_sample_data(
        np.array([0], dtype=np.int64),
        ["F1"],
        ["I1"],
        phenotype_path,
        "trait_a",
        None,
        None,
        False,
        "bad-mode",
    )
    raise AssertionError("invalid sample key mode should fail")
except ValueError:
    pass
try:
    _core.RegeniePredictionSource(
        os.path.join(os.path.dirname(run_directory), "missing.list"),
        "trait_a",
        ["F1"],
        ["I1"],
        "fid_iid",
    )
    raise AssertionError("missing prediction list should fail")
except FileNotFoundError:
    pass
missing_loco_list = os.path.join(os.path.dirname(run_directory), "missing-loco.list")
with open(missing_loco_list, "w", encoding="utf-8") as prediction_list_file:
    prediction_list_file.write("trait_a missing.loco\n")
try:
    _core.RegeniePredictionSource(missing_loco_list, "trait_a", ["F1"], ["I1"], "fid_iid")
    raise AssertionError("missing LOCO file should fail")
except FileNotFoundError:
    pass

engine = _core.Regenie2RunEngine(bgen_path, 2)
assert engine.sample_count == 4
assert engine.variant_count == 4
sample_identifiers = engine.sample_identifiers()
if engine.contains_embedded_samples:
    assert len(sample_identifiers) == engine.sample_count
else:
    assert sample_identifiers == []
assert engine.chromosome_boundary_indices() == [0, 4]
chromosome, variant_identifiers, position, allele_one, allele_two = engine.variant_metadata_slice(0, 1)
assert len(chromosome) == 1
assert len(variant_identifiers) == 1
assert len(position) == 1
assert len(allele_one) == 1
assert len(allele_two) == 1
try:
    engine.align_sample_data(None, phenotype_path, "trait_a")
except ValueError:
    pass
engine_sample_aligned = engine.align_sample_data(
    engine_sample_path,
    phenotype_path,
    "trait_a",
    covariate_path,
    ["age"],
    False,
    "fid_iid",
)
assert engine_sample_aligned.sample_indices.tolist() == [0, 1]
engine_multi_aligned = engine.align_multi_sample_data(
    engine_sample_path,
    phenotype_path,
    ["trait_a", "trait_b"],
    None,
    None,
    False,
    "fid_iid",
)
assert engine_multi_aligned.family_identifiers == ["F1", "F2"]
assert engine_multi_aligned.individual_identifiers == ["I1", "I2"]
try:
    engine.align_multi_sample_data(None, phenotype_path, ["trait_a"])
except ValueError:
    pass
try:
    engine.validate_trusted_no_missing_diploid()
    raise AssertionError("phased fixture should not validate as trusted diploid")
except ValueError:
    pass
no_sample_engine = _core.Regenie2RunEngine(no_sample_bgen_path, 1)
try:
    no_sample_engine.align_sample_data(None, phenotype_path, "trait_a")
    raise AssertionError("BGEN without embedded samples should require a sample file")
except ValueError:
    pass
try:
    no_sample_engine.align_multi_sample_data(None, phenotype_path, ["trait_a"])
    raise AssertionError("BGEN without embedded samples should require a sample file for multi alignment")
except ValueError:
    pass

class RecordingCallback:
    def __init__(self, writer=None):
        self.writer = writer
        self.row_major_shapes = []
        self.variant_major_shapes = []
        self.free_row_major_buffers = []
        self.free_variant_major_buffers = []

    def acquire_dosage_buffer(self, sample_count, variant_count):
        if self.free_row_major_buffers:
            return self.free_row_major_buffers.pop()
        return np.empty((sample_count, variant_count), dtype=np.float32, order="C")

    def compute_preprocessed_dosage_chunk(self, metadata, genotype_matrix, chunk_stats):
        self.row_major_shapes.append((metadata.variant_start_index, genotype_matrix.shape))
        assert len(metadata.chromosome) == genotype_matrix.shape[1]
        assert len(metadata.variant_identifiers) == genotype_matrix.shape[1]
        assert metadata.position.shape == (genotype_matrix.shape[1],)
        assert len(metadata.allele_one) == genotype_matrix.shape[1]
        assert len(metadata.allele_two) == genotype_matrix.shape[1]
        assert chunk_stats.allele_one_frequency.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.observation_count.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.dosage_square_sum.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.imputed_dosage_square_sum.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.info_score.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.minor_allele_count.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.zero_count.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.nonzero_count.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.is_sparse_candidate.shape == (genotype_matrix.shape[1],)
        assert chunk_stats.is_rare_sparse_firth_candidate.shape == (genotype_matrix.shape[1],)
        assert isinstance(chunk_stats.has_missing_values, bool)
        if self.writer is not None:
            variant_count = metadata.variant_stop_index - metadata.variant_start_index
            self.writer.write_regenie2_native_chunk(
                metadata,
                chunk_stats,
                np.full(variant_count, 0.1, dtype=np.float32),
                np.full(variant_count, 0.01, dtype=np.float32),
                np.full(variant_count, 10.0, dtype=np.float32),
                np.full(variant_count, 5.0, dtype=np.float32),
                None,
            )
        self.free_row_major_buffers.append(genotype_matrix)

    def acquire_variant_major_dosage_buffer(self, variant_count, sample_count):
        if self.free_variant_major_buffers:
            return self.free_variant_major_buffers.pop()
        return np.empty((variant_count, sample_count), dtype=np.float32, order="C")

    def compute_preprocessed_variant_major_dosage_chunk(self, metadata, genotype_matrix, chunk_stats):
        self.variant_major_shapes.append((metadata.variant_start_index, genotype_matrix.shape))
        assert chunk_stats.allele_one_frequency.shape == (genotype_matrix.shape[0],)
        self.free_variant_major_buffers.append(genotype_matrix)

callback = RecordingCallback()
assert engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback) == 2
assert callback.row_major_shapes == [(0, (4, 2)), (2, (4, 2))]
resume_callback = RecordingCallback()
assert engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), resume_callback, [0]) == 1
assert resume_callback.row_major_shapes == [(2, (4, 2))]
assert engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), callback) == 2
assert callback.variant_major_shapes == [(0, (2, 4)), (2, (2, 4))]
profile = engine.profile_snapshot()
assert "variant_decode_count" in profile
engine.reset_profile()
assert engine.profile_snapshot()["variant_decode_count"] == 0

class BadRowMajorShapeCallback:
    def acquire_dosage_buffer(self, sample_count, variant_count):
        return np.empty((sample_count, variant_count + 1), dtype=np.float32)

    def compute_preprocessed_dosage_chunk(self, metadata, genotype_matrix, chunk_stats):
        raise AssertionError("bad buffer shape should fail before compute")

try:
    engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), BadRowMajorShapeCallback())
    raise AssertionError("bad row-major buffer shape should fail")
except ValueError:
    pass

class BadVariantMajorShapeCallback:
    def acquire_variant_major_dosage_buffer(self, variant_count, sample_count):
        return np.empty((variant_count + 1, sample_count), dtype=np.float32)

    def compute_preprocessed_variant_major_dosage_chunk(self, metadata, genotype_matrix, chunk_stats):
        raise AssertionError("bad variant-major buffer shape should fail before compute")

try:
    engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), BadVariantMajorShapeCallback())
    raise AssertionError("bad variant-major buffer shape should fail")
except ValueError:
    pass

class FortranRowMajorBufferCallback:
    def acquire_dosage_buffer(self, sample_count, variant_count):
        return np.empty((sample_count, variant_count), dtype=np.float32, order="F")

    def compute_preprocessed_dosage_chunk(self, metadata, genotype_matrix, chunk_stats):
        raise AssertionError("non-C row-major buffer should fail before compute")

try:
    engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), FortranRowMajorBufferCallback())
    raise AssertionError("non-C row-major buffer should fail")
except ValueError:
    pass

class FortranVariantMajorBufferCallback:
    def acquire_variant_major_dosage_buffer(self, variant_count, sample_count):
        return np.empty((variant_count, sample_count), dtype=np.float32, order="F")

    def compute_preprocessed_variant_major_dosage_chunk(self, metadata, genotype_matrix, chunk_stats):
        raise AssertionError("non-C variant-major buffer should fail before compute")

try:
    engine.run_bgen_variant_major_dosage_buffered_chunks(np.arange(4, dtype=np.int64), FortranVariantMajorBufferCallback())
    raise AssertionError("non-C variant-major buffer should fail")
except ValueError:
    pass

writer = _core.OutputWriterSession(
    run_directory,
    chunks_directory,
    "regenie2_linear",
    1,
    1,
    False,
    1,
    "none",
    True,
)
writer_callback = RecordingCallback(writer)
assert engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), writer_callback) == 2
assert writer.finish() is None
assert _core.scan_committed_chunk_identifiers(chunks_directory) == [0, 2]
assert _core.finalize_output_run_chunks(run_directory, chunks_directory, "regenie2_linear").endswith("final.parquet")

interrupted_run_directory = os.path.join(os.path.dirname(run_directory), "interrupted")
interrupted_chunks_directory = os.path.join(interrupted_run_directory, "chunks")
os.makedirs(interrupted_chunks_directory, exist_ok=True)
with open(os.path.join(interrupted_run_directory, "run_manifest.json"), "w", encoding="utf-8") as manifest_file:
    manifest_file.write("{}\n")
interrupted_writer = _core.OutputWriterSession(
    interrupted_run_directory,
    interrupted_chunks_directory,
    "regenie2_linear",
    1,
    1,
    False,
    1,
    "none",
    False,
)
interrupted_callback = RecordingCallback(interrupted_writer)
assert engine.run_bgen_dosage_buffered_chunks(np.arange(4, dtype=np.int64), interrupted_callback) == 2
interrupted_writer.finish_interrupted("SIGINT")

abort_run_directory = os.path.join(os.path.dirname(run_directory), "abort")
abort_chunks_directory = os.path.join(abort_run_directory, "chunks")
os.makedirs(abort_chunks_directory, exist_ok=True)
abort_writer = _core.OutputWriterSession(
    abort_run_directory,
    abort_chunks_directory,
    "regenie2_linear",
    1,
    1,
    False,
    1,
    "none",
    False,
)
abort_writer.abort()

try:
    _core.OutputWriterSession(run_directory, chunks_directory, "regenie2_linear", 0, 1, False, 1, "none", False)
    raise AssertionError("invalid writer thread count should fail")
except ValueError:
    pass
try:
    _core.validate_strict_manifest_chunks(chunks_directory, "{}")
    raise AssertionError("invalid manifest should fail")
except ValueError:
    pass

try:
    _core.configure_bgen_decode_tile_variant_count(0)
    raise AssertionError("invalid tile size should fail")
except ValueError:
    pass
try:
    _core.configure_rayon_global_thread_pool(0)
    raise AssertionError("invalid rayon thread count should fail")
except ValueError:
    pass
try:
    _core.configure_rayon_global_thread_pool(1)
    _core.configure_rayon_global_thread_pool(1)
except RuntimeError:
    pass
"#
            ),
            Some(&globals),
            None,
        )
    })
}
