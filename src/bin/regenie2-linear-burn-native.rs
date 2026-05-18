use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::time::Instant;

use _core::genotype::bgen::BgenReaderCore;
use _core::genotype::common::VariantMetadataColumns;
use _core::genotype::planner;
use _core::native::alignment;
#[cfg(feature = "cuda-kernel")]
use _core::native::cuda_linear;
use _core::native::linear;
use _core::output::{NativeOutputWriterSession, NativeRegenieStep2Chunk};
use _core::regenie;
use clap::{Parser, ValueEnum};
use serde::Serialize;

const DEFAULT_CHUNK_SIZE: usize = 65_536;

#[derive(Clone, Debug, ValueEnum)]
enum NativeBackend {
    Wgpu,
    Cuda,
    CudaKernel,
}

#[derive(Clone, Debug, Eq, PartialEq, ValueEnum)]
enum OutputMode {
    FullParquet,
    ChunksOnly,
    NoOutput,
    DecodeOnly,
}

#[derive(Debug, Parser)]
#[command(name = "regenie2-linear-burn-native")]
struct Cli {
    #[arg(long)]
    bgen: PathBuf,
    #[arg(long)]
    sample: Option<PathBuf>,
    #[arg(long)]
    pheno: PathBuf,
    #[arg(long = "pheno-name")]
    pheno_name: String,
    #[arg(long)]
    covar: Option<PathBuf>,
    #[arg(long = "covar-names", value_delimiter = ',')]
    covar_names: Vec<String>,
    #[arg(long)]
    pred: PathBuf,
    #[arg(long)]
    out: PathBuf,
    #[arg(long = "chunk-size", default_value_t = DEFAULT_CHUNK_SIZE)]
    chunk_size: usize,
    #[arg(long = "variant-limit")]
    variant_limit: Option<usize>,
    #[arg(long, value_enum, default_value_t = NativeBackend::Wgpu)]
    backend: NativeBackend,
    #[arg(long = "cuda-block-size", default_value_t = 256)]
    cuda_block_size: u32,
    #[arg(long = "writer-threads", default_value_t = 1)]
    writer_threads: usize,
    #[arg(long = "writer-queue-depth", default_value_t = 1)]
    writer_queue_depth: usize,
    #[arg(long = "output-mode", value_enum, default_value_t = OutputMode::FullParquet)]
    output_mode: OutputMode,
    #[arg(long = "report-json")]
    report_json: Option<PathBuf>,
}

#[derive(Debug, Serialize)]
struct NativeRunReport {
    backend: String,
    output_mode: String,
    chunk_size: usize,
    variant_limit: Option<usize>,
    sample_count: usize,
    variant_count: usize,
    processed_variant_count: usize,
    processed_chunk_count: usize,
    checksum: f64,
    output_run_directory: String,
    final_parquet: Option<String>,
    cuda_block_size: Option<u32>,
    stage_seconds: BTreeMap<String, f64>,
    total_wall_seconds: f64,
}

struct StageTimer {
    start_time: Instant,
    stage_seconds: BTreeMap<String, f64>,
}

impl StageTimer {
    fn new() -> Self {
        Self { start_time: Instant::now(), stage_seconds: BTreeMap::new() }
    }

    fn measure<T, E>(&mut self, stage_name: &str, operation: impl FnOnce() -> Result<T, E>) -> Result<T, E> {
        let start_time = Instant::now();
        let result = operation();
        self.add_seconds(stage_name, start_time.elapsed().as_secs_f64());
        result
    }

    fn add_seconds(&mut self, stage_name: &str, seconds: f64) {
        *self.stage_seconds.entry(stage_name.to_string()).or_insert(0.0) += seconds;
    }

    fn total_wall_seconds(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64()
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    match cli.backend {
        NativeBackend::Wgpu => run_wgpu(cli)?,
        NativeBackend::Cuda => run_cuda(cli)?,
        NativeBackend::CudaKernel => run_cuda_kernel(cli)?,
    };
    Ok(())
}

fn run_wgpu(cli: Cli) -> Result<NativeRunReport, Box<dyn std::error::Error>> {
    type Backend = burn::backend::Wgpu;
    let device = Default::default();
    run_with_backend::<Backend>(cli, &device, "wgpu")
}

fn run_cuda(cli: Cli) -> Result<NativeRunReport, Box<dyn std::error::Error>> {
    #[cfg(feature = "burn-cuda")]
    {
        type Backend = burn::backend::Cuda;
        let device = Default::default();
        return run_with_backend::<Backend>(cli, &device, "cuda");
    }
    #[cfg(not(feature = "burn-cuda"))]
    {
        let _ = cli;
        Err("The native binary was not built with the burn-cuda feature.".into())
    }
}

fn run_cuda_kernel(cli: Cli) -> Result<NativeRunReport, Box<dyn std::error::Error>> {
    #[cfg(feature = "cuda-kernel")]
    {
        return run_with_cuda_kernel(cli);
    }
    #[cfg(not(feature = "cuda-kernel"))]
    {
        let _ = cli;
        Err("The native binary was not built with the cuda-kernel feature.".into())
    }
}

fn run_with_backend<Backend: burn::tensor::backend::Backend>(
    cli: Cli,
    device: &Backend::Device,
    backend_name: &str,
) -> Result<NativeRunReport, Box<dyn std::error::Error>> {
    let report_json_path = cli.report_json.clone();
    let mut timer = StageTimer::new();
    let reader = timer.measure("startup_config_bgen_open", || BgenReaderCore::open(&cli.bgen, true))?;
    let aligned_sample_data = timer.measure("sample_alignment", || {
        alignment::load_bgen_aligned_sample_data(
            &reader,
            cli.sample.as_deref(),
            &cli.pheno,
            &cli.pheno_name,
            cli.covar.as_deref(),
            &cli.covar_names,
        )
    })?;
    let sample_count = aligned_sample_data.sample_indices.len();
    let sample_indices = aligned_sample_data.sample_indices.clone();
    timer.measure("bgen_sample_selection", || reader.prepare_sample_selection(&sample_indices))?;

    let prediction_source = if cli.output_mode == OutputMode::DecodeOnly {
        None
    } else {
        Some(timer.measure("prediction_load", || {
            regenie::PredictionSource::load(
                &cli.pred,
                &cli.pheno_name,
                &aligned_sample_data.family_identifiers,
                &aligned_sample_data.individual_identifiers,
            )
        })?)
    };
    let linear_state = if cli.output_mode == OutputMode::DecodeOnly {
        None
    } else {
        Some(timer.measure("linear_state_prepare", || {
            linear::prepare_linear_state(
                &aligned_sample_data.covariate_values,
                &aligned_sample_data.phenotype_values,
                aligned_sample_data.covariate_count,
            )
        })?)
    };

    let run_directory = build_run_directory(&cli.out);
    let chunks_directory = run_directory.join("chunks");
    let writer_session = match cli.output_mode {
        OutputMode::FullParquet | OutputMode::ChunksOnly => Some(timer.measure("output_writer_prepare", || {
            NativeOutputWriterSession::new(
                run_directory.clone(),
                chunks_directory.clone(),
                "regenie2_linear".to_string(),
                cli.writer_threads,
                cli.writer_queue_depth,
                cli.output_mode == OutputMode::FullParquet,
            )
        })?),
        OutputMode::NoOutput | OutputMode::DecodeOnly => None,
    };

    let chunk_specs = planner::plan_chromosome_homogeneous_chunks(
        reader.variant_count(),
        cli.chunk_size,
        cli.variant_limit,
        &reader.chromosome_boundary_indices(),
        &BTreeSet::new(),
    )?;

    let mut current_chromosome = String::new();
    let mut current_chromosome_state: Option<linear::LinearDeviceChromosomeState<Backend>> = None;
    let mut processed_variant_count = 0usize;
    let mut checksum = 0.0_f64;
    for chunk_spec in &chunk_specs {
        let variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
        let mut genotype_values = vec![0.0_f32; sample_count * variant_count];
        let decode_start_time = Instant::now();
        let chunk_stats = reader.read_preprocessed_dosage_f32_into_address_prepared(
            chunk_spec.variant_start_index,
            chunk_spec.variant_stop_index,
            genotype_values.as_mut_ptr() as usize,
            genotype_values.len(),
        )?;
        timer.add_seconds("bgen_decode_preprocess", decode_start_time.elapsed().as_secs_f64());

        let metadata = timer.measure("metadata_load", || {
            reader.variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
        })?;
        let metadata = convert_variant_metadata(metadata);
        if cli.output_mode == OutputMode::DecodeOnly {
            checksum += chunk_stats
                .allele_one_frequency
                .iter()
                .chain(genotype_values.iter().take(16))
                .map(|value| f64::from(value.to_bits()))
                .sum::<f64>();
            processed_variant_count += variant_count;
            continue;
        }

        let chromosome = metadata.chromosome.first().cloned().unwrap_or_default();
        if current_chromosome_state.is_none() || chromosome != current_chromosome {
            current_chromosome = chromosome.clone();
            let prediction_values = prediction_source
                .as_ref()
                .ok_or("Prediction source was not loaded.")?
                .chromosome_predictions(&chromosome)?;
            let prepared_linear_state = linear_state.as_ref().ok_or("Linear state was not prepared.")?;
            current_chromosome_state = Some(timer.measure("chromosome_state_prepare", || {
                let chromosome_state =
                    linear::prepare_linear_chromosome_state(prepared_linear_state, prediction_values)?;
                Ok::<_, linear::LinearError>(linear::prepare_linear_device_chromosome_state::<Backend>(
                    &chromosome_state,
                    device,
                ))
            })?);
        }

        let compute_start_time = Instant::now();
        let result = linear::compute_linear_chunk_burn::<Backend>(
            current_chromosome_state.as_ref().ok_or("Chromosome state was not prepared.")?,
            genotype_values,
            variant_count,
            device,
        )?;
        timer.add_seconds("burn_upload_compute_materialize_cpu_pvalue", compute_start_time.elapsed().as_secs_f64());
        for (stage_name, seconds) in &result.timing_seconds {
            timer.add_seconds(stage_name, *seconds);
        }
        checksum += result.checksum;
        if let Some(writer) = writer_session.as_ref() {
            let enqueue_start_time = Instant::now();
            writer.write_regenie2_chunk(NativeRegenieStep2Chunk {
                variant_start_index: i64::try_from(chunk_spec.variant_start_index).unwrap_or(i64::MAX),
                variant_stop_index: i64::try_from(chunk_spec.variant_stop_index).unwrap_or(i64::MAX),
                metadata,
                allele_one_frequency: chunk_stats.allele_one_frequency,
                observation_count: chunk_stats.observation_count,
                beta: result.beta,
                standard_error: result.standard_error,
                chi_squared: result.chi_squared,
                log10_p_value: result.log10_p_value,
                extra_code: None,
            })?;
            timer.add_seconds("output_enqueue", enqueue_start_time.elapsed().as_secs_f64());
        }
        processed_variant_count += variant_count;
    }

    let final_parquet = if let Some(writer) = writer_session.as_ref() {
        timer.measure("parquet_finalization", || writer.finish())?.map(|path| path.display().to_string())
    } else {
        None
    };
    timer.measure("bgen_sample_selection_clear", || reader.clear_prepared_sample_selection())?;
    let total_wall_seconds = timer.total_wall_seconds();
    let report = NativeRunReport {
        backend: backend_name.to_string(),
        output_mode: format!("{:?}", cli.output_mode),
        chunk_size: cli.chunk_size,
        variant_limit: cli.variant_limit,
        sample_count,
        variant_count: reader.variant_count(),
        processed_variant_count,
        processed_chunk_count: chunk_specs.len(),
        checksum,
        output_run_directory: run_directory.display().to_string(),
        final_parquet,
        cuda_block_size: None,
        stage_seconds: timer.stage_seconds,
        total_wall_seconds,
    };
    if let Some(report_path) = report_json_path {
        let report_json = serde_json::to_string_pretty(&report)?;
        std::fs::write(report_path, report_json)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(report)
}

#[cfg(feature = "cuda-kernel")]
fn run_with_cuda_kernel(cli: Cli) -> Result<NativeRunReport, Box<dyn std::error::Error>> {
    let report_json_path = cli.report_json.clone();
    let mut timer = StageTimer::new();
    let reader = timer.measure("startup_config_bgen_open", || BgenReaderCore::open(&cli.bgen, true))?;
    let aligned_sample_data = timer.measure("sample_alignment", || {
        alignment::load_bgen_aligned_sample_data(
            &reader,
            cli.sample.as_deref(),
            &cli.pheno,
            &cli.pheno_name,
            cli.covar.as_deref(),
            &cli.covar_names,
        )
    })?;
    let sample_count = aligned_sample_data.sample_indices.len();
    let sample_indices = aligned_sample_data.sample_indices.clone();
    timer.measure("bgen_sample_selection", || reader.prepare_sample_selection(&sample_indices))?;

    let prediction_source = if cli.output_mode == OutputMode::DecodeOnly {
        None
    } else {
        Some(timer.measure("prediction_load", || {
            regenie::PredictionSource::load(
                &cli.pred,
                &cli.pheno_name,
                &aligned_sample_data.family_identifiers,
                &aligned_sample_data.individual_identifiers,
            )
        })?)
    };
    let linear_state = if cli.output_mode == OutputMode::DecodeOnly {
        None
    } else {
        Some(timer.measure("linear_state_prepare", || {
            linear::prepare_linear_state(
                &aligned_sample_data.covariate_values,
                &aligned_sample_data.phenotype_values,
                aligned_sample_data.covariate_count,
            )
        })?)
    };
    let mut cuda_session = if cli.output_mode == OutputMode::DecodeOnly {
        None
    } else {
        let (session, timing_seconds) = timer.measure("cuda_session_prepare", || {
            cuda_linear::CudaLinearKernelSession::new(cuda_linear::CudaLinearKernelConfig {
                block_size: cli.cuda_block_size,
            })
        })?;
        for (stage_name, seconds) in timing_seconds {
            timer.add_seconds(&stage_name, seconds);
        }
        Some(session)
    };

    let run_directory = build_run_directory(&cli.out);
    let chunks_directory = run_directory.join("chunks");
    let writer_session = match cli.output_mode {
        OutputMode::FullParquet | OutputMode::ChunksOnly => Some(timer.measure("output_writer_prepare", || {
            NativeOutputWriterSession::new(
                run_directory.clone(),
                chunks_directory.clone(),
                "regenie2_linear".to_string(),
                cli.writer_threads,
                cli.writer_queue_depth,
                cli.output_mode == OutputMode::FullParquet,
            )
        })?),
        OutputMode::NoOutput | OutputMode::DecodeOnly => None,
    };

    let chunk_specs = planner::plan_chromosome_homogeneous_chunks(
        reader.variant_count(),
        cli.chunk_size,
        cli.variant_limit,
        &reader.chromosome_boundary_indices(),
        &BTreeSet::new(),
    )?;

    let mut current_chromosome = String::new();
    let mut current_chromosome_state: Option<cuda_linear::CudaLinearChromosomeState> = None;
    let mut processed_variant_count = 0usize;
    let mut checksum = 0.0_f64;
    for chunk_spec in &chunk_specs {
        let variant_count = chunk_spec.variant_stop_index - chunk_spec.variant_start_index;
        let mut genotype_values = vec![0.0_f32; sample_count * variant_count];
        let decode_start_time = Instant::now();
        let chunk_stats = reader.read_preprocessed_dosage_f32_into_address_prepared(
            chunk_spec.variant_start_index,
            chunk_spec.variant_stop_index,
            genotype_values.as_mut_ptr() as usize,
            genotype_values.len(),
        )?;
        timer.add_seconds("bgen_decode_preprocess", decode_start_time.elapsed().as_secs_f64());

        let metadata = timer.measure("metadata_load", || {
            reader.variant_metadata_slice(chunk_spec.variant_start_index, chunk_spec.variant_stop_index)
        })?;
        let metadata = convert_variant_metadata(metadata);
        if cli.output_mode == OutputMode::DecodeOnly {
            checksum += chunk_stats
                .allele_one_frequency
                .iter()
                .chain(genotype_values.iter().take(16))
                .map(|value| f64::from(value.to_bits()))
                .sum::<f64>();
            processed_variant_count += variant_count;
            continue;
        }

        let chromosome = metadata.chromosome.first().cloned().unwrap_or_default();
        if current_chromosome_state.is_none() || chromosome != current_chromosome {
            current_chromosome = chromosome.clone();
            let prediction_values = prediction_source
                .as_ref()
                .ok_or("Prediction source was not loaded.")?
                .chromosome_predictions(&chromosome)?;
            let prepared_linear_state = linear_state.as_ref().ok_or("Linear state was not prepared.")?;
            let chromosome_state = timer.measure("chromosome_state_prepare", || {
                linear::prepare_linear_chromosome_state(prepared_linear_state, prediction_values)
            })?;
            current_chromosome_state = Some(timer.measure("cuda_chromosome_state_h2d", || {
                cuda_session
                    .as_ref()
                    .ok_or_else(|| linear::LinearError::Backend("CUDA session was not prepared.".to_string()))?
                    .prepare_chromosome_state(&chromosome_state)
            })?);
        }

        let compute_start_time = Instant::now();
        let result = cuda_session.as_mut().ok_or("CUDA session was not prepared.")?.compute_chunk(
            current_chromosome_state.as_ref().ok_or("Chromosome state was not prepared.")?,
            &genotype_values,
            variant_count,
        )?;
        timer.add_seconds("cuda_upload_compute_materialize_cpu_pvalue", compute_start_time.elapsed().as_secs_f64());
        for (stage_name, seconds) in &result.timing_seconds {
            timer.add_seconds(stage_name, *seconds);
        }
        checksum += result.checksum;
        if let Some(writer) = writer_session.as_ref() {
            let enqueue_start_time = Instant::now();
            writer.write_regenie2_chunk(NativeRegenieStep2Chunk {
                variant_start_index: i64::try_from(chunk_spec.variant_start_index).unwrap_or(i64::MAX),
                variant_stop_index: i64::try_from(chunk_spec.variant_stop_index).unwrap_or(i64::MAX),
                metadata,
                allele_one_frequency: chunk_stats.allele_one_frequency,
                observation_count: chunk_stats.observation_count,
                beta: result.beta,
                standard_error: result.standard_error,
                chi_squared: result.chi_squared,
                log10_p_value: result.log10_p_value,
                extra_code: None,
            })?;
            timer.add_seconds("output_enqueue", enqueue_start_time.elapsed().as_secs_f64());
        }
        processed_variant_count += variant_count;
    }

    let final_parquet = if let Some(writer) = writer_session.as_ref() {
        timer.measure("parquet_finalization", || writer.finish())?.map(|path| path.display().to_string())
    } else {
        None
    };
    timer.measure("bgen_sample_selection_clear", || reader.clear_prepared_sample_selection())?;
    let total_wall_seconds = timer.total_wall_seconds();
    let report = NativeRunReport {
        backend: "cuda-kernel".to_string(),
        output_mode: format!("{:?}", cli.output_mode),
        chunk_size: cli.chunk_size,
        variant_limit: cli.variant_limit,
        sample_count,
        variant_count: reader.variant_count(),
        processed_variant_count,
        processed_chunk_count: chunk_specs.len(),
        checksum,
        output_run_directory: run_directory.display().to_string(),
        final_parquet,
        cuda_block_size: Some(cli.cuda_block_size),
        stage_seconds: timer.stage_seconds,
        total_wall_seconds,
    };
    if let Some(report_path) = report_json_path {
        let report_json = serde_json::to_string_pretty(&report)?;
        std::fs::write(report_path, report_json)?;
    } else {
        println!("{}", serde_json::to_string_pretty(&report)?);
    }
    Ok(report)
}

type VariantMetadataTuple = (Vec<String>, Vec<String>, Vec<i64>, Vec<String>, Vec<String>);

fn convert_variant_metadata(metadata: VariantMetadataTuple) -> VariantMetadataColumns {
    let (chromosome, variant_identifier, position, allele_one, allele_two) = metadata;
    VariantMetadataColumns { chromosome, variant_identifier, position, allele_one, allele_two }
}

fn build_run_directory(output_prefix: &Path) -> PathBuf {
    PathBuf::from(format!("{}.regenie2_linear_burn_native.run", output_prefix.display()))
}
