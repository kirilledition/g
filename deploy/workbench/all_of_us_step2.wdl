version 1.1

struct ChromosomeJob {
  File manifest
  File bgen
  File sample
  File phenotype
  File? covariates
  Array[File]+ predictions
}

struct LocalizedInputs {
  File bgen
  File sample
  File pheno_file
  File? covar_file
}

struct LocalizationBindings {
  File manifest
  LocalizedInputs inputs
  Array[File]+ predictions
  Int cpu_threads
  Int memory_gib
}

# Portable CPU template. GPU pilot runs directly on a GPU Workbench app until
# the actual workspace execution backend and its GPU resource keys are qualified.
workflow all_of_us_step2 {
  input {
    Array[ChromosomeJob]+ chromosomes
    String image
    Int cpu_threads = 8
    Int memory_gib = 32
  }

  scatter (chromosome in chromosomes) {
    call chromosome_step2 {
      input:
        job = chromosome,
        image = image,
        cpu_threads = cpu_threads,
        memory_gib = memory_gib
    }
  }

  output {
    Array[File] result_bundles = chromosome_step2.result_bundle
    Array[File] source_manifests = chromosome_step2.source_manifest
    Array[File] localized_manifests = chromosome_step2.localized_manifest
  }
}

task chromosome_step2 {
  input {
    ChromosomeJob job
    String image
    Int cpu_threads
    Int memory_gib
  }

  File bindings = write_json(LocalizationBindings {
    manifest: job.manifest,
    inputs: LocalizedInputs {
      bgen: job.bgen,
      sample: job.sample,
      pheno_file: job.phenotype,
      covar_file: job.covariates
    },
    predictions: job.predictions,
    cpu_threads: cpu_threads,
    memory_gib: memory_gib
  })

  command <<<
    set -euo pipefail
    python -m deploy.workbench.localize_wdl '~{bindings}'
    python -m tooling.cli.workbench --config-name workbench \
      tool.action=run tool.manifest="$PWD/job.local.json" \
      tool.work_directory="$PWD/attempts" tool.dry_run=false
    python - <<'PYTHON'
    import pathlib
    import tarfile
    markers = list(pathlib.Path("published").glob("*/COMMITTED.json"))
    if len(markers) != 1:
        raise SystemExit("Expected exactly one validated committed result bundle.")
    with tarfile.open("result.tar", "w") as archive:
        archive.add(markers[0].parent, arcname=markers[0].parent.name)
    PYTHON
  >>>

  runtime {
    docker: image
    cpu: cpu_threads
    memory: "~{memory_gib} GiB"
  }

  output {
    File result_bundle = "result.tar"
    File source_manifest = job.manifest
    File localized_manifest = "job.local.json"
  }
}
