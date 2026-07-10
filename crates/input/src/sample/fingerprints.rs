use std::fmt::Write as _;

use sha2::{Digest, Sha256};

use super::types::{AlignedPhenotypeGroupDraft, PhenotypeGroupLoadRequest};

pub(super) fn build_phenotype_compute_group(
    request: &PhenotypeGroupLoadRequest,
    draft: &AlignedPhenotypeGroupDraft,
    family_identifiers: &[&str],
    individual_identifiers: &[&str],
) -> Result<g_plan::PhenotypeComputeGroup, String> {
    let phenotype_indices = draft
        .phenotype_indices
        .iter()
        .map(|phenotype_index| {
            u32::try_from(*phenotype_index).map_err(|_| "Phenotype index exceeds the supported u32 range.".to_string())
        })
        .collect::<Result<Vec<_>, _>>()?;
    let phenotype_names = draft
        .phenotype_indices
        .iter()
        .map(|phenotype_index| request.phenotype_names[*phenotype_index].clone())
        .collect::<Vec<_>>();
    let group_mode = if request.phenotype_names.len() == 1 {
        g_plan::PhenotypeComputeGroupMode::SinglePhenotype
    } else if request.sample_mode == g_plan::MultiPhenotypeSampleMode::CompleteCase {
        g_plan::PhenotypeComputeGroupMode::CompleteCase
    } else {
        g_plan::PhenotypeComputeGroupMode::PerPhenotypeCompatible
    };
    let sample_mode = if request.phenotype_names.len() == 1 {
        g_plan::MultiPhenotypeSampleMode::PerPhenotype
    } else {
        request.sample_mode
    };
    let sample_set_fingerprint =
        fingerprint_sample_set(&draft.sample_indices, family_identifiers, individual_identifiers)?;
    let covariate_design_fingerprint =
        fingerprint_covariate_design(&draft.covariate_names, draft.sample_indices.len(), &draft.covariate_values);
    let prediction_alignment_fingerprint = fingerprint_prediction_alignment(
        &request.prediction_list_path,
        request.sample_key_mode,
        &sample_set_fingerprint,
        &phenotype_names,
    );
    Ok(g_plan::PhenotypeComputeGroup {
        group_mode,
        phenotype_indices,
        phenotype_names,
        sample_mode,
        sample_set_fingerprint: Some(sample_set_fingerprint),
        covariate_design_fingerprint: Some(covariate_design_fingerprint),
        prediction_alignment_fingerprint: Some(prediction_alignment_fingerprint),
    })
}

fn fingerprint_sample_set(
    sample_indices: &[usize],
    family_identifiers: &[&str],
    individual_identifiers: &[&str],
) -> Result<String, String> {
    let mut fingerprint_hash = Sha256::new();
    update_fingerprint(&mut fingerprint_hash, "sample-set-v1");
    update_usize_as_i64_array_fingerprint(&mut fingerprint_hash, "int64", &[sample_indices.len()], sample_indices)?;
    update_str_sequence_fingerprint(&mut fingerprint_hash, family_identifiers);
    update_str_sequence_fingerprint(&mut fingerprint_hash, individual_identifiers);
    Ok(finalize_sha256_hex(fingerprint_hash))
}

fn fingerprint_covariate_design(covariate_names: &[String], sample_count: usize, covariate_values: &[f32]) -> String {
    let mut fingerprint_hash = Sha256::new();
    update_fingerprint(&mut fingerprint_hash, "covariate-design-v1");
    update_string_sequence_fingerprint(&mut fingerprint_hash, covariate_names);
    update_f32_array_fingerprint(
        &mut fingerprint_hash,
        "float32",
        &[sample_count, covariate_names.len()],
        covariate_values,
    );
    finalize_sha256_hex(fingerprint_hash)
}

fn fingerprint_prediction_alignment(
    prediction_list_path: &str,
    sample_key_mode: g_plan::SampleKeyMode,
    sample_set_fingerprint: &str,
    phenotype_names: &[String],
) -> String {
    let mut fingerprint_hash = Sha256::new();
    update_fingerprint(&mut fingerprint_hash, "prediction-alignment-v1");
    update_fingerprint(&mut fingerprint_hash, prediction_list_path);
    update_fingerprint(&mut fingerprint_hash, sample_key_mode.as_str());
    update_fingerprint(&mut fingerprint_hash, sample_set_fingerprint);
    update_string_sequence_fingerprint(&mut fingerprint_hash, phenotype_names);
    finalize_sha256_hex(fingerprint_hash)
}

fn finalize_sha256_hex(fingerprint_hash: Sha256) -> String {
    let digest_bytes = fingerprint_hash.finalize();
    let mut digest_text = String::with_capacity(digest_bytes.len() * 2);
    for digest_byte in digest_bytes {
        write!(&mut digest_text, "{digest_byte:02x}").expect("writing to String must succeed");
    }
    digest_text
}

fn update_usize_as_i64_array_fingerprint(
    fingerprint_hash: &mut Sha256,
    dtype_name: &str,
    shape: &[usize],
    values: &[usize],
) -> Result<(), String> {
    update_fingerprint(fingerprint_hash, dtype_name);
    update_fingerprint(fingerprint_hash, &python_shape_repr(shape));
    for value in values {
        let schema_value =
            i64::try_from(*value).map_err(|_| "Sample index exceeds the fingerprint schema i64 range.".to_string())?;
        fingerprint_hash.update(schema_value.to_ne_bytes());
    }
    Ok(())
}

fn update_f32_array_fingerprint(fingerprint_hash: &mut Sha256, dtype_name: &str, shape: &[usize], values: &[f32]) {
    update_fingerprint(fingerprint_hash, dtype_name);
    update_fingerprint(fingerprint_hash, &python_shape_repr(shape));
    for value in values {
        fingerprint_hash.update(value.to_ne_bytes());
    }
}

fn update_string_sequence_fingerprint(fingerprint_hash: &mut Sha256, values: &[String]) {
    update_fingerprint(fingerprint_hash, &values.len().to_string());
    for value in values {
        update_fingerprint(fingerprint_hash, value);
    }
}

fn update_str_sequence_fingerprint(fingerprint_hash: &mut Sha256, values: &[&str]) {
    update_fingerprint(fingerprint_hash, &values.len().to_string());
    for value in values {
        update_fingerprint(fingerprint_hash, value);
    }
}

fn update_fingerprint(fingerprint_hash: &mut Sha256, value: &str) {
    let encoded_value = value.as_bytes();
    fingerprint_hash.update(encoded_value.len().to_string().as_bytes());
    fingerprint_hash.update(b":");
    fingerprint_hash.update(encoded_value);
}

fn python_shape_repr(shape: &[usize]) -> String {
    match shape {
        [] => "()".to_string(),
        [axis_length] => format!("({axis_length},)"),
        _ => format!("({})", shape.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")),
    }
}
