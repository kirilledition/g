//! Host-side genotype preprocessing shared by native readers.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::missing_errors_doc)]

use crate::genotype::common::{ChunkStats, GenotypeError};

pub fn preprocess_row_major_dosage_matrix(
    dosage_values: &mut [f32],
    selected_sample_count: usize,
    selected_variant_count: usize,
) -> Result<ChunkStats, GenotypeError> {
    let expected_value_count = selected_sample_count.checked_mul(selected_variant_count).ok_or_else(|| {
        GenotypeError::InvalidInput("Integer overflow while validating genotype matrix shape.".to_string())
    })?;
    if dosage_values.len() != expected_value_count {
        return Err(GenotypeError::InvalidInput(format!(
            "Genotype matrix value count mismatch: expected {expected_value_count}, observed {}.",
            dosage_values.len(),
        )));
    }

    let mut observed_genotype_totals = vec![0.0_f32; selected_variant_count];
    let mut observation_count = vec![0_i32; selected_variant_count];
    let mut has_missing_values = false;

    for sample_index in 0..selected_sample_count {
        let row_offset = sample_index
            .checked_mul(selected_variant_count)
            .ok_or_else(|| GenotypeError::InvalidInput("Integer overflow while scanning genotype rows.".to_string()))?;
        for variant_index in 0..selected_variant_count {
            let dosage_value = dosage_values[row_offset + variant_index];
            if dosage_value.is_nan() {
                has_missing_values = true;
                continue;
            }
            observed_genotype_totals[variant_index] += dosage_value;
            observation_count[variant_index] += 1;
        }
    }

    let mut imputed_column_means = vec![0.0_f32; selected_variant_count];
    let mut allele_one_frequency = vec![0.0_f32; selected_variant_count];
    for variant_index in 0..selected_variant_count {
        let count = observation_count[variant_index];
        if count > 0 {
            let column_mean = observed_genotype_totals[variant_index] / count as f32;
            imputed_column_means[variant_index] = column_mean;
            allele_one_frequency[variant_index] = column_mean / 2.0;
        }
    }

    if has_missing_values {
        for sample_index in 0..selected_sample_count {
            let row_offset = sample_index.checked_mul(selected_variant_count).ok_or_else(|| {
                GenotypeError::InvalidInput("Integer overflow while imputing genotype rows.".to_string())
            })?;
            for variant_index in 0..selected_variant_count {
                let output_value = &mut dosage_values[row_offset + variant_index];
                if output_value.is_nan() {
                    *output_value = imputed_column_means[variant_index];
                }
            }
        }
    }

    Ok(ChunkStats { allele_one_frequency, observation_count, has_missing_values })
}

#[cfg(test)]
mod tests {
    use super::preprocess_row_major_dosage_matrix;

    #[test]
    fn preprocess_imputes_missing_values_and_computes_stats() {
        let mut dosage_values = vec![0.0, f32::NAN, 2.0, 1.0, 2.0, f32::NAN];

        let stats = preprocess_row_major_dosage_matrix(&mut dosage_values, 3, 2).expect("preprocess should succeed");

        assert_eq!(dosage_values, vec![0.0, 1.0, 2.0, 1.0, 2.0, 1.0]);
        assert_eq!(stats.observation_count, vec![3, 1]);
        assert_eq!(stats.allele_one_frequency, vec![2.0 / 3.0, 0.5]);
        assert!(stats.has_missing_values);
    }

    #[test]
    fn preprocess_handles_all_missing_column_as_zero_mean() {
        let mut dosage_values = vec![f32::NAN, 1.0, f32::NAN, 2.0];

        let stats = preprocess_row_major_dosage_matrix(&mut dosage_values, 2, 2).expect("preprocess should succeed");

        assert_eq!(dosage_values, vec![0.0, 1.0, 0.0, 2.0]);
        assert_eq!(stats.observation_count, vec![0, 2]);
        assert_eq!(stats.allele_one_frequency, vec![0.0, 0.75]);
        assert!(stats.has_missing_values);
    }
}
