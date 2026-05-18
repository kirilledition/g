//! Sample, phenotype, and covariate alignment for native REGENIE step 2.

#![allow(clippy::missing_errors_doc)]

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use thiserror::Error;

use crate::genotype::bgen::BgenReaderCore;

#[derive(Clone, Debug)]
pub struct AlignedSampleData {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_values: Vec<f32>,
    pub covariate_values: Vec<f32>,
    pub covariate_count: usize,
}

#[derive(Clone, Debug)]
struct SampleRecord {
    sample_index: i64,
    family_identifier: String,
    individual_identifier: String,
}

#[derive(Debug)]
struct TabularData {
    column_names: Vec<String>,
    rows: Vec<HashMap<String, String>>,
}

#[derive(Debug, Error)]
pub enum AlignmentError {
    #[error("{0}")]
    InvalidInput(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub fn load_bgen_aligned_sample_data(
    reader: &BgenReaderCore,
    sample_path: Option<&Path>,
    phenotype_path: &Path,
    phenotype_name: &str,
    covariate_path: Option<&Path>,
    covariate_names: &[String],
) -> Result<AlignedSampleData, AlignmentError> {
    let sample_records = load_sample_records(reader, sample_path)?;
    if sample_records.len() != reader.sample_count() {
        return Err(AlignmentError::InvalidInput(format!(
            "Sample source contains {} rows, but BGEN contains {} samples.",
            sample_records.len(),
            reader.sample_count(),
        )));
    }

    let phenotype_table = load_tabular_data(phenotype_path)?;
    if !phenotype_table.column_names.iter().any(|column_name| column_name == phenotype_name) {
        return Err(AlignmentError::InvalidInput(format!(
            "Phenotype column '{phenotype_name}' was not found in {}.",
            phenotype_path.display(),
        )));
    }
    let phenotype_by_individual = build_row_lookup_by_individual_identifier(&phenotype_table)?;

    let covariate_table = covariate_path.map(|path| load_tabular_data(path).map(|table| (path, table))).transpose()?;
    let mut selected_covariate_names = covariate_names.to_vec();
    let covariate_by_individual = if let Some((path, table)) = covariate_table.as_ref() {
        if selected_covariate_names.is_empty() {
            selected_covariate_names = infer_covariate_names(table)?;
        }
        let covariate_column_set = table.column_names.iter().collect::<std::collections::HashSet<_>>();
        let missing_covariates = selected_covariate_names
            .iter()
            .filter(|covariate_name| !covariate_column_set.contains(covariate_name))
            .cloned()
            .collect::<Vec<_>>();
        if !missing_covariates.is_empty() {
            return Err(AlignmentError::InvalidInput(format!(
                "Covariate columns are missing from {}: {missing_covariates:?}.",
                path.display(),
            )));
        }
        Some(build_row_lookup_by_individual_identifier(table)?)
    } else {
        if !selected_covariate_names.is_empty() {
            return Err(AlignmentError::InvalidInput(
                "Covariate names cannot be provided without a covariate table.".to_string(),
            ));
        }
        None
    };

    build_aligned_sample_data(
        &sample_records,
        &phenotype_by_individual,
        phenotype_name,
        covariate_by_individual.as_ref(),
        &selected_covariate_names,
    )
}

fn load_sample_records(
    reader: &BgenReaderCore,
    sample_path: Option<&Path>,
) -> Result<Vec<SampleRecord>, AlignmentError> {
    if let Some(path) = sample_path {
        return load_oxford_sample_records(path);
    }
    if reader.contains_embedded_samples() {
        return Ok(reader
            .sample_identifiers()
            .into_iter()
            .enumerate()
            .map(|(sample_index, sample_identifier)| SampleRecord {
                sample_index: i64::try_from(sample_index).unwrap_or(i64::MAX),
                family_identifier: sample_identifier.clone(),
                individual_identifier: sample_identifier,
            })
            .collect());
    }
    Err(AlignmentError::InvalidInput(
        "BGEN file does not contain embedded samples and no .sample file was provided.".to_string(),
    ))
}

fn load_oxford_sample_records(sample_path: &Path) -> Result<Vec<SampleRecord>, AlignmentError> {
    let lines = read_non_empty_lines(sample_path)?;
    if lines.len() < 2 {
        return Err(AlignmentError::InvalidInput(format!(
            "Sample file '{}' must contain at least two header lines.",
            sample_path.display(),
        )));
    }
    let column_names = split_whitespace_line(&lines[0]);
    let column_types = split_whitespace_line(&lines[1]);
    if column_names.len() != column_types.len() {
        return Err(AlignmentError::InvalidInput(format!(
            "Sample file '{}' header and type lines have different column counts.",
            sample_path.display(),
        )));
    }
    if column_names.is_empty() {
        return Err(AlignmentError::InvalidInput(format!(
            "Sample file '{}' does not contain any columns.",
            sample_path.display(),
        )));
    }
    if column_types[0] != "0" {
        return Err(AlignmentError::InvalidInput(format!(
            "Sample file '{}' must mark the first identifier column with type '0'.",
            sample_path.display(),
        )));
    }
    let individual_identifier_index = column_names.iter().position(|column_name| column_name == "ID_2").unwrap_or(0);
    if column_names[individual_identifier_index] == "ID_2" && column_types[individual_identifier_index] != "0" {
        return Err(AlignmentError::InvalidInput(format!(
            "Sample file '{}' must mark 'ID_2' with type '0'.",
            sample_path.display(),
        )));
    }

    let mut sample_records = Vec::with_capacity(lines.len().saturating_sub(2));
    for (row_offset, line) in lines[2..].iter().enumerate() {
        let row_values = split_whitespace_line(line);
        if row_values.len() != column_names.len() {
            return Err(AlignmentError::InvalidInput(format!(
                "Sample file '{}' line {} has {} values, but the header declares {} columns.",
                sample_path.display(),
                row_offset + 3,
                row_values.len(),
                column_names.len(),
            )));
        }
        sample_records.push(SampleRecord {
            sample_index: i64::try_from(row_offset).unwrap_or(i64::MAX),
            family_identifier: row_values[0].clone(),
            individual_identifier: row_values[individual_identifier_index].clone(),
        });
    }
    Ok(sample_records)
}

fn load_tabular_data(table_path: &Path) -> Result<TabularData, AlignmentError> {
    let lines = read_non_empty_lines(table_path)?;
    let Some(header_line) = lines.first() else {
        return Err(AlignmentError::InvalidInput(format!("Table '{}' is empty.", table_path.display())));
    };
    let column_names = header_line.split('\t').map(str::to_string).collect::<Vec<_>>();
    if !column_names.iter().any(|column_name| column_name == "IID") {
        return Err(AlignmentError::InvalidInput(format!(
            "Table '{}' must contain an IID column.",
            table_path.display(),
        )));
    }
    let mut rows = Vec::with_capacity(lines.len().saturating_sub(1));
    for (row_offset, line) in lines[1..].iter().enumerate() {
        let row_values = line.split('\t').map(str::to_string).collect::<Vec<_>>();
        if row_values.len() != column_names.len() {
            return Err(AlignmentError::InvalidInput(format!(
                "Table '{}' line {} has {} values, but the header declares {} columns.",
                table_path.display(),
                row_offset + 2,
                row_values.len(),
                column_names.len(),
            )));
        }
        rows.push(column_names.iter().cloned().zip(row_values).collect());
    }
    Ok(TabularData { column_names, rows })
}

fn build_row_lookup_by_individual_identifier(
    table: &TabularData,
) -> Result<HashMap<&str, &HashMap<String, String>>, AlignmentError> {
    let mut lookup = HashMap::with_capacity(table.rows.len());
    for row in &table.rows {
        let individual_identifier =
            row.get("IID").ok_or_else(|| AlignmentError::InvalidInput("Table row is missing IID.".to_string()))?;
        lookup.insert(individual_identifier.as_str(), row);
    }
    Ok(lookup)
}

fn infer_covariate_names(covariate_table: &TabularData) -> Result<Vec<String>, AlignmentError> {
    let covariate_names = covariate_table
        .column_names
        .iter()
        .filter(|column_name| column_name.as_str() != "FID" && column_name.as_str() != "IID")
        .cloned()
        .collect::<Vec<_>>();
    if covariate_names.is_empty() {
        return Err(AlignmentError::InvalidInput(
            "Covariate table must contain at least one non-identifier column.".to_string(),
        ));
    }
    Ok(covariate_names)
}

fn build_aligned_sample_data(
    sample_records: &[SampleRecord],
    phenotype_by_individual: &HashMap<&str, &HashMap<String, String>>,
    phenotype_name: &str,
    covariate_by_individual: Option<&HashMap<&str, &HashMap<String, String>>>,
    covariate_names: &[String],
) -> Result<AlignedSampleData, AlignmentError> {
    let covariate_count = covariate_names.len() + 1;
    let mut sample_indices = Vec::new();
    let mut family_identifiers = Vec::new();
    let mut individual_identifiers = Vec::new();
    let mut phenotype_values = Vec::new();
    let mut covariate_values = Vec::new();

    for sample_record in sample_records {
        let Some(phenotype_row) = phenotype_by_individual.get(sample_record.individual_identifier.as_str()) else {
            continue;
        };
        let Some(phenotype_value) = parse_required_float(phenotype_row, phenotype_name)? else {
            continue;
        };
        let mut row_covariates = Vec::with_capacity(covariate_count);
        row_covariates.push(1.0_f32);
        if let Some(covariate_lookup) = covariate_by_individual {
            let Some(covariate_row) = covariate_lookup.get(sample_record.individual_identifier.as_str()) else {
                continue;
            };
            let mut should_drop_sample = false;
            for covariate_name in covariate_names {
                if let Some(covariate_value) = parse_required_float(covariate_row, covariate_name)? {
                    row_covariates.push(covariate_value);
                } else {
                    should_drop_sample = true;
                    break;
                }
            }
            if should_drop_sample {
                continue;
            }
        }
        sample_indices.push(sample_record.sample_index);
        family_identifiers.push(sample_record.family_identifier.clone());
        individual_identifiers.push(sample_record.individual_identifier.clone());
        phenotype_values.push(phenotype_value);
        covariate_values.extend(row_covariates);
    }

    if sample_indices.is_empty() {
        return Err(AlignmentError::InvalidInput(
            "No aligned samples remain after joining phenotype and covariate tables.".to_string(),
        ));
    }

    Ok(AlignedSampleData {
        sample_indices,
        family_identifiers,
        individual_identifiers,
        phenotype_values,
        covariate_values,
        covariate_count,
    })
}

fn parse_required_float(row: &HashMap<String, String>, column_name: &str) -> Result<Option<f32>, AlignmentError> {
    let value = row
        .get(column_name)
        .ok_or_else(|| AlignmentError::InvalidInput(format!("Column '{column_name}' is missing from a table row.")))?;
    if is_null_token(value) {
        return Ok(None);
    }
    let parsed_value = value.parse::<f32>().map_err(|source| {
        AlignmentError::InvalidInput(format!("Failed to parse value '{value}' in column '{column_name}': {source}"))
    })?;
    Ok(Some(parsed_value))
}

fn is_null_token(value: &str) -> bool {
    matches!(value, "NA" | "NaN" | "nan" | "-9")
}

fn read_non_empty_lines(path: &Path) -> Result<Vec<String>, AlignmentError> {
    let file = File::open(PathBuf::from(path))?;
    let lines = BufReader::new(file)
        .lines()
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .filter(|line| !line.trim().is_empty())
        .collect();
    Ok(lines)
}

fn split_whitespace_line(line: &str) -> Vec<String> {
    line.split_whitespace().map(str::to_string).collect()
}
