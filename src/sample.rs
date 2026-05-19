//! Native sample alignment and Oxford sample-file parsing.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::float_cmp)]
#![allow(clippy::single_match_else)]
#![allow(clippy::too_many_arguments)]

use std::collections::HashMap;
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SampleKeyMode {
    Iid,
    FidIid,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlignedSampleData {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_name: String,
    pub phenotype_vector: Vec<f32>,
    pub covariate_names: Vec<String>,
    pub covariate_matrix_values: Vec<f32>,
    pub covariate_row_count: usize,
    pub covariate_column_count: usize,
    pub is_binary_trait: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct MultiAlignedSampleData {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_names: Vec<String>,
    pub phenotype_matrix_values: Vec<f32>,
    pub phenotype_row_count: usize,
    pub phenotype_column_count: usize,
    pub covariate_names: Vec<String>,
    pub covariate_matrix_values: Vec<f32>,
    pub covariate_row_count: usize,
    pub covariate_column_count: usize,
    pub is_binary_trait: bool,
}

#[derive(Clone, Debug)]
pub struct AlignmentInputs {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_path: String,
    pub phenotype_name: String,
    pub covariate_path: Option<String>,
    pub covariate_names: Option<Vec<String>>,
    pub is_binary_trait: bool,
    pub sample_key_mode: SampleKeyMode,
    pub allow_duplicate_iid_alignment: bool,
}

#[derive(Clone, Debug)]
pub struct MultiAlignmentInputs {
    pub sample_indices: Vec<i64>,
    pub family_identifiers: Vec<String>,
    pub individual_identifiers: Vec<String>,
    pub phenotype_path: String,
    pub phenotype_names: Vec<String>,
    pub covariate_path: Option<String>,
    pub covariate_names: Option<Vec<String>>,
    pub is_binary_trait: bool,
    pub sample_key_mode: SampleKeyMode,
    pub allow_duplicate_iid_alignment: bool,
}

struct TabularTable {
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

struct RawPhenotypeRecord {
    phenotype_value: String,
}

struct RawCovariateRecord {
    covariate_values: Vec<String>,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum SampleKey {
    Iid(String),
    FidIid { family_identifier: String, individual_identifier: String },
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct AlignedSampleKey {
    sample_index: i64,
    family_identifier: String,
    individual_identifier: String,
}

struct SampleIdentifierData {
    sample_indices: Vec<i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
}

struct AlignedSampleRow {
    sample_index: i64,
    family_identifier: String,
    individual_identifier: String,
    phenotype_value: f32,
    covariate_values: Vec<f32>,
}

impl AlignedSampleData {
    fn new(
        phenotype_name: String,
        covariate_names: Vec<String>,
        rows: Vec<AlignedSampleRow>,
        is_binary_trait: bool,
    ) -> Self {
        let covariate_column_count = covariate_names.len();
        let covariate_row_count = rows.len();
        let mut sample_indices = Vec::with_capacity(covariate_row_count);
        let mut family_identifiers = Vec::with_capacity(covariate_row_count);
        let mut individual_identifiers = Vec::with_capacity(covariate_row_count);
        let mut phenotype_vector = Vec::with_capacity(covariate_row_count);
        let mut covariate_matrix_values = Vec::with_capacity(covariate_row_count * covariate_column_count);

        for row in rows {
            sample_indices.push(row.sample_index);
            family_identifiers.push(row.family_identifier);
            individual_identifiers.push(row.individual_identifier);
            phenotype_vector.push(row.phenotype_value);
            covariate_matrix_values.push(1.0);
            covariate_matrix_values.extend(row.covariate_values);
        }

        Self {
            sample_indices,
            family_identifiers,
            individual_identifiers,
            phenotype_name,
            phenotype_vector,
            covariate_names,
            covariate_matrix_values,
            covariate_row_count,
            covariate_column_count,
            is_binary_trait,
        }
    }
}

impl MultiAlignedSampleData {
    fn new(
        phenotype_names: Vec<String>,
        aligned_sample_data_by_trait: &[AlignedSampleData],
        common_positions_by_trait: &[Vec<usize>],
        is_binary_trait: bool,
    ) -> Self {
        let first_aligned_sample_data = &aligned_sample_data_by_trait[0];
        let first_common_positions = &common_positions_by_trait[0];
        let sample_count = first_common_positions.len();
        let trait_count = aligned_sample_data_by_trait.len();
        let covariate_column_count = first_aligned_sample_data.covariate_column_count;
        let mut sample_indices = Vec::with_capacity(sample_count);
        let mut family_identifiers = Vec::with_capacity(sample_count);
        let mut individual_identifiers = Vec::with_capacity(sample_count);
        let mut phenotype_matrix_values = Vec::with_capacity(trait_count * sample_count);
        let mut covariate_matrix_values = Vec::with_capacity(sample_count * covariate_column_count);

        for position in first_common_positions {
            sample_indices.push(first_aligned_sample_data.sample_indices[*position]);
            family_identifiers.push(first_aligned_sample_data.family_identifiers[*position].clone());
            individual_identifiers.push(first_aligned_sample_data.individual_identifiers[*position].clone());
            let covariate_row_start = position * covariate_column_count;
            covariate_matrix_values.extend(
                &first_aligned_sample_data.covariate_matrix_values
                    [covariate_row_start..covariate_row_start + covariate_column_count],
            );
        }
        for (aligned_sample_data, common_positions) in
            aligned_sample_data_by_trait.iter().zip(common_positions_by_trait.iter())
        {
            for position in common_positions {
                phenotype_matrix_values.push(aligned_sample_data.phenotype_vector[*position]);
            }
        }

        Self {
            sample_indices,
            family_identifiers,
            individual_identifiers,
            phenotype_names,
            phenotype_matrix_values,
            phenotype_row_count: trait_count,
            phenotype_column_count: sample_count,
            covariate_names: first_aligned_sample_data.covariate_names.clone(),
            covariate_matrix_values,
            covariate_row_count: sample_count,
            covariate_column_count,
            is_binary_trait,
        }
    }
}

pub fn align_sample_data(inputs: AlignmentInputs) -> Result<AlignedSampleData, String> {
    validate_alignment_input_lengths(&inputs)?;
    validate_alignment_config(inputs.sample_key_mode, inputs.allow_duplicate_iid_alignment)?;
    validate_sample_identifier_keys(&inputs)?;

    let phenotype_table = read_tabular_table(Path::new(&inputs.phenotype_path))?;
    validate_required_column(&phenotype_table, "IID", &inputs.phenotype_path)?;
    if inputs.sample_key_mode == SampleKeyMode::FidIid {
        validate_required_column(&phenotype_table, "FID", &inputs.phenotype_path)?;
    }
    validate_required_column(&phenotype_table, &inputs.phenotype_name, &inputs.phenotype_path)?;
    let phenotype_records_by_identifier = build_phenotype_records_by_key(
        &phenotype_table,
        &inputs.phenotype_name,
        inputs.sample_key_mode,
        inputs.allow_duplicate_iid_alignment,
    )?;

    let (selected_covariate_names, covariate_records_by_identifier) = match inputs.covariate_path.as_ref() {
        Some(covariate_path) => {
            let covariate_table = read_tabular_table(Path::new(covariate_path))?;
            validate_required_column(&covariate_table, "IID", covariate_path)?;
            if inputs.sample_key_mode == SampleKeyMode::FidIid {
                validate_required_column(&covariate_table, "FID", covariate_path)?;
            }
            let selected_covariate_names =
                select_covariate_names(&covariate_table, inputs.covariate_names.as_deref(), covariate_path)?;
            let covariate_records_by_identifier = build_covariate_records_by_key(
                &covariate_table,
                &selected_covariate_names,
                inputs.sample_key_mode,
                inputs.allow_duplicate_iid_alignment,
            )?;
            (selected_covariate_names, covariate_records_by_identifier)
        }
        None => {
            if inputs.covariate_names.is_some() {
                return Err("Covariate names cannot be provided without a covariate table.".to_string());
            }
            (Vec::new(), HashMap::new())
        }
    };

    let mut aligned_rows = Vec::new();
    for sample_array_index in 0..inputs.sample_indices.len() {
        let sample_key = build_sample_key(
            inputs.sample_key_mode,
            &inputs.family_identifiers[sample_array_index],
            &inputs.individual_identifiers[sample_array_index],
        );
        let Some(phenotype_records) = phenotype_records_by_identifier.get(&sample_key) else {
            continue;
        };
        if inputs.covariate_path.is_none() {
            append_intercept_only_aligned_rows(&inputs, sample_array_index, phenotype_records, &mut aligned_rows)?;
            continue;
        }
        let Some(covariate_records) = covariate_records_by_identifier.get(&sample_key) else {
            continue;
        };
        append_covariate_aligned_rows(
            &inputs,
            sample_array_index,
            phenotype_records,
            covariate_records,
            &mut aligned_rows,
        )?;
    }

    aligned_rows.sort_by_key(|row| row.sample_index);
    if aligned_rows.is_empty() {
        return Err("No aligned samples remain after joining phenotype and covariate tables.".to_string());
    }

    let mut returned_covariate_names = Vec::with_capacity(selected_covariate_names.len() + 1);
    returned_covariate_names.push("intercept".to_string());
    returned_covariate_names.extend(selected_covariate_names);
    Ok(AlignedSampleData::new(inputs.phenotype_name, returned_covariate_names, aligned_rows, inputs.is_binary_trait))
}

pub fn align_multi_sample_data(inputs: MultiAlignmentInputs) -> Result<MultiAlignedSampleData, String> {
    if inputs.phenotype_names.is_empty() {
        return Err("At least one phenotype is required for multi-phenotype alignment.".to_string());
    }
    let mut aligned_sample_data_by_trait = Vec::with_capacity(inputs.phenotype_names.len());
    for phenotype_name in &inputs.phenotype_names {
        aligned_sample_data_by_trait.push(align_sample_data(AlignmentInputs {
            sample_indices: inputs.sample_indices.clone(),
            family_identifiers: inputs.family_identifiers.clone(),
            individual_identifiers: inputs.individual_identifiers.clone(),
            phenotype_path: inputs.phenotype_path.clone(),
            phenotype_name: phenotype_name.clone(),
            covariate_path: inputs.covariate_path.clone(),
            covariate_names: inputs.covariate_names.clone(),
            is_binary_trait: inputs.is_binary_trait,
            sample_key_mode: inputs.sample_key_mode,
            allow_duplicate_iid_alignment: inputs.allow_duplicate_iid_alignment,
        })?);
    }
    let common_positions_by_trait = build_complete_case_positions(&aligned_sample_data_by_trait)?;
    Ok(MultiAlignedSampleData::new(
        inputs.phenotype_names,
        &aligned_sample_data_by_trait,
        &common_positions_by_trait,
        inputs.is_binary_trait,
    ))
}

pub fn align_sample_data_from_sample_file(
    sample_path: &Path,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_name: String,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: SampleKeyMode,
    allow_duplicate_iid_alignment: bool,
) -> Result<AlignedSampleData, String> {
    let sample_identifier_data = load_sample_identifier_data_from_sample_file(sample_path, expected_sample_count)?;
    let inputs = AlignmentInputs {
        sample_indices: sample_identifier_data.sample_indices,
        family_identifiers: sample_identifier_data.family_identifiers,
        individual_identifiers: sample_identifier_data.individual_identifiers,
        phenotype_path,
        phenotype_name,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode,
        allow_duplicate_iid_alignment,
    };
    align_sample_data(inputs)
}

pub fn align_multi_sample_data_from_sample_file(
    sample_path: &Path,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: SampleKeyMode,
    allow_duplicate_iid_alignment: bool,
) -> Result<MultiAlignedSampleData, String> {
    let sample_identifier_data = load_sample_identifier_data_from_sample_file(sample_path, expected_sample_count)?;
    let inputs = MultiAlignmentInputs {
        sample_indices: sample_identifier_data.sample_indices,
        family_identifiers: sample_identifier_data.family_identifiers,
        individual_identifiers: sample_identifier_data.individual_identifiers,
        phenotype_path,
        phenotype_names,
        covariate_path,
        covariate_names,
        is_binary_trait,
        sample_key_mode,
        allow_duplicate_iid_alignment,
    };
    align_multi_sample_data(inputs)
}

fn build_complete_case_positions(
    aligned_sample_data_by_trait: &[AlignedSampleData],
) -> Result<Vec<Vec<usize>>, String> {
    let mut positions_by_trait = Vec::with_capacity(aligned_sample_data_by_trait.len());
    let mut key_sets = Vec::with_capacity(aligned_sample_data_by_trait.len());
    for aligned_sample_data in aligned_sample_data_by_trait {
        let mut positions_by_key = HashMap::with_capacity(aligned_sample_data.sample_indices.len());
        for row_index in 0..aligned_sample_data.sample_indices.len() {
            let aligned_sample_key = AlignedSampleKey {
                sample_index: aligned_sample_data.sample_indices[row_index],
                family_identifier: aligned_sample_data.family_identifiers[row_index].clone(),
                individual_identifier: aligned_sample_data.individual_identifiers[row_index].clone(),
            };
            if positions_by_key.insert(aligned_sample_key.clone(), row_index).is_some() {
                return Err(format!(
                    "Duplicate aligned sample key '{}_{}' at BGEN sample index {} prevents unambiguous multi-phenotype complete-case alignment.",
                    aligned_sample_key.family_identifier,
                    aligned_sample_key.individual_identifier,
                    aligned_sample_key.sample_index,
                ));
            }
        }
        key_sets.push(positions_by_key.keys().cloned().collect::<Vec<_>>());
        positions_by_trait.push(positions_by_key);
    }

    let first_aligned_sample_data = &aligned_sample_data_by_trait[0];
    let mut common_positions_by_trait = vec![Vec::new(); aligned_sample_data_by_trait.len()];
    for row_index in 0..first_aligned_sample_data.sample_indices.len() {
        let aligned_sample_key = AlignedSampleKey {
            sample_index: first_aligned_sample_data.sample_indices[row_index],
            family_identifier: first_aligned_sample_data.family_identifiers[row_index].clone(),
            individual_identifier: first_aligned_sample_data.individual_identifiers[row_index].clone(),
        };
        if !key_sets.iter().all(|key_set| key_set.contains(&aligned_sample_key)) {
            continue;
        }
        for (trait_index, positions_by_key) in positions_by_trait.iter().enumerate() {
            let position = positions_by_key
                .get(&aligned_sample_key)
                .ok_or_else(|| "Internal multi-phenotype alignment key mismatch.".to_string())?;
            common_positions_by_trait[trait_index].push(*position);
        }
    }
    if common_positions_by_trait[0].is_empty() {
        return Err("No aligned samples remain after complete-case multi-phenotype intersection.".to_string());
    }
    Ok(common_positions_by_trait)
}

fn validate_alignment_input_lengths(inputs: &AlignmentInputs) -> Result<(), String> {
    if inputs.sample_indices.len() != inputs.family_identifiers.len()
        || inputs.sample_indices.len() != inputs.individual_identifiers.len()
    {
        return Err(format!(
            "Sample alignment arrays must have equal length: sample_indices={}, family_identifiers={}, individual_identifiers={}.",
            inputs.sample_indices.len(),
            inputs.family_identifiers.len(),
            inputs.individual_identifiers.len(),
        ));
    }
    Ok(())
}

fn validate_alignment_config(
    sample_key_mode: SampleKeyMode,
    allow_duplicate_iid_alignment: bool,
) -> Result<(), String> {
    if sample_key_mode == SampleKeyMode::FidIid && allow_duplicate_iid_alignment {
        return Err("allow_duplicate_iid_alignment is only supported when sample_key_mode='iid'.".to_string());
    }
    Ok(())
}

fn validate_sample_identifier_keys(inputs: &AlignmentInputs) -> Result<(), String> {
    match inputs.sample_key_mode {
        SampleKeyMode::Iid => {
            if !inputs.allow_duplicate_iid_alignment {
                reject_duplicate_individual_identifiers(&inputs.individual_identifiers, "BGEN/sample identifiers")?;
            }
        }
        SampleKeyMode::FidIid => {
            reject_duplicate_sample_keys(
                &inputs.family_identifiers,
                &inputs.individual_identifiers,
                "BGEN/sample identifiers",
            )?;
        }
    }
    Ok(())
}

fn reject_duplicate_individual_identifiers(individual_identifiers: &[String], source_name: &str) -> Result<(), String> {
    let mut observed_identifiers: HashMap<&str, usize> = HashMap::new();
    for individual_identifier in individual_identifiers {
        if individual_identifier.is_empty() {
            continue;
        }
        let occurrence_count = observed_identifiers.entry(individual_identifier.as_str()).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(format!(
                "Duplicate IID '{individual_identifier}' found in {source_name}; sample_key_mode='iid' requires unique non-null IID values. Use sample_key_mode='fid_iid' for datasets with non-globally-unique IID, or allow_duplicate_iid_alignment for legacy IID alignment."
            ));
        }
    }
    Ok(())
}

fn reject_duplicate_sample_keys(
    family_identifiers: &[String],
    individual_identifiers: &[String],
    source_name: &str,
) -> Result<(), String> {
    let mut observed_identifiers: HashMap<(&str, &str), usize> = HashMap::new();
    for (family_identifier, individual_identifier) in family_identifiers.iter().zip(individual_identifiers.iter()) {
        if individual_identifier.is_empty() {
            continue;
        }
        let sample_key = (family_identifier.as_str(), individual_identifier.as_str());
        let occurrence_count = observed_identifiers.entry(sample_key).or_insert(0);
        *occurrence_count += 1;
        if *occurrence_count > 1 {
            return Err(format!(
                "Duplicate sample key '{family_identifier}_{individual_identifier}' found in {source_name}; sample_key_mode='fid_iid' requires unique (FID, IID) values."
            ));
        }
    }
    Ok(())
}

fn load_sample_identifier_data_from_sample_file(
    sample_path: &Path,
    expected_sample_count: usize,
) -> Result<SampleIdentifierData, String> {
    let sample_content = std::fs::read_to_string(sample_path)
        .map_err(|error| format!("Failed to read sample file '{}': {error}.", sample_path.display()))?;
    let non_empty_lines: Vec<String> = sample_content
        .lines()
        .map(|line| line.trim_end_matches('\r').to_string())
        .filter(|line| !line.trim().is_empty())
        .collect();
    if non_empty_lines.len() < 2 {
        return Err(format!("Sample file '{}' must contain at least two header lines.", sample_path.display()));
    }

    let column_names = split_sample_file_line(&non_empty_lines[0]);
    let column_types = split_sample_file_line(&non_empty_lines[1]);
    validate_sample_file_header(sample_path, &column_names, &column_types)?;
    let family_identifier_column_index = 0;
    let individual_identifier_column_index =
        column_names.iter().position(|column_name| column_name == "ID_2").unwrap_or(family_identifier_column_index);
    let sample_count = non_empty_lines.len() - 2;
    if sample_count != expected_sample_count {
        return Err(format!(
            "Expect number of samples in file to match BGEN sample count. Sample file '{}' contains {sample_count} rows, but the BGEN contains {expected_sample_count} samples.",
            sample_path.display()
        ));
    }

    let mut sample_indices = Vec::with_capacity(sample_count);
    let mut family_identifiers = Vec::with_capacity(sample_count);
    let mut individual_identifiers = Vec::with_capacity(sample_count);
    for (line_index, raw_line) in non_empty_lines[2..].iter().enumerate() {
        let row_values = split_sample_file_line(raw_line);
        if row_values.len() != column_names.len() {
            return Err(format!(
                "Sample file '{}' line {} has {} values, but the header declares {} columns.",
                sample_path.display(),
                line_index + 3,
                row_values.len(),
                column_names.len(),
            ));
        }
        sample_indices.push(i64::try_from(line_index).map_err(|error| error.to_string())?);
        family_identifiers.push(row_values[family_identifier_column_index].clone());
        individual_identifiers.push(row_values[individual_identifier_column_index].clone());
    }
    Ok(SampleIdentifierData { sample_indices, family_identifiers, individual_identifiers })
}

fn split_sample_file_line(raw_line: &str) -> Vec<String> {
    raw_line.split_whitespace().map(ToString::to_string).collect()
}

fn validate_sample_file_header(
    sample_path: &Path,
    column_names: &[String],
    column_types: &[String],
) -> Result<(), String> {
    if column_names.len() != column_types.len() {
        return Err(format!(
            "Sample file '{}' header and type lines have different column counts.",
            sample_path.display()
        ));
    }
    if column_names.is_empty() {
        return Err(format!("Sample file '{}' does not contain any columns.", sample_path.display()));
    }
    if column_types[0] != "0" {
        return Err(format!(
            "Sample file '{}' must mark the first identifier column with type '0'.",
            sample_path.display()
        ));
    }
    if let Some(individual_identifier_column_index) = column_names.iter().position(|column_name| column_name == "ID_2")
        && column_types[individual_identifier_column_index] != "0"
    {
        return Err(format!("Sample file '{}' must mark 'ID_2' with type '0'.", sample_path.display()));
    }
    Ok(())
}

fn read_tabular_table(table_path: &Path) -> Result<TabularTable, String> {
    let table_content = std::fs::read_to_string(table_path)
        .map_err(|error| format!("Failed to read table '{}': {error}.", table_path.display()))?;
    let mut lines = table_content.lines();
    let Some(header_line) = lines.next() else {
        return Err(format!("Table '{}' is empty.", table_path.display()));
    };
    let headers = split_tabular_line(header_line);
    if headers.is_empty() {
        return Err(format!("Table '{}' must contain a header row.", table_path.display()));
    }
    let rows = lines.map(split_tabular_line).collect();
    Ok(TabularTable { headers, rows })
}

fn split_tabular_line(line: &str) -> Vec<String> {
    line.trim_end_matches('\r').split('\t').map(ToString::to_string).collect()
}

fn validate_required_column(table: &TabularTable, column_name: &str, table_path: &str) -> Result<(), String> {
    column_index(table, column_name).map(|_| ()).ok_or_else(|| {
        if column_name == "FID" || column_name == "IID" {
            format!("Identifier column '{column_name}' was not found in {table_path}.")
        } else {
            format!("Phenotype column '{column_name}' was not found in {table_path}.")
        }
    })
}

fn column_index(table: &TabularTable, column_name: &str) -> Option<usize> {
    table.headers.iter().position(|header| header == column_name)
}

fn row_value(row: &[String], column_index: usize) -> &str {
    row.get(column_index).map_or("", String::as_str)
}

fn is_tabular_null_value(value: &str) -> bool {
    matches!(value, "" | "NA" | "NaN" | "nan" | "-9")
}

fn build_sample_key(sample_key_mode: SampleKeyMode, family_identifier: &str, individual_identifier: &str) -> SampleKey {
    match sample_key_mode {
        SampleKeyMode::Iid => SampleKey::Iid(individual_identifier.to_string()),
        SampleKeyMode::FidIid => SampleKey::FidIid {
            family_identifier: family_identifier.to_string(),
            individual_identifier: individual_identifier.to_string(),
        },
    }
}

fn build_phenotype_records_by_key(
    phenotype_table: &TabularTable,
    phenotype_name: &str,
    sample_key_mode: SampleKeyMode,
    allow_duplicate_iid_alignment: bool,
) -> Result<HashMap<SampleKey, Vec<RawPhenotypeRecord>>, String> {
    let family_identifier_index = column_index(phenotype_table, "FID");
    let individual_identifier_index = column_index(phenotype_table, "IID")
        .ok_or_else(|| "Identifier column 'IID' was not found in phenotype table.".to_string())?;
    let phenotype_index = column_index(phenotype_table, phenotype_name)
        .ok_or_else(|| format!("Phenotype column '{phenotype_name}' was not found in phenotype table."))?;
    let mut records_by_key: HashMap<SampleKey, Vec<RawPhenotypeRecord>> = HashMap::new();
    for row in &phenotype_table.rows {
        let individual_identifier = row_value(row, individual_identifier_index);
        let phenotype_value = row_value(row, phenotype_index);
        if individual_identifier.is_empty() || is_tabular_null_value(phenotype_value) {
            continue;
        }
        let family_identifier = family_identifier_index.map_or("", |column| row_value(row, column));
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if sample_key_mode == SampleKeyMode::Iid
            && !allow_duplicate_iid_alignment
            && records_by_key.contains_key(&sample_key)
        {
            return Err(format!(
                "Duplicate IID '{individual_identifier}' found in phenotype table; sample_key_mode='iid' requires unique non-null IID values."
            ));
        }
        if sample_key_mode == SampleKeyMode::FidIid && records_by_key.contains_key(&sample_key) {
            return Err(format!(
                "Duplicate sample key '{family_identifier}_{individual_identifier}' found in phenotype table; sample_key_mode='fid_iid' requires unique (FID, IID) values."
            ));
        }
        records_by_key
            .entry(sample_key)
            .or_default()
            .push(RawPhenotypeRecord { phenotype_value: phenotype_value.to_string() });
    }
    Ok(records_by_key)
}

fn select_covariate_names(
    covariate_table: &TabularTable,
    requested_covariate_names: Option<&[String]>,
    covariate_path: &str,
) -> Result<Vec<String>, String> {
    match requested_covariate_names {
        Some(covariate_names) => {
            let missing_covariates: Vec<String> = covariate_names
                .iter()
                .filter(|covariate_name| column_index(covariate_table, covariate_name).is_none())
                .cloned()
                .collect();
            if !missing_covariates.is_empty() {
                return Err(format!("Covariate columns are missing from {covariate_path}: {missing_covariates:?}."));
            }
            Ok(covariate_names.to_vec())
        }
        None => {
            let inferred_covariate_names: Vec<String> = covariate_table
                .headers
                .iter()
                .filter(|column_name| column_name.as_str() != "FID" && column_name.as_str() != "IID")
                .cloned()
                .collect();
            if inferred_covariate_names.is_empty() {
                return Err("Covariate table must contain at least one non-identifier column.".to_string());
            }
            Ok(inferred_covariate_names)
        }
    }
}

fn build_covariate_records_by_key(
    covariate_table: &TabularTable,
    selected_covariate_names: &[String],
    sample_key_mode: SampleKeyMode,
    allow_duplicate_iid_alignment: bool,
) -> Result<HashMap<SampleKey, Vec<RawCovariateRecord>>, String> {
    let family_identifier_index = column_index(covariate_table, "FID");
    let individual_identifier_index = column_index(covariate_table, "IID")
        .ok_or_else(|| "Identifier column 'IID' was not found in covariate table.".to_string())?;
    let covariate_indices: Vec<usize> = selected_covariate_names
        .iter()
        .map(|covariate_name| {
            column_index(covariate_table, covariate_name)
                .ok_or_else(|| format!("Covariate column '{covariate_name}' was not found."))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut records_by_key: HashMap<SampleKey, Vec<RawCovariateRecord>> = HashMap::new();
    for row in &covariate_table.rows {
        let individual_identifier = row_value(row, individual_identifier_index);
        if individual_identifier.is_empty() {
            continue;
        }
        let covariate_values: Vec<String> = covariate_indices
            .iter()
            .map(|covariate_index| row_value(row, *covariate_index))
            .filter(|covariate_value| !is_tabular_null_value(covariate_value))
            .map(ToString::to_string)
            .collect();
        if covariate_values.len() != selected_covariate_names.len() {
            continue;
        }
        let family_identifier = family_identifier_index.map_or("", |column| row_value(row, column));
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if sample_key_mode == SampleKeyMode::Iid
            && !allow_duplicate_iid_alignment
            && records_by_key.contains_key(&sample_key)
        {
            return Err(format!(
                "Duplicate IID '{individual_identifier}' found in covariate table; sample_key_mode='iid' requires unique non-null IID values."
            ));
        }
        if sample_key_mode == SampleKeyMode::FidIid && records_by_key.contains_key(&sample_key) {
            return Err(format!(
                "Duplicate sample key '{family_identifier}_{individual_identifier}' found in covariate table; sample_key_mode='fid_iid' requires unique (FID, IID) values."
            ));
        }
        records_by_key.entry(sample_key).or_default().push(RawCovariateRecord { covariate_values });
    }
    Ok(records_by_key)
}

fn append_intercept_only_aligned_rows(
    inputs: &AlignmentInputs,
    sample_array_index: usize,
    phenotype_records: &[RawPhenotypeRecord],
    aligned_rows: &mut Vec<AlignedSampleRow>,
) -> Result<(), String> {
    for phenotype_record in phenotype_records {
        let phenotype_value =
            parse_phenotype_value(&phenotype_record.phenotype_value, &inputs.phenotype_name, inputs.is_binary_trait)?;
        aligned_rows.push(build_aligned_sample_row(inputs, sample_array_index, phenotype_value, Vec::new()));
    }
    Ok(())
}

fn append_covariate_aligned_rows(
    inputs: &AlignmentInputs,
    sample_array_index: usize,
    phenotype_records: &[RawPhenotypeRecord],
    covariate_records: &[RawCovariateRecord],
    aligned_rows: &mut Vec<AlignedSampleRow>,
) -> Result<(), String> {
    for phenotype_record in phenotype_records {
        let phenotype_value =
            parse_phenotype_value(&phenotype_record.phenotype_value, &inputs.phenotype_name, inputs.is_binary_trait)?;
        for covariate_record in covariate_records {
            let covariate_values = parse_covariate_values(&covariate_record.covariate_values)?;
            aligned_rows.push(build_aligned_sample_row(inputs, sample_array_index, phenotype_value, covariate_values));
        }
    }
    Ok(())
}

fn build_aligned_sample_row(
    inputs: &AlignmentInputs,
    sample_array_index: usize,
    phenotype_value: f32,
    covariate_values: Vec<f32>,
) -> AlignedSampleRow {
    AlignedSampleRow {
        sample_index: inputs.sample_indices[sample_array_index],
        family_identifier: inputs.family_identifiers[sample_array_index].clone(),
        individual_identifier: inputs.individual_identifiers[sample_array_index].clone(),
        phenotype_value,
        covariate_values,
    }
}

fn parse_phenotype_value(phenotype_value: &str, phenotype_name: &str, is_binary_trait: bool) -> Result<f32, String> {
    let parsed_value = phenotype_value.parse::<f32>().map_err(|error| {
        format!("Failed to parse phenotype column '{phenotype_name}' value '{phenotype_value}': {error}.")
    })?;
    if !is_binary_trait {
        return Ok(parsed_value);
    }
    if parsed_value == 1.0 {
        return Ok(0.0);
    }
    if parsed_value == 2.0 {
        return Ok(1.0);
    }
    Err(format!("Binary phenotype must contain only values 1 and 2, found value {parsed_value}."))
}

fn parse_covariate_values(covariate_values: &[String]) -> Result<Vec<f32>, String> {
    covariate_values
        .iter()
        .map(|covariate_value| {
            covariate_value
                .parse::<f32>()
                .map_err(|error| format!("Failed to parse covariate value '{covariate_value}': {error}."))
        })
        .collect()
}
