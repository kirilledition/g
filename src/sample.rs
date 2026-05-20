//! Native sample alignment and Oxford sample-file parsing.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::float_cmp)]
#![allow(clippy::single_match_else)]
#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

const TABULAR_MISSING_VALUE_TOKENS: &[&str] = &["", "NA", "NaN", "nan", "-9"];

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
}

struct RawPhenotypeRecord {
    phenotype_value: String,
}

struct RawMultiPhenotypeRecord {
    phenotype_values: Vec<Option<String>>,
}

struct RawCovariateRecord {
    covariate_values: Vec<String>,
}

#[derive(Clone, Debug)]
struct SelectedTabularColumn {
    column_index: usize,
    selected_value_index: usize,
}

struct TabularColumnSelection {
    family_identifier_value_index: Option<usize>,
    individual_identifier_value_index: usize,
    data_value_indices: Vec<usize>,
    selected_columns: Vec<SelectedTabularColumn>,
}

#[derive(Clone, Copy)]
enum TabularDelimiter {
    Tab,
    Space,
}

impl TabularDelimiter {
    fn byte(self) -> u8 {
        match self {
            Self::Tab => b'\t',
            Self::Space => b' ',
        }
    }
}

struct StreamingTabularReader<R: Read> {
    path_text: String,
    source_label: &'static str,
    reader: csv::Reader<R>,
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
    validate_sample_identifier_keys(&inputs)?;

    let phenotype_records_by_identifier = read_phenotype_records_by_key(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_name,
        inputs.sample_key_mode,
    )?;

    let (selected_covariate_names, covariate_records_by_identifier) = match inputs.covariate_path.as_ref() {
        Some(covariate_path) => read_covariate_records_by_key(
            Path::new(covariate_path),
            inputs.covariate_names.as_deref(),
            inputs.sample_key_mode,
        )?,
        None => {
            if inputs.covariate_names.is_some() {
                return Err("Covariate names cannot be provided without a covariate table.".to_string());
            }
            (Vec::new(), HashMap::new())
        }
    };

    build_single_aligned_sample_data(
        inputs,
        &phenotype_records_by_identifier,
        &selected_covariate_names,
        &covariate_records_by_identifier,
    )
}

/// Align several phenotypes to one shared complete-case sample set.
///
/// This intentionally intersects all per-trait valid sample sets and therefore
/// is not equivalent to running each phenotype through `align_sample_data`.
pub fn align_multi_sample_data(inputs: MultiAlignmentInputs) -> Result<MultiAlignedSampleData, String> {
    if inputs.phenotype_names.is_empty() {
        return Err("At least one phenotype is required for multi-phenotype alignment.".to_string());
    }
    validate_multi_alignment_input_lengths(&inputs)?;
    validate_multi_sample_identifier_keys(&inputs)?;

    let phenotype_records_by_identifier = read_multi_phenotype_records_by_key(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_names,
        inputs.sample_key_mode,
    )?;
    let (selected_covariate_names, covariate_records_by_identifier) = match inputs.covariate_path.as_ref() {
        Some(covariate_path) => read_covariate_records_by_key(
            Path::new(covariate_path),
            inputs.covariate_names.as_deref(),
            inputs.sample_key_mode,
        )?,
        None => {
            if inputs.covariate_names.is_some() {
                return Err("Covariate names cannot be provided without a covariate table.".to_string());
            }
            (Vec::new(), HashMap::new())
        }
    };

    let mut aligned_sample_data_by_trait = Vec::with_capacity(inputs.phenotype_names.len());
    for phenotype_index in 0..inputs.phenotype_names.len() {
        aligned_sample_data_by_trait.push(build_multi_trait_aligned_sample_data(
            &inputs,
            phenotype_index,
            &phenotype_records_by_identifier,
            &selected_covariate_names,
            &covariate_records_by_identifier,
        )?);
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

fn validate_multi_alignment_input_lengths(inputs: &MultiAlignmentInputs) -> Result<(), String> {
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

fn validate_sample_identifier_keys(inputs: &AlignmentInputs) -> Result<(), String> {
    match inputs.sample_key_mode {
        SampleKeyMode::Iid => {
            reject_duplicate_individual_identifiers(&inputs.individual_identifiers, "BGEN/sample identifiers")?;
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

fn validate_multi_sample_identifier_keys(inputs: &MultiAlignmentInputs) -> Result<(), String> {
    match inputs.sample_key_mode {
        SampleKeyMode::Iid => {
            reject_duplicate_individual_identifiers(&inputs.individual_identifiers, "BGEN/sample identifiers")?;
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
                "Duplicate IID '{individual_identifier}' found in {source_name}; sample_key_mode='iid' requires unique non-null IID values. Use sample_key_mode='fid_iid' for datasets with non-globally-unique IID."
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
    let mut reader = open_sample_file_reader(sample_path)?;
    let column_names = space_delimited_record_to_strings(&reader.read_required_record(format!(
        "Sample file '{}' must contain at least two header lines.",
        sample_path.display()
    ))?);
    let column_types = space_delimited_record_to_strings(&reader.read_required_record(format!(
        "Sample file '{}' must contain at least two header lines.",
        sample_path.display()
    ))?);
    validate_sample_file_header(sample_path, &column_names, &column_types)?;
    let family_identifier_column_index = 0;
    let individual_identifier_column_index =
        column_names.iter().position(|column_name| column_name == "ID_2").unwrap_or(family_identifier_column_index);

    let mut sample_indices = Vec::with_capacity(expected_sample_count);
    let mut family_identifiers = Vec::with_capacity(expected_sample_count);
    let mut individual_identifiers = Vec::with_capacity(expected_sample_count);
    let mut sample_count = 0usize;
    while let Some(record) = reader.read_next_record()? {
        sample_count += 1;
        let row_values = space_delimited_record_to_strings(&record);
        if row_values.len() != column_names.len() {
            return Err(format!(
                "Sample file '{}' line {} has {} values, but the header declares {} columns.",
                sample_path.display(),
                sample_count + 2,
                row_values.len(),
                column_names.len(),
            ));
        }
        sample_indices.push(i64::try_from(sample_count - 1).map_err(|error| error.to_string())?);
        family_identifiers.push(row_values[family_identifier_column_index].clone());
        individual_identifiers.push(row_values[individual_identifier_column_index].clone());
    }
    if sample_count != expected_sample_count {
        return Err(format!(
            "Expect number of samples in file to match BGEN sample count. Sample file '{}' contains {sample_count} rows, but the BGEN contains {expected_sample_count} samples.",
            sample_path.display()
        ));
    }
    Ok(SampleIdentifierData { sample_indices, family_identifiers, individual_identifiers })
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

fn open_sample_file_reader(sample_path: &Path) -> Result<StreamingTabularReader<File>, String> {
    open_tabular_reader(sample_path, "sample file", TabularDelimiter::Space)
}

fn open_tsv_table_reader(table_path: &Path) -> Result<StreamingTabularReader<File>, String> {
    open_tabular_reader(table_path, "table", TabularDelimiter::Tab)
}

fn open_tabular_reader(
    table_path: &Path,
    source_label: &'static str,
    delimiter: TabularDelimiter,
) -> Result<StreamingTabularReader<File>, String> {
    let table_file = File::open(table_path)
        .map_err(|error| format!("Failed to read {source_label} '{}': {error}.", table_path.display()))?;
    Ok(StreamingTabularReader::new(table_path.display().to_string(), source_label, table_file, delimiter))
}

fn read_tabular_header<R: Read>(
    reader: &mut StreamingTabularReader<R>,
    table_path: &Path,
) -> Result<Vec<String>, String> {
    let headers =
        record_to_strings(&reader.read_required_record(format!("Table '{}' is empty.", table_path.display()))?);
    if headers.is_empty() {
        return Err(format!("Table '{}' must contain a header row.", table_path.display()));
    }
    Ok(headers)
}

impl<R: Read> StreamingTabularReader<R> {
    fn new(path_text: String, source_label: &'static str, source: R, delimiter: TabularDelimiter) -> Self {
        let reader = csv::ReaderBuilder::new()
            .delimiter(delimiter.byte())
            .flexible(true)
            .has_headers(false)
            .trim(csv::Trim::All)
            .from_reader(source);
        Self { path_text, source_label, reader }
    }

    fn read_required_record(&mut self, empty_error_message: String) -> Result<csv::StringRecord, String> {
        self.read_next_record()?.ok_or(empty_error_message)
    }

    fn read_next_record(&mut self) -> Result<Option<csv::StringRecord>, String> {
        let mut record = csv::StringRecord::new();
        loop {
            let has_record = self
                .reader
                .read_record(&mut record)
                .map_err(|error| format!("Failed to read {} '{}': {error}.", self.source_label, self.path_text))?;
            if !has_record {
                return Ok(None);
            }
            if !is_empty_tabular_record(&record) {
                return Ok(Some(record));
            }
            record.clear();
        }
    }
}

fn is_empty_tabular_record(record: &csv::StringRecord) -> bool {
    record.iter().all(str::is_empty)
}

fn record_to_strings(record: &csv::StringRecord) -> Vec<String> {
    record.iter().map(ToString::to_string).collect()
}

fn space_delimited_record_to_strings(record: &csv::StringRecord) -> Vec<String> {
    record.iter().filter(|field_value| !field_value.is_empty()).map(ToString::to_string).collect()
}

fn required_column_index(headers: &[String], column_name: &str, table_path: &str) -> Result<usize, String> {
    column_index(headers, column_name).ok_or_else(|| {
        if column_name == "FID" || column_name == "IID" {
            format!("Identifier column '{column_name}' was not found in {table_path}.")
        } else {
            format!("Phenotype column '{column_name}' was not found in {table_path}.")
        }
    })
}

fn column_index(headers: &[String], column_name: &str) -> Option<usize> {
    headers.iter().position(|header| header == column_name)
}

impl TabularColumnSelection {
    fn new(
        family_identifier_column_index: Option<usize>,
        individual_identifier_column_index: usize,
        data_column_indices: &[usize],
    ) -> Self {
        let mut selected_columns = Vec::with_capacity(data_column_indices.len() + 2);
        let family_identifier_value_index = family_identifier_column_index
            .map(|column_index| push_selected_column(&mut selected_columns, column_index));
        let individual_identifier_value_index =
            push_selected_column(&mut selected_columns, individual_identifier_column_index);
        let data_value_indices = data_column_indices
            .iter()
            .map(|column_index| push_selected_column(&mut selected_columns, *column_index))
            .collect();
        selected_columns.sort_by_key(|selected_column| selected_column.column_index);
        Self { family_identifier_value_index, individual_identifier_value_index, data_value_indices, selected_columns }
    }
}

fn push_selected_column(selected_columns: &mut Vec<SelectedTabularColumn>, column_index: usize) -> usize {
    let selected_value_index = selected_columns.len();
    selected_columns.push(SelectedTabularColumn { column_index, selected_value_index });
    selected_value_index
}

fn select_tabular_record_values<'a>(record: &'a csv::StringRecord, selection: &TabularColumnSelection) -> Vec<&'a str> {
    let mut selected_values = vec![""; selection.selected_columns.len()];
    for selected_column in &selection.selected_columns {
        if let Some(field_value) = record.get(selected_column.column_index) {
            selected_values[selected_column.selected_value_index] = field_value;
        }
    }
    selected_values
}

fn is_tabular_null_value(value: &str) -> bool {
    TABULAR_MISSING_VALUE_TOKENS.contains(&value)
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

fn read_phenotype_records_by_key(
    phenotype_path: &Path,
    phenotype_name: &str,
    sample_key_mode: SampleKeyMode,
) -> Result<HashMap<SampleKey, RawPhenotypeRecord>, String> {
    let mut reader = open_tsv_table_reader(phenotype_path)?;
    let headers = read_tabular_header(&mut reader, phenotype_path)?;
    let phenotype_path_text = phenotype_path.display().to_string();
    let family_identifier_index = column_index(&headers, "FID");
    if sample_key_mode == SampleKeyMode::FidIid {
        required_column_index(&headers, "FID", &phenotype_path_text)?;
    }
    let individual_identifier_index = required_column_index(&headers, "IID", &phenotype_path_text)?;
    let phenotype_index = required_column_index(&headers, phenotype_name, &phenotype_path_text)?;
    let selection =
        TabularColumnSelection::new(family_identifier_index, individual_identifier_index, &[phenotype_index]);
    let mut observed_sample_keys: HashSet<SampleKey> = HashSet::new();
    let mut records_by_key: HashMap<SampleKey, RawPhenotypeRecord> = HashMap::new();
    while let Some(record) = reader.read_next_record()? {
        let selected_values = select_tabular_record_values(&record, &selection);
        let individual_identifier = selected_values[selection.individual_identifier_value_index];
        if individual_identifier.is_empty() {
            continue;
        }
        let family_identifier =
            selection.family_identifier_value_index.map_or("", |value_index| selected_values[value_index]);
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if !observed_sample_keys.insert(sample_key.clone()) {
            if sample_key_mode == SampleKeyMode::FidIid {
                return Err(format!(
                    "Duplicate sample key '{family_identifier}_{individual_identifier}' found in phenotype table; sample_key_mode='fid_iid' requires unique (FID, IID) values."
                ));
            }
            return Err(format!(
                "Duplicate IID '{individual_identifier}' found in phenotype table; sample_key_mode='iid' requires unique non-null IID values."
            ));
        }
        let phenotype_value = selected_values[selection.data_value_indices[0]];
        if is_tabular_null_value(phenotype_value) {
            continue;
        }
        records_by_key.insert(sample_key, RawPhenotypeRecord { phenotype_value: phenotype_value.to_string() });
    }
    Ok(records_by_key)
}

fn read_multi_phenotype_records_by_key(
    phenotype_path: &Path,
    phenotype_names: &[String],
    sample_key_mode: SampleKeyMode,
) -> Result<HashMap<SampleKey, RawMultiPhenotypeRecord>, String> {
    let mut reader = open_tsv_table_reader(phenotype_path)?;
    let headers = read_tabular_header(&mut reader, phenotype_path)?;
    let phenotype_path_text = phenotype_path.display().to_string();
    let family_identifier_index = column_index(&headers, "FID");
    if sample_key_mode == SampleKeyMode::FidIid {
        required_column_index(&headers, "FID", &phenotype_path_text)?;
    }
    let individual_identifier_index = required_column_index(&headers, "IID", &phenotype_path_text)?;
    let phenotype_indices = phenotype_names
        .iter()
        .map(|phenotype_name| required_column_index(&headers, phenotype_name, &phenotype_path_text))
        .collect::<Result<Vec<_>, _>>()?;
    let selection =
        TabularColumnSelection::new(family_identifier_index, individual_identifier_index, &phenotype_indices);
    let mut observed_sample_keys: HashSet<SampleKey> = HashSet::new();
    let mut records_by_key: HashMap<SampleKey, RawMultiPhenotypeRecord> = HashMap::new();
    while let Some(record) = reader.read_next_record()? {
        let selected_values = select_tabular_record_values(&record, &selection);
        let individual_identifier = selected_values[selection.individual_identifier_value_index];
        if individual_identifier.is_empty() {
            continue;
        }
        let family_identifier =
            selection.family_identifier_value_index.map_or("", |value_index| selected_values[value_index]);
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if !observed_sample_keys.insert(sample_key.clone()) {
            if sample_key_mode == SampleKeyMode::FidIid {
                return Err(format!(
                    "Duplicate sample key '{family_identifier}_{individual_identifier}' found in phenotype table; sample_key_mode='fid_iid' requires unique (FID, IID) values."
                ));
            }
            return Err(format!(
                "Duplicate IID '{individual_identifier}' found in phenotype table; sample_key_mode='iid' requires unique non-null IID values."
            ));
        }
        let phenotype_values = selection
            .data_value_indices
            .iter()
            .map(|value_index| {
                let phenotype_value = selected_values[*value_index];
                if is_tabular_null_value(phenotype_value) { None } else { Some(phenotype_value.to_string()) }
            })
            .collect();
        records_by_key.insert(sample_key, RawMultiPhenotypeRecord { phenotype_values });
    }
    Ok(records_by_key)
}

fn select_covariate_names(
    covariate_headers: &[String],
    requested_covariate_names: Option<&[String]>,
    covariate_path: &str,
) -> Result<Vec<String>, String> {
    match requested_covariate_names {
        Some(covariate_names) => {
            let missing_covariates: Vec<String> = covariate_names
                .iter()
                .filter(|covariate_name| column_index(covariate_headers, covariate_name).is_none())
                .cloned()
                .collect();
            if !missing_covariates.is_empty() {
                return Err(format!("Covariate columns are missing from {covariate_path}: {missing_covariates:?}."));
            }
            Ok(covariate_names.to_vec())
        }
        None => {
            let inferred_covariate_names: Vec<String> = covariate_headers
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

fn read_covariate_records_by_key(
    covariate_path: &Path,
    requested_covariate_names: Option<&[String]>,
    sample_key_mode: SampleKeyMode,
) -> Result<(Vec<String>, HashMap<SampleKey, RawCovariateRecord>), String> {
    let mut reader = open_tsv_table_reader(covariate_path)?;
    let headers = read_tabular_header(&mut reader, covariate_path)?;
    let covariate_path_text = covariate_path.display().to_string();
    let family_identifier_index = column_index(&headers, "FID");
    if sample_key_mode == SampleKeyMode::FidIid {
        required_column_index(&headers, "FID", &covariate_path_text)?;
    }
    let individual_identifier_index = required_column_index(&headers, "IID", &covariate_path_text)?;
    let selected_covariate_names = select_covariate_names(&headers, requested_covariate_names, &covariate_path_text)?;
    let covariate_indices: Vec<usize> = selected_covariate_names
        .iter()
        .map(|covariate_name| {
            column_index(&headers, covariate_name)
                .ok_or_else(|| format!("Covariate column '{covariate_name}' was not found."))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let selection =
        TabularColumnSelection::new(family_identifier_index, individual_identifier_index, &covariate_indices);
    let mut observed_sample_keys: HashSet<SampleKey> = HashSet::new();
    let mut records_by_key: HashMap<SampleKey, RawCovariateRecord> = HashMap::new();
    while let Some(record) = reader.read_next_record()? {
        let selected_values = select_tabular_record_values(&record, &selection);
        let individual_identifier = selected_values[selection.individual_identifier_value_index];
        if individual_identifier.is_empty() {
            continue;
        }
        let family_identifier =
            selection.family_identifier_value_index.map_or("", |value_index| selected_values[value_index]);
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if !observed_sample_keys.insert(sample_key.clone()) {
            if sample_key_mode == SampleKeyMode::FidIid {
                return Err(format!(
                    "Duplicate sample key '{family_identifier}_{individual_identifier}' found in covariate table; sample_key_mode='fid_iid' requires unique (FID, IID) values."
                ));
            }
            return Err(format!(
                "Duplicate IID '{individual_identifier}' found in covariate table; sample_key_mode='iid' requires unique non-null IID values."
            ));
        }
        let covariate_values: Vec<String> = covariate_indices
            .iter()
            .enumerate()
            .map(|(covariate_index, _column_index)| selected_values[selection.data_value_indices[covariate_index]])
            .filter(|covariate_value| !is_tabular_null_value(covariate_value))
            .map(ToString::to_string)
            .collect();
        if covariate_values.len() != selected_covariate_names.len() {
            continue;
        }
        records_by_key.insert(sample_key, RawCovariateRecord { covariate_values });
    }
    Ok((selected_covariate_names, records_by_key))
}

fn build_single_aligned_sample_data(
    inputs: AlignmentInputs,
    phenotype_records_by_identifier: &HashMap<SampleKey, RawPhenotypeRecord>,
    selected_covariate_names: &[String],
    covariate_records_by_identifier: &HashMap<SampleKey, RawCovariateRecord>,
) -> Result<AlignedSampleData, String> {
    let mut aligned_rows = Vec::new();
    for sample_array_index in 0..inputs.sample_indices.len() {
        let sample_key = build_sample_key(
            inputs.sample_key_mode,
            &inputs.family_identifiers[sample_array_index],
            &inputs.individual_identifiers[sample_array_index],
        );
        let Some(phenotype_record) = phenotype_records_by_identifier.get(&sample_key) else {
            continue;
        };
        let phenotype_value =
            parse_phenotype_value(&phenotype_record.phenotype_value, &inputs.phenotype_name, inputs.is_binary_trait)?;
        let covariate_values = if inputs.covariate_path.is_some() {
            let Some(covariate_record) = covariate_records_by_identifier.get(&sample_key) else {
                continue;
            };
            parse_covariate_values(&covariate_record.covariate_values)?
        } else {
            Vec::new()
        };
        aligned_rows.push(AlignedSampleRow {
            sample_index: inputs.sample_indices[sample_array_index],
            family_identifier: inputs.family_identifiers[sample_array_index].clone(),
            individual_identifier: inputs.individual_identifiers[sample_array_index].clone(),
            phenotype_value,
            covariate_values,
        });
    }

    aligned_rows.sort_by_key(|row| row.sample_index);
    if aligned_rows.is_empty() {
        return Err("No aligned samples remain after joining phenotype and covariate tables.".to_string());
    }

    Ok(AlignedSampleData::new(
        inputs.phenotype_name,
        returned_covariate_names(selected_covariate_names),
        aligned_rows,
        inputs.is_binary_trait,
    ))
}

fn build_multi_trait_aligned_sample_data(
    inputs: &MultiAlignmentInputs,
    phenotype_index: usize,
    phenotype_records_by_identifier: &HashMap<SampleKey, RawMultiPhenotypeRecord>,
    selected_covariate_names: &[String],
    covariate_records_by_identifier: &HashMap<SampleKey, RawCovariateRecord>,
) -> Result<AlignedSampleData, String> {
    let phenotype_name = &inputs.phenotype_names[phenotype_index];
    let mut aligned_rows = Vec::new();
    for sample_array_index in 0..inputs.sample_indices.len() {
        let sample_key = build_sample_key(
            inputs.sample_key_mode,
            &inputs.family_identifiers[sample_array_index],
            &inputs.individual_identifiers[sample_array_index],
        );
        let Some(phenotype_record) = phenotype_records_by_identifier.get(&sample_key) else {
            continue;
        };
        let Some(phenotype_value_text) =
            phenotype_record.phenotype_values.get(phenotype_index).and_then(Option::as_deref)
        else {
            continue;
        };
        let phenotype_value = parse_phenotype_value(phenotype_value_text, phenotype_name, inputs.is_binary_trait)?;
        let covariate_values = if inputs.covariate_path.is_some() {
            let Some(covariate_record) = covariate_records_by_identifier.get(&sample_key) else {
                continue;
            };
            parse_covariate_values(&covariate_record.covariate_values)?
        } else {
            Vec::new()
        };
        aligned_rows.push(AlignedSampleRow {
            sample_index: inputs.sample_indices[sample_array_index],
            family_identifier: inputs.family_identifiers[sample_array_index].clone(),
            individual_identifier: inputs.individual_identifiers[sample_array_index].clone(),
            phenotype_value,
            covariate_values,
        });
    }

    aligned_rows.sort_by_key(|row| row.sample_index);
    if aligned_rows.is_empty() {
        return Err("No aligned samples remain after joining phenotype and covariate tables.".to_string());
    }

    Ok(AlignedSampleData::new(
        phenotype_name.clone(),
        returned_covariate_names(selected_covariate_names),
        aligned_rows,
        inputs.is_binary_trait,
    ))
}

fn returned_covariate_names(selected_covariate_names: &[String]) -> Vec<String> {
    let mut covariate_names = Vec::with_capacity(selected_covariate_names.len() + 1);
    covariate_names.push("intercept".to_string());
    covariate_names.extend(selected_covariate_names.iter().cloned());
    covariate_names
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
