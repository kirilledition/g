//! Native sample alignment and Oxford sample-file parsing.

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::float_cmp)]
#![allow(clippy::single_match_else)]
#![allow(clippy::too_many_arguments)]

use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, Read};
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

#[derive(Clone, Debug, PartialEq)]
pub struct AlignedPhenotypeGroup {
    pub phenotype_indices: Vec<usize>,
    pub aligned_sample_data: MultiAlignedSampleData,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupedAlignedSampleData {
    pub groups: Vec<AlignedPhenotypeGroup>,
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

struct TabularColumnSelection {
    family_identifier_value_index: Option<usize>,
    individual_identifier_value_index: usize,
    data_value_indices: Vec<usize>,
    selected_columns: Vec<SelectedTabularColumn>,
}

struct TabularColumnDefinition<'column> {
    column_name: &'column str,
    column_index: usize,
}

struct SelectedTabularColumn {
    column_name: String,
    column_index: usize,
}

struct SampleFileReader<R: BufRead> {
    path_text: String,
    reader: R,
    line_buffer: String,
}

struct StreamingTabularReader<R: Read> {
    path_text: String,
    source_label: &'static str,
    reader: csv::Reader<R>,
    current_line_number: usize,
}

#[derive(Clone, Debug, Eq, Hash, PartialEq)]
enum SampleKey {
    Iid(String),
    FidIid { family_identifier: String, individual_identifier: String },
}

struct SampleIdentifierData {
    sample_indices: Vec<i64>,
    family_identifiers: Vec<String>,
    individual_identifiers: Vec<String>,
}

struct SinglePhenotypeTable {
    phenotype_values: Vec<f32>,
    phenotype_mask: Vec<bool>,
}

struct MultiPhenotypeTable {
    phenotype_values: Vec<f32>,
    phenotype_masks: Vec<bool>,
    phenotype_count: usize,
    sample_count: usize,
}

struct CovariateTable {
    selected_covariate_names: Vec<String>,
    covariate_values: Vec<f32>,
    covariate_mask: Vec<bool>,
    selected_covariate_count: usize,
}

pub fn align_sample_data(inputs: AlignmentInputs) -> Result<AlignedSampleData, String> {
    validate_alignment_input_lengths(&inputs)?;
    validate_sample_identifier_keys(&inputs)?;

    let sample_row_indices_by_key = build_sample_row_indices_by_key(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    );
    let phenotype_table = read_single_phenotype_table(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_name,
        inputs.is_binary_trait,
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        inputs.sample_indices.len(),
    )?;

    let covariate_table = match inputs.covariate_path.as_ref() {
        Some(covariate_path) => read_covariate_table(
            Path::new(covariate_path),
            inputs.covariate_names.as_deref(),
            inputs.sample_key_mode,
            &sample_row_indices_by_key,
            &phenotype_table.phenotype_mask,
            inputs.sample_indices.len(),
        )?,
        None => {
            if inputs.covariate_names.is_some() {
                return Err("Covariate names cannot be provided without a covariate table.".to_string());
            }
            empty_covariate_table(inputs.sample_indices.len())
        }
    };

    build_single_aligned_sample_data(inputs, &phenotype_table, &covariate_table)
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

    let sample_row_indices_by_key = build_sample_row_indices_by_key(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    );
    let phenotype_table = read_multi_phenotype_table(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_names,
        inputs.is_binary_trait,
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        inputs.sample_indices.len(),
    )?;
    let parse_candidate_mask = multi_phenotype_parse_candidate_mask(&phenotype_table);
    let covariate_table = match inputs.covariate_path.as_ref() {
        Some(covariate_path) => read_covariate_table(
            Path::new(covariate_path),
            inputs.covariate_names.as_deref(),
            inputs.sample_key_mode,
            &sample_row_indices_by_key,
            &parse_candidate_mask,
            inputs.sample_indices.len(),
        )?,
        None => {
            if inputs.covariate_names.is_some() {
                return Err("Covariate names cannot be provided without a covariate table.".to_string());
            }
            empty_covariate_table(inputs.sample_indices.len())
        }
    };

    build_multi_aligned_sample_data(inputs, &phenotype_table, &covariate_table)
}

/// Align several phenotypes independently, then group traits that share one
/// sample/covariate layout.
pub fn align_grouped_sample_data(inputs: &MultiAlignmentInputs) -> Result<GroupedAlignedSampleData, String> {
    if inputs.phenotype_names.is_empty() {
        return Err("At least one phenotype is required for grouped phenotype alignment.".to_string());
    }
    validate_multi_alignment_input_lengths(inputs)?;
    validate_multi_sample_identifier_keys(inputs)?;

    let sample_row_indices_by_key = build_sample_row_indices_by_key(
        inputs.sample_key_mode,
        &inputs.family_identifiers,
        &inputs.individual_identifiers,
    );
    let phenotype_table = read_multi_phenotype_table(
        Path::new(&inputs.phenotype_path),
        &inputs.phenotype_names,
        inputs.is_binary_trait,
        inputs.sample_key_mode,
        &sample_row_indices_by_key,
        inputs.sample_indices.len(),
    )?;
    let parse_candidate_mask = multi_phenotype_parse_candidate_mask(&phenotype_table);
    let covariate_table = match inputs.covariate_path.as_ref() {
        Some(covariate_path) => read_covariate_table(
            Path::new(covariate_path),
            inputs.covariate_names.as_deref(),
            inputs.sample_key_mode,
            &sample_row_indices_by_key,
            &parse_candidate_mask,
            inputs.sample_indices.len(),
        )?,
        None => {
            if inputs.covariate_names.is_some() {
                return Err("Covariate names cannot be provided without a covariate table.".to_string());
            }
            empty_covariate_table(inputs.sample_indices.len())
        }
    };

    build_grouped_aligned_sample_data(inputs, &phenotype_table, &covariate_table)
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

pub fn align_grouped_sample_data_from_sample_file(
    sample_path: &Path,
    expected_sample_count: usize,
    phenotype_path: String,
    phenotype_names: Vec<String>,
    covariate_path: Option<String>,
    covariate_names: Option<Vec<String>>,
    is_binary_trait: bool,
    sample_key_mode: SampleKeyMode,
) -> Result<GroupedAlignedSampleData, String> {
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
    align_grouped_sample_data(&inputs)
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
    let column_names = reader.read_required_fields(format!(
        "Sample file '{}' must contain at least two header lines.",
        sample_path.display()
    ))?;
    let column_types = reader.read_required_fields(format!(
        "Sample file '{}' must contain at least two header lines.",
        sample_path.display()
    ))?;
    validate_sample_file_header(sample_path, &column_names, &column_types)?;
    let family_identifier_column_index = 0;
    let individual_identifier_column_index =
        column_names.iter().position(|column_name| column_name == "ID_2").unwrap_or(family_identifier_column_index);

    let mut sample_indices = Vec::with_capacity(expected_sample_count);
    let mut family_identifiers = Vec::with_capacity(expected_sample_count);
    let mut individual_identifiers = Vec::with_capacity(expected_sample_count);
    let mut sample_count = 0usize;
    while let Some(row_values) = reader.read_next_fields()? {
        sample_count += 1;
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

fn open_sample_file_reader(sample_path: &Path) -> Result<SampleFileReader<BufReader<File>>, String> {
    let sample_file = File::open(sample_path)
        .map_err(|error| format!("Failed to read sample file '{}': {error}.", sample_path.display()))?;
    Ok(SampleFileReader::new(sample_path.display().to_string(), BufReader::new(sample_file)))
}

fn open_phenotype_table_reader(table_path: &Path) -> Result<StreamingTabularReader<File>, String> {
    open_tabular_reader(table_path, "phenotype table", b'\t')
}

fn open_covariate_table_reader(table_path: &Path) -> Result<StreamingTabularReader<File>, String> {
    open_tabular_reader(table_path, "covariate table", b'\t')
}

// Phenotype and covariate tables intentionally use tab-only parsing.
fn open_tabular_reader(
    table_path: &Path,
    source_label: &'static str,
    delimiter: u8,
) -> Result<StreamingTabularReader<File>, String> {
    let table_file = File::open(table_path)
        .map_err(|error| format!("Failed to read {source_label} '{}': {error}.", table_path.display()))?;
    Ok(StreamingTabularReader::new(table_path.display().to_string(), source_label, table_file, delimiter))
}

fn read_tabular_header<R: Read>(
    reader: &mut StreamingTabularReader<R>,
    table_path: &Path,
) -> Result<Vec<String>, String> {
    let headers = record_to_strings(&reader.read_required_record(format!(
        "{} '{}' is empty.",
        reader.display_source_label(),
        table_path.display()
    ))?);
    if headers.is_empty() {
        return Err(format!("{} '{}' must contain a header row.", reader.display_source_label(), table_path.display()));
    }
    Ok(headers)
}

impl<R: BufRead> SampleFileReader<R> {
    fn new(path_text: String, reader: R) -> Self {
        Self { path_text, reader, line_buffer: String::new() }
    }

    fn read_required_fields(&mut self, empty_error_message: String) -> Result<Vec<String>, String> {
        self.read_next_fields()?.ok_or(empty_error_message)
    }

    fn read_next_fields(&mut self) -> Result<Option<Vec<String>>, String> {
        loop {
            self.line_buffer.clear();
            let read_byte_count = self
                .reader
                .read_line(&mut self.line_buffer)
                .map_err(|error| format!("Failed to read sample file '{}': {error}.", self.path_text))?;
            if read_byte_count == 0 {
                return Ok(None);
            }
            let field_values = self.line_buffer.split_whitespace().map(ToString::to_string).collect::<Vec<_>>();
            if !field_values.is_empty() {
                return Ok(Some(field_values));
            }
        }
    }
}

impl<R: Read> StreamingTabularReader<R> {
    fn new(path_text: String, source_label: &'static str, source: R, delimiter: u8) -> Self {
        let reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .flexible(true)
            .has_headers(false)
            .trim(csv::Trim::All)
            .from_reader(source);
        Self { path_text, source_label, reader, current_line_number: 0 }
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
            self.current_line_number += 1;
            if !is_empty_tabular_record(&record) {
                return Ok(Some(record));
            }
            record.clear();
        }
    }

    fn display_source_label(&self) -> &'static str {
        match self.source_label {
            "phenotype table" => "Phenotype table",
            "covariate table" => "Covariate table",
            _ => "Table",
        }
    }

    fn missing_selected_column_error(
        &self,
        selected_column: &SelectedTabularColumn,
        record: &csv::StringRecord,
    ) -> String {
        format!(
            "{} '{}' line {} is missing selected column '{}' at column index {}; row has {} fields.",
            self.display_source_label(),
            self.path_text,
            self.current_line_number,
            selected_column.column_name,
            selected_column.column_index,
            record.len()
        )
    }
}

fn is_empty_tabular_record(record: &csv::StringRecord) -> bool {
    record.iter().all(str::is_empty)
}

fn record_to_strings(record: &csv::StringRecord) -> Vec<String> {
    record.iter().map(ToString::to_string).collect()
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
        data_column_definitions: &[TabularColumnDefinition<'_>],
    ) -> Self {
        let mut selected_columns = Vec::with_capacity(data_column_definitions.len() + 2);
        let family_identifier_value_index = family_identifier_column_index
            .map(|column_index| push_selected_column(&mut selected_columns, "FID", column_index));
        let individual_identifier_value_index =
            push_selected_column(&mut selected_columns, "IID", individual_identifier_column_index);
        let data_value_indices = data_column_definitions
            .iter()
            .map(|column_definition| {
                push_selected_column(
                    &mut selected_columns,
                    column_definition.column_name,
                    column_definition.column_index,
                )
            })
            .collect();
        Self { family_identifier_value_index, individual_identifier_value_index, data_value_indices, selected_columns }
    }

    fn validate_record<R: Read>(
        &self,
        reader: &StreamingTabularReader<R>,
        record: &csv::StringRecord,
    ) -> Result<(), String> {
        for selected_column in &self.selected_columns {
            if selected_column.column_index >= record.len() {
                return Err(reader.missing_selected_column_error(selected_column, record));
            }
        }
        Ok(())
    }

    fn record_value<'record>(&self, record: &'record csv::StringRecord, selected_value_index: usize) -> &'record str {
        let selected_column =
            self.selected_columns.get(selected_value_index).expect("selected tabular value index should exist");
        record.get(selected_column.column_index).expect("selected tabular column should be validated before access")
    }
}

fn push_selected_column(
    selected_columns: &mut Vec<SelectedTabularColumn>,
    column_name: &str,
    column_index: usize,
) -> usize {
    let selected_value_index = selected_columns.len();
    selected_columns.push(SelectedTabularColumn { column_name: column_name.to_string(), column_index });
    selected_value_index
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

fn build_sample_row_indices_by_key(
    sample_key_mode: SampleKeyMode,
    family_identifiers: &[String],
    individual_identifiers: &[String],
) -> HashMap<SampleKey, usize> {
    let mut sample_row_indices_by_key = HashMap::with_capacity(individual_identifiers.len());
    for (sample_array_index, (family_identifier, individual_identifier)) in
        family_identifiers.iter().zip(individual_identifiers.iter()).enumerate()
    {
        if individual_identifier.is_empty() {
            continue;
        }
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        sample_row_indices_by_key.insert(sample_key, sample_array_index);
    }
    sample_row_indices_by_key
}

fn duplicate_table_sample_key_error(
    source_name: &str,
    sample_key_mode: SampleKeyMode,
    family_identifier: &str,
    individual_identifier: &str,
) -> String {
    if sample_key_mode == SampleKeyMode::FidIid {
        return format!(
            "Duplicate sample key '{family_identifier}_{individual_identifier}' found in {source_name}; sample_key_mode='fid_iid' requires unique (FID, IID) values."
        );
    }
    format!(
        "Duplicate IID '{individual_identifier}' found in {source_name}; sample_key_mode='iid' requires unique non-null IID values."
    )
}

fn read_single_phenotype_table(
    phenotype_path: &Path,
    phenotype_name: &str,
    is_binary_trait: bool,
    sample_key_mode: SampleKeyMode,
    sample_row_indices_by_key: &HashMap<SampleKey, usize>,
    sample_count: usize,
) -> Result<SinglePhenotypeTable, String> {
    let mut reader = open_phenotype_table_reader(phenotype_path)?;
    let headers = read_tabular_header(&mut reader, phenotype_path)?;
    let phenotype_path_text = phenotype_path.display().to_string();
    let family_identifier_index = if sample_key_mode == SampleKeyMode::FidIid {
        Some(required_column_index(&headers, "FID", &phenotype_path_text)?)
    } else {
        None
    };
    let individual_identifier_index = required_column_index(&headers, "IID", &phenotype_path_text)?;
    let phenotype_index = required_column_index(&headers, phenotype_name, &phenotype_path_text)?;
    let phenotype_column_definition =
        [TabularColumnDefinition { column_name: phenotype_name, column_index: phenotype_index }];
    let selection =
        TabularColumnSelection::new(family_identifier_index, individual_identifier_index, &phenotype_column_definition);
    let mut observed_sample_keys: HashSet<SampleKey> = HashSet::new();
    let mut phenotype_values = vec![0.0; sample_count];
    let mut phenotype_mask = vec![false; sample_count];
    while let Some(record) = reader.read_next_record()? {
        selection.validate_record(&reader, &record)?;
        let individual_identifier = selection.record_value(&record, selection.individual_identifier_value_index);
        if individual_identifier.is_empty() {
            continue;
        }
        let family_identifier = selection
            .family_identifier_value_index
            .map_or("", |value_index| selection.record_value(&record, value_index));
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if !observed_sample_keys.insert(sample_key.clone()) {
            return Err(duplicate_table_sample_key_error(
                "phenotype table",
                sample_key_mode,
                family_identifier,
                individual_identifier,
            ));
        }
        let phenotype_value = selection.record_value(&record, selection.data_value_indices[0]);
        if is_tabular_null_value(phenotype_value) {
            continue;
        }
        if let Some(sample_array_index) = sample_row_indices_by_key.get(&sample_key) {
            phenotype_values[*sample_array_index] =
                parse_phenotype_value(phenotype_value, phenotype_name, is_binary_trait)?;
            phenotype_mask[*sample_array_index] = true;
        }
    }
    Ok(SinglePhenotypeTable { phenotype_values, phenotype_mask })
}

fn read_multi_phenotype_table(
    phenotype_path: &Path,
    phenotype_names: &[String],
    is_binary_trait: bool,
    sample_key_mode: SampleKeyMode,
    sample_row_indices_by_key: &HashMap<SampleKey, usize>,
    sample_count: usize,
) -> Result<MultiPhenotypeTable, String> {
    let mut reader = open_phenotype_table_reader(phenotype_path)?;
    let headers = read_tabular_header(&mut reader, phenotype_path)?;
    let phenotype_path_text = phenotype_path.display().to_string();
    let family_identifier_index = if sample_key_mode == SampleKeyMode::FidIid {
        Some(required_column_index(&headers, "FID", &phenotype_path_text)?)
    } else {
        None
    };
    let individual_identifier_index = required_column_index(&headers, "IID", &phenotype_path_text)?;
    let phenotype_indices = phenotype_names
        .iter()
        .map(|phenotype_name| required_column_index(&headers, phenotype_name, &phenotype_path_text))
        .collect::<Result<Vec<_>, _>>()?;
    let phenotype_column_definitions = phenotype_names
        .iter()
        .zip(phenotype_indices.iter())
        .map(|(phenotype_name, phenotype_index)| TabularColumnDefinition {
            column_name: phenotype_name.as_str(),
            column_index: *phenotype_index,
        })
        .collect::<Vec<_>>();
    let selection = TabularColumnSelection::new(
        family_identifier_index,
        individual_identifier_index,
        &phenotype_column_definitions,
    );
    let mut observed_sample_keys: HashSet<SampleKey> = HashSet::new();
    let phenotype_count = phenotype_names.len();
    let mut phenotype_values = vec![0.0; phenotype_count * sample_count];
    let mut phenotype_masks = vec![false; phenotype_count * sample_count];
    while let Some(record) = reader.read_next_record()? {
        selection.validate_record(&reader, &record)?;
        let individual_identifier = selection.record_value(&record, selection.individual_identifier_value_index);
        if individual_identifier.is_empty() {
            continue;
        }
        let family_identifier = selection
            .family_identifier_value_index
            .map_or("", |value_index| selection.record_value(&record, value_index));
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if !observed_sample_keys.insert(sample_key.clone()) {
            return Err(duplicate_table_sample_key_error(
                "phenotype table",
                sample_key_mode,
                family_identifier,
                individual_identifier,
            ));
        }
        let Some(sample_array_index) = sample_row_indices_by_key.get(&sample_key) else {
            continue;
        };
        for (phenotype_index, phenotype_name) in phenotype_names.iter().enumerate() {
            let phenotype_value = selection.record_value(&record, selection.data_value_indices[phenotype_index]);
            if is_tabular_null_value(phenotype_value) {
                continue;
            }
            let value_index = phenotype_index * sample_count + sample_array_index;
            phenotype_values[value_index] = parse_phenotype_value(phenotype_value, phenotype_name, is_binary_trait)?;
            phenotype_masks[value_index] = true;
        }
    }
    Ok(MultiPhenotypeTable { phenotype_values, phenotype_masks, phenotype_count, sample_count })
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

fn read_covariate_table(
    covariate_path: &Path,
    requested_covariate_names: Option<&[String]>,
    sample_key_mode: SampleKeyMode,
    sample_row_indices_by_key: &HashMap<SampleKey, usize>,
    parse_candidate_mask: &[bool],
    sample_count: usize,
) -> Result<CovariateTable, String> {
    let mut reader = open_covariate_table_reader(covariate_path)?;
    let headers = read_tabular_header(&mut reader, covariate_path)?;
    let covariate_path_text = covariate_path.display().to_string();
    let family_identifier_index = if sample_key_mode == SampleKeyMode::FidIid {
        Some(required_column_index(&headers, "FID", &covariate_path_text)?)
    } else {
        None
    };
    let individual_identifier_index = required_column_index(&headers, "IID", &covariate_path_text)?;
    let selected_covariate_names = select_covariate_names(&headers, requested_covariate_names, &covariate_path_text)?;
    let covariate_indices: Vec<usize> = selected_covariate_names
        .iter()
        .map(|covariate_name| {
            column_index(&headers, covariate_name)
                .ok_or_else(|| format!("Covariate column '{covariate_name}' was not found."))
        })
        .collect::<Result<Vec<_>, _>>()?;
    let covariate_column_definitions = selected_covariate_names
        .iter()
        .zip(covariate_indices.iter())
        .map(|(covariate_name, covariate_index)| TabularColumnDefinition {
            column_name: covariate_name.as_str(),
            column_index: *covariate_index,
        })
        .collect::<Vec<_>>();
    let selection = TabularColumnSelection::new(
        family_identifier_index,
        individual_identifier_index,
        &covariate_column_definitions,
    );
    let mut observed_sample_keys: HashSet<SampleKey> = HashSet::new();
    let selected_covariate_count = selected_covariate_names.len();
    let mut covariate_values = vec![0.0; sample_count * selected_covariate_count];
    let mut covariate_mask = vec![false; sample_count];
    while let Some(record) = reader.read_next_record()? {
        selection.validate_record(&reader, &record)?;
        let individual_identifier = selection.record_value(&record, selection.individual_identifier_value_index);
        if individual_identifier.is_empty() {
            continue;
        }
        let family_identifier = selection
            .family_identifier_value_index
            .map_or("", |value_index| selection.record_value(&record, value_index));
        let sample_key = build_sample_key(sample_key_mode, family_identifier, individual_identifier);
        if !observed_sample_keys.insert(sample_key.clone()) {
            return Err(duplicate_table_sample_key_error(
                "covariate table",
                sample_key_mode,
                family_identifier,
                individual_identifier,
            ));
        }
        let Some(sample_array_index) = sample_row_indices_by_key.get(&sample_key) else {
            continue;
        };
        if !parse_candidate_mask[*sample_array_index] {
            continue;
        }
        let mut row_has_missing_covariates = false;
        for covariate_index in 0..selected_covariate_count {
            let covariate_value = selection.record_value(&record, selection.data_value_indices[covariate_index]);
            if is_tabular_null_value(covariate_value) {
                row_has_missing_covariates = true;
                break;
            }
            let value_index = sample_array_index * selected_covariate_count + covariate_index;
            covariate_values[value_index] = parse_covariate_value(covariate_value)?;
        }
        if !row_has_missing_covariates {
            covariate_mask[*sample_array_index] = true;
        }
    }
    Ok(CovariateTable { selected_covariate_names, covariate_values, covariate_mask, selected_covariate_count })
}

fn empty_covariate_table(sample_count: usize) -> CovariateTable {
    CovariateTable {
        selected_covariate_names: Vec::new(),
        covariate_values: Vec::new(),
        covariate_mask: vec![true; sample_count],
        selected_covariate_count: 0,
    }
}

fn build_single_aligned_sample_data(
    inputs: AlignmentInputs,
    phenotype_table: &SinglePhenotypeTable,
    covariate_table: &CovariateTable,
) -> Result<AlignedSampleData, String> {
    let complete_sample_array_indices = complete_single_sample_array_indices(&inputs, phenotype_table, covariate_table);
    if complete_sample_array_indices.is_empty() {
        return Err("No aligned samples remain after joining phenotype and covariate tables.".to_string());
    }

    let aligned_sample_count = complete_sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let covariate_column_count = covariate_names.len();
    let mut sample_indices = Vec::with_capacity(aligned_sample_count);
    let mut family_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut individual_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut phenotype_vector = Vec::with_capacity(aligned_sample_count);
    let mut covariate_matrix_values = Vec::with_capacity(aligned_sample_count * covariate_column_count);

    for sample_array_index in complete_sample_array_indices {
        sample_indices.push(inputs.sample_indices[sample_array_index]);
        family_identifiers.push(inputs.family_identifiers[sample_array_index].clone());
        individual_identifiers.push(inputs.individual_identifiers[sample_array_index].clone());
        phenotype_vector.push(phenotype_table.phenotype_values[sample_array_index]);
        push_covariate_matrix_row(&mut covariate_matrix_values, covariate_table, sample_array_index);
    }

    Ok(AlignedSampleData {
        sample_indices,
        family_identifiers,
        individual_identifiers,
        phenotype_name: inputs.phenotype_name,
        phenotype_vector,
        covariate_names,
        covariate_matrix_values,
        covariate_row_count: aligned_sample_count,
        covariate_column_count,
        is_binary_trait: inputs.is_binary_trait,
    })
}

fn build_multi_aligned_sample_data(
    inputs: MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> Result<MultiAlignedSampleData, String> {
    let complete_sample_array_indices = complete_multi_sample_array_indices(&inputs, phenotype_table, covariate_table);
    if complete_sample_array_indices.is_empty() {
        return Err("No aligned samples remain after complete-case multi-phenotype intersection.".to_string());
    }

    let aligned_sample_count = complete_sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let covariate_column_count = covariate_names.len();
    let mut sample_indices = Vec::with_capacity(aligned_sample_count);
    let mut family_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut individual_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut phenotype_matrix_values = Vec::with_capacity(phenotype_table.phenotype_count * aligned_sample_count);
    let mut covariate_matrix_values = Vec::with_capacity(aligned_sample_count * covariate_column_count);

    for sample_array_index in &complete_sample_array_indices {
        sample_indices.push(inputs.sample_indices[*sample_array_index]);
        family_identifiers.push(inputs.family_identifiers[*sample_array_index].clone());
        individual_identifiers.push(inputs.individual_identifiers[*sample_array_index].clone());
        push_covariate_matrix_row(&mut covariate_matrix_values, covariate_table, *sample_array_index);
    }
    for phenotype_index in 0..phenotype_table.phenotype_count {
        for sample_array_index in &complete_sample_array_indices {
            let value_index = phenotype_index * phenotype_table.sample_count + sample_array_index;
            phenotype_matrix_values.push(phenotype_table.phenotype_values[value_index]);
        }
    }

    Ok(MultiAlignedSampleData {
        sample_indices,
        family_identifiers,
        individual_identifiers,
        phenotype_names: inputs.phenotype_names,
        phenotype_matrix_values,
        phenotype_row_count: phenotype_table.phenotype_count,
        phenotype_column_count: aligned_sample_count,
        covariate_names,
        covariate_matrix_values,
        covariate_row_count: aligned_sample_count,
        covariate_column_count,
        is_binary_trait: inputs.is_binary_trait,
    })
}

fn build_grouped_aligned_sample_data(
    inputs: &MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> Result<GroupedAlignedSampleData, String> {
    let mut group_indices_by_sample_indices: HashMap<Vec<usize>, usize> = HashMap::new();
    let mut group_sample_array_indices: Vec<Vec<usize>> = Vec::new();
    let mut phenotype_indices_by_group: Vec<Vec<usize>> = Vec::new();
    let sorted_sample_array_indices = sorted_sample_array_indices_by_sample_index(&inputs.sample_indices);
    let mut complete_sample_array_indices = Vec::with_capacity(inputs.sample_indices.len());

    for phenotype_index in 0..phenotype_table.phenotype_count {
        collect_complete_grouped_trait_sample_array_indices(
            &sorted_sample_array_indices,
            phenotype_table,
            covariate_table,
            phenotype_index,
            &mut complete_sample_array_indices,
        );
        if complete_sample_array_indices.is_empty() {
            return Err(format!(
                "No aligned samples remain after joining phenotype '{}' and covariate tables.",
                inputs.phenotype_names[phenotype_index]
            ));
        }
        let group_index = match group_indices_by_sample_indices.get(&complete_sample_array_indices) {
            Some(existing_group_index) => *existing_group_index,
            None => {
                let new_group_index = group_sample_array_indices.len();
                let stored_sample_array_indices = std::mem::take(&mut complete_sample_array_indices);
                group_indices_by_sample_indices.insert(stored_sample_array_indices.clone(), new_group_index);
                group_sample_array_indices.push(stored_sample_array_indices);
                phenotype_indices_by_group.push(Vec::new());
                complete_sample_array_indices = Vec::with_capacity(inputs.sample_indices.len());
                new_group_index
            }
        };
        phenotype_indices_by_group[group_index].push(phenotype_index);
    }

    let groups = phenotype_indices_by_group
        .into_iter()
        .zip(group_sample_array_indices)
        .map(|(phenotype_indices, complete_sample_array_indices)| {
            build_aligned_phenotype_group(
                inputs,
                phenotype_table,
                covariate_table,
                phenotype_indices,
                &complete_sample_array_indices,
            )
        })
        .collect::<Vec<_>>();
    Ok(GroupedAlignedSampleData { groups })
}

fn build_aligned_phenotype_group(
    inputs: &MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
    phenotype_indices: Vec<usize>,
    complete_sample_array_indices: &[usize],
) -> AlignedPhenotypeGroup {
    let aligned_sample_count = complete_sample_array_indices.len();
    let covariate_names = returned_covariate_names(&covariate_table.selected_covariate_names);
    let covariate_column_count = covariate_names.len();
    let mut sample_indices = Vec::with_capacity(aligned_sample_count);
    let mut family_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut individual_identifiers = Vec::with_capacity(aligned_sample_count);
    let mut phenotype_matrix_values = Vec::with_capacity(phenotype_indices.len() * aligned_sample_count);
    let mut covariate_matrix_values = Vec::with_capacity(aligned_sample_count * covariate_column_count);

    for sample_array_index in complete_sample_array_indices {
        sample_indices.push(inputs.sample_indices[*sample_array_index]);
        family_identifiers.push(inputs.family_identifiers[*sample_array_index].clone());
        individual_identifiers.push(inputs.individual_identifiers[*sample_array_index].clone());
        push_covariate_matrix_row(&mut covariate_matrix_values, covariate_table, *sample_array_index);
    }
    for phenotype_index in &phenotype_indices {
        for sample_array_index in complete_sample_array_indices {
            let value_index = phenotype_index * phenotype_table.sample_count + sample_array_index;
            phenotype_matrix_values.push(phenotype_table.phenotype_values[value_index]);
        }
    }

    let phenotype_names =
        phenotype_indices.iter().map(|phenotype_index| inputs.phenotype_names[*phenotype_index].clone()).collect();
    let phenotype_row_count = phenotype_indices.len();
    AlignedPhenotypeGroup {
        phenotype_indices,
        aligned_sample_data: MultiAlignedSampleData {
            sample_indices,
            family_identifiers,
            individual_identifiers,
            phenotype_names,
            phenotype_matrix_values,
            phenotype_row_count,
            phenotype_column_count: aligned_sample_count,
            covariate_names,
            covariate_matrix_values,
            covariate_row_count: aligned_sample_count,
            covariate_column_count,
            is_binary_trait: inputs.is_binary_trait,
        },
    }
}

fn complete_single_sample_array_indices(
    inputs: &AlignmentInputs,
    phenotype_table: &SinglePhenotypeTable,
    covariate_table: &CovariateTable,
) -> Vec<usize> {
    sorted_sample_array_indices_by_sample_index(&inputs.sample_indices)
        .into_iter()
        .filter(|sample_array_index| {
            phenotype_table.phenotype_mask[*sample_array_index] && covariate_table.covariate_mask[*sample_array_index]
        })
        .collect()
}

fn complete_multi_sample_array_indices(
    inputs: &MultiAlignmentInputs,
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
) -> Vec<usize> {
    sorted_sample_array_indices_by_sample_index(&inputs.sample_indices)
        .into_iter()
        .filter(|sample_array_index| {
            is_complete_multi_phenotype_sample(phenotype_table, *sample_array_index)
                && covariate_table.covariate_mask[*sample_array_index]
        })
        .collect()
}

fn sorted_sample_array_indices_by_sample_index(sample_indices: &[i64]) -> Vec<usize> {
    let mut sorted_sample_array_indices: Vec<usize> = (0..sample_indices.len()).collect();
    sorted_sample_array_indices.sort_by_key(|sample_array_index| sample_indices[*sample_array_index]);
    sorted_sample_array_indices
}

fn collect_complete_grouped_trait_sample_array_indices(
    sorted_sample_array_indices: &[usize],
    phenotype_table: &MultiPhenotypeTable,
    covariate_table: &CovariateTable,
    phenotype_index: usize,
    complete_sample_array_indices: &mut Vec<usize>,
) {
    complete_sample_array_indices.clear();
    complete_sample_array_indices.extend(sorted_sample_array_indices.iter().copied().filter(|sample_array_index| {
        let phenotype_mask_index = phenotype_index * phenotype_table.sample_count + sample_array_index;
        phenotype_table.phenotype_masks[phenotype_mask_index] && covariate_table.covariate_mask[*sample_array_index]
    }));
}

fn is_complete_multi_phenotype_sample(phenotype_table: &MultiPhenotypeTable, sample_array_index: usize) -> bool {
    (0..phenotype_table.phenotype_count).all(|phenotype_index| {
        phenotype_table.phenotype_masks[phenotype_index * phenotype_table.sample_count + sample_array_index]
    })
}

fn multi_phenotype_parse_candidate_mask(phenotype_table: &MultiPhenotypeTable) -> Vec<bool> {
    (0..phenotype_table.sample_count)
        .map(|sample_array_index| {
            (0..phenotype_table.phenotype_count).any(|phenotype_index| {
                phenotype_table.phenotype_masks[phenotype_index * phenotype_table.sample_count + sample_array_index]
            })
        })
        .collect()
}

fn push_covariate_matrix_row(
    covariate_matrix_values: &mut Vec<f32>,
    covariate_table: &CovariateTable,
    sample_array_index: usize,
) {
    covariate_matrix_values.push(1.0);
    if covariate_table.selected_covariate_count == 0 {
        return;
    }
    let row_start = sample_array_index * covariate_table.selected_covariate_count;
    covariate_matrix_values
        .extend(&covariate_table.covariate_values[row_start..row_start + covariate_table.selected_covariate_count]);
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

fn parse_covariate_value(covariate_value: &str) -> Result<f32, String> {
    covariate_value
        .parse::<f32>()
        .map_err(|error| format!("Failed to parse covariate value '{covariate_value}': {error}."))
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{
        AlignmentInputs, MultiAlignmentInputs, SampleKeyMode, align_grouped_sample_data, align_multi_sample_data,
        align_multi_sample_data_from_sample_file, align_sample_data, align_sample_data_from_sample_file,
        validate_sample_file_header,
    };

    static NEXT_FIXTURE_ID: AtomicUsize = AtomicUsize::new(0);

    struct FixtureDirectory {
        path: PathBuf,
    }

    impl FixtureDirectory {
        fn new() -> Self {
            let fixture_id = NEXT_FIXTURE_ID.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!("g-sample-tests-{}-{fixture_id}", std::process::id()));
            fs::create_dir_all(&path).expect("sample test fixture directory should be created");
            Self { path }
        }

        fn write_file(&self, file_name: &str, contents: &str) -> String {
            let path = self.path.join(file_name);
            fs::write(&path, contents).expect("sample test fixture should be written");
            path.to_string_lossy().into_owned()
        }
    }

    impl Drop for FixtureDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.path);
        }
    }

    fn strings(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_string()).collect()
    }

    fn base_alignment_inputs(phenotype_path: String, phenotype_name: &str) -> AlignmentInputs {
        AlignmentInputs {
            sample_indices: vec![2, 0, 1],
            family_identifiers: strings(&["F3", "F1", "F2"]),
            individual_identifiers: strings(&["I3", "I1", "I2"]),
            phenotype_path,
            phenotype_name: phenotype_name.to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        }
    }

    #[test]
    fn aligns_quantitative_samples_with_covariate_complete_cases_by_fid_iid() {
        let fixture = FixtureDirectory::new();
        let phenotype_path =
            fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1.5\nF2\tI2\tNA\nF3\tI3\t2.5\n");
        let covariate_path =
            fixture.write_file("covariates.tsv", "FID\tIID\tage\tbatch\nF1\tI1\t41\t0\nF3\tI3\t63\t1\n");
        let mut inputs = base_alignment_inputs(phenotype_path, "trait");
        inputs.covariate_path = Some(covariate_path);

        let aligned = align_sample_data(inputs).expect("quantitative sample alignment should succeed");

        assert_eq!(aligned.sample_indices, vec![0, 2]);
        assert_eq!(aligned.family_identifiers, strings(&["F1", "F3"]));
        assert_eq!(aligned.individual_identifiers, strings(&["I1", "I3"]));
        assert_eq!(aligned.phenotype_name, "trait");
        assert_eq!(aligned.phenotype_vector, vec![1.5, 2.5]);
        assert_eq!(aligned.covariate_names, strings(&["intercept", "age", "batch"]));
        assert_eq!(aligned.covariate_row_count, 2);
        assert_eq!(aligned.covariate_column_count, 3);
        assert_eq!(aligned.covariate_matrix_values, vec![1.0, 41.0, 0.0, 1.0, 63.0, 1.0]);
        assert!(!aligned.is_binary_trait);
    }

    #[test]
    fn ignores_invalid_covariate_values_for_phenotype_missing_samples() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1.5\nF2\tI2\tNA\n");
        let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t41\nF2\tI2\tbad\n");
        let mut inputs = base_alignment_inputs(phenotype_path, "trait");
        inputs.sample_indices = vec![0, 1];
        inputs.family_identifiers = strings(&["F1", "F2"]);
        inputs.individual_identifiers = strings(&["I1", "I2"]);
        inputs.covariate_path = Some(covariate_path);

        let aligned = align_sample_data(inputs).expect("unused invalid covariate should not fail alignment");

        assert_eq!(aligned.sample_indices, vec![0]);
        assert_eq!(aligned.phenotype_vector, vec![1.5]);
        assert_eq!(aligned.covariate_matrix_values, vec![1.0, 41.0]);
    }

    #[test]
    fn aligns_binary_samples_and_recodes_regenie_case_control_values() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "IID\tcase\nI1\t1\nI2\t2\nI3\t1\n");
        let inputs = AlignmentInputs {
            sample_indices: vec![0, 1, 2],
            family_identifiers: strings(&["F1", "F2", "F3"]),
            individual_identifiers: strings(&["I1", "I2", "I3"]),
            phenotype_path,
            phenotype_name: "case".to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: true,
            sample_key_mode: SampleKeyMode::Iid,
        };

        let aligned = align_sample_data(inputs).expect("binary sample alignment should succeed");

        assert_eq!(aligned.phenotype_vector, vec![0.0, 1.0, 0.0]);
        assert_eq!(aligned.covariate_names, strings(&["intercept"]));
        assert!(aligned.is_binary_trait);
    }

    #[test]
    fn rejects_invalid_binary_phenotype_value() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "IID\tcase\nI1\t3\n");
        let inputs = AlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path,
            phenotype_name: "case".to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: true,
            sample_key_mode: SampleKeyMode::Iid,
        };

        let error = align_sample_data(inputs).expect_err("invalid binary phenotype should be rejected");

        assert!(error.contains("Binary phenotype must contain only values 1 and 2"));
    }

    #[test]
    fn aligns_multi_phenotype_complete_cases_to_shared_sample_set() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file(
            "phenotypes.tsv",
            "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t10\t20\nF2\tI2\t11\tNA\nF3\tI3\tNA\t22\n",
        );
        let inputs = MultiAlignmentInputs {
            sample_indices: vec![0, 1, 2],
            family_identifiers: strings(&["F1", "F2", "F3"]),
            individual_identifiers: strings(&["I1", "I2", "I3"]),
            phenotype_path,
            phenotype_names: strings(&["trait_a", "trait_b"]),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };

        let aligned = align_multi_sample_data(inputs).expect("multi-phenotype alignment should succeed");

        assert_eq!(aligned.sample_indices, vec![0]);
        assert_eq!(aligned.phenotype_names, strings(&["trait_a", "trait_b"]));
        assert_eq!(aligned.phenotype_row_count, 2);
        assert_eq!(aligned.phenotype_column_count, 1);
        assert_eq!(aligned.phenotype_matrix_values, vec![10.0, 20.0]);
        assert_eq!(aligned.covariate_matrix_values, vec![1.0]);
    }

    #[test]
    fn rejects_short_single_phenotype_selected_row() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("short-phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\n");
        let inputs = AlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path: phenotype_path.clone(),
            phenotype_name: "trait".to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };

        let error = align_sample_data(inputs).expect_err("short selected phenotype row should fail");

        assert!(error.contains(&format!(
            "Phenotype table '{phenotype_path}' line 2 is missing selected column 'trait' at column index 2; row has 2 fields."
        )));
    }

    #[test]
    fn rejects_short_selected_covariate_row() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
        let covariate_path = fixture.write_file("short-covariates.tsv", "FID\tIID\tage\tsex\nF1\tI1\t40\n");
        let inputs = AlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path,
            phenotype_name: "trait".to_string(),
            covariate_path: Some(covariate_path.clone()),
            covariate_names: Some(strings(&["sex"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };

        let error = align_sample_data(inputs).expect_err("short selected covariate row should fail");

        assert!(error.contains(&format!(
            "Covariate table '{covariate_path}' line 2 is missing selected column 'sex' at column index 3; row has 3 fields."
        )));
    }

    #[test]
    fn rejects_short_multi_phenotype_selected_row() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("short-multi.tsv", "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t10\n");
        let inputs = MultiAlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path: phenotype_path.clone(),
            phenotype_names: strings(&["trait_a", "trait_b"]),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };

        let error = align_multi_sample_data(inputs).expect_err("short selected multi-phenotype row should fail");

        assert!(error.contains(&format!(
            "Phenotype table '{phenotype_path}' line 2 is missing selected column 'trait_b' at column index 3; row has 3 fields."
        )));
    }

    #[test]
    fn accepts_explicit_empty_selected_fields_as_missing_values() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t\nF2\tI2\t2\nF3\tI3\t3\n");
        let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF2\tI2\t\nF3\tI3\t50\n");
        let inputs = AlignmentInputs {
            sample_indices: vec![0, 1, 2],
            family_identifiers: strings(&["F1", "F2", "F3"]),
            individual_identifiers: strings(&["I1", "I2", "I3"]),
            phenotype_path,
            phenotype_name: "trait".to_string(),
            covariate_path: Some(covariate_path),
            covariate_names: Some(strings(&["age"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };

        let aligned = align_sample_data(inputs).expect("explicit empty selected fields should remain missing values");

        assert_eq!(aligned.sample_indices, vec![2]);
        assert_eq!(aligned.phenotype_vector, vec![3.0]);
        assert_eq!(aligned.covariate_matrix_values, vec![1.0, 50.0]);
    }

    #[test]
    fn groups_per_phenotype_alignments_by_identical_sample_sets() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file(
            "phenotypes.tsv",
            "FID\tIID\ttrait_a\ttrait_b\ttrait_c\nF1\tI1\t10\t20\t30\nF2\tI2\t11\t21\tNA\nF3\tI3\tNA\tNA\t32\n",
        );
        let covariate_path =
            fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\nF2\tI2\t50\nF3\tI3\t60\n");
        let inputs = MultiAlignmentInputs {
            sample_indices: vec![2, 0, 1],
            family_identifiers: strings(&["F3", "F1", "F2"]),
            individual_identifiers: strings(&["I3", "I1", "I2"]),
            phenotype_path,
            phenotype_names: strings(&["trait_a", "trait_b", "trait_c"]),
            covariate_path: Some(covariate_path),
            covariate_names: Some(strings(&["age"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };

        let grouped = align_grouped_sample_data(&inputs).expect("grouped alignment should succeed");

        assert_eq!(grouped.groups.len(), 2);
        assert_eq!(grouped.groups[0].phenotype_indices, vec![0, 1]);
        assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_names, strings(&["trait_a", "trait_b"]));
        assert_eq!(grouped.groups[0].aligned_sample_data.sample_indices, vec![0, 1]);
        assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_row_count, 2);
        assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_column_count, 2);
        assert_eq!(grouped.groups[0].aligned_sample_data.phenotype_matrix_values, vec![10.0, 11.0, 20.0, 21.0]);
        assert_eq!(grouped.groups[0].aligned_sample_data.covariate_matrix_values, vec![1.0, 40.0, 1.0, 50.0]);
        assert_eq!(grouped.groups[1].phenotype_indices, vec![2]);
        assert_eq!(grouped.groups[1].aligned_sample_data.phenotype_names, strings(&["trait_c"]));
        assert_eq!(grouped.groups[1].aligned_sample_data.sample_indices, vec![0, 2]);
        assert_eq!(grouped.groups[1].aligned_sample_data.phenotype_matrix_values, vec![30.0, 32.0]);
        assert_eq!(grouped.groups[1].aligned_sample_data.covariate_matrix_values, vec![1.0, 40.0, 1.0, 60.0]);
    }

    #[test]
    fn rejects_empty_multi_phenotype_request() {
        let inputs = MultiAlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path: "unused.tsv".to_string(),
            phenotype_names: Vec::new(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::Iid,
        };

        let error = align_multi_sample_data(inputs).expect_err("empty phenotype list should be rejected");

        assert!(error.contains("At least one phenotype is required"));
    }

    #[test]
    fn rejects_duplicate_iids_when_iid_mode_is_requested() {
        let inputs = AlignmentInputs {
            sample_indices: vec![0, 1],
            family_identifiers: strings(&["F1", "F2"]),
            individual_identifiers: strings(&["same", "same"]),
            phenotype_path: "unused.tsv".to_string(),
            phenotype_name: "trait".to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::Iid,
        };

        let error = align_sample_data(inputs).expect_err("duplicate IID should be rejected before file IO");

        assert!(error.contains("Duplicate IID 'same'"));
    }

    #[test]
    fn rejects_covariate_names_without_covariate_table() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
        let mut inputs = base_alignment_inputs(phenotype_path, "trait");
        inputs.covariate_names = Some(strings(&["age"]));

        let error = align_sample_data(inputs).expect_err("covariate names require a covariate table");

        assert!(error.contains("Covariate names cannot be provided without a covariate table"));
    }

    #[test]
    fn rejects_missing_requested_covariate_column() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
        let covariate_path = fixture.write_file("covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\n");
        let mut inputs = AlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path,
            phenotype_name: "trait".to_string(),
            covariate_path: Some(covariate_path),
            covariate_names: Some(strings(&["missing"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };

        let error = align_sample_data(inputs.clone()).expect_err("missing covariate should be rejected");
        inputs.covariate_names = Some(strings(&["age"]));
        assert!(align_sample_data(inputs).is_ok());
        assert!(error.contains("Covariate columns are missing"));
    }

    #[test]
    fn covers_sample_file_header_and_count_errors() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\n");
        let sample_path = fixture.write_file("study.sample", "ID_1 ID_2 missing\n0 0 0\nF1 I1 0\nF2 I2 0\n");

        assert!(
            align_sample_data_from_sample_file(
                Path::new(&sample_path),
                1,
                phenotype_path.clone(),
                "trait".to_string(),
                None,
                None,
                false,
                SampleKeyMode::FidIid,
            )
            .expect_err("sample count mismatch should fail")
            .contains("BGEN contains 1 samples")
        );
        assert!(
            align_multi_sample_data_from_sample_file(
                Path::new(&sample_path),
                1,
                phenotype_path,
                strings(&["trait"]),
                None,
                None,
                false,
                SampleKeyMode::FidIid,
            )
            .expect_err("multi sample count mismatch should fail")
            .contains("BGEN contains 1 samples")
        );

        assert!(
            validate_sample_file_header(Path::new("empty.sample"), &[], &[])
                .expect_err("empty sample header should fail")
                .contains("does not contain any columns")
        );
        assert!(
            validate_sample_file_header(Path::new("bad-first-type.sample"), &strings(&["ID_1"]), &strings(&["D"]))
                .expect_err("first identifier type should be zero")
                .contains("first identifier column")
        );
        assert!(
            validate_sample_file_header(
                Path::new("bad-id2-type.sample"),
                &strings(&["ID_1", "ID_2"]),
                &strings(&["0", "D"]),
            )
            .expect_err("ID_2 type should be zero")
            .contains("'ID_2'")
        );
    }

    #[test]
    fn covers_table_identifier_duplicate_and_missing_value_edges() {
        let fixture = FixtureDirectory::new();

        let empty_identifier_phenotype_path = fixture.write_file("empty-iid.tsv", "IID\ttrait\n\t1\n");
        let empty_identifier_inputs = AlignmentInputs {
            sample_indices: vec![0, 1],
            family_identifiers: strings(&["F1", "F2"]),
            individual_identifiers: strings(&["", ""]),
            phenotype_path: empty_identifier_phenotype_path,
            phenotype_name: "trait".to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::Iid,
        };
        assert!(
            align_sample_data(empty_identifier_inputs)
                .expect_err("empty IIDs should not align")
                .contains("No aligned samples")
        );

        let missing_fid_phenotype_path = fixture.write_file("missing-fid.tsv", "IID\ttrait\nI1\t1\n");
        let missing_fid_inputs = AlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path: missing_fid_phenotype_path,
            phenotype_name: "trait".to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };
        assert!(
            align_sample_data(missing_fid_inputs)
                .expect_err("FID is required in fid_iid mode")
                .contains("Identifier column 'FID'")
        );

        let nonnumeric_phenotype_path = fixture.write_file("nonnumeric.tsv", "IID\ttrait\nI1\tbad\n");
        let nonnumeric_inputs = AlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path: nonnumeric_phenotype_path,
            phenotype_name: "trait".to_string(),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::Iid,
        };
        assert!(
            align_sample_data(nonnumeric_inputs)
                .expect_err("nonnumeric phenotype should fail")
                .contains("Failed to parse phenotype")
        );

        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait\nF1\tI1\t1\nF2\tI2\t2\n");
        let duplicate_covariate_path =
            fixture.write_file("duplicate-covariates.tsv", "FID\tIID\tage\nF1\tI1\t40\nF1\tI1\t41\n");
        let duplicate_covariate_inputs = AlignmentInputs {
            sample_indices: vec![0, 1],
            family_identifiers: strings(&["F1", "F2"]),
            individual_identifiers: strings(&["I1", "I2"]),
            phenotype_path: phenotype_path.clone(),
            phenotype_name: "trait".to_string(),
            covariate_path: Some(duplicate_covariate_path),
            covariate_names: Some(strings(&["age"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };
        assert!(
            align_sample_data(duplicate_covariate_inputs)
                .expect_err("duplicate covariate sample should fail")
                .contains("covariate table")
        );

        let missing_covariate_path =
            fixture.write_file("missing-covariates.tsv", "FID\tIID\tage\nF1\tI1\tNA\nF2\tI2\t50\n");
        let missing_covariate_inputs = AlignmentInputs {
            sample_indices: vec![0, 1],
            family_identifiers: strings(&["F1", "F2"]),
            individual_identifiers: strings(&["I1", "I2"]),
            phenotype_path,
            phenotype_name: "trait".to_string(),
            covariate_path: Some(missing_covariate_path),
            covariate_names: Some(strings(&["age"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };
        let aligned = align_sample_data(missing_covariate_inputs).expect("missing covariate should drop one sample");
        assert_eq!(aligned.sample_indices, vec![1]);
        assert_eq!(aligned.covariate_matrix_values, vec![1.0, 50.0]);
    }

    #[test]
    fn covers_multi_alignment_covariate_and_duplicate_edges() {
        let fixture = FixtureDirectory::new();
        let phenotype_path = fixture.write_file("phenotypes.tsv", "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t1\t2\n");
        let covariate_names_without_table = MultiAlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path: phenotype_path.clone(),
            phenotype_names: strings(&["trait_a", "trait_b"]),
            covariate_path: None,
            covariate_names: Some(strings(&["age"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };
        assert!(
            align_multi_sample_data(covariate_names_without_table)
                .expect_err("multi covariate names without table should fail")
                .contains("Covariate names cannot be provided")
        );

        let duplicate_multi_phenotype_path =
            fixture.write_file("duplicate-multi.tsv", "FID\tIID\ttrait_a\ttrait_b\nF1\tI1\t1\t2\nF1\tI1\t3\t4\n");
        let duplicate_multi_inputs = MultiAlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path: duplicate_multi_phenotype_path,
            phenotype_names: strings(&["trait_a", "trait_b"]),
            covariate_path: None,
            covariate_names: None,
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };
        assert!(
            align_multi_sample_data(duplicate_multi_inputs)
                .expect_err("duplicate multi phenotype keys should fail")
                .contains("phenotype table")
        );

        let missing_fid_covariate_path = fixture.write_file("missing-fid-covariates.tsv", "IID\tage\nI1\t40\n");
        let missing_fid_covariate_inputs = MultiAlignmentInputs {
            sample_indices: vec![0],
            family_identifiers: strings(&["F1"]),
            individual_identifiers: strings(&["I1"]),
            phenotype_path,
            phenotype_names: strings(&["trait_a", "trait_b"]),
            covariate_path: Some(missing_fid_covariate_path),
            covariate_names: Some(strings(&["age"])),
            is_binary_trait: false,
            sample_key_mode: SampleKeyMode::FidIid,
        };
        assert!(
            align_multi_sample_data(missing_fid_covariate_inputs)
                .expect_err("FID is required in covariate table")
                .contains("Identifier column 'FID'")
        );
    }
}
