use std::fs::File;
use std::io::Read;
use std::path::Path;

use crate::error::InputError;

use super::SampleAlignmentResult;
use super::keys::{ObservedTableSampleKeys, SampleRowIndicesByKey, duplicate_table_sample_key_error};

const TABULAR_MISSING_VALUE_TOKENS: &[&str] = &["", "NA", "NaN", "nan", "-9"];

struct TabularColumnSelection {
    family_identifier_value_index: usize,
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

struct StreamingTabularReader<R: Read> {
    path_text: String,
    source_label: &'static str,
    reader: csv::Reader<R>,
    current_record: csv::StringRecord,
    current_line_number: usize,
}

pub(super) struct MultiPhenotypeTable {
    pub(super) phenotype_values: Vec<f32>,
    pub(super) phenotype_masks: Vec<bool>,
    pub(super) phenotype_count: usize,
    pub(super) sample_count: usize,
}

pub(super) struct CovariateTable {
    pub(super) selected_covariate_names: Vec<String>,
    pub(super) covariate_values: Vec<f32>,
    pub(super) covariate_mask: Vec<bool>,
    pub(super) selected_covariate_count: usize,
}

pub(super) fn read_multi_phenotype_table(
    phenotype_path: &Path,
    phenotype_names: &[String],
    is_binary_trait: bool,
    sample_row_indices_by_key: &SampleRowIndicesByKey<'_>,
    sample_count: usize,
) -> SampleAlignmentResult<MultiPhenotypeTable> {
    let mut reader = open_tabular_reader(phenotype_path, "phenotype table", b'\t')?;
    let headers = read_tabular_header(&mut reader, phenotype_path)?;
    let phenotype_path_text = phenotype_path.display().to_string();
    let family_identifier_index = required_column_index(&headers, "FID", &phenotype_path_text)?;
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
    let mut observed_sample_keys = ObservedTableSampleKeys::new(sample_count);
    let phenotype_count = phenotype_names.len();
    let mut phenotype_values = vec![0.0; phenotype_count * sample_count];
    let mut phenotype_masks = vec![false; phenotype_count * sample_count];
    while reader.read_next_record()? {
        let record = &reader.current_record;
        selection.validate_record(&reader, record)?;
        let family_identifier = selection.record_value(record, selection.family_identifier_value_index);
        let individual_identifier = selection.record_value(record, selection.individual_identifier_value_index);
        validate_nonempty_sample_key(&reader, family_identifier, individual_identifier)?;
        let sample_array_index = sample_row_indices_by_key.sample_row_index(family_identifier, individual_identifier);
        if !observed_sample_keys.insert(family_identifier, individual_identifier, sample_array_index) {
            return Err(
                duplicate_table_sample_key_error("phenotype table", family_identifier, individual_identifier).into()
            );
        }
        let Some(sample_array_index) = sample_array_index else {
            continue;
        };
        for (phenotype_index, phenotype_name) in phenotype_names.iter().enumerate() {
            let phenotype_value = selection.record_value(record, selection.data_value_indices[phenotype_index]);
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

pub(super) fn load_covariate_table(
    covariate_path: Option<&str>,
    covariate_names: Option<&[String]>,
    sample_row_indices_by_key: &SampleRowIndicesByKey<'_>,
    parse_candidate_mask: &[bool],
    sample_count: usize,
) -> SampleAlignmentResult<CovariateTable> {
    if let Some(covariate_path) = covariate_path {
        return read_covariate_table(
            Path::new(covariate_path),
            covariate_names,
            sample_row_indices_by_key,
            parse_candidate_mask,
            sample_count,
        );
    }
    if covariate_names.is_some() {
        return Err("Covariate names cannot be provided without a covariate table.".to_string().into());
    }
    Ok(empty_covariate_table(sample_count))
}

pub(super) fn is_complete_multi_phenotype_sample(
    phenotype_table: &MultiPhenotypeTable,
    sample_array_index: usize,
) -> bool {
    (0..phenotype_table.phenotype_count).all(|phenotype_index| {
        phenotype_table.phenotype_masks[phenotype_index * phenotype_table.sample_count + sample_array_index]
    })
}

pub(super) fn multi_phenotype_parse_candidate_mask(phenotype_table: &MultiPhenotypeTable) -> Vec<bool> {
    (0..phenotype_table.sample_count)
        .map(|sample_array_index| {
            (0..phenotype_table.phenotype_count).any(|phenotype_index| {
                phenotype_table.phenotype_masks[phenotype_index * phenotype_table.sample_count + sample_array_index]
            })
        })
        .collect()
}

// Phenotype and covariate tables intentionally use tab-only parsing.
fn open_tabular_reader(
    table_path: &Path,
    source_label: &'static str,
    delimiter: u8,
) -> SampleAlignmentResult<StreamingTabularReader<File>> {
    let table_file = File::open(table_path)
        .map_err(|error| format!("Failed to read {source_label} '{}': {error}.", table_path.display()))?;
    Ok(StreamingTabularReader::new(table_path.display().to_string(), source_label, table_file, delimiter))
}

fn read_tabular_header<R: Read>(
    reader: &mut StreamingTabularReader<R>,
    table_path: &Path,
) -> SampleAlignmentResult<Vec<String>> {
    if !reader.read_next_record()? {
        return Err(format!("{} '{}' is empty.", reader.display_source_label(), table_path.display()).into());
    }
    let header_record = &reader.current_record;
    let headers = header_record.iter().map(ToString::to_string).collect::<Vec<_>>();
    if headers.is_empty() {
        return Err(
            format!("{} '{}' must contain a header row.", reader.display_source_label(), table_path.display()).into()
        );
    }
    Ok(headers)
}

impl<R: Read> StreamingTabularReader<R> {
    fn new(path_text: String, source_label: &'static str, source: R, delimiter: u8) -> Self {
        let reader = csv::ReaderBuilder::new()
            .delimiter(delimiter)
            .flexible(true)
            .has_headers(false)
            .trim(csv::Trim::All)
            .from_reader(source);
        Self { path_text, source_label, reader, current_record: csv::StringRecord::new(), current_line_number: 0 }
    }

    fn read_next_record(&mut self) -> SampleAlignmentResult<bool> {
        loop {
            self.current_record.clear();
            let has_record = self
                .reader
                .read_record(&mut self.current_record)
                .map_err(|error| format!("Failed to read {} '{}': {error}.", self.source_label, self.path_text))?;
            if !has_record {
                return Ok(false);
            }
            self.current_line_number += 1;
            if !is_empty_tabular_record(&self.current_record) {
                return Ok(true);
            }
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

fn validate_nonempty_sample_key<R: Read>(
    reader: &StreamingTabularReader<R>,
    family_identifier: &str,
    individual_identifier: &str,
) -> SampleAlignmentResult<()> {
    if family_identifier.is_empty() {
        return Err(format!(
            "{} '{}' line {} contains an empty FID; FID and IID must both be non-empty.",
            reader.display_source_label(),
            reader.path_text,
            reader.current_line_number,
        )
        .into());
    }
    if individual_identifier.is_empty() {
        return Err(format!(
            "{} '{}' line {} contains an empty IID; FID and IID must both be non-empty.",
            reader.display_source_label(),
            reader.path_text,
            reader.current_line_number,
        )
        .into());
    }
    Ok(())
}

fn required_column_index(headers: &[String], column_name: &str, table_path: &str) -> SampleAlignmentResult<usize> {
    Ok(column_index(headers, column_name).ok_or_else(|| {
        if column_name == "FID" || column_name == "IID" {
            format!("Identifier column '{column_name}' was not found in {table_path}.")
        } else {
            format!("Phenotype column '{column_name}' was not found in {table_path}.")
        }
    })?)
}

fn column_index(headers: &[String], column_name: &str) -> Option<usize> {
    headers.iter().position(|header| header == column_name)
}

impl TabularColumnSelection {
    fn new(
        family_identifier_column_index: usize,
        individual_identifier_column_index: usize,
        data_column_definitions: &[TabularColumnDefinition<'_>],
    ) -> Self {
        let mut selected_columns = Vec::with_capacity(data_column_definitions.len() + 2);
        let family_identifier_value_index =
            push_selected_column(&mut selected_columns, "FID", family_identifier_column_index);
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
    ) -> SampleAlignmentResult<()> {
        for selected_column in &self.selected_columns {
            if selected_column.column_index >= record.len() {
                return Err(reader.missing_selected_column_error(selected_column, record).into());
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

fn select_covariate_names(
    covariate_headers: &[String],
    requested_covariate_names: Option<&[String]>,
    covariate_path: &str,
) -> SampleAlignmentResult<Vec<String>> {
    if let Some(covariate_names) = requested_covariate_names {
        let missing_covariates: Vec<String> = covariate_names
            .iter()
            .filter(|covariate_name| column_index(covariate_headers, covariate_name).is_none())
            .cloned()
            .collect();
        if !missing_covariates.is_empty() {
            return Err(format!("Covariate columns are missing from {covariate_path}: {missing_covariates:?}.").into());
        }
        return Ok(covariate_names.to_vec());
    }
    let inferred_covariate_names: Vec<String> = covariate_headers
        .iter()
        .filter(|column_name| column_name.as_str() != "FID" && column_name.as_str() != "IID")
        .cloned()
        .collect();
    if inferred_covariate_names.is_empty() {
        return Err("Covariate table must contain at least one non-identifier column.".to_string().into());
    }
    Ok(inferred_covariate_names)
}

fn read_covariate_table(
    covariate_path: &Path,
    requested_covariate_names: Option<&[String]>,
    sample_row_indices_by_key: &SampleRowIndicesByKey<'_>,
    parse_candidate_mask: &[bool],
    sample_count: usize,
) -> SampleAlignmentResult<CovariateTable> {
    let mut reader = open_tabular_reader(covariate_path, "covariate table", b'\t')?;
    let headers = read_tabular_header(&mut reader, covariate_path)?;
    let covariate_path_text = covariate_path.display().to_string();
    let family_identifier_index = required_column_index(&headers, "FID", &covariate_path_text)?;
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
    let mut observed_sample_keys = ObservedTableSampleKeys::new(sample_count);
    let selected_covariate_count = selected_covariate_names.len();
    let mut covariate_values = vec![0.0; sample_count * selected_covariate_count];
    let mut covariate_mask = vec![false; sample_count];
    while reader.read_next_record()? {
        let record = &reader.current_record;
        selection.validate_record(&reader, record)?;
        let family_identifier = selection.record_value(record, selection.family_identifier_value_index);
        let individual_identifier = selection.record_value(record, selection.individual_identifier_value_index);
        validate_nonempty_sample_key(&reader, family_identifier, individual_identifier)?;
        let sample_array_index = sample_row_indices_by_key.sample_row_index(family_identifier, individual_identifier);
        if !observed_sample_keys.insert(family_identifier, individual_identifier, sample_array_index) {
            return Err(
                duplicate_table_sample_key_error("covariate table", family_identifier, individual_identifier).into()
            );
        }
        let Some(sample_array_index) = sample_array_index else {
            continue;
        };
        if !parse_candidate_mask[sample_array_index] {
            continue;
        }
        let mut row_has_missing_covariates = false;
        for (covariate_index, covariate_name) in selected_covariate_names.iter().enumerate() {
            let covariate_value = selection.record_value(record, selection.data_value_indices[covariate_index]);
            if is_tabular_null_value(covariate_value) {
                row_has_missing_covariates = true;
                break;
            }
            let value_index = sample_array_index * selected_covariate_count + covariate_index;
            covariate_values[value_index] = parse_covariate_value(covariate_value, covariate_name)?;
        }
        if !row_has_missing_covariates {
            covariate_mask[sample_array_index] = true;
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

fn parse_phenotype_value(
    phenotype_value: &str,
    phenotype_name: &str,
    is_binary_trait: bool,
) -> SampleAlignmentResult<f32> {
    let parsed_value = phenotype_value.parse::<f32>().map_err(|error| {
        format!("Failed to parse phenotype column '{phenotype_name}' value '{phenotype_value}': {error}.")
    })?;
    if !parsed_value.is_finite() {
        return Err(InputError::NonFinitePhenotypeValue {
            phenotype_name: phenotype_name.to_string(),
            value: phenotype_value.to_string(),
        });
    }
    if !is_binary_trait {
        return Ok(parsed_value);
    }
    // Binary phenotype coding is discrete: approximate comparisons would
    // incorrectly admit values other than the specified 1 and 2 tokens.
    if parsed_value.to_bits() == 1.0_f32.to_bits() {
        return Ok(0.0);
    }
    if parsed_value.to_bits() == 2.0_f32.to_bits() {
        return Ok(1.0);
    }
    Err(format!("Binary phenotype must contain only values 1 and 2, found value {parsed_value}.").into())
}

fn parse_covariate_value(covariate_value: &str, covariate_name: &str) -> SampleAlignmentResult<f32> {
    let parsed_value = covariate_value
        .parse::<f32>()
        .map_err(|error| format!("Failed to parse covariate value '{covariate_value}': {error}."))?;
    if !parsed_value.is_finite() {
        return Err(InputError::NonFiniteCovariateValue {
            covariate_name: covariate_name.to_string(),
            value: covariate_value.to_string(),
        });
    }
    Ok(parsed_value)
}

#[cfg(test)]
mod tests {
    use crate::error::InputError;

    use super::{TABULAR_MISSING_VALUE_TOKENS, is_tabular_null_value, parse_covariate_value, parse_phenotype_value};

    #[test]
    fn tabular_missing_tokens_are_exact_and_documented() {
        for missing_value in TABULAR_MISSING_VALUE_TOKENS {
            assert!(is_tabular_null_value(missing_value));
        }
        for observed_value in ["NAN", "Inf", "-8", " NA "] {
            assert!(!is_tabular_null_value(observed_value));
        }
    }

    #[test]
    fn quantitative_phenotypes_and_covariates_reject_nonfinite_values() {
        for nonfinite_value in ["inf", "-inf", "Infinity", "NAN"] {
            assert!(matches!(
                parse_phenotype_value(nonfinite_value, "trait-a", false),
                Err(InputError::NonFinitePhenotypeValue { .. })
            ));
            assert!(matches!(
                parse_covariate_value(nonfinite_value, "age"),
                Err(InputError::NonFiniteCovariateValue { .. })
            ));
        }
    }
}
