use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::types::{SampleAlignmentError, SampleIdentifierData};

struct SampleFileReader<R: BufRead> {
    path_text: String,
    reader: R,
    line_buffer: String,
}

pub fn load_sample_identifier_data_from_sample_file(
    sample_path: &Path,
    expected_sample_count: usize,
) -> Result<SampleIdentifierData, SampleAlignmentError> {
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
            return Err(SampleAlignmentError::new(format!(
                "Sample file '{}' line {} has {} values, but the header declares {} columns.",
                sample_path.display(),
                sample_count + 2,
                row_values.len(),
                column_names.len(),
            )));
        }
        sample_indices.push(sample_count - 1);
        family_identifiers.push(row_values[family_identifier_column_index].clone());
        individual_identifiers.push(row_values[individual_identifier_column_index].clone());
    }
    if sample_count != expected_sample_count {
        return Err(SampleAlignmentError::new(format!(
            "Expect number of samples in file to match BGEN sample count. Sample file '{}' contains {sample_count} rows, but the BGEN contains {expected_sample_count} samples.",
            sample_path.display()
        )));
    }
    Ok(SampleIdentifierData { sample_indices, family_identifiers, individual_identifiers })
}

fn validate_sample_file_header(
    sample_path: &Path,
    column_names: &[String],
    column_types: &[String],
) -> Result<(), SampleAlignmentError> {
    if column_names.len() != column_types.len() {
        return Err(SampleAlignmentError::new(format!(
            "Sample file '{}' header and type lines have different column counts.",
            sample_path.display()
        )));
    }
    if column_names.is_empty() {
        return Err(SampleAlignmentError::new(format!(
            "Sample file '{}' does not contain any columns.",
            sample_path.display()
        )));
    }
    if column_types[0] != "0" {
        return Err(SampleAlignmentError::new(format!(
            "Sample file '{}' must mark the first identifier column with type '0'.",
            sample_path.display()
        )));
    }
    if let Some(individual_identifier_column_index) = column_names.iter().position(|column_name| column_name == "ID_2")
        && column_types[individual_identifier_column_index] != "0"
    {
        return Err(SampleAlignmentError::new(format!(
            "Sample file '{}' must mark 'ID_2' with type '0'.",
            sample_path.display()
        )));
    }
    Ok(())
}

fn open_sample_file_reader(sample_path: &Path) -> Result<SampleFileReader<BufReader<File>>, SampleAlignmentError> {
    let sample_file = File::open(sample_path).map_err(|error| {
        SampleAlignmentError::new(format!("Failed to read sample file '{}': {error}.", sample_path.display()))
    })?;
    Ok(SampleFileReader {
        path_text: sample_path.display().to_string(),
        reader: BufReader::new(sample_file),
        line_buffer: String::new(),
    })
}

impl<R: BufRead> SampleFileReader<R> {
    fn read_required_fields(&mut self, empty_error_message: String) -> Result<Vec<String>, SampleAlignmentError> {
        self.read_next_fields()?.ok_or_else(|| SampleAlignmentError::new(empty_error_message))
    }

    fn read_next_fields(&mut self) -> Result<Option<Vec<String>>, SampleAlignmentError> {
        loop {
            self.line_buffer.clear();
            let read_byte_count = self.reader.read_line(&mut self.line_buffer).map_err(|error| {
                SampleAlignmentError::new(format!("Failed to read sample file '{}': {error}.", self.path_text))
            })?;
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
