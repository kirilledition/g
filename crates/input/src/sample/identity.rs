use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use crate::error::InputResult;

use super::types::SampleIdentifierData;

struct SampleFileReader<R: BufRead> {
    path_text: String,
    reader: R,
    line_buffer: String,
}

pub fn load_sample_identifier_data_from_sample_file(
    sample_path: &Path,
    expected_sample_count: usize,
) -> InputResult<SampleIdentifierData> {
    let mut reader = open_sample_file_reader(sample_path)?;
    let missing_header_error =
        || format!("Sample file '{}' must contain at least two header lines.", sample_path.display());
    let column_names = reader
        .read_next_nonempty_line()?
        .ok_or_else(missing_header_error)?
        .split_ascii_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    let column_types = reader
        .read_next_nonempty_line()?
        .ok_or_else(missing_header_error)?
        .split_ascii_whitespace()
        .map(str::to_string)
        .collect::<Vec<_>>();
    validate_sample_file_header(sample_path, &column_names, &column_types)?;
    let family_identifier_column_index = column_names
        .iter()
        .position(|column_name| column_name == "ID_1")
        .ok_or_else(|| format!("Sample file '{}' must contain the identifier column 'ID_1'.", sample_path.display()))?;
    let individual_identifier_column_index = column_names
        .iter()
        .position(|column_name| column_name == "ID_2")
        .ok_or_else(|| format!("Sample file '{}' must contain the identifier column 'ID_2'.", sample_path.display()))?;

    let mut family_identifiers = Vec::with_capacity(expected_sample_count);
    let mut individual_identifiers = Vec::with_capacity(expected_sample_count);
    let mut sample_count = 0usize;
    while let Some(row_text) = reader.read_next_nonempty_line()? {
        sample_count += 1;
        let mut row_value_count = 0;
        let mut family_identifier = None;
        let mut individual_identifier = None;
        for (column_index, value) in row_text.split_ascii_whitespace().enumerate() {
            row_value_count += 1;
            if column_index == family_identifier_column_index {
                family_identifier = Some(value.to_owned());
            }
            if column_index == individual_identifier_column_index {
                individual_identifier = Some(value.to_owned());
            }
        }
        if row_value_count != column_names.len() {
            return Err(format!(
                "Sample file '{}' line {} has {} values, but the header declares {} columns.",
                sample_path.display(),
                sample_count + 2,
                row_value_count,
                column_names.len(),
            )
            .into());
        }
        let family_identifier = family_identifier.ok_or_else(|| {
            format!("Sample file '{}' line {} is missing ID_1.", sample_path.display(), sample_count + 2)
        })?;
        let individual_identifier = individual_identifier.ok_or_else(|| {
            format!("Sample file '{}' line {} is missing ID_2.", sample_path.display(), sample_count + 2)
        })?;
        if family_identifier.is_empty() {
            return Err(format!(
                "Sample file '{}' line {} contains an empty ID_1; ID_1 and ID_2 must both be non-empty.",
                sample_path.display(),
                sample_count + 2,
            )
            .into());
        }
        if individual_identifier.is_empty() {
            return Err(format!(
                "Sample file '{}' line {} contains an empty ID_2; ID_1 and ID_2 must both be non-empty.",
                sample_path.display(),
                sample_count + 2,
            )
            .into());
        }
        family_identifiers.push(family_identifier);
        individual_identifiers.push(individual_identifier);
    }
    if sample_count != expected_sample_count {
        return Err(format!(
            "Expect number of samples in file to match BGEN sample count. Sample file '{}' contains {sample_count} rows, but the BGEN contains {expected_sample_count} samples.",
            sample_path.display()
        )
        .into());
    }
    Ok(SampleIdentifierData { family_identifiers, individual_identifiers })
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
    for identifier_column_name in ["ID_1", "ID_2"] {
        let identifier_column_index =
            column_names.iter().position(|column_name| column_name == identifier_column_name).ok_or_else(|| {
                format!(
                    "Sample file '{}' must contain the identifier column '{identifier_column_name}'.",
                    sample_path.display(),
                )
            })?;
        if column_types[identifier_column_index] != "0" {
            return Err(format!(
                "Sample file '{}' must mark '{identifier_column_name}' with type '0'.",
                sample_path.display(),
            ));
        }
    }
    Ok(())
}

fn open_sample_file_reader(sample_path: &Path) -> Result<SampleFileReader<BufReader<File>>, String> {
    let sample_file = File::open(sample_path)
        .map_err(|error| format!("Failed to read sample file '{}': {error}.", sample_path.display()))?;
    Ok(SampleFileReader {
        path_text: sample_path.display().to_string(),
        reader: BufReader::new(sample_file),
        line_buffer: String::new(),
    })
}

impl<R: BufRead> SampleFileReader<R> {
    fn read_next_nonempty_line(&mut self) -> Result<Option<&str>, String> {
        loop {
            self.line_buffer.clear();
            let read_byte_count = self
                .reader
                .read_line(&mut self.line_buffer)
                .map_err(|error| format!("Failed to read sample file '{}': {error}.", self.path_text))?;
            if read_byte_count == 0 {
                return Ok(None);
            }
            if self.line_buffer.split_ascii_whitespace().next().is_some() {
                break;
            }
        }
        Ok(Some(&self.line_buffer))
    }
}
