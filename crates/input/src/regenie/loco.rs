use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use super::{PredictionError, normalize_chromosome};

#[derive(Debug)]
pub(super) struct LocoFileIndex {
    pub(super) file_path: PathBuf,
    pub(super) sample_count: usize,
    pub(super) header_digest: [u8; 32],
    pub(super) source_digest: [u8; 32],
    pub(super) chromosome_rows: HashMap<String, LocoRowIndex>,
    source_identity: LocoSourceIdentity,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LocoSourceIdentity {
    device_identifier: u64,
    inode_identifier: u64,
    change_time_nanoseconds: i128,
    modification_time_nanoseconds: i128,
    file_size: u64,
}

#[derive(Debug)]
pub(super) struct IndexedLocoFile {
    pub(super) file_index: LocoFileIndex,
    pub(super) header_line: String,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct LocoRowIndex {
    byte_offset: u64,
    line_number: usize,
    raw_digest: [u8; 32],
}

#[derive(Debug)]
pub(super) struct LocoSampleIndex {
    pub(super) family_identifiers: Vec<String>,
    pub(super) individual_identifiers: Vec<String>,
}

pub(super) fn index_loco_file(loco_file_path: &Path) -> Result<IndexedLocoFile, PredictionError> {
    if !loco_file_path.exists() {
        return Err(PredictionError::LocoFileNotFound(loco_file_path.to_path_buf()));
    }

    let file = File::open(loco_file_path)?;
    let metadata = file.metadata()?;
    let source_identity = LocoSourceIdentity::from_metadata(&metadata);
    let mut reader = BufReader::new(file);
    let mut line = String::new();
    let mut line_number = 0_usize;
    let mut sample_count = None;
    let mut header_line = None;
    let mut header_digest = None;
    let mut chromosome_rows = HashMap::new();
    loop {
        let byte_offset = reader.stream_position()?;
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            break;
        }
        line_number += 1;
        let stripped_line = line.trim();
        if stripped_line.is_empty() {
            continue;
        }
        if line_number == 1 {
            header_digest = Some(Sha256::digest(line.as_bytes()).into());
            sample_count = Some(validate_loco_header(stripped_line)?);
            header_line = Some(std::mem::take(&mut line));
            continue;
        }
        let mut fields = stripped_line.split_whitespace();
        let Some(chromosome_field) = fields.next() else {
            return Err(PredictionError::InvalidLocoDataLine { line_number, field_count: 0 });
        };
        if fields.next().is_none() {
            return Err(PredictionError::InvalidLocoDataLine { line_number, field_count: 1 });
        }
        let expected_prediction_count =
            sample_count.ok_or_else(|| PredictionError::MissingLocoHeader(loco_file_path.to_path_buf()))?;
        let observed_prediction_count = 1 + fields.count();
        if observed_prediction_count != expected_prediction_count {
            return Err(PredictionError::LocoPredictionCountMismatch {
                line_number,
                expected_count: expected_prediction_count,
                observed_count: observed_prediction_count,
            });
        }
        let chromosome = normalize_chromosome(chromosome_field);
        let raw_digest = Sha256::digest(line.as_bytes()).into();
        if chromosome_rows.insert(chromosome.clone(), LocoRowIndex { byte_offset, line_number, raw_digest }).is_some() {
            return Err(PredictionError::DuplicateChromosome { chromosome });
        }
    }

    let sample_count = sample_count.ok_or_else(|| PredictionError::MissingLocoHeader(loco_file_path.to_path_buf()))?;
    if chromosome_rows.is_empty() {
        return Err(PredictionError::MissingChromosomePredictions(loco_file_path.to_path_buf()));
    }
    let header_digest = header_digest.expect("a parsed LOCO sample index comes from the physical header line");
    let header_line = header_line.expect("a parsed LOCO sample index retains the physical header line");
    ensure_loco_source_unchanged(loco_file_path, &source_identity, reader.get_ref())?;
    let source_digest = build_indexed_source_digest(header_digest, &chromosome_rows);
    Ok(IndexedLocoFile {
        file_index: LocoFileIndex {
            file_path: loco_file_path.to_path_buf(),
            sample_count,
            header_digest,
            source_digest,
            chromosome_rows,
            source_identity,
        },
        header_line,
    })
}

fn build_indexed_source_digest(header_digest: [u8; 32], chromosome_rows: &HashMap<String, LocoRowIndex>) -> [u8; 32] {
    let mut source_hash = Sha256::new();
    source_hash.update(b"loco-indexed-source-v1");
    source_hash.update(header_digest);
    let mut ordered_rows = chromosome_rows.iter().collect::<Vec<_>>();
    ordered_rows.sort_unstable_by_key(|(chromosome, _)| *chromosome);
    source_hash.update(
        u64::try_from(ordered_rows.len()).expect("supported Rust targets represent usize within u64").to_le_bytes(),
    );
    for (chromosome, row) in ordered_rows {
        let chromosome_bytes = chromosome.as_bytes();
        source_hash.update(
            u64::try_from(chromosome_bytes.len())
                .expect("supported Rust targets represent usize within u64")
                .to_le_bytes(),
        );
        source_hash.update(chromosome_bytes);
        source_hash.update(row.raw_digest);
    }
    source_hash.finalize().into()
}

pub(super) fn read_loco_chromosome_predictions_into(
    loco_file_index: &LocoFileIndex,
    chromosome: &str,
    prediction_values: &mut Vec<f32>,
) -> Result<(), PredictionError> {
    let row_index = loco_file_index
        .chromosome_rows
        .get(chromosome)
        .expect("planned LOCO chromosomes are validated against every file index");
    let file = File::open(&loco_file_index.file_path)?;
    if LocoSourceIdentity::from_metadata(&file.metadata()?) != loco_file_index.source_identity {
        return Err(PredictionError::IndexedLocoFileChanged { path: loco_file_index.file_path.clone() });
    }
    let mut reader = BufReader::new(file);
    reader.seek(SeekFrom::Start(row_index.byte_offset))?;
    let mut line = String::new();
    let read_byte_count = reader.read_line(&mut line)?;
    let metadata_changed =
        LocoSourceIdentity::from_metadata(&reader.get_ref().metadata()?) != loco_file_index.source_identity;
    if read_byte_count == 0 {
        if metadata_changed {
            return Err(PredictionError::IndexedLocoFileChanged { path: loco_file_index.file_path.clone() });
        }
        return Err(PredictionError::IndexedLocoRowChanged {
            path: loco_file_index.file_path.clone(),
            line_number: row_index.line_number,
            expected_chromosome: chromosome.to_string(),
            observed_chromosome: "end of file".to_string(),
        });
    }

    let mut fields = line.split_whitespace();
    let Some(chromosome_field) = fields.next() else {
        return Err(PredictionError::InvalidLocoDataLine { line_number: row_index.line_number, field_count: 0 });
    };
    let Some(first_prediction_field) = fields.next() else {
        return Err(PredictionError::InvalidLocoDataLine { line_number: row_index.line_number, field_count: 1 });
    };
    let expected_prediction_count = loco_file_index.sample_count;
    let mut observed_prediction_count = 0_usize;
    let mut first_prediction_error = None;
    for value in std::iter::once(first_prediction_field).chain(fields) {
        observed_prediction_count += 1;
        if first_prediction_error.is_some() {
            continue;
        }
        match value.parse::<f32>() {
            Ok(prediction_value) if prediction_value.is_finite() => prediction_values.push(prediction_value),
            Ok(_) => {
                first_prediction_error = Some(PredictionError::NonFinitePredictionValue {
                    line_number: row_index.line_number,
                    value: value.to_string(),
                });
            }
            Err(source) => {
                first_prediction_error = Some(PredictionError::InvalidPredictionValue {
                    line_number: row_index.line_number,
                    value: value.to_string(),
                    source,
                });
            }
        }
    }
    if observed_prediction_count != expected_prediction_count {
        return Err(PredictionError::LocoPredictionCountMismatch {
            line_number: row_index.line_number,
            expected_count: expected_prediction_count,
            observed_count: observed_prediction_count,
        });
    }
    let observed_chromosome = normalize_chromosome(chromosome_field);
    if observed_chromosome != chromosome {
        return Err(PredictionError::IndexedLocoRowChanged {
            path: loco_file_index.file_path.clone(),
            line_number: row_index.line_number,
            expected_chromosome: chromosome.to_string(),
            observed_chromosome,
        });
    }
    if metadata_changed || !loco_source_path_matches(&loco_file_index.file_path, &loco_file_index.source_identity)? {
        return Err(PredictionError::IndexedLocoFileChanged { path: loco_file_index.file_path.clone() });
    }
    let observed_raw_digest: [u8; 32] = Sha256::digest(line.as_bytes()).into();
    if observed_raw_digest != row_index.raw_digest {
        return Err(PredictionError::IndexedLocoRowContentChanged {
            path: loco_file_index.file_path.clone(),
            line_number: row_index.line_number,
            chromosome: chromosome.to_string(),
        });
    }
    if let Some(prediction_error) = first_prediction_error {
        return Err(prediction_error);
    }
    Ok(())
}

impl LocoSourceIdentity {
    fn from_metadata(metadata: &std::fs::Metadata) -> Self {
        Self {
            device_identifier: metadata.dev(),
            inode_identifier: metadata.ino(),
            change_time_nanoseconds: metadata_timestamp_nanoseconds(metadata.ctime(), metadata.ctime_nsec()),
            modification_time_nanoseconds: metadata_timestamp_nanoseconds(metadata.mtime(), metadata.mtime_nsec()),
            file_size: metadata.len(),
        }
    }
}

fn ensure_loco_source_unchanged(
    path: &Path,
    expected_identity: &LocoSourceIdentity,
    opened_file: &File,
) -> Result<(), PredictionError> {
    let opened_file_matches = LocoSourceIdentity::from_metadata(&opened_file.metadata()?) == *expected_identity;
    if opened_file_matches && loco_source_path_matches(path, expected_identity)? {
        return Ok(());
    }
    Err(PredictionError::IndexedLocoFileChanged { path: path.to_path_buf() })
}

fn loco_source_path_matches(path: &Path, expected_identity: &LocoSourceIdentity) -> Result<bool, PredictionError> {
    match path.metadata() {
        Ok(metadata) => Ok(LocoSourceIdentity::from_metadata(&metadata) == *expected_identity),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error.into()),
    }
}

fn metadata_timestamp_nanoseconds(seconds: i64, nanoseconds: i64) -> i128 {
    i128::from(seconds) * 1_000_000_000_i128 + i128::from(nanoseconds)
}

fn validate_loco_header(header_line: &str) -> Result<usize, PredictionError> {
    let mut fields = header_line.split_whitespace();
    let Some(observed_marker) = fields.next() else {
        return Err(PredictionError::EmptyLocoHeader);
    };
    let sample_identifier_count = fields.clone().count();
    if sample_identifier_count == 0 {
        return Err(PredictionError::EmptyLocoHeader);
    }
    if observed_marker != "FID_IID" {
        return Err(PredictionError::InvalidLocoHeaderMarker { observed_marker: observed_marker.to_string() });
    }

    for (sample_index, sample_identifier) in fields.enumerate() {
        let valid_sample_identifier =
            sample_identifier.split_once('_').is_some_and(|(family_identifier, individual_identifier)| {
                !family_identifier.is_empty() && !individual_identifier.is_empty()
            });
        if !valid_sample_identifier {
            return Err(PredictionError::InvalidLocoSampleIdentifier {
                sample_index,
                sample_identifier: sample_identifier.to_string(),
            });
        }
    }
    Ok(sample_identifier_count)
}

pub(super) fn parse_loco_sample_identifiers(header_line: &str) -> Result<LocoSampleIndex, PredictionError> {
    let sample_identifier_count = validate_loco_header(header_line)?;
    let mut fields = header_line.split_whitespace();
    let _ = fields.next().expect("validated LOCO headers contain their marker");

    let mut family_identifiers = Vec::with_capacity(sample_identifier_count);
    let mut individual_identifiers = Vec::with_capacity(sample_identifier_count);
    for (sample_index, sample_identifier) in fields.enumerate() {
        let (family_identifier, individual_identifier) = sample_identifier
            .split_once('_')
            .unwrap_or_else(|| unreachable!("LOCO sample identifier {sample_index} was validated immediately above"));
        family_identifiers.push(family_identifier.to_string());
        individual_identifiers.push(individual_identifier.to_string());
    }

    Ok(LocoSampleIndex { family_identifiers, individual_identifiers })
}
