use std::error::Error;
use std::fmt;

use serde::Serialize;

#[derive(Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Serialize)]
pub struct TransferMetadataKey {
    pub transfer_name: String,
    pub array_role: String,
    pub dtype_name: String,
    pub dimension_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TransferMetadataAccumulator {
    pub observation_count: i64,
    pub total_bytes: i64,
    pub max_bytes: i64,
    pub total_elements: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransferMetadataObservation {
    pub key: TransferMetadataKey,
    pub byte_count: i64,
    pub element_count: i64,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct TransferMetadataSnapshot {
    pub transfer_name: String,
    pub array_role: String,
    pub dtype_name: String,
    pub dimension_count: i64,
    pub observation_count: i64,
    pub total_bytes: i64,
    pub max_bytes: i64,
    pub total_elements: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TransferMetadataError {
    NegativeDimension { dimension: i64 },
    NonPositiveItemSize { item_size: i64 },
    DimensionCountOverflow { dimension_count: usize },
    ElementCountOverflow,
    ByteCountOverflow,
}

impl fmt::Display for TransferMetadataError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NegativeDimension { dimension } => {
                write!(formatter, "Transfer metadata shape dimensions must be nonnegative: {dimension}")
            }
            Self::NonPositiveItemSize { item_size } => {
                write!(formatter, "Transfer metadata dtype item size must be positive: {item_size}")
            }
            Self::DimensionCountOverflow { dimension_count } => {
                write!(formatter, "Transfer metadata dimension count exceeds platform capacity: {dimension_count}")
            }
            Self::ElementCountOverflow => write!(formatter, "Transfer metadata element count exceeds i64 capacity."),
            Self::ByteCountOverflow => write!(formatter, "Transfer metadata byte count exceeds i64 capacity."),
        }
    }
}

impl Error for TransferMetadataError {}

/// Build one transfer metadata observation from array adapter fields.
///
/// # Errors
///
/// Returns an error when the dtype item size is non-positive, any dimension is
/// negative, or the dimension/element/byte counts exceed `i64`.
pub fn build_transfer_metadata_observation(
    transfer_name: &str,
    array_role: &str,
    dtype_name: &str,
    shape_dimensions: &[i64],
    item_size: i64,
) -> Result<TransferMetadataObservation, TransferMetadataError> {
    if item_size <= 0 {
        return Err(TransferMetadataError::NonPositiveItemSize { item_size });
    }
    let dimension_count = i64::try_from(shape_dimensions.len())
        .map_err(|_| TransferMetadataError::DimensionCountOverflow { dimension_count: shape_dimensions.len() })?;
    let mut element_count = 1_i64;
    for dimension in shape_dimensions {
        if *dimension < 0 {
            return Err(TransferMetadataError::NegativeDimension { dimension: *dimension });
        }
        element_count = element_count.checked_mul(*dimension).ok_or(TransferMetadataError::ElementCountOverflow)?;
    }
    let byte_count = element_count.checked_mul(item_size).ok_or(TransferMetadataError::ByteCountOverflow)?;
    Ok(TransferMetadataObservation {
        key: TransferMetadataKey {
            transfer_name: transfer_name.to_string(),
            array_role: array_role.to_string(),
            dtype_name: dtype_name.to_string(),
            dimension_count,
        },
        byte_count,
        element_count,
    })
}
