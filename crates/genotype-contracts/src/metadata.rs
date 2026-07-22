//! Shared, allocation-preserving variant metadata columns.

use std::fmt;
use std::ops::Range;
use std::sync::Arc;

/// Violation of the structural invariants required by shared variant metadata.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VariantMetadataInvariantError {
    /// The chromosome-code column does not match the canonical variant count.
    ChromosomeCodeCountMismatch { variant_count: usize, chromosome_code_count: usize },
    /// The allele-one-code column does not match the canonical variant count.
    AlleleOneCodeCountMismatch { variant_count: usize, allele_one_code_count: usize },
    /// The allele-two-code column does not match the canonical variant count.
    AlleleTwoCodeCountMismatch { variant_count: usize, allele_two_code_count: usize },
    /// The identifier-offset column does not contain one terminal offset beyond the variants.
    VariantIdentifierOffsetCountMismatch { variant_count: usize, expected_offset_count: usize, offset_count: usize },
    /// The identifier-offset column does not begin at the start of the text arena.
    VariantIdentifierOffsetStartMismatch { observed_offset: u32 },
    /// The identifier-offset column does not end at the end of the text arena.
    VariantIdentifierOffsetEndMismatch { text_length: usize, observed_offset: u32 },
    /// An identifier offset addresses bytes beyond the text arena.
    VariantIdentifierOffsetOutOfBounds { offset_index: usize, offset: u32, text_length: usize },
    /// An identifier offset divides a multibyte UTF-8 code point.
    VariantIdentifierOffsetNotUtf8Boundary { offset_index: usize, offset: u32 },
    /// Two adjacent identifier offsets are decreasing.
    VariantIdentifierOffsetOrder {
        preceding_offset_index: usize,
        following_offset_index: usize,
        preceding_offset: u32,
        following_offset: u32,
    },
    /// A chromosome code does not address the shared text dictionary.
    ChromosomeCodeOutOfBounds { variant_index: usize, code: u32, dictionary_length: usize },
    /// An allele-one code does not address the shared text dictionary.
    AlleleOneCodeOutOfBounds { variant_index: usize, code: u32, dictionary_length: usize },
    /// An allele-two code does not address the shared text dictionary.
    AlleleTwoCodeOutOfBounds { variant_index: usize, code: u32, dictionary_length: usize },
    /// A requested metadata range has a start greater than its end.
    RangeStartAfterEnd { range_start: usize, range_end: usize },
    /// A requested metadata range extends beyond the shared store.
    RangeOutOfBounds { range_start: usize, range_end: usize, variant_count: usize },
}

impl fmt::Display for VariantMetadataInvariantError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChromosomeCodeCountMismatch { variant_count, chromosome_code_count } => write!(
                formatter,
                "chromosome code count {chromosome_code_count} does not match variant count {variant_count}"
            ),
            Self::AlleleOneCodeCountMismatch { variant_count, allele_one_code_count } => write!(
                formatter,
                "allele-one code count {allele_one_code_count} does not match variant count {variant_count}"
            ),
            Self::AlleleTwoCodeCountMismatch { variant_count, allele_two_code_count } => write!(
                formatter,
                "allele-two code count {allele_two_code_count} does not match variant count {variant_count}"
            ),
            Self::VariantIdentifierOffsetCountMismatch { variant_count, expected_offset_count, offset_count } => {
                write!(
                    formatter,
                    "variant identifier offset count {offset_count} does not match required count {expected_offset_count} for {variant_count} variants"
                )
            }
            Self::VariantIdentifierOffsetStartMismatch { observed_offset } => {
                write!(formatter, "first variant identifier offset must be zero, observed {observed_offset}")
            }
            Self::VariantIdentifierOffsetEndMismatch { text_length, observed_offset } => write!(
                formatter,
                "last variant identifier offset must equal text length {text_length}, observed {observed_offset}"
            ),
            Self::VariantIdentifierOffsetOutOfBounds { offset_index, offset, text_length } => write!(
                formatter,
                "variant identifier offset {offset} at index {offset_index} exceeds text length {text_length}"
            ),
            Self::VariantIdentifierOffsetNotUtf8Boundary { offset_index, offset } => write!(
                formatter,
                "variant identifier offset {offset} at index {offset_index} is not a UTF-8 character boundary"
            ),
            Self::VariantIdentifierOffsetOrder {
                preceding_offset_index,
                following_offset_index,
                preceding_offset,
                following_offset,
            } => write!(
                formatter,
                "variant identifier offsets decrease between indices {preceding_offset_index} and {following_offset_index}: {preceding_offset} exceeds {following_offset}"
            ),
            Self::ChromosomeCodeOutOfBounds { variant_index, code, dictionary_length } => write!(
                formatter,
                "chromosome code {code} at variant index {variant_index} is outside a dictionary of length {dictionary_length}"
            ),
            Self::AlleleOneCodeOutOfBounds { variant_index, code, dictionary_length } => write!(
                formatter,
                "allele-one code {code} at variant index {variant_index} is outside a dictionary of length {dictionary_length}"
            ),
            Self::AlleleTwoCodeOutOfBounds { variant_index, code, dictionary_length } => write!(
                formatter,
                "allele-two code {code} at variant index {variant_index} is outside a dictionary of length {dictionary_length}"
            ),
            Self::RangeStartAfterEnd { range_start, range_end } => {
                write!(formatter, "metadata range start {range_start} exceeds end {range_end}")
            }
            Self::RangeOutOfBounds { range_start, range_end, variant_count } => {
                write!(formatter, "metadata range {range_start}..{range_end} exceeds variant count {variant_count}")
            }
        }
    }
}

impl std::error::Error for VariantMetadataInvariantError {}

#[derive(Clone, Debug)]
pub struct VariantMetadataColumns {
    store: Arc<VariantMetadataStore>,
    range: Range<usize>,
}

#[derive(Debug)]
pub struct VariantMetadataStore {
    text_dictionary: Box<[Arc<str>]>,
    chromosome_codes: Box<[u32]>,
    variant_identifier_text: Box<str>,
    variant_identifier_offsets: Box<[u32]>,
    position: Box<[i64]>,
    allele_one_codes: Box<[u32]>,
    allele_two_codes: Box<[u32]>,
}

impl VariantMetadataColumns {
    /// Select a validated half-open range from one shared metadata store.
    ///
    /// # Errors
    ///
    /// Returns an invariant error when the range is reversed or extends beyond
    /// the store.
    pub fn new(store: Arc<VariantMetadataStore>, range: Range<usize>) -> Result<Self, VariantMetadataInvariantError> {
        if range.start > range.end {
            return Err(VariantMetadataInvariantError::RangeStartAfterEnd {
                range_start: range.start,
                range_end: range.end,
            });
        }
        if range.end > store.position.len() {
            return Err(VariantMetadataInvariantError::RangeOutOfBounds {
                range_start: range.start,
                range_end: range.end,
                variant_count: store.position.len(),
            });
        }
        Ok(Self { store, range })
    }

    #[must_use]
    pub fn chromosomes(&self) -> impl ExactSizeIterator<Item = &str> {
        self.store.chromosome_codes[self.range.clone()].iter().map(|code| self.store.dictionary_value(*code))
    }

    #[must_use]
    pub fn shared_chromosome(&self, relative_variant_index: usize) -> Option<Arc<str>> {
        let absolute_variant_index = self.range.start.checked_add(relative_variant_index)?;
        if absolute_variant_index >= self.range.end {
            return None;
        }
        let chromosome_code = *self.store.chromosome_codes.get(absolute_variant_index)?;
        let dictionary_index = usize::try_from(chromosome_code).ok()?;
        self.store.text_dictionary.get(dictionary_index).map(Arc::clone)
    }

    #[must_use]
    pub fn variant_identifiers(&self) -> impl ExactSizeIterator<Item = &str> {
        self.range.clone().map(|variant_index| self.store.variant_identifier(variant_index))
    }

    #[must_use]
    pub fn position(&self) -> &[i64] {
        &self.store.position[self.range.clone()]
    }

    #[must_use]
    pub fn allele_ones(&self) -> impl ExactSizeIterator<Item = &str> {
        self.store.allele_one_codes[self.range.clone()].iter().map(|code| self.store.dictionary_value(*code))
    }

    #[must_use]
    pub fn allele_twos(&self) -> impl ExactSizeIterator<Item = &str> {
        self.store.allele_two_codes[self.range.clone()].iter().map(|code| self.store.dictionary_value(*code))
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.range.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.range.is_empty()
    }
}

impl VariantMetadataStore {
    /// Construct a shared metadata store after validating every structural invariant.
    ///
    /// Validation is linear in the variant and offset counts and occurs only
    /// while the immutable store is constructed.
    ///
    /// # Errors
    ///
    /// Returns a typed invariant error when parallel columns, identifier
    /// offsets, or dictionary codes cannot form a valid store.
    pub fn from_parts(
        text_dictionary: Box<[Arc<str>]>,
        chromosome_codes: Box<[u32]>,
        variant_identifier_text: Box<str>,
        variant_identifier_offsets: Box<[u32]>,
        position: Box<[i64]>,
        allele_one_codes: Box<[u32]>,
        allele_two_codes: Box<[u32]>,
    ) -> Result<Self, VariantMetadataInvariantError> {
        validate_parallel_column_lengths(&chromosome_codes, &position, &allele_one_codes, &allele_two_codes)?;
        validate_variant_identifier_offsets(&variant_identifier_text, &variant_identifier_offsets, position.len())?;
        validate_dictionary_codes(&text_dictionary, &chromosome_codes, &allele_one_codes, &allele_two_codes)?;
        Ok(Self {
            text_dictionary,
            chromosome_codes,
            variant_identifier_text,
            variant_identifier_offsets,
            position,
            allele_one_codes,
            allele_two_codes,
        })
    }

    fn dictionary_value(&self, code: u32) -> &str {
        let dictionary_index = usize::try_from(code).expect("u32 metadata dictionary code must fit usize");
        &self.text_dictionary[dictionary_index]
    }

    /// Return one identifier by its absolute index in the shared store.
    ///
    /// # Panics
    ///
    /// Panics when the index is out of bounds.
    #[must_use]
    pub fn variant_identifier(&self, variant_index: usize) -> &str {
        let start = usize::try_from(self.variant_identifier_offsets[variant_index])
            .expect("u32 variant identifier offset must fit usize");
        let stop = usize::try_from(self.variant_identifier_offsets[variant_index + 1])
            .expect("u32 variant identifier offset must fit usize");
        &self.variant_identifier_text[start..stop]
    }
}

fn validate_parallel_column_lengths(
    chromosome_codes: &[u32],
    position: &[i64],
    allele_one_codes: &[u32],
    allele_two_codes: &[u32],
) -> Result<(), VariantMetadataInvariantError> {
    let variant_count = position.len();
    if chromosome_codes.len() != variant_count {
        return Err(VariantMetadataInvariantError::ChromosomeCodeCountMismatch {
            variant_count,
            chromosome_code_count: chromosome_codes.len(),
        });
    }
    if allele_one_codes.len() != variant_count {
        return Err(VariantMetadataInvariantError::AlleleOneCodeCountMismatch {
            variant_count,
            allele_one_code_count: allele_one_codes.len(),
        });
    }
    if allele_two_codes.len() != variant_count {
        return Err(VariantMetadataInvariantError::AlleleTwoCodeCountMismatch {
            variant_count,
            allele_two_code_count: allele_two_codes.len(),
        });
    }
    Ok(())
}

fn validate_variant_identifier_offsets(
    variant_identifier_text: &str,
    variant_identifier_offsets: &[u32],
    variant_count: usize,
) -> Result<(), VariantMetadataInvariantError> {
    let expected_offset_count =
        variant_count.checked_add(1).expect("allocated metadata cannot contain usize::MAX rows");
    if variant_identifier_offsets.len() != expected_offset_count {
        return Err(VariantMetadataInvariantError::VariantIdentifierOffsetCountMismatch {
            variant_count,
            expected_offset_count,
            offset_count: variant_identifier_offsets.len(),
        });
    }

    let first_offset = variant_identifier_offsets[0];
    if first_offset != 0 {
        return Err(VariantMetadataInvariantError::VariantIdentifierOffsetStartMismatch {
            observed_offset: first_offset,
        });
    }
    let last_offset = variant_identifier_offsets[expected_offset_count - 1];
    if usize::try_from(last_offset).expect("u32 metadata offset must fit usize") != variant_identifier_text.len() {
        return Err(VariantMetadataInvariantError::VariantIdentifierOffsetEndMismatch {
            text_length: variant_identifier_text.len(),
            observed_offset: last_offset,
        });
    }

    for (offset_index, offset) in variant_identifier_offsets.iter().copied().enumerate() {
        let byte_offset = usize::try_from(offset).expect("u32 metadata offset must fit usize");
        if byte_offset > variant_identifier_text.len() {
            return Err(VariantMetadataInvariantError::VariantIdentifierOffsetOutOfBounds {
                offset_index,
                offset,
                text_length: variant_identifier_text.len(),
            });
        }
        if !variant_identifier_text.is_char_boundary(byte_offset) {
            return Err(VariantMetadataInvariantError::VariantIdentifierOffsetNotUtf8Boundary { offset_index, offset });
        }
    }

    for (preceding_offset_index, adjacent_offsets) in variant_identifier_offsets.windows(2).enumerate() {
        if adjacent_offsets[0] > adjacent_offsets[1] {
            return Err(VariantMetadataInvariantError::VariantIdentifierOffsetOrder {
                preceding_offset_index,
                following_offset_index: preceding_offset_index
                    .checked_add(1)
                    .expect("an adjacent metadata offset index must fit usize"),
                preceding_offset: adjacent_offsets[0],
                following_offset: adjacent_offsets[1],
            });
        }
    }
    Ok(())
}

fn validate_dictionary_codes(
    text_dictionary: &[Arc<str>],
    chromosome_codes: &[u32],
    allele_one_codes: &[u32],
    allele_two_codes: &[u32],
) -> Result<(), VariantMetadataInvariantError> {
    let dictionary_length = text_dictionary.len();
    for (variant_index, code) in chromosome_codes.iter().copied().enumerate() {
        if usize::try_from(code).expect("u32 metadata dictionary code must fit usize") >= dictionary_length {
            return Err(VariantMetadataInvariantError::ChromosomeCodeOutOfBounds {
                variant_index,
                code,
                dictionary_length,
            });
        }
    }
    for (variant_index, code) in allele_one_codes.iter().copied().enumerate() {
        if usize::try_from(code).expect("u32 metadata dictionary code must fit usize") >= dictionary_length {
            return Err(VariantMetadataInvariantError::AlleleOneCodeOutOfBounds {
                variant_index,
                code,
                dictionary_length,
            });
        }
    }
    for (variant_index, code) in allele_two_codes.iter().copied().enumerate() {
        if usize::try_from(code).expect("u32 metadata dictionary code must fit usize") >= dictionary_length {
            return Err(VariantMetadataInvariantError::AlleleTwoCodeOutOfBounds {
                variant_index,
                code,
                dictionary_length,
            });
        }
    }
    Ok(())
}
