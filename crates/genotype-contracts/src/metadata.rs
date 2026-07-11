//! Shared, allocation-preserving variant metadata columns.

use std::ops::Range;
use std::sync::Arc;

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
    #[must_use]
    pub fn new(store: Arc<VariantMetadataStore>, range: Range<usize>) -> Self {
        debug_assert!(range.start <= range.end, "metadata range start must not exceed its end");
        debug_assert!(range.end <= store.position.len(), "metadata range must stay within the shared store");
        Self { store, range }
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
    #[must_use]
    pub fn from_parts(
        text_dictionary: Box<[Arc<str>]>,
        chromosome_codes: Box<[u32]>,
        variant_identifier_text: Box<str>,
        variant_identifier_offsets: Box<[u32]>,
        position: Box<[i64]>,
        allele_one_codes: Box<[u32]>,
        allele_two_codes: Box<[u32]>,
    ) -> Self {
        debug_assert_eq!(chromosome_codes.len(), position.len());
        debug_assert_eq!(allele_one_codes.len(), position.len());
        debug_assert_eq!(allele_two_codes.len(), position.len());
        debug_assert_eq!(variant_identifier_offsets.len(), position.len().saturating_add(1));
        debug_assert_eq!(variant_identifier_offsets.first(), Some(&0));
        debug_assert_eq!(
            variant_identifier_offsets.last().and_then(|offset| usize::try_from(*offset).ok()),
            Some(variant_identifier_text.len()),
        );
        debug_assert!(
            variant_identifier_offsets.windows(2).all(|offsets| offsets[0] <= offsets[1]),
            "variant identifier offsets must be monotonic",
        );
        debug_assert!(
            variant_identifier_offsets.iter().all(|offset| {
                usize::try_from(*offset).is_ok_and(|byte_offset| variant_identifier_text.is_char_boundary(byte_offset))
            }),
            "variant identifier offsets must address UTF-8 boundaries",
        );
        debug_assert!(
            chromosome_codes.iter().chain(&allele_one_codes).chain(&allele_two_codes).all(|code| {
                usize::try_from(*code).is_ok_and(|dictionary_index| dictionary_index < text_dictionary.len())
            }),
            "metadata dictionary codes must address the shared dictionary",
        );
        Self {
            text_dictionary,
            chromosome_codes,
            variant_identifier_text,
            variant_identifier_offsets,
            position,
            allele_one_codes,
            allele_two_codes,
        }
    }

    fn dictionary_value(&self, code: u32) -> &str {
        let dictionary_index = usize::try_from(code).expect("u32 metadata dictionary code must fit usize");
        &self.text_dictionary[dictionary_index]
    }

    /// Return one identifier by its absolute index in the shared store.
    ///
    /// # Panics
    ///
    /// Panics when the index is out of bounds or the producer supplied invalid
    /// identifier offsets.
    #[must_use]
    pub fn variant_identifier(&self, variant_index: usize) -> &str {
        let start = usize::try_from(self.variant_identifier_offsets[variant_index])
            .expect("u32 variant identifier offset must fit usize");
        let stop = usize::try_from(self.variant_identifier_offsets[variant_index + 1])
            .expect("u32 variant identifier offset must fit usize");
        &self.variant_identifier_text[start..stop]
    }
}
