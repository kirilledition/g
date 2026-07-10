use super::error::BgenError;

pub(super) const VARIANT_IDENTIFIER_LENGTH_SIZE_IN_BYTES: usize = 2;
pub(super) const ALLELE_LENGTH_SIZE_IN_BYTES: usize = 4;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompressionType {
    None,
    Zlib,
}

impl TryFrom<u32> for CompressionType {
    type Error = BgenError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Zlib),
            unsupported_value => Err(BgenError::UnsupportedFormat(format!(
                "Unsupported BGEN compression flag {unsupported_value}. Only uncompressed and zlib-compressed blocks are supported.",
            ))),
        }
    }
}
