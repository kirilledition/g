use super::error::BgenError;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum CompressionType {
    None,
    Zlib,
    Zstandard,
}

impl TryFrom<u32> for CompressionType {
    type Error = BgenError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Zlib),
            2 => Ok(Self::Zstandard),
            unsupported_value => Err(BgenError::UnsupportedFormat(format!(
                "Unsupported BGEN compression flag {unsupported_value}. Only uncompressed, zlib-compressed, and Zstandard-compressed blocks are supported.",
            ))),
        }
    }
}
