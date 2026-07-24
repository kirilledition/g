use std::fmt;
use std::path::PathBuf;
use std::str::FromStr;

use serde::de::Visitor;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

const SHA256_BYTE_COUNT: usize = 32;
const SHA256_HEX_CHARACTER_COUNT: usize = SHA256_BYTE_COUNT * 2;

/// A canonical lowercase hexadecimal SHA-256 content address for BGEN bytes.
#[derive(Clone, Copy, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct BgenContentSha256([u8; SHA256_BYTE_COUNT]);

/// Failure to parse a canonical BGEN content SHA-256 value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenContentSha256ParseError {
    /// The input does not contain exactly 64 ASCII bytes.
    InvalidLength {
        /// Number of bytes present in the rejected representation.
        observed_byte_count: usize,
    },
    /// The input contains a byte outside lowercase hexadecimal ASCII.
    InvalidCharacter {
        /// Byte index of the rejected character.
        character_index: usize,
        /// Rejected byte value.
        character_byte: u8,
    },
}

/// Detached identity of one exact BGEN byte sequence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BgenContentFingerprint {
    /// SHA-256 of the exact bytes.
    pub content_sha256: BgenContentSha256,
    /// Number of exact bytes authenticated by the digest.
    pub byte_count: u64,
}

/// Evidence retained by a BGEN reader about the bytes it serves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BgenContentEvidence {
    /// The reader owns immutable bytes with an authoritative fingerprint.
    OwnedSnapshot(BgenContentFingerprint),
    /// The reader performs positioned I/O without content authentication.
    PositionedUnattested(BgenSourceIdentity),
}

/// How one BGEN reader request resolved its source.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BgenSnapshotResolution {
    /// The request acquired its source from the supplied locator.
    CapturedFromLocator,
    /// The request reused an already authenticated process snapshot.
    ProcessSnapshotCache,
}

/// Identity of the exact BGEN file opened by the native reader.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BgenSourceIdentity {
    pub configured_path: PathBuf,
    pub canonical_path: Option<PathBuf>,
    pub device_identifier: u64,
    pub inode_identifier: u64,
    pub change_time_nanoseconds: i64,
    pub modification_time_nanoseconds: i64,
    pub file_size: u64,
}

/// Acquisition provenance for one BGEN reader request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BgenSourceProvenance {
    /// Locator supplied for this reader request.
    pub requested_path: PathBuf,
    /// Descriptor metadata recorded when the source was opened.
    ///
    /// This is acquisition provenance, not content authority. In particular,
    /// positioned sources remain mutable and unattested.
    pub captured_source_identity: BgenSourceIdentity,
    /// Whether this request acquired its locator or reused a process snapshot.
    pub resolution: BgenSnapshotResolution,
}

impl BgenContentSha256 {
    /// Construct a content address from its exact digest bytes.
    #[must_use]
    pub const fn from_bytes(bytes: [u8; SHA256_BYTE_COUNT]) -> Self {
        Self(bytes)
    }

    /// Borrow the exact digest bytes.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; SHA256_BYTE_COUNT] {
        &self.0
    }
}

impl fmt::Debug for BgenContentSha256 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_tuple("BgenContentSha256").field(&format_args!("{self}")).finish()
    }
}

impl fmt::Display for BgenContentSha256 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(formatter, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for BgenContentSha256 {
    type Err = BgenContentSha256ParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let input_bytes = value.as_bytes();
        if input_bytes.len() != SHA256_HEX_CHARACTER_COUNT {
            return Err(BgenContentSha256ParseError::InvalidLength { observed_byte_count: input_bytes.len() });
        }
        let mut digest_bytes = [0_u8; SHA256_BYTE_COUNT];
        for (character_index, character_pair) in input_bytes.chunks_exact(2).enumerate() {
            let high_character_index = character_index * 2;
            let high_nibble = parse_lowercase_hexadecimal_nibble(character_pair[0], high_character_index)?;
            let low_nibble = parse_lowercase_hexadecimal_nibble(character_pair[1], high_character_index + 1)?;
            digest_bytes[character_index] = (high_nibble << 4) | low_nibble;
        }
        Ok(Self(digest_bytes))
    }
}

impl fmt::Display for BgenContentSha256ParseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { observed_byte_count } => write!(
                formatter,
                "BGEN content SHA-256 must contain exactly {SHA256_HEX_CHARACTER_COUNT} lowercase hexadecimal bytes; observed {observed_byte_count}"
            ),
            Self::InvalidCharacter { character_index, character_byte } => write!(
                formatter,
                "BGEN content SHA-256 contains non-lowercase-hexadecimal byte 0x{character_byte:02x} at index {character_index}"
            ),
        }
    }
}

impl std::error::Error for BgenContentSha256ParseError {}

impl Serialize for BgenContentSha256 {
    fn serialize<SerializerType>(&self, serializer: SerializerType) -> Result<SerializerType::Ok, SerializerType::Error>
    where
        SerializerType: Serializer,
    {
        serializer.collect_str(self)
    }
}

impl<'de> Deserialize<'de> for BgenContentSha256 {
    fn deserialize<DeserializerType>(deserializer: DeserializerType) -> Result<Self, DeserializerType::Error>
    where
        DeserializerType: Deserializer<'de>,
    {
        deserializer.deserialize_str(BgenContentSha256Visitor)
    }
}

struct BgenContentSha256Visitor;

impl Visitor<'_> for BgenContentSha256Visitor {
    type Value = BgenContentSha256;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("exactly 64 lowercase hexadecimal SHA-256 characters")
    }

    fn visit_str<ErrorType>(self, value: &str) -> Result<Self::Value, ErrorType>
    where
        ErrorType: serde::de::Error,
    {
        value.parse().map_err(ErrorType::custom)
    }
}

fn parse_lowercase_hexadecimal_nibble(
    character_byte: u8,
    character_index: usize,
) -> Result<u8, BgenContentSha256ParseError> {
    match character_byte {
        b'0'..=b'9' => Ok(character_byte - b'0'),
        b'a'..=b'f' => Ok(character_byte - b'a' + 10),
        _ => Err(BgenContentSha256ParseError::InvalidCharacter { character_index, character_byte }),
    }
}
