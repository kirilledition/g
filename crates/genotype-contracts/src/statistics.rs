//! Output-facing genotype statistics.

#[derive(Debug, PartialEq)]
pub struct ChunkOutputStatistics {
    pub allele_one_frequency: Vec<f32>,
    pub observation_count: Vec<i32>,
    pub info_score: NullableFloat32Column,
}

#[derive(Debug, PartialEq)]
pub struct NullableFloat32Column {
    pub values: Vec<f32>,
    pub validity_bytes: Vec<u8>,
}

impl NullableFloat32Column {
    pub fn push(&mut self, value: f32, is_valid: bool) {
        let value_index = self.values.len();
        let validity_bit_index = value_index & 7;
        if validity_bit_index == 0 {
            self.validity_bytes.push(0);
        }
        if is_valid {
            self.validity_bytes[value_index >> 3] |= 1_u8 << validity_bit_index;
        }
        self.values.push(value);
    }
}
