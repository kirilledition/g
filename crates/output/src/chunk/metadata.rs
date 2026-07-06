#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VariantMetadataColumns {
    pub chromosome: Vec<String>,
    pub variant_identifier: Vec<String>,
    pub position: Vec<i64>,
    pub allele_one: Vec<String>,
    pub allele_two: Vec<String>,
}
