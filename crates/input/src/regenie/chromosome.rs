#[must_use]
pub(super) fn normalize_chromosome(chromosome: &str) -> String {
    let normalized = chromosome.to_ascii_lowercase();
    let without_prefix = normalized.strip_prefix("chr").unwrap_or(&normalized);
    if without_prefix.chars().all(|character| character.is_ascii_digit()) {
        without_prefix.parse::<u64>().map_or_else(|_| without_prefix.to_string(), |value| value.to_string())
    } else {
        without_prefix.to_string()
    }
}
