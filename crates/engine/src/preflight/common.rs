use super::error::PreflightError;

pub(super) fn validate_shape_counts(label: &'static str, shape_counts: &[i64]) -> Result<(), PreflightError> {
    for &shape_count in shape_counts {
        validate_non_negative_count(label, shape_count)?;
    }
    Ok(())
}

pub(super) fn format_python_shape(shape_counts: &[i64]) -> String {
    match shape_counts {
        [] => "()".to_string(),
        [shape_count] => format!("({shape_count},)"),
        _ => format!("({})", shape_counts.iter().map(std::string::ToString::to_string).collect::<Vec<_>>().join(", ")),
    }
}

pub(super) fn validate_non_negative_count(label: &'static str, count: i64) -> Result<(), PreflightError> {
    if count < 0 {
        return Err(PreflightError::NegativeCount { label, count });
    }
    Ok(())
}
