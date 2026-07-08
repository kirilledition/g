//! Binary correction compilation.

use g_plan as plan;

use crate::ConfigResult;
use crate::domain::RegenieTraitTypeValue;
use crate::resolved::RegenieConfigData;

use super::conversion;

pub(super) fn build_correction_plan(config: &RegenieConfigData) -> ConfigResult<plan::CorrectionPlan> {
    if config.trait_config.trait_type == RegenieTraitTypeValue::Quantitative {
        return Ok(plan::CorrectionPlan {
            method: plan::BinaryFallbackMethod::ScoreOnly,
            p_threshold: 0.05,
            firth_se: false,
        });
    }
    plan::normalize_binary_correction(
        config.binary.firth,
        config.binary.approx,
        config.binary.spa,
        f64::from(config.binary.p_threshold),
        config.binary.firth_se,
    )
    .map_err(conversion::plan_error_to_config_error)
}
