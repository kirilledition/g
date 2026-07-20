use g_plan::{DosageThreshold, PositiveF32, PositiveF64, Probability, ProbabilityFloor, StepScale};

const F32_TOLERANCE: f32 = 1.0e-6;
const F64_TOLERANCE: f64 = 1.0e-12;

fn assert_f32_within_tolerance(actual: f32, expected: f32) {
    let absolute_difference = (actual - expected).abs();
    assert!(
        absolute_difference < F32_TOLERANCE,
        "expected {actual} and {expected} to differ by less than {F32_TOLERANCE}, got {absolute_difference}",
    );
}

fn assert_f64_within_tolerance(actual: f64, expected: f64) {
    let absolute_difference = (actual - expected).abs();
    assert!(
        absolute_difference < F64_TOLERANCE,
        "expected {actual} and {expected} to differ by less than {F64_TOLERANCE}, got {absolute_difference}",
    );
}

#[test]
fn positive_wrappers_accept_only_finite_positive_values() {
    let positive_f32 = PositiveF32::try_from(0.25).expect("positive finite f32 should be accepted");
    let positive_f64 = PositiveF64::try_from(0.125).expect("positive finite f64 should be accepted");
    assert_f32_within_tolerance(positive_f32.get(), 0.25);
    assert_f64_within_tolerance(positive_f64.get(), 0.125);

    for invalid_value in [0.0, -0.0, -1.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(PositiveF32::try_from(invalid_value).is_err());
    }
    for invalid_value in [0.0, -0.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(PositiveF64::try_from(invalid_value).is_err());
    }
}

#[test]
fn step_scale_enforces_its_open_interval() {
    let smallest_value = StepScale::try_from(f64::MIN_POSITIVE).expect("positive lower interior should be accepted");
    let largest_value =
        StepScale::try_from(f64::from_bits(1.0_f64.to_bits() - 1)).expect("upper interior should be accepted");
    assert_f64_within_tolerance(smallest_value.get(), f64::MIN_POSITIVE);
    assert!(largest_value.get() < 1.0);

    for invalid_value in [0.0, -0.0, -0.25, 1.0, 1.25, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(StepScale::try_from(invalid_value).is_err());
    }
}

#[test]
fn dosage_threshold_enforces_its_half_open_interval() {
    let smallest_value =
        DosageThreshold::try_from(f64::MIN_POSITIVE).expect("positive lower interior should be accepted");
    let inclusive_upper_bound = DosageThreshold::try_from(2.0).expect("inclusive upper bound should be accepted");
    assert_f64_within_tolerance(smallest_value.get(), f64::MIN_POSITIVE);
    assert_f64_within_tolerance(inclusive_upper_bound.get(), 2.0);

    for invalid_value in [0.0, -0.0, -0.25, 2.000_001, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(DosageThreshold::try_from(invalid_value).is_err());
    }
}

#[test]
fn probability_enforces_its_open_unit_interval() {
    let smallest_value = Probability::try_from(f32::MIN_POSITIVE).expect("positive lower interior should be accepted");
    let largest_value =
        Probability::try_from(f32::from_bits(1.0_f32.to_bits() - 1)).expect("upper interior should be accepted");
    assert_f32_within_tolerance(smallest_value.get(), f32::MIN_POSITIVE);
    assert!(largest_value.get() < 1.0);

    for invalid_value in [0.0, -0.0, -0.25, 1.0, 1.25, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(Probability::try_from(invalid_value).is_err());
    }
}

#[test]
fn probability_floor_enforces_its_open_half_interval() {
    let smallest_value =
        ProbabilityFloor::try_from(f32::MIN_POSITIVE).expect("positive lower interior should be accepted");
    let largest_value =
        ProbabilityFloor::try_from(f32::from_bits(0.5_f32.to_bits() - 1)).expect("upper interior should be accepted");
    assert_f32_within_tolerance(smallest_value.get(), f32::MIN_POSITIVE);
    assert!(largest_value.get() < 0.5);

    for invalid_value in [0.0, -0.0, -0.25, 0.5, 0.75, f32::INFINITY, f32::NEG_INFINITY, f32::NAN] {
        assert!(ProbabilityFloor::try_from(invalid_value).is_err());
    }
}

#[test]
fn numeric_wrappers_parse_display_and_round_trip_as_json_numbers() {
    let positive_f32 = "0.25".parse::<PositiveF32>().expect("valid positive f32 should parse");
    let probability = "0.125".parse::<Probability>().expect("valid probability should parse");
    let probability_floor = "0.0625".parse::<ProbabilityFloor>().expect("valid floor should parse");
    let positive_f64 = "1.25".parse::<PositiveF64>().expect("valid positive f64 should parse");
    let step_scale = "0.5".parse::<StepScale>().expect("valid step scale should parse");
    let dosage_threshold = "2".parse::<DosageThreshold>().expect("valid dosage threshold should parse");

    assert_eq!(positive_f32.to_string(), "0.25");
    assert_eq!(probability.to_string(), "0.125");
    assert_eq!(probability_floor.to_string(), "0.0625");
    assert_eq!(positive_f64.to_string(), "1.25");
    assert_eq!(step_scale.to_string(), "0.5");
    assert_eq!(dosage_threshold.to_string(), "2");

    let decoded_positive_f32 = round_trip_json(&positive_f32);
    let decoded_probability = round_trip_json(&probability);
    let decoded_probability_floor = round_trip_json(&probability_floor);
    let decoded_positive_f64 = round_trip_json(&positive_f64);
    let decoded_step_scale = round_trip_json(&step_scale);
    let decoded_dosage_threshold = round_trip_json(&dosage_threshold);
    assert_f32_within_tolerance(decoded_positive_f32.get(), 0.25);
    assert_f32_within_tolerance(decoded_probability.get(), 0.125);
    assert_f32_within_tolerance(decoded_probability_floor.get(), 0.0625);
    assert_f64_within_tolerance(decoded_positive_f64.get(), 1.25);
    assert_f64_within_tolerance(decoded_step_scale.get(), 0.5);
    assert_f64_within_tolerance(decoded_dosage_threshold.get(), 2.0);
}

fn round_trip_json<ValueType>(value: &ValueType) -> ValueType
where
    ValueType: serde::Serialize + serde::de::DeserializeOwned,
{
    let serialized = serde_json::to_string(value).expect("validated value serialization should succeed");
    serde_json::from_str(&serialized).expect("serialized validated value should deserialize")
}

#[test]
fn numeric_wrappers_report_parse_and_range_failures() {
    assert_eq!("not-a-number".parse::<PositiveF32>().expect_err("text should be rejected"), "must be a number");
    assert_eq!(PositiveF32::try_from(0.0).expect_err("zero should be rejected"), "must be positive");
    assert_eq!(PositiveF64::try_from(f64::NAN).expect_err("NaN should be rejected"), "must be finite");
    assert_eq!(StepScale::try_from(1.0).expect_err("upper bound should be rejected"), "must be in (0, 1)");
    assert_eq!(DosageThreshold::try_from(2.1).expect_err("upper excess should be rejected"), "must be in (0, 2]");
    assert_eq!(Probability::try_from(1.0).expect_err("upper bound should be rejected"), "must be in (0, 1)");
    assert_eq!(ProbabilityFloor::try_from(0.5).expect_err("upper bound should be rejected"), "must be in (0, 0.5)",);
}

#[test]
fn deserialization_revalidates_numeric_values() {
    assert!(serde_json::from_str::<PositiveF32>("0.0").is_err());
    assert!(serde_json::from_str::<PositiveF64>("-1.0").is_err());
    assert!(serde_json::from_str::<StepScale>("1.0").is_err());
    assert!(serde_json::from_str::<DosageThreshold>("2.1").is_err());
    assert!(serde_json::from_str::<Probability>("1.0").is_err());
    assert!(serde_json::from_str::<ProbabilityFloor>("0.5").is_err());
}
