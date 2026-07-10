//! Closed string-valued planning domains.

use std::str::FromStr;

use serde::{Deserialize, Serialize};

macro_rules! string_enum {
    (
        $name:ident {
            $($variant:ident => $value:literal),+ $(,)?
        }
    ) => {
        #[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
        pub enum $name {
            $(
                #[serde(rename = $value)]
                $variant,
            )+
        }

        impl $name {
            #[must_use]
            pub fn as_str(self) -> &'static str {
                match self {
                    $(Self::$variant => $value,)+
                }
            }

        }

        impl FromStr for $name {
            type Err = String;

            fn from_str(raw_value: &str) -> Result<Self, Self::Err> {
                match raw_value {
                    $($value => Ok(Self::$variant),)+
                    _ => Err(format!("invalid value {raw_value:?}")),
                }
            }
        }
    };
}

string_enum!(AssociationMode {
    Regenie2Linear => "regenie2_linear",
    Regenie2Binary => "regenie2_binary",
});

string_enum!(RegenieTraitType {
    Quantitative => "quantitative",
    Binary => "binary",
});

string_enum!(Device {
    Cpu => "cpu",
    Gpu => "gpu",
});

string_enum!(TrustedBgenValidationMode {
    CacheOnMiss => "cache_on_miss",
    ForceValidate => "force_validate",
});

string_enum!(SampleKeyMode {
    Iid => "iid",
    FidIid => "fid_iid",
});

string_enum!(MultiPhenotypeSampleMode {
    PerPhenotype => "per-phenotype",
    CompleteCase => "complete-case",
});

string_enum!(PhenotypeComputeGroupMode {
    SinglePhenotype => "single-phenotype",
    CompleteCase => "complete-case",
    PerPhenotypeCompatible => "per-phenotype-compatible",
});

string_enum!(GpuGenotypeFormat {
    Auto => "auto",
    Dosage => "dosage",
    Packed8 => "packed8",
});

string_enum!(FloatingPointDtype {
    Float32 => "float32",
    Float64 => "float64",
});

string_enum!(JaxMatmulPrecision {
    Float32 => "float32",
    TensorFloat32 => "tensorfloat32",
    BrainFloat16 => "bfloat16",
    Highest => "highest",
});

string_enum!(BinaryFallbackMethod {
    ScoreOnly => "score_only",
    FirthApproximate => "firth_approximate",
});

string_enum!(NullLogisticNonconvergencePolicy {
    Fail => "fail",
    Warn => "warn",
});

string_enum!(TelemetryMode {
    Off => "off",
    Progress => "progress",
    Profile => "profile",
    Trace => "trace",
});

string_enum!(ResumeMode {
    Fast => "fast",
    Strict => "strict",
});

string_enum!(ParquetCompression {
    Zstd => "zstd",
    None => "none",
});
