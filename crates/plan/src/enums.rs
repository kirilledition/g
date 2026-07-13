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
    Dosage => "dosage",
    Packed8 => "packed8",
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
});
