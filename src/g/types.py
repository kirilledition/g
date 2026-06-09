"""Enumerated types for configuration and mode selection."""

import enum
from dataclasses import dataclass


class Device(enum.StrEnum):
    """JAX execution device."""

    CPU = "cpu"
    GPU = "gpu"


class AssociationMode(enum.StrEnum):
    """Statistical association model."""

    REGENIE2_LINEAR = "regenie2_linear"
    REGENIE2_BINARY = "regenie2_binary"


class AssociationBackendKind(enum.StrEnum):
    """Concrete backend selected for association execution."""

    JAX_DOSAGE = "jax_dosage"
    JAX_PACKED8 = "jax_packed8"


class ResumeMode(enum.StrEnum):
    """Resume validation mode."""

    FAST = "fast"
    STRICT = "strict"


class OutputFormat(enum.StrEnum):
    """User-facing output materialization format."""

    PARQUET = "parquet"
    ARROW = "arrow"
    REGENIE = "regenie"


class JaxMatmulPrecision(enum.StrEnum):
    """JAX matrix multiplication precision selector."""

    FLOAT32 = "float32"
    TENSORFLOAT32 = "tensorfloat32"
    BFLOAT16 = "bfloat16"
    HIGHEST = "highest"


class FloatingPointDtype(enum.StrEnum):
    """Floating-point dtype selector for JAX compute kernels."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"


class GpuGenotypeFormat(enum.StrEnum):
    """Host-to-device genotype representation for GPU kernels."""

    DOSAGE = "dosage"
    PACKED8 = "packed8"


class ArrowCompression(enum.StrEnum):
    """Arrow IPC compression codec for internal chunk files."""

    ZSTD = "zstd"
    NONE = "none"


class ParquetCompression(enum.StrEnum):
    """Parquet compression codec for dataset part files."""

    ZSTD = "zstd"
    NONE = "none"


class TelemetryMode(enum.StrEnum):
    """Run telemetry detail level."""

    OFF = "off"
    PROGRESS = "progress"
    PROFILE = "profile"
    TRACE = "trace"


class TrustedBgenValidationMode(enum.StrEnum):
    """Trusted BGEN validation behavior."""

    CACHE_ON_MISS = "cache_on_miss"
    FORCE_VALIDATE = "force_validate"
    ASSUME_VALIDATED = "assume_validated"


class RegenieTraitType(enum.StrEnum):
    """REGENIE trait family."""

    QUANTITATIVE = "quantitative"
    BINARY = "binary"


class BinaryFallbackMethod(enum.StrEnum):
    """Internal binary fallback method."""

    SCORE_ONLY = "score_only"
    FIRTH = "firth"
    FIRTH_APPROXIMATE = "firth_approximate"
    SPA = "spa"


class BinaryExtraCode(enum.IntEnum):
    """Integer correction labels used by binary REGENIE step 2 output."""

    SCORE = 0
    FIRTH = 1
    SPA = 2
    TEST_FAIL = 3


class FirthFailureCode(enum.IntEnum):
    """Integer failure labels for binary Firth fallback rows."""

    NONE = 0
    NUMERICAL = 1
    MAX_ITERATIONS = 2
    INVALID_STATISTIC = 3
    STEP_HALVING = 4


class FirthCorrectionCode(enum.IntEnum):
    """Integer labels for the final binary approximate-Firth branch."""

    NONE = 0
    PSEUDO_FIRTH = 1
    NEWTON_RAPHSON_ZERO_START = 2
    NEWTON_RAPHSON_WARM_START = 3


@dataclass(frozen=True)
class BinaryCorrectionPlan:
    """Normalized binary fallback execution plan.

    Attributes:
        method: Binary fallback method to run.
        p_threshold: Score-test p-value threshold for fallback candidates.
        firth_se: Whether successful Firth rows use LRT-derived standard errors.

    """

    method: BinaryFallbackMethod = BinaryFallbackMethod.SCORE_ONLY
    p_threshold: float = 0.05
    firth_se: bool = False


class SampleIdentifierSource(enum.StrEnum):
    """Origin of BGEN sample identifiers."""

    EMBEDDED = "embedded"
    EXTERNAL = "external"
    GENERATED = "generated"


class SampleKeyMode(enum.StrEnum):
    """Sample key used for phenotype, covariate, and prediction alignment."""

    IID = "iid"
    FID_IID = "fid_iid"


class MultiPhenotypeSampleMode(enum.StrEnum):
    """Sample handling for requests containing multiple phenotypes."""

    PER_PHENOTYPE = "per-phenotype"
    COMPLETE_CASE = "complete-case"


class PhenotypeComputeGroupMode(enum.StrEnum):
    """Planning mode for one phenotype compute group."""

    SINGLE_PHENOTYPE = "single-phenotype"
    COMPLETE_CASE = "complete-case"
    PER_PHENOTYPE_COMPATIBLE = "per-phenotype-compatible"


class NullLogisticNonconvergencePolicy(enum.StrEnum):
    """Host policy for binary null-logistic non-convergence."""

    FAIL = "fail"
    WARN = "warn"


class ArrayMemoryOrder(enum.StrEnum):
    """NumPy array memory layout selector."""

    KEEP = "K"
    ANY = "A"
    C_CONTIGUOUS = "C"
    FORTRAN_CONTIGUOUS = "F"
