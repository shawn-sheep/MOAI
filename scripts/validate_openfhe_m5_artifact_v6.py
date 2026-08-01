#!/usr/bin/env python3
"""Validate the fail-closed schema-v6 M5 OpenFHE evidence bundle.

This validator is intentionally separate from the frozen schema-v2 M3/M4
validator.  It accepts exactly one server-only, twelve-layer ciphertext-chain
correctness run.  Timing fields are retained only as non-benchmark diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import shlex
import stat
import struct
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_ROOT = REPO_ROOT / "build-openfhe"
OPENFHE_PREFIX = Path("/home/shawnsheep/opt/openfhe_v1_5_1")
SYSTEM_CMAKE = "/usr/bin/cmake"
SYSTEM_CTEST = "/usr/bin/ctest"
SYSTEM_CXX = "/usr/bin/c++"
SYSTEM_GIT = "/usr/bin/git"
SYSTEM_LDD = "/usr/bin/ldd"
SYSTEM_MAKE = "/usr/bin/gmake"
SYSTEM_ENV = "/usr/bin/env"
CONFIGURE_ENV_UNSET = (
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "CMAKE_PREFIX_PATH",
    "CMAKE_TOOLCHAIN_FILE",
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "LIBRARY_PATH",
    "LD_AUDIT",
    "LD_DEBUG",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
    "MAKEFLAGS",
    "MFLAGS",
    "BASH_ENV",
    "ENV",
    "PYTHONHOME",
    "PYTHONPATH",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_GENERATOR_INSTANCE",
    "CMAKE_BUILD_PARALLEL_LEVEL",
    "CTEST_PARALLEL_LEVEL",
    "CTEST_TEST_LOAD",
    "CTEST_RESOURCE_SPEC_FILE",
)
FORCED_SUBPROCESS_ENVIRONMENT = {"LANG": "C", "LC_ALL": "C"}
OPENFHE_CMAKE_PACKAGE_FILES = (
    "OpenFHEConfig.cmake",
    "OpenFHEConfigVersion.cmake",
    "OpenFHETargets.cmake",
    "OpenFHETargets-release.cmake",
)
OPENFHE_LINKED_LIBRARY_SONAMES = (
    "libOPENFHEbinfhe.so.1",
    "libOPENFHEcore.so.1",
    "libOPENFHEpke.so.1",
)
OPENFHE_LINKED_LIBRARY_BASENAMES = {
    "libOPENFHEbinfhe.so.1": "libOPENFHEbinfhe.so.1.5.1",
    "libOPENFHEcore.so.1": "libOPENFHEcore.so.1.5.1",
    "libOPENFHEpke.so.1": "libOPENFHEpke.so.1.5.1",
}
OPENFHE_INCLUDE_TREE_CANONICALIZATION = (
    "relative-posix-path-nul-size-nul-sha256-newline-v1"
)
OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
DEFAULT_SCHEMA = REPO_ROOT / "docs" / "openfhe-m5-artifact-schema-v6.json"
SCHEMA_URI = "https://json-schema.org/draft/2020-12/schema"
SCHEMA_ID = "https://local.moai/openfhe-m5-artifact-schema-v6.json"
SCHEMA_TITLE = "MOAI OpenFHE M5 evidence manifest v6"
M5_SCHEMA_VERSION = 6
VALIDATOR_ID = "moai.openfhe.m5.artifact-validator.v6"
RUNNER_ID = "moai.openfhe.m5.artifact-runner.v6"
CANONICALIZATION = "MOAI-json-sort-keys-compact-utf8-v1"
LEGACY_R16_CALIBRATION_EVIDENCE = {
    "run_id": "20260731T090736+0900-m5-single-scale-mask-calibration-339daa5-r16",
    "manifest_sha256": (
        "008718bfd1c75dda4853c7a3a59214b68b44116cea626dc7ff3c4778e6e5aa87"
    ),
    "sha256sums_sha256": (
        "ffc1f181e5b81427f9232cf4257aa41c58e89147a41c92cca5573bb40d9b75e4"
    ),
}
EXACT3_EVIDENCE: dict[str, object] | None = {
    "run_id": "20260801T124248+0900-m5-runnable-prototype-exact3-339daa5-r23",
    "relative_path": (
        "results/openfhe/"
        "20260801T124248+0900-m5-runnable-prototype-exact3-339daa5-r23"
    ),
    "manifest_sha256": (
        "a19a578b3381343ad602713b528310dee52cbe09bccfd39c560ebc33159203e4"
    ),
    "sha256sums_sha256": (
        "aa9d0c37bb2bfbfa86b33b9d4c3bdeaefd7cffa17b17c26710b78524f60bad8e"
    ),
    "layer_count": 3,
    "artifact_eligible": False,
    "exact3_gate_passed": True,
    "schedule_evidence_eligible": True,
    "formal_schedule_sealed": True,
}
EXACT3_BUNDLE_PATHS = (
    "stdout.log",
    "stderr.log",
    "time.log",
    "metrics.csv",
    "manifest.json",
    "SHA256SUMS",
)
EXACT3_CHECKSUM_PATHS = EXACT3_BUNDLE_PATHS[:-1]
EXACT3_MANIFEST_SCHEMA_ID = "moai.openfhe.diagnostic.exact3.schedule-evidence.v2"
EXACT3_MANIFEST_SCHEMA_VERSION = 2
TRACE_SCALE_SOURCE_PATH = "config/openfhe_approximations.json"
TRACE_SCALE_LOCATOR = "operators.layernorm.feature_packed_trace_scale_contract"
TRACE_SCALE_CONTRACT_ID = "layernorm_layer_token_power2_scale_v1"
TRACE_SCALE_CONTRACT_SHA256 = (
    "b493b032e461e15d7436efe7fff5948436afa8e302646daa97db78fa1d59be3e"
)
TRACE_SCALE_VALUES_SHA256 = (
    "9436f05ce80b427de47700d924869b0dd6f13cc515e586446fcd2dc56faec28a"
)
TRACE_SCALE_RAW_VARIANCE_SHA256 = (
    "940f92de81915c2121b427b72827e4ed87d072fda0847266d0cb5521d60f93b6"
)
TRACE_SCALE_SHAPE = (2, 12, 5)
LAYERNORM_REGISTERED_INTERVAL = [0.5, 1536.0]
PROFILE_SCHEDULE_SEALED_STATUS = "sealed"
PROFILE_CANDIDATE_SEALED_STATUS = "sealed_exact3"
PROFILE_SCHEDULE_SEALED_SOURCE = (
    "approved exact-three-layer live OpenFHE schedule evidence"
)
PROFILE_VALIDATION_STATUS_SEALED = (
    "exact-three live M5 metadata schedule sealed; live M4 regression and "
    "full 12-layer MOAI-observability-compatible prototype gates remain required"
)
PROFILE_PRESEAL_TRANSITION = (
    "candidate profile captured by the exact3 sealer before the one-way "
    "schedule-seal transition"
)
PROFILE_TRANSITION_CANONICALIZATION = CANONICALIZATION
PROFILE_TRANSITION_ALLOWED_JSON_POINTERS = (
    "/validation_status",
    "/m5_schedule_candidate/status",
    "/m5_schedule_candidate/source",
    "/m5_schedule_candidate/formal_schedule_sealed",
    "/m5_schedule_candidate/calibrated_layers",
    "/m5_schedule_candidate/exact3_seal",
    "/feature_packed_layernorm_override/schedule_status",
)
PROFILE_TRANSITION_PRE_STATE_VALUES = {
    "/validation_status": (
        "two-layer live M5 schedule recorded as candidate; formal schedule "
        "unsealed until exact-three succeeds; live M4 regression and 12-layer "
        "MOAI-observability-compatible prototype gates remain required"
    ),
    "/m5_schedule_candidate/status": "candidate_unsealed",
    "/m5_schedule_candidate/source": (
        "reviewed live two-layer metadata calibration output; no sealed "
        "artifact identity claimed"
    ),
    "/m5_schedule_candidate/formal_schedule_sealed": False,
    "/m5_schedule_candidate/calibrated_layers": [0, 1],
    "/feature_packed_layernorm_override/schedule_status": (
        "two-layer live M5 candidate recorded but unsealed; exact-three, live "
        "M4 regression, and 12-layer MOAI-observability-compatible prototype "
        "gates remain required"
    ),
}
PROFILE_TRANSITION_PRE_STATE_ABSENT = (
    "/m5_schedule_candidate/exact3_seal",
)
PROFILE_TRANSITION_APPLICATION_ORDER = (
    "candidate pre-state and immutable bytes are verified before and after the "
    "sole exact3 child; the fsynced six-file bundle is published before the "
    "one-way profile transition is applied"
)
PROFILE_TRANSITION_POST_STATE_VALUES = {
    "/validation_status": PROFILE_VALIDATION_STATUS_SEALED,
    "/m5_schedule_candidate/status": PROFILE_CANDIDATE_SEALED_STATUS,
    "/m5_schedule_candidate/source": PROFILE_SCHEDULE_SEALED_SOURCE,
    "/m5_schedule_candidate/formal_schedule_sealed": True,
    "/m5_schedule_candidate/calibrated_layers": [0, 1, 2],
    "/feature_packed_layernorm_override/schedule_status": (
        PROFILE_SCHEDULE_SEALED_STATUS
    ),
}
PROFILE_PATH = "config/paper_compat_feature_packed.json"
PROFILE_LOCATOR = "/effective_profile"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
WARNING = "Research reproduction parameters only. Do not claim 128-bit security."
EXACT3_CONFIG_PATHS = (
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    "config/openfhe_approximations.json",
    PROFILE_PATH,
)
EXACT3_SOURCE_PATHS = (
    "CMakeLists.txt",
    "include/moai/openfhe/approximation_registry.hpp",
    "include/moai/openfhe/client_runtime.hpp",
    "include/moai/openfhe/context_factory.hpp",
    "include/moai/openfhe/encoder_layer.hpp",
    "include/moai/openfhe/evaluation_key_registry.hpp",
    "include/moai/openfhe/feature_packed_attention.hpp",
    "include/moai/openfhe/feature_packed_ops.hpp",
    "include/moai/openfhe/linear_ops.hpp",
    "include/moai/openfhe/nonlinear_ops.hpp",
    "include/moai/openfhe/packing.hpp",
    "include/moai/openfhe/server_runtime.hpp",
    "include/moai/openfhe/types.hpp",
    "scripts/seal_openfhe_exact3_evidence.py",
    "src/openfhe/approximation_registry.cpp",
    "src/openfhe/client_runtime.cpp",
    "src/openfhe/context_factory.cpp",
    "src/openfhe/encoder_layer.cpp",
    "src/openfhe/evaluation_key_registry.cpp",
    "src/openfhe/feature_packed_attention.cpp",
    "src/openfhe/feature_packed_ops.cpp",
    "src/openfhe/linear_ops.cpp",
    "src/openfhe/nonlinear_ops.cpp",
    "src/openfhe/packing.cpp",
    "src/openfhe/server_runtime.cpp",
    "tests/openfhe_encoder_12_layer_smoke.cpp",
    "tests/support/moai_encoder_fixture.cpp",
    "tests/support/moai_encoder_fixture.hpp",
    "tests/support/moai_encoder_plaintext_oracle.cpp",
    "tests/support/moai_encoder_plaintext_oracle.hpp",
)
EXACT3_EXECUTABLE_PATH = "build-openfhe/openfhe_encoder_12_layer_smoke"
EXACT3_PROVENANCE_SNAPSHOT_SEMANTICS = (
    "source, config, executable, trace-scale, and git/worktree provenance "
    "captured immediately before the sole exact3 child launch and rechecked "
    "byte-for-byte and identity-for-identity after child exit"
)
EXACT3_PROFILE_WARNING = (
    f'profile=paper_compat security_claim=none warning="{WARNING}"'
)
EXACT3_RUN_ID_LABEL_SEMANTICS = (
    "operator-assigned directory label; not treated as a measured process start "
    "timestamp"
)
EXACT3_EXECUTION_TIMESTAMP_SEMANTICS = (
    "launcher wall clocks sampled immediately before the sole child invocation "
    "and immediately after it returned; GNU time supplies the independently "
    "parsed elapsed duration"
)
EXACT3_RAW_TIMESTAMP_SEMANTICS = (
    "mtime_ns and ctime_ns are filesystem metadata, not process start or finish "
    "timestamps"
)
EXACT3_SEALED_AT_SEMANTICS = (
    "sealer wall clock sampled while preparing derived outputs after initial "
    "validation, rounded to whole seconds; not an execution finish timestamp"
)
EXACT3_GIT_SNAPSHOT_SEMANTICS = (
    "Git HEAD, branch, porcelain status, tracked diff, and untracked source "
    "manifest captured immediately before the sole exact3 child launch and "
    "rechecked byte-for-byte after child exit"
)
EXACT3_GIT_UNTRACKED_SELECTION = (
    "git ls-files --others --exclude-standard under CMakeLists.txt, config, docs, "
    "include, scripts, src, tests"
)
EXACT3_GIT_CANONICALIZATION = "sorted-key compact UTF-8 JSON"
M5_PROTOTYPE_INACTIVE_MAX_ABS = 1e-3
M5_PROTOTYPE_CONTRACT_ID = "moai_observability_compatible_prototype_v1"
M5_PROTOTYPE_CONTRACT_STATUS = "user_authorized_2026-08-01"
M5_PROTOTYPE_PROFILE_THRESHOLDS = {
    "per_layer_relative_l2_max": 5e-2,
    "per_layer_cosine_min": 0.99,
    "final_relative_l2_max": 5e-2,
    "final_cosine_min": 0.99,
    "inactive_or_cross_lane_max_absolute": M5_PROTOTYPE_INACTIVE_MAX_ABS,
    "finite_required": True,
}
M5_PROTOTYPE_ACCEPTANCE = {
    "contract_id": M5_PROTOTYPE_CONTRACT_ID,
    "status": M5_PROTOTYPE_CONTRACT_STATUS,
    "scope": (
        "M5 exact-three and full-12 encrypted encoder runtime only; M2 packing, "
        "M3 nonlinear, and M4 single-layer gates remain unchanged"
    ),
    "legacy_basis": (
        "legacy MOAI has no inactive/cross-lane numerical assertion and observes "
        "only active slots; 1e-3 is a new prototype engineering bound, not a legacy "
        "MOAI threshold"
    ),
    "thresholds": M5_PROTOTYPE_PROFILE_THRESHOLDS,
    "active_quality_policy": "unchanged from the original M5 contract",
    "sentinel_range_policy": (
        "registered polynomial intervals and inactive guard range checks remain "
        "unchanged"
    ),
    "claim_boundary": (
        "prototype_only; MOAI-observability-compatible; not strict numerical parity "
        "and not evidence that 1e-3 was used by legacy MOAI"
    ),
}
EXACT3_GATE_CONTRACT = {
    "prototype_acceptance": M5_PROTOTYPE_ACCEPTANCE,
    "raw_input_bounds": {
        "maximum_bytes": {
            "stdout.log": 2 * 1024 * 1024,
            "stderr.log": 64 * 1024,
            "time.log": 64 * 1024,
        },
        "stdout_nonempty_line_count": 5,
        "stdout_maximum_line_bytes": 512 * 1024,
    },
    "quality": {
        "relative_l2_max": 5e-2,
        "cosine_min": 0.99,
        "finite_required": True,
    },
    "inactive": {
        "inactive_max_abs": M5_PROTOTYPE_INACTIVE_MAX_ABS,
        "zero_checkpoint_count": 28,
        "sentinel_deviation_from_one": "diagnostic_only",
        "sentinel_ranges_must_stay_in_registered_intervals": True,
    },
    "polynomial_intervals": {
        "softmax_shifted_logits": [-16.0, 5.0],
        "softmax_denominator": [0.01, 80.0],
        "ln1_normalized_variance": [0.5, 1536.0],
        "gelu_input": [-80.0, 128.0],
        "ln2_normalized_variance": [0.5, 1536.0],
    },
}
BRANCH = "refactor/openfhe-cpu"
TRACKING_REF = "refs/remotes/origin/refactor/openfhe-cpu"
REMOTE_NAME = "origin"
REMOTE_URL = "https://github.com/shawn-sheep/MOAI.git"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
EXECUTABLE_PATH = "build-openfhe/openfhe_encoder_12_layer_smoke"
SCHEMA_PATH = DEFAULT_SCHEMA
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m5_artifact_v6.py"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_encoder12_artifact_v6.py"
SCHEMA_BINDING_SPECS = {
    "schema": (
        SCHEMA_ID,
        "docs/openfhe-m5-artifact-schema-v6.json",
    ),
    "validator": (
        VALIDATOR_ID,
        "scripts/validate_openfhe_m5_artifact_v6.py",
    ),
    "runner": (
        RUNNER_ID,
        "scripts/run_openfhe_encoder12_artifact_v6.py",
    ),
}
GIT_OBJECT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
CONFIG_INPUTS = (
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    "config/openfhe_approximations.json",
    PROFILE_PATH,
)
LAYER_IDS = tuple(range(12))
TRACE_SHAPE = [5, 768]
TRACE_VALUE_COUNT = 5 * 768
FEATURE_BLOCK_SIZE = 1024
CSV_FILE_COUNT = 444
INPUT_FILE_COUNT = CSV_FILE_COUNT + len(CONFIG_INPUTS)
WEIGHT_FILES_PER_LAYER = 16
TRACE_FILES_PER_LAYER = 21

# These operation-count and metadata contracts are intentionally centralized.
# The metadata tuples remain static candidates until a sealed profile schedule
# and approved exact-three-layer live evidence are both available.
OPERATION_COUNT_KEYS = (
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
)
ZERO_COUNTS = dict.fromkeys(OPERATION_COUNT_KEYS, 0)
LAYER_COUNTS = {
    "rotations": 6300,
    "ct_pt_multiplications": 51885,
    "ct_ct_multiplications": 95,
    "explicit_rescale_requests": 810,
    "chebyshev_evaluations": 55,
    "estimated_polynomial_multiplications": 1150,
    "bootstraps": 25,
    "bootstrap_iterations": 50,
}
REFRESH_COUNTS = {
    "rotations": 0,
    "ct_pt_multiplications": 5,
    "ct_ct_multiplications": 0,
    "explicit_rescale_requests": 5,
    "chebyshev_evaluations": 0,
    "estimated_polynomial_multiplications": 0,
    "bootstraps": 5,
    "bootstrap_iterations": 10,
}
TOTAL_COUNTS = {
    "rotations": 75600,
    "ct_pt_multiplications": 622675,
    "ct_ct_multiplications": 1140,
    "explicit_rescale_requests": 9775,
    "chebyshev_evaluations": 660,
    "estimated_polynomial_multiplications": 13800,
    "bootstraps": 355,
    "bootstrap_iterations": 710,
}
EXACT3_FINAL_COUNTS = {
    key: LAYER_COUNTS[key] * 3 + REFRESH_COUNTS[key] * 2 for key in OPERATION_COUNT_KEYS
}
# Live crypto preflight: actual scale_bits=50.000000079945785.  Freeze the
# schedule at 2^50 while retaining the 1e-3 tolerance for future live values.
LAYER0_INPUT_METADATA: dict[str, Any] | None = {
    "level": 29,
    "noise_scale_degree": 1,
    "remaining_levels": 18,
    "scale_bits": 50,
    "expected_scale_bits": 50,
    "ciphertext_count": 5,
}
LAYER_HANDOFF_INPUT_METADATA: dict[str, Any] | None = {
    "level": 19,
    "noise_scale_degree": 2,
    "remaining_levels": 27,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
RAW_OUTPUT_METADATA = {
    "level": 30,
    "noise_scale_degree": 2,
    "remaining_levels": 16,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
SOFTMAX_CHECKPOINT_METADATA: dict[str, Any] | None = {
    "level": 18,
    "noise_scale_degree": 2,
    "remaining_levels": 28,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYERNORM_CHECKPOINT_METADATA: dict[str, Any] | None = {
    "level": 19,
    "noise_scale_degree": 2,
    "remaining_levels": 27,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYER0_ATTENTION_OUTPUT_METADATA = {
    "level": 40,
    "noise_scale_degree": 2,
    "remaining_levels": 6,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
POST_REFRESH_ATTENTION_OUTPUT_METADATA = {
    "level": 31,
    "noise_scale_degree": 2,
    "remaining_levels": 15,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYER0_LN1_OUTPUT_METADATA = {
    "level": 32,
    "noise_scale_degree": 2,
    "remaining_levels": 14,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
POST_REFRESH_LN1_OUTPUT_METADATA = {
    "level": 30,
    "noise_scale_degree": 2,
    "remaining_levels": 16,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYER0_FFN_OUTPUT_METADATA = {
    "level": 45,
    "noise_scale_degree": 2,
    "remaining_levels": 1,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
POST_REFRESH_FFN_OUTPUT_METADATA = {
    "level": 43,
    "noise_scale_degree": 2,
    "remaining_levels": 3,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
RELATIVE_USED_LEVEL_DELTAS = {
    "input_to_softmax_checkpoint_net_recovered": {
        "layer_0": 10,
        "layers_1_to_11": 1,
    },
    "softmax_checkpoint_to_attention_output_consumed": {
        "layer_0": 22,
        "layers_1_to_11": 13,
    },
    "attention_output_to_ln1_checkpoint_net_recovered": {
        "layer_0": 21,
        "layers_1_to_11": 12,
    },
    "ln1_checkpoint_to_ln1_output_consumed": {
        "layer_0": 13,
        "layers_1_to_11": 11,
    },
    "ln1_output_to_ffn_output_consumed": {
        "layer_0": 13,
        "layers_1_to_11": 13,
    },
    "ffn_output_to_ln2_checkpoint_net_recovered": {
        "layer_0": 26,
        "layers_1_to_11": 24,
    },
    "ln2_checkpoint_to_raw_output_consumed": {
        "layer_0": 11,
        "layers_1_to_11": 11,
    },
    "previous_raw_output_to_input_recovered": {
        "layer_0": None,
        "layers_1_to_11": 11,
    },
}
CHECKPOINT_METADATA_FIELDS = (
    "input_metadata",
    "softmax_denominator_metadata",
    "attention_output_metadata",
    "ln1_variance_metadata",
    "ln1_output_metadata",
    "ffn_output_metadata",
    "ln2_variance_metadata",
    "raw_output_metadata",
)
CHECKPOINT_METADATA_HASH_FIELDS = ("layer_id", *CHECKPOINT_METADATA_FIELDS)
SCHEDULE_PREFLIGHT_KEYS = {
    "test",
    "profile",
    "security_claim",
    "encoder_layers",
    "layer_ids",
    "chain_mode",
    "expected_fixture_files",
    "metadata_schedule_preflight",
    "formal_metadata_schedule_source",
    "formal_schedule_sealed",
    "he_metadata_verified_by_preflight",
    "trace_hash_gate",
    "finite_and_in_range",
    "final_relative_l2",
    "final_cosine",
    "final_max_absolute",
    "load_and_oracle_ms",
}
CRYPTO_PREFLIGHT_KEYS = {
    "test",
    "profile",
    "security_claim",
    "input_metadata",
    "server_private_key_present",
    "server_decryptions",
    "server_plaintext_activations",
    "additive_he_operations",
    "passed",
}
POLYNOMIAL_INTERVALS = {
    "softmax_shifted_logits": [-16.0, 5.0],
    "softmax_denominator": [0.01, 80.0],
    "ln1_normalized_variance": [0.5, 1536.0],
    "gelu_input": [-80.0, 128.0],
    "ln2_normalized_variance": [0.5, 1536.0],
}
THRESHOLDS = {
    "per_layer_relative_l2_max": 5e-2,
    "per_layer_cosine_min": 0.99,
    "final_relative_l2_max": 5e-2,
    "final_cosine_min": 0.99,
    "inactive_max_abs": M5_PROTOTYPE_INACTIVE_MAX_ABS,
    "inactive_sentinel_in_interval_required": True,
    "finite_required": True,
}
INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS = {
    "softmax_denominator": POLYNOMIAL_INTERVALS["softmax_denominator"],
    "ln1_normalized_variance": LAYERNORM_REGISTERED_INTERVAL,
    "ln2_normalized_variance": LAYERNORM_REGISTERED_INTERVAL,
}
EXPECTED_PROFILE_FIELDS = {
    "effective_profile_schema_version": 1,
    "ring_dimension": 65536,
    "slot_count": 32768,
    "scaling_modulus_bits": 50,
    "first_modulus_bits": 55,
    "multiplicative_depth": 47,
    "levels_available_after_bootstrap": 28,
    "bootstrap_slots": 1024,
    "bootstrap_iterations": 2,
    "bootstrap_precision": 14,
    "scaling_technique": "FLEXIBLEAUTO",
    "security_claim": "none",
    "security_level": "HEStd_NotSet",
}
EXPECTED_EXECUTION = {
    "mode": "server-only",
    "encoder_layers": 12,
    "layer_ids": list(LAYER_IDS),
    "trace_shape": TRACE_SHAPE,
    "feature_block_size": FEATURE_BLOCK_SIZE,
    "chain_mode": "ciphertext_output_to_next_input",
    "client_encrypt_calls": 1,
    "encrypted_input_ciphertexts": 5,
    "plaintext_activation_resets": 0,
    "server_layer_evaluations": 12,
    "inter_layer_refreshes": 11,
    "checkpoint_decryption_owner": "client",
    "final_decryption_owner": "client",
    "server_private_key_present": False,
    "server_decryptions": 0,
    "server_plaintext_activations": False,
    "approximation_range_status": "all_client_validated",
    "quality_reference": (
        "12-layer chained frozen polynomial oracle from config/moai_encoder_trace.json"
    ),
}
M5_CTEST_PATTERN = (
    "^(moai_trace_contract|openfhe_(server_trust_boundary|"
    "server_trust_boundary_negative|profile_contract|"
    "profile_validator_contract|artifact_schema_contract|"
    "artifact_validator_contract|encoder_artifact_runner_contract|"
    "m4_v5_artifact_schema_contract|m4_v5_artifact_validator_contract|"
    "m4_v5_encoder_artifact_runner_contract|m5_artifact_schema_contract|"
    "m5_artifact_validator_contract|encoder12_artifact_runner_contract|"
    "m5_v6_artifact_schema_contract|m5_v6_artifact_validator_contract|"
    "m5_v6_encoder12_artifact_runner_contract|evaluation_key_bundle_smoke|"
    "feature_packed_smoke|feature_packed_attention_smoke|"
    "feature_bootstrap_smoke|feature_layernorm_smoke|encoder_fixture_contract|"
    "encoder_plaintext_oracle_smoke|encoder_layer_smoke|encoder_12_layer_preflight|"
    "encoder_12_layer_crypto_preflight|"
    "encoder_trace_contract|encoder_trace_validator_contract))$"
)
CSV_FIELDS = (
    "run",
    "exit_code",
    "elapsed_seconds",
    "peak_rss_kib",
    "fixture_load_oracle_ms",
    "setup_keygen_ms",
    "client_encrypt_ms",
    "server_online_diagnostic_ms",
    "client_checkpoint_validate_ms",
    "final_rel_l2",
    "final_cosine",
    "final_max_absolute",
    "inactive_max_abs",
    "inactive_sentinel_max_error",
    "checkpoint_metadata_sha256",
    "final_level",
    "final_noise_scale_degree",
    "final_remaining_levels",
    "final_scale_bits",
    "final_ciphertext_count",
    *OPERATION_COUNT_KEYS,
    "timing_claim",
    "latency_kind",
    "finite",
    "passed",
)
CLAIM_BOUNDARY = [
    "Five-token M5 12-layer encoder trace replay only; not task-level inference.",
    "server-only is an API/target trust boundary, not operating-system process "
    "isolation; correctness executable links client observer for checkpoints",
    "paper_compat OpenFHE CKKS CPU parameters with security_claim=none.",
    "M5 correctness is prototype-only with inactive/cross-lane <=1e-3; legacy "
    "MOAI defined no such threshold and strict numerical parity is not claimed.",
    "Timing is a non-benchmark diagnostic; no speedup claim.",
    "GPU, Discrete CKKS/FBT, QDQ, tokenizer, and classifier are excluded.",
]
REQUIRED_TOP_LEVEL = {
    "schema_version",
    "schema_binding",
    "run_id",
    "milestone",
    "started_at",
    "finished_at",
    "git",
    "commands",
    "build_configuration",
    "environment",
    "profile",
    "inputs",
    "workload",
    "contracts",
    "metrics",
    "gate",
    "artifacts",
    "claim_boundary",
    "verdict",
}
SUPPORTED_SCHEMA_KEYWORDS = {
    "$schema",
    "$id",
    "$defs",
    "$ref",
    "title",
    "description",
    "type",
    "additionalProperties",
    "required",
    "properties",
    "const",
    "enum",
    "pattern",
    "format",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "uniqueItems",
    "items",
    "prefixItems",
    "allOf",
    "minProperties",
}


class ValidationError(RuntimeError):
    """Raised when an M5 schema, manifest, or evidence file drifts."""


def _artifact_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for name in CONFIGURE_ENV_UNSET:
        environment.pop(name, None)
    environment.update(FORCED_SUBPROCESS_ENVIRONMENT)
    return environment


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("GIT_")
    }
    environment.update(
        {
            "PATH": "/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "HOME": "/nonexistent/moai-m5-git-home",
            "XDG_CONFIG_HOME": "/nonexistent/moai-m5-git-config",
        }
    )
    return environment


def _expected_metadata_schedule_contract() -> dict[str, Any]:
    if LAYER0_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-0 input metadata is not sealed by the crypto preflight"
        )
    if LAYER_HANDOFF_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-handoff input metadata is not sealed by the two-layer seam run"
        )
    if SOFTMAX_CHECKPOINT_METADATA is None:
        raise ValidationError(
            "M5 Softmax checkpoint metadata is not sealed by the live seam run"
        )
    if LAYERNORM_CHECKPOINT_METADATA is None:
        raise ValidationError(
            "M5 LayerNorm checkpoint metadata is not sealed by the live seam run"
        )
    schedule = {
        "multiplicative_depth": EXPECTED_PROFILE_FIELDS["multiplicative_depth"],
        "used_level_formula": "level + noise_scale_degree - 1",
        "remaining_levels_formula": "multiplicative_depth - used_level",
        "initial_layer_input": dict(LAYER0_INPUT_METADATA),
        "post_refresh_layer_input": dict(LAYER_HANDOFF_INPUT_METADATA),
        "post_bootstrap_softmax_denominator_checkpoint": dict(
            SOFTMAX_CHECKPOINT_METADATA
        ),
        "post_bootstrap_canonicalized_layernorm_variance_checkpoint": dict(
            LAYERNORM_CHECKPOINT_METADATA
        ),
        "raw_layer_output": dict(RAW_OUTPUT_METADATA),
        "attention_output_by_layer_regime": {
            "layer_0": dict(LAYER0_ATTENTION_OUTPUT_METADATA),
            "layers_1_to_11": dict(POST_REFRESH_ATTENTION_OUTPUT_METADATA),
        },
        "ln1_output_by_layer_regime": {
            "layer_0": dict(LAYER0_LN1_OUTPUT_METADATA),
            "layers_1_to_11": dict(POST_REFRESH_LN1_OUTPUT_METADATA),
        },
        "ffn_output_by_layer_regime": {
            "layer_0": dict(LAYER0_FFN_OUTPUT_METADATA),
            "layers_1_to_11": dict(POST_REFRESH_FFN_OUTPUT_METADATA),
        },
        "relative_used_level_deltas": dict(RELATIVE_USED_LEVEL_DELTAS),
        "checkpoint_metadata_digest": {
            "algorithm": "SHA-256",
            "canonicalization": CANONICALIZATION,
            "layer_order": "0 through 11",
            "fields": list(CHECKPOINT_METADATA_HASH_FIELDS),
        },
    }
    for name in (
        "initial_layer_input",
        "post_refresh_layer_input",
        "post_bootstrap_softmax_denominator_checkpoint",
        "post_bootstrap_canonicalized_layernorm_variance_checkpoint",
        "raw_layer_output",
    ):
        _require_metadata(schedule[name], schedule[name], f"metadata_schedule.{name}")
    for field in (
        "attention_output_by_layer_regime",
        "ln1_output_by_layer_regime",
        "ffn_output_by_layer_regime",
    ):
        for regime, metadata in schedule[field].items():
            _require_metadata(
                metadata,
                metadata,
                f"metadata_schedule.{field}.{regime}",
            )
    for regime, input_field in (
        ("layer_0", "initial_layer_input"),
        ("layers_1_to_11", "post_refresh_layer_input"),
    ):
        _require_used_level_delta(
            schedule[input_field],
            schedule["post_bootstrap_softmax_denominator_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["input_to_softmax_checkpoint_net_recovered"][
                regime
            ],
            f"metadata_schedule.{regime}_input_to_softmax_checkpoint_recovery",
        )
        _require_used_level_delta(
            schedule["attention_output_by_layer_regime"][regime],
            schedule["post_bootstrap_softmax_denominator_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS[
                "softmax_checkpoint_to_attention_output_consumed"
            ][regime],
            f"metadata_schedule.{regime}_softmax_to_attention_consumption",
        )
        _require_used_level_delta(
            schedule["attention_output_by_layer_regime"][regime],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS[
                "attention_output_to_ln1_checkpoint_net_recovered"
            ][regime],
            f"metadata_schedule.{regime}_attention_to_ln1_checkpoint_recovery",
        )
        _require_used_level_delta(
            schedule["ln1_output_by_layer_regime"][regime],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["ln1_checkpoint_to_ln1_output_consumed"][regime],
            f"metadata_schedule.{regime}_ln1",
        )
        _require_used_level_delta(
            schedule["ffn_output_by_layer_regime"][regime],
            schedule["ln1_output_by_layer_regime"][regime],
            RELATIVE_USED_LEVEL_DELTAS["ln1_output_to_ffn_output_consumed"][regime],
            f"metadata_schedule.{regime}_ffn",
        )
        _require_used_level_delta(
            schedule["ffn_output_by_layer_regime"][regime],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["ffn_output_to_ln2_checkpoint_net_recovered"][
                regime
            ],
            f"metadata_schedule.{regime}_ffn_to_ln2_checkpoint_recovery",
        )
        _require_used_level_delta(
            schedule["raw_layer_output"],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["ln2_checkpoint_to_raw_output_consumed"][regime],
            f"metadata_schedule.{regime}_ln2_to_raw_output_consumption",
        )
    _require_used_level_delta(
        schedule["raw_layer_output"],
        schedule["post_refresh_layer_input"],
        RELATIVE_USED_LEVEL_DELTAS["previous_raw_output_to_input_recovered"][
            "layers_1_to_11"
        ],
        "metadata_schedule.previous_raw_output_to_input_recovered",
    )
    return schedule


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValidationError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise ValidationError(f"non-finite JSON number is forbidden: {token}")


def _parse_finite_float(token: str) -> float:
    value = float(token)
    if not math.isfinite(value):
        raise ValidationError(f"non-finite JSON number is forbidden: {token}")
    return value


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(
                handle,
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_non_finite,
                parse_float=_parse_finite_float,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValidationError(f"cannot read JSON {path}: {error}") from error


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValidationError(f"value cannot be canonicalized: {error}") from error


def _binary64_tensor_sha256(value: Any, shape: tuple[int, ...], label: str) -> str:
    digest = hashlib.sha256()

    def visit(
        node: Any, dimensions: tuple[int, ...], coordinate: tuple[int, ...]
    ) -> None:
        if not dimensions:
            if (
                not isinstance(node, (int, float))
                or isinstance(node, bool)
                or not math.isfinite(node)
            ):
                raise ValidationError(
                    f"{label}{coordinate} must be a finite binary64 value"
                )
            digest.update(struct.pack("<d", float(node)))
            return
        if not isinstance(node, list) or len(node) != dimensions[0]:
            raise ValidationError(f"{label}{coordinate} shape differs from {shape}")
        for index, child in enumerate(node):
            visit(child, dimensions[1:], (*coordinate, index))

    visit(value, shape, ())
    return digest.hexdigest()


def _trace_scale_contract_provenance(
    repository_root: Path = REPO_ROOT,
) -> dict[str, str]:
    approximation = load_json(repository_root / TRACE_SCALE_SOURCE_PATH)
    try:
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"trace-scale contract locator is missing: {TRACE_SCALE_LOCATOR}"
        ) from error
    if not isinstance(contract, dict):
        raise ValidationError("trace-scale contract must be an object")
    payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    computed_contract_hash = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    computed_values_hash = _binary64_tensor_sha256(
        contract.get("values"),
        TRACE_SCALE_SHAPE,
        "trace-scale values",
    )
    hashes = {
        "contract_sha256": contract.get("contract_sha256"),
        "values_sha256": contract.get("values_sha256"),
        "raw_variance_sha256": contract.get("raw_variance_sha256"),
    }
    if any(
        not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
        for value in hashes.values()
    ):
        raise ValidationError("trace-scale contract hashes must be 64 lowercase hex")
    if (
        contract.get("contract_id") != TRACE_SCALE_CONTRACT_ID
        or tuple(contract.get("shape", ())) != TRACE_SCALE_SHAPE
        or computed_contract_hash != TRACE_SCALE_CONTRACT_SHA256
        or hashes["contract_sha256"] != computed_contract_hash
        or computed_values_hash != TRACE_SCALE_VALUES_SHA256
        or hashes["values_sha256"] != computed_values_hash
        or hashes["raw_variance_sha256"] != TRACE_SCALE_RAW_VARIANCE_SHA256
    ):
        raise ValidationError("trace-scale contract identity or hash drifted")
    try:
        hard_interval = contract["normalized_variance_range"]["hard_interval"]
        runtime_dependency = contract["scope"]["runtime_activation_dependency"]
        exact_identity_gate = contract["inactive_guard"]["exact_identity_gate"]
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"trace-scale semantic gate is incomplete: {error}"
        ) from error
    if (
        hard_interval != LAYERNORM_REGISTERED_INTERVAL
        or runtime_dependency != "none"
        or exact_identity_gate is not False
    ):
        raise ValidationError("trace-scale semantic gate drifted")

    profile = load_json(repository_root / PROFILE_PATH)
    try:
        profile_binding = profile["feature_packed_layernorm_override"][
            "trace_scale_contract"
        ]
    except (KeyError, TypeError) as error:
        raise ValidationError(
            "feature profile lacks the trace-scale cross-binding"
        ) from error
    provenance = {
        "source_path": TRACE_SCALE_SOURCE_PATH,
        "json_locator": TRACE_SCALE_LOCATOR,
        "contract_id": TRACE_SCALE_CONTRACT_ID,
        "contract_sha256": computed_contract_hash,
        "values_sha256": computed_values_hash,
        "raw_variance_sha256": TRACE_SCALE_RAW_VARIANCE_SHA256,
    }
    if profile_binding != provenance:
        raise ValidationError(
            "feature profile trace-scale cross-binding differs from the source contract"
        )
    return provenance


def _profile_pointer_parent(
    document: dict[str, Any],
    pointer: str,
) -> tuple[dict[str, Any], str]:
    if not pointer.startswith("/") or pointer == "/":
        raise ValidationError(
            f"profile transition JSON pointer is malformed: {pointer!r}"
        )
    tokens = [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    ]
    if any(not token for token in tokens):
        raise ValidationError(
            f"profile transition JSON pointer is malformed: {pointer!r}"
        )
    current: Any = document
    for token in tokens[:-1]:
        if not isinstance(current, dict) or token not in current:
            raise ValidationError(
                f"profile transition JSON pointer is missing: {pointer}"
            )
        current = current[token]
    if not isinstance(current, dict):
        raise ValidationError(
            f"profile transition JSON pointer parent is not an object: {pointer}"
        )
    return current, tokens[-1]


def _profile_immutable_projection_sha256(profile: Any) -> str:
    if not isinstance(profile, dict):
        raise ValidationError("feature profile must be a JSON object")
    projection = json.loads(canonical_json_bytes(profile).decode("utf-8"))
    for pointer in PROFILE_TRANSITION_ALLOWED_JSON_POINTERS:
        parent, key = _profile_pointer_parent(projection, pointer)
        parent.pop(key, None)
    return hashlib.sha256(canonical_json_bytes(projection)).hexdigest()


def _validate_profile_transition(
    value: Any,
    preseal_profile_record: dict[str, Any],
) -> dict[str, Any]:
    transition = _require_exact_keys(
        value,
        {
            "path",
            "canonicalization",
            "allowed_json_pointers",
            "pre_state",
            "pre_seal_file_sha256",
            "pre_seal_size_bytes",
            "immutable_projection_sha256",
            "application_order",
            "post_state_contract",
        },
        "exact3 manifest.provenance.profile_transition",
    )
    _exact3_require_value(
        transition["path"],
        PROFILE_PATH,
        "exact3 profile transition path",
    )
    _exact3_require_value(
        transition["canonicalization"],
        PROFILE_TRANSITION_CANONICALIZATION,
        "exact3 profile transition canonicalization",
    )
    _exact3_require_value(
        transition["allowed_json_pointers"],
        list(PROFILE_TRANSITION_ALLOWED_JSON_POINTERS),
        "exact3 profile transition allowed_json_pointers",
    )
    _exact3_require_value(
        transition["pre_state"],
        {
            "values": PROFILE_TRANSITION_PRE_STATE_VALUES,
            "absent": list(PROFILE_TRANSITION_PRE_STATE_ABSENT),
        },
        "exact3 profile transition pre_state",
    )
    _exact3_require_value(
        transition["application_order"],
        PROFILE_TRANSITION_APPLICATION_ORDER,
        "exact3 profile transition application_order",
    )
    _exact3_require_value(
        transition["post_state_contract"],
        {
            "values": PROFILE_TRANSITION_POST_STATE_VALUES,
            "exact3_seal_pointer": "/m5_schedule_candidate/exact3_seal",
            "exact3_seal_keys": ["source", "evidence", "pre_seal_profile"],
        },
        "exact3 profile transition post_state_contract",
    )
    for name in ("pre_seal_file_sha256", "immutable_projection_sha256"):
        _exact3_require_sha256(
            transition[name],
            f"exact3 profile transition {name}",
        )
    size = transition["pre_seal_size_bytes"]
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ValidationError(
            "exact3 profile transition pre_seal_size_bytes must be positive"
        )
    _exact3_require_value(
        transition["pre_seal_file_sha256"],
        preseal_profile_record["sha256"],
        "exact3 profile transition pre-seal SHA versus config provenance",
    )
    _exact3_require_value(
        transition["pre_seal_size_bytes"],
        preseal_profile_record["size_bytes"],
        "exact3 profile transition pre-seal size versus config provenance",
    )
    return transition


def _require_profile_schedule_sealed(
    repository_root: Path = REPO_ROOT,
    evidence: dict[str, object] | None = None,
    profile_transition: dict[str, Any] | None = None,
) -> dict[str, object]:
    profile = load_json(repository_root / PROFILE_PATH)
    try:
        status = profile["feature_packed_layernorm_override"]["schedule_status"]
        validation_status = profile["validation_status"]
        candidate = profile["m5_schedule_candidate"]
        candidate_status = candidate["status"]
        candidate_source = candidate["source"]
        formal_schedule_sealed = candidate["formal_schedule_sealed"]
        calibrated_layers = candidate["calibrated_layers"]
        exact3_seal = candidate["exact3_seal"]
    except (KeyError, TypeError) as error:
        raise ValidationError(
            "feature profile metadata schedule is not sealed or is incomplete"
        ) from error
    expected_scalars = {
        "feature_packed_layernorm_override.schedule_status": (
            status,
            PROFILE_SCHEDULE_SEALED_STATUS,
        ),
        "validation_status": (validation_status, PROFILE_VALIDATION_STATUS_SEALED),
        "m5_schedule_candidate.status": (
            candidate_status,
            PROFILE_CANDIDATE_SEALED_STATUS,
        ),
        "m5_schedule_candidate.source": (
            candidate_source,
            PROFILE_SCHEDULE_SEALED_SOURCE,
        ),
        "m5_schedule_candidate.formal_schedule_sealed": (
            formal_schedule_sealed,
            True,
        ),
        "m5_schedule_candidate.calibrated_layers": (
            calibrated_layers,
            [0, 1, 2],
        ),
    }
    for label, (actual, expected) in expected_scalars.items():
        if not _exact3_typed_equal(actual, expected):
            raise ValidationError(
                "feature profile metadata schedule is not sealed: "
                f"{label} expected={expected!r} actual={actual!r}"
            )
    seal = _require_exact_keys(
        exact3_seal,
        {"source", "evidence", "pre_seal_profile"},
        "feature profile m5_schedule_candidate.exact3_seal",
    )
    _exact3_require_value(
        seal["source"],
        PROFILE_SCHEDULE_SEALED_SOURCE,
        "feature profile exact3 seal source",
    )
    approved = _validate_exact3_evidence(
        seal["evidence"],
        "feature profile exact3 seal evidence",
    )
    if evidence is not None and not _exact3_typed_equal(approved, evidence):
        raise ValidationError(
            "feature profile exact3 evidence differs from the approved evidence"
        )
    preseal = _require_exact_keys(
        seal["pre_seal_profile"],
        {"path", "sha256", "size_bytes", "transition"},
        "feature profile exact3 pre-seal binding",
    )
    _exact3_require_value(
        preseal["path"],
        PROFILE_PATH,
        "feature profile exact3 pre-seal path",
    )
    _exact3_require_sha256(
        preseal["sha256"],
        "feature profile exact3 pre-seal sha256",
    )
    if (
        not isinstance(preseal["size_bytes"], int)
        or isinstance(preseal["size_bytes"], bool)
        or preseal["size_bytes"] <= 0
    ):
        raise ValidationError(
            "feature profile exact3 pre-seal size_bytes must be a positive integer"
        )
    _exact3_require_value(
        preseal["transition"],
        PROFILE_PRESEAL_TRANSITION,
        "feature profile exact3 pre-seal transition",
    )
    if profile_transition is not None:
        _exact3_require_value(
            profile_transition["path"],
            preseal["path"],
            "exact3 provenance pre-seal profile path",
        )
        _exact3_require_value(
            profile_transition["pre_seal_file_sha256"],
            preseal["sha256"],
            "exact3 provenance pre-seal profile sha256",
        )
        _exact3_require_value(
            profile_transition["pre_seal_size_bytes"],
            preseal["size_bytes"],
            "exact3 provenance pre-seal profile size",
        )
        _exact3_require_value(
            _profile_immutable_projection_sha256(profile),
            profile_transition["immutable_projection_sha256"],
            "sealed profile immutable projection",
        )
    return {
        "source_path": PROFILE_PATH,
        "json_locator": "feature_packed_layernorm_override.schedule_status",
        "status": status,
        "exact3_evidence": approved,
        "pre_seal_profile": dict(preseal),
    }


def _validate_exact3_evidence(value: Any, label: str) -> dict[str, object]:
    record = _require_exact_keys(
        value,
        {
            "run_id",
            "relative_path",
            "manifest_sha256",
            "sha256sums_sha256",
            "layer_count",
            "artifact_eligible",
            "exact3_gate_passed",
            "schedule_evidence_eligible",
            "formal_schedule_sealed",
        },
        label,
    )
    run_id = record["run_id"]
    if (
        not isinstance(run_id, str)
        or RUN_ID_PATTERN.fullmatch(run_id) is None
        or run_id == LEGACY_R16_CALIBRATION_EVIDENCE["run_id"]
    ):
        raise ValidationError(f"{label}.run_id is missing, malformed, or retired r16")
    expected_relative_path = f"results/openfhe/{run_id}"
    if record["relative_path"] != expected_relative_path:
        raise ValidationError(
            f"{label}.relative_path must be exactly {expected_relative_path!r}"
        )
    for key in ("manifest_sha256", "sha256sums_sha256"):
        digest = record[key]
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ValidationError(f"{label}.{key} must be 64 lowercase hex")
        if digest == LEGACY_R16_CALIBRATION_EVIDENCE[key]:
            raise ValidationError(f"{label}.{key} reuses retired r16 evidence")
    for key, expected in {
        "layer_count": 3,
        "artifact_eligible": False,
        "exact3_gate_passed": True,
        "schedule_evidence_eligible": True,
        "formal_schedule_sealed": True,
    }.items():
        _require_typed_equal(record[key], expected, f"{label}.{key}")
    return dict(record)


def _exact3_require_value(value: Any, expected: Any, label: str) -> None:
    if not _exact3_typed_equal(value, expected):
        raise ValidationError(
            f"{label} mismatch: expected={expected!r} actual={value!r}"
        )


def _exact3_typed_equal(actual: Any, expected: Any) -> bool:
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _exact3_typed_equal(actual[key], expected[key]) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _exact3_typed_equal(actual_value, expected_value)
            for actual_value, expected_value in zip(actual, expected, strict=True)
        )
    return actual == expected


def _exact3_require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ValidationError(f"{label} must be 64 lowercase hex")
    return value


def _exact3_require_relative_file_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValidationError(f"{label} must be a relative POSIX path")
    raw = Path(value)
    if (
        not value
        or raw.is_absolute()
        or "\\" in value
        or ".." in raw.parts
        or raw.as_posix() != value
    ):
        raise ValidationError(f"{label} must be a canonical relative POSIX path")
    return value


def _exact3_provenance_file_record(value: Any, label: str) -> dict[str, Any]:
    record = _require_exact_keys(
        value,
        {
            "path",
            "sha256",
            "size_bytes",
            "device",
            "inode",
            "mode",
            "link_count",
            "mtime_ns",
            "ctime_ns",
        },
        label,
    )
    _exact3_require_relative_file_path(record["path"], f"{label}.path")
    _exact3_require_sha256(record["sha256"], f"{label}.sha256")
    for name in (
        "size_bytes",
        "device",
        "inode",
        "mode",
        "link_count",
        "mtime_ns",
        "ctime_ns",
    ):
        number = record[name]
        if not isinstance(number, int) or isinstance(number, bool) or number < 0:
            raise ValidationError(f"{label}.{name} must be a nonnegative integer")
    if (
        record["inode"] == 0
        or record["link_count"] != 1
        or not stat.S_ISREG(record["mode"])
        or stat.S_ISLNK(record["mode"])
    ):
        raise ValidationError(
            f"{label} must describe one ordinary singly-linked historical file"
        )
    return record


def _exact3_validate_live_file_record(
    record: dict[str, Any],
    expected_path: str,
    repository_root: Path,
    label: str,
    *,
    require_executable: bool = False,
) -> None:
    _exact3_require_value(record["path"], expected_path, f"{label}.path")
    try:
        resolved_repository = repository_root.resolve(strict=True)
        path = resolved_repository / expected_path
        result = path.lstat()
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValidationError(
            f"cannot inspect live exact3 provenance file {expected_path}: {error}"
        ) from error
    if (
        resolved != path
        or not stat.S_ISREG(result.st_mode)
        or stat.S_ISLNK(result.st_mode)
        or result.st_nlink != 1
    ):
        raise ValidationError(
            "live exact3 provenance file must be ordinary, non-symlink, and "
            f"singly linked: {expected_path}"
        )
    _exact3_require_value(result.st_size, record["size_bytes"], f"{label}.size_bytes")
    _exact3_require_value(sha256_file(path), record["sha256"], f"{label}.sha256")
    if require_executable and not os.access(path, os.X_OK):
        raise ValidationError(
            f"live exact3 provenance executable is not executable: {expected_path}"
        )


def _exact3_bundle_files(
    evidence: dict[str, object],
    repository_root: Path,
) -> tuple[Path, dict[str, Path]]:
    try:
        resolved_repository = repository_root.resolve(strict=True)
    except OSError as error:
        raise ValidationError(
            f"cannot resolve exact3 repository root: {error}"
        ) from error
    bundle_root = resolved_repository / str(evidence["relative_path"])
    expected_parent = resolved_repository / "results" / "openfhe"
    try:
        root_stat = bundle_root.lstat()
        resolved_bundle = bundle_root.resolve(strict=True)
    except OSError as error:
        raise ValidationError(
            f"cannot open approved exact3 evidence directory: {error}"
        ) from error
    if (
        bundle_root.parent != expected_parent
        or bundle_root.name != evidence["run_id"]
        or resolved_bundle != bundle_root
        or not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
    ):
        raise ValidationError(
            "approved exact3 evidence path must be one canonical non-symlink "
            "results/openfhe/<run_id> directory"
        )
    try:
        entries = {entry.name: entry for entry in bundle_root.iterdir()}
    except OSError as error:
        raise ValidationError(
            f"cannot inspect approved exact3 evidence directory: {error}"
        ) from error
    expected_inventory = set(EXACT3_BUNDLE_PATHS)
    if set(entries) != expected_inventory:
        raise ValidationError(
            "approved exact3 evidence must contain exactly the six-file bundle: "
            f"missing={sorted(expected_inventory - set(entries))} "
            f"extra={sorted(set(entries) - expected_inventory)}"
        )
    files: dict[str, Path] = {}
    for name in EXACT3_BUNDLE_PATHS:
        path = entries[name]
        try:
            result = path.lstat()
            resolved = path.resolve(strict=True)
        except OSError as error:
            raise ValidationError(
                f"cannot inspect exact3 evidence file {name}: {error}"
            ) from error
        if (
            resolved != path
            or not stat.S_ISREG(result.st_mode)
            or stat.S_ISLNK(result.st_mode)
            or result.st_nlink != 1
        ):
            raise ValidationError(
                "exact3 evidence file must be ordinary, non-symlink, and singly "
                f"linked: {name}"
            )
        files[name] = path
    return bundle_root, files


def _exact3_checksum_entries(
    checksum_path: Path,
    approved_sha256: str,
    files: dict[str, Path],
) -> dict[str, str]:
    if sha256_file(checksum_path) != approved_sha256:
        raise ValidationError(
            "exact3 SHA256SUMS hash differs from the approved evidence object"
        )
    try:
        data = checksum_path.read_bytes()
        text = data.decode("ascii")
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read exact3 SHA256SUMS: {error}") from error
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        raise ValidationError("exact3 SHA256SUMS must be NUL-free ASCII with final LF")
    lines = text[:-1].split("\n")
    if len(lines) != len(EXACT3_CHECKSUM_PATHS) or any(not line for line in lines):
        raise ValidationError("exact3 SHA256SUMS must contain exactly five lines")
    entries: dict[str, str] = {}
    ordered_names: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)", line)
        if match is None:
            raise ValidationError(f"malformed exact3 SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        if name in entries:
            raise ValidationError(f"duplicate exact3 SHA256SUMS entry: {name}")
        entries[name] = digest
        ordered_names.append(name)
    if tuple(ordered_names) != EXACT3_CHECKSUM_PATHS:
        raise ValidationError(
            "exact3 SHA256SUMS paths/order must be stdout.log, stderr.log, "
            "time.log, metrics.csv, manifest.json"
        )
    for name, digest in entries.items():
        if sha256_file(files[name]) != digest:
            raise ValidationError(f"exact3 SHA256SUMS digest mismatch for {name}")
    return entries


def _exact3_load_json_line(text: str, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValidationError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ValidationError(f"{label} contains non-finite JSON number {token}")

    def parse_finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ValidationError(f"{label} contains non-finite JSON number {token}")
        return value

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
            parse_float=parse_finite_float,
        )
    except json.JSONDecodeError as error:
        raise ValidationError(f"{label} is malformed JSON: {error}") from error
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be a JSON object")
    return value


def _exact3_validate_range_map(
    value: Any,
    intervals: dict[str, tuple[float, float]],
    label: str,
) -> dict[str, list[float]]:
    record = _require_exact_keys(value, set(intervals), label)
    validated: dict[str, list[float]] = {}
    for name, interval in intervals.items():
        observed = record[name]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ValidationError(f"{label}.{name} must be [minimum,maximum]")
        minimum = _require_finite(
            observed[0],
            f"{label}.{name}[0]",
            interval[0],
            interval[1],
        )
        maximum = _require_finite(
            observed[1],
            f"{label}.{name}[1]",
            interval[0],
            interval[1],
        )
        if minimum > maximum:
            raise ValidationError(f"{label}.{name} minimum exceeds maximum")
        validated[name] = [minimum, maximum]
    return validated


def _exact3_expected_counts(layer_id: int) -> dict[str, int]:
    return {
        key: LAYER_COUNTS[key] * (layer_id + 1)
        + REFRESH_COUNTS[key] * min(layer_id + 1, 2)
        for key in OPERATION_COUNT_KEYS
    }


def _exact3_validate_raw_layer(
    value: Any,
    layer_id: int,
    previous: dict[str, Any] | None,
) -> dict[str, Any]:
    diagnostic_keys = {
        "claim_scope",
        "artifact_eligible",
        "formal_schedule_sealed",
        "metadata_validation_mode",
        "metadata_used_levels",
        "metadata_used_level_deltas",
        "diagnostic_gates_passed",
    }
    layer = _require_exact_keys(
        value,
        LAYER_STDOUT_KEYS | diagnostic_keys,
        f"exact3 stdout layer {layer_id}",
    )
    expected_scalars = {
        "test": "openfhe_encoder_exact_prefix_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "layer_id": layer_id,
        "claim_scope": "diagnostic_prefix_exact_schedule",
        "artifact_eligible": False,
        "formal_schedule_sealed": False,
        "metadata_validation_mode": "exact",
        "weights_layer_id": layer_id,
        "handoff_refresh_performed": layer_id < 2,
        "chain_input_source": (
            "client_encrypted_trace_input"
            if layer_id == 0
            else "previous_ciphertext_output_after_refresh"
        ),
        "range_validation_owner": "client",
        "checkpoint_decryption_owner": "client",
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "finite": True,
        "range_status": "passed",
        "diagnostic_gates_passed": True,
        "inactive_zero_checkpoint_count": 28,
        "inactive_sentinel_range_status": "passed",
    }
    for key, expected in expected_scalars.items():
        _exact3_require_value(
            layer[key],
            expected,
            f"exact3 stdout layer {layer_id}.{key}",
        )
    metadata_expected = {
        "input_metadata": (
            LAYER0_INPUT_METADATA if layer_id == 0 else LAYER_HANDOFF_INPUT_METADATA
        ),
        "raw_output_metadata": RAW_OUTPUT_METADATA,
        "softmax_denominator_metadata": SOFTMAX_CHECKPOINT_METADATA,
        "attention_output_metadata": (
            LAYER0_ATTENTION_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_ATTENTION_OUTPUT_METADATA
        ),
        "ln1_variance_metadata": LAYERNORM_CHECKPOINT_METADATA,
        "ln1_output_metadata": (
            LAYER0_LN1_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_LN1_OUTPUT_METADATA
        ),
        "ffn_output_metadata": (
            LAYER0_FFN_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_FFN_OUTPUT_METADATA
        ),
        "ln2_variance_metadata": LAYERNORM_CHECKPOINT_METADATA,
    }
    if any(expected is None for expected in metadata_expected.values()):
        raise ValidationError("exact3 metadata schedule contains an unsealed tuple")
    for name, expected in metadata_expected.items():
        assert expected is not None
        _require_metadata(
            layer[name],
            expected,
            f"exact3 stdout layer {layer_id}.{name}",
        )
    used = {
        "input": _metadata_used_level(
            layer["input_metadata"], "exact3 input metadata"
        ),
        "softmax_denominator": _metadata_used_level(
            layer["softmax_denominator_metadata"],
            "exact3 softmax metadata",
        ),
        "attention_output": _metadata_used_level(
            layer["attention_output_metadata"],
            "exact3 attention metadata",
        ),
        "ln1_variance": _metadata_used_level(
            layer["ln1_variance_metadata"],
            "exact3 ln1 variance metadata",
        ),
        "ln1_output": _metadata_used_level(
            layer["ln1_output_metadata"],
            "exact3 ln1 output metadata",
        ),
        "ffn_output": _metadata_used_level(
            layer["ffn_output_metadata"],
            "exact3 ffn metadata",
        ),
        "ln2_variance": _metadata_used_level(
            layer["ln2_variance_metadata"],
            "exact3 ln2 variance metadata",
        ),
        "raw_output": _metadata_used_level(
            layer["raw_output_metadata"],
            "exact3 raw output metadata",
        ),
    }
    _exact3_require_value(
        layer["metadata_used_levels"],
        used,
        f"exact3 stdout layer {layer_id}.metadata_used_levels",
    )
    previous_recovered = (
        None
        if previous is None
        else _metadata_used_level(
            previous["raw_output_metadata"],
            "exact3 previous raw output metadata",
        )
        - used["input"]
    )
    deltas = {
        "input_to_softmax_checkpoint_net_recovered": (
            used["input"] - used["softmax_denominator"]
        ),
        "softmax_checkpoint_to_attention_output_consumed": (
            used["attention_output"] - used["softmax_denominator"]
        ),
        "attention_output_to_ln1_checkpoint_net_recovered": (
            used["attention_output"] - used["ln1_variance"]
        ),
        "ln1_checkpoint_to_ln1_output_consumed": (
            used["ln1_output"] - used["ln1_variance"]
        ),
        "ln1_output_to_ffn_output_consumed": (
            used["ffn_output"] - used["ln1_output"]
        ),
        "ffn_output_to_ln2_checkpoint_net_recovered": (
            used["ffn_output"] - used["ln2_variance"]
        ),
        "ln2_checkpoint_to_raw_output_consumed": (
            used["raw_output"] - used["ln2_variance"]
        ),
        "previous_raw_output_to_input_recovered": previous_recovered,
    }
    _exact3_require_value(
        layer["metadata_used_level_deltas"],
        deltas,
        f"exact3 stdout layer {layer_id}.metadata_used_level_deltas",
    )
    for name in ("input_quality", "output_quality", "exact_trace_diagnostic"):
        _require_quality(layer[name], f"exact3 stdout layer {layer_id}.{name}")
    _require_finite(
        layer["inactive_max_abs"],
        f"exact3 stdout layer {layer_id}.inactive_max_abs",
        0.0,
        THRESHOLDS["inactive_max_abs"],
    )
    sentinel_maximum = _require_finite(
        layer["inactive_sentinel_max_error"],
        f"exact3 stdout layer {layer_id}.inactive_sentinel_max_error",
        0.0,
    )
    sentinel_ranges = _exact3_validate_range_map(
        layer["inactive_polynomial_sentinel_ranges"],
        INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS,
        f"exact3 stdout layer {layer_id}.inactive_polynomial_sentinel_ranges",
    )
    derived_sentinel_maximum = max(
        abs(endpoint - 1.0)
        for observed in sentinel_ranges.values()
        for endpoint in observed
    )
    if not math.isclose(
        sentinel_maximum,
        derived_sentinel_maximum,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValidationError(
            f"exact3 stdout layer {layer_id} inactive sentinel maximum is inconsistent"
        )
    _exact3_validate_range_map(
        layer["encrypted_polynomial_input_ranges"],
        POLYNOMIAL_INTERVALS,
        f"exact3 stdout layer {layer_id}.encrypted_polynomial_input_ranges",
    )
    _require_counts(
        layer["refresh_operation_counts"],
        REFRESH_COUNTS if layer_id < 2 else ZERO_COUNTS,
        f"exact3 stdout layer {layer_id}.refresh_operation_counts",
    )
    _require_counts(
        layer["layer_operation_counts"],
        LAYER_COUNTS,
        f"exact3 stdout layer {layer_id}.layer_operation_counts",
    )
    _require_counts(
        layer["cumulative_operation_counts"],
        _exact3_expected_counts(layer_id),
        f"exact3 stdout layer {layer_id}.cumulative_operation_counts",
    )
    return layer


def _exact3_validate_raw_summary(
    value: Any,
    layers: list[dict[str, Any]],
) -> dict[str, Any]:
    diagnostic_keys = {
        "artifact_eligible",
        "formal_schedule_sealed",
        "metadata_validation_mode",
        "diagnostic_gates_passed",
    }
    summary = _require_exact_keys(
        value,
        SUMMARY_STDOUT_KEYS | diagnostic_keys,
        "exact3 stdout final summary",
    )
    expected_scalars = {
        "test": "openfhe_encoder_exact_prefix",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": PROFILE_SHA256,
        "execution_mode": "server-only",
        "claim_scope": "diagnostic_prefix_exact_schedule",
        "artifact_eligible": False,
        "formal_schedule_sealed": True,
        "metadata_validation_mode": "exact",
        "encoder_layers": 3,
        "chain_mode": "ciphertext_output_to_next_input",
        "client_encrypt_calls": 1,
        "encrypted_input_ciphertexts": 5,
        "plaintext_activation_resets": 0,
        "server_layer_evaluations": 3,
        "inter_layer_refreshes": 2,
        "checkpoint_decryption_owner": "client",
        "final_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "approximation_range_status": "all_client_validated",
        "inactive_sentinel_range_status": "all_client_validated",
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
        "diagnostic_gates_passed": True,
    }
    for key, expected in expected_scalars.items():
        _exact3_require_value(
            summary[key],
            expected,
            f"exact3 stdout final summary.{key}",
        )
    for name in (
        "fixture_load_oracle_ms",
        "setup_keygen_ms",
        "client_encrypt_ms",
        "client_checkpoint_validate_ms",
    ):
        _require_finite(summary[name], f"exact3 stdout final summary.{name}", 0.0)
    _require_finite(
        summary["server_online_diagnostic_ms"],
        "exact3 stdout final summary.server_online_diagnostic_ms",
        1e-12,
    )
    quality = {
        "relative_l2": summary["relative_l2"],
        "cosine": summary["cosine"],
        "max_absolute": summary["max_absolute"],
    }
    _require_quality(quality, "exact3 stdout final summary quality")
    _exact3_require_value(
        quality,
        layers[-1]["output_quality"],
        "exact3 stdout final summary quality",
    )
    _require_finite(
        summary["inactive_max_abs"],
        "exact3 stdout final summary.inactive_max_abs",
        0.0,
        THRESHOLDS["inactive_max_abs"],
    )
    _exact3_require_value(
        summary["inactive_max_abs"],
        max(layer["inactive_max_abs"] for layer in layers),
        "exact3 stdout final summary.inactive_max_abs",
    )
    _require_finite(
        summary["inactive_sentinel_max_error"],
        "exact3 stdout final summary.inactive_sentinel_max_error",
        0.0,
    )
    _exact3_require_value(
        summary["inactive_sentinel_max_error"],
        max(layer["inactive_sentinel_max_error"] for layer in layers),
        "exact3 stdout final summary.inactive_sentinel_max_error",
    )
    _require_metadata(
        summary["final_metadata"],
        RAW_OUTPUT_METADATA,
        "exact3 stdout final summary.final_metadata",
    )
    _exact3_require_value(
        summary["final_metadata"],
        layers[-1]["raw_output_metadata"],
        "exact3 stdout final summary.final_metadata",
    )
    _require_counts(
        summary["operation_counts"],
        EXACT3_FINAL_COUNTS,
        "exact3 stdout final summary.operation_counts",
    )
    peak = summary["peak_rss_bytes"]
    if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
        raise ValidationError(
            "exact3 stdout final summary.peak_rss_bytes must be a positive integer"
        )
    return summary


def _exact3_parse_stdout(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = path.read_bytes()
    maximum = int(EXACT3_GATE_CONTRACT["raw_input_bounds"]["maximum_bytes"]["stdout.log"])
    if not data or len(data) > maximum:
        raise ValidationError("exact3 stdout.log size is outside the frozen bound")
    if not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise ValidationError(
            "exact3 stdout.log must be NUL-free UTF-8 with LF line endings"
        )
    lines_bytes = data[:-1].split(b"\n")
    if len(lines_bytes) != 5 or any(not line for line in lines_bytes):
        raise ValidationError(
            "exact3 stdout.log must contain one warning and exactly four JSON lines"
        )
    maximum_line = int(
        EXACT3_GATE_CONTRACT["raw_input_bounds"]["stdout_maximum_line_bytes"]
    )
    if any(len(line) > maximum_line for line in lines_bytes):
        raise ValidationError("exact3 stdout.log contains an over-sized line")
    try:
        lines = [line.decode("utf-8") for line in lines_bytes]
    except UnicodeDecodeError as error:
        raise ValidationError("exact3 stdout.log is not UTF-8") from error
    _exact3_require_value(
        lines[0],
        EXACT3_PROFILE_WARNING,
        "exact3 stdout.log profile warning",
    )
    records = [
        _exact3_load_json_line(line, f"exact3 stdout.log line {index}")
        for index, line in enumerate(lines[1:], start=2)
    ]
    layers: list[dict[str, Any]] = []
    for layer_id, record in enumerate(records[:3]):
        layers.append(
            _exact3_validate_raw_layer(
                record,
                layer_id,
                layers[-1] if layers else None,
            )
        )
    for field in CHECKPOINT_METADATA_FIELDS:
        for key in set(layers[1][field]) - {"scale_bits"}:
            _exact3_require_value(
                layers[2][field][key],
                layers[1][field][key],
                f"exact3 layer-2 steady-state {field}.{key}",
            )
    summary = _exact3_validate_raw_summary(records[3], layers)
    return layers, summary


def _exact3_parse_time(path: Path, repository_root: Path) -> dict[str, Any]:
    data = path.read_bytes()
    maximum = int(EXACT3_GATE_CONTRACT["raw_input_bounds"]["maximum_bytes"]["time.log"])
    if not data or len(data) > maximum:
        raise ValidationError("exact3 time.log size is outside the frozen bound")
    if not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise ValidationError(
            "exact3 time.log must be NUL-free UTF-8 with LF line endings"
        )
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ValidationError("exact3 time.log is not UTF-8") from error
    if len(lines) > 64:
        raise ValidationError("exact3 time.log contains too many lines")
    fields: dict[str, str | None] = {
        "Command being timed": None,
        "Elapsed (wall clock) time (h:mm:ss or m:ss)": None,
        "Maximum resident set size (kbytes)": None,
        "Swaps": None,
        "Exit status": None,
    }
    for line in lines:
        stripped = line.strip()
        for key in fields:
            prefix = f"{key}: "
            if stripped.startswith(prefix):
                if fields[key] is not None:
                    raise ValidationError(
                        f"exact3 time.log repeats GNU time field {key!r}"
                    )
                fields[key] = stripped[len(prefix) :]
    missing = [key for key, value in fields.items() if value is None]
    if missing:
        raise ValidationError(
            f"exact3 time.log is missing GNU time fields: {missing!r}"
        )
    command_field = fields["Command being timed"]
    assert command_field is not None
    if len(command_field) < 2 or not (
        command_field.startswith('"') and command_field.endswith('"')
    ):
        raise ValidationError(
            "exact3 GNU time command must use its quoted verbose format"
        )
    command_text = command_field[1:-1]
    try:
        command_argv = shlex.split(command_text)
    except ValueError as error:
        raise ValidationError(
            "exact3 GNU time command is not shell-parseable"
        ) from error
    expected_argv = [
        f"./{EXACT3_EXECUTABLE_PATH}",
        "--data-root",
        str(repository_root.resolve(strict=True) / "data"),
        "--diagnostic-layer-count",
        "3",
    ]
    _exact3_require_value(command_argv, expected_argv, "exact3 GNU time command")
    elapsed_text = fields["Elapsed (wall clock) time (h:mm:ss or m:ss)"]
    peak_text = fields["Maximum resident set size (kbytes)"]
    swaps_text = fields["Swaps"]
    exit_text = fields["Exit status"]
    assert elapsed_text is not None
    assert peak_text is not None
    assert swaps_text is not None
    assert exit_text is not None
    elapsed_seconds = _exact3_parse_elapsed(elapsed_text)
    try:
        peak_rss_kib = int(peak_text)
        swaps = int(swaps_text)
        exit_status = int(exit_text)
    except ValueError as error:
        raise ValidationError("exact3 GNU time integer field is malformed") from error
    if peak_rss_kib <= 0 or swaps != 0 or exit_status != 0:
        raise ValidationError(
            "exact3 GNU time requires positive RSS, zero swaps, and exit status 0"
        )
    return {
        "command_text": command_text,
        "elapsed_text": elapsed_text,
        "elapsed_seconds": elapsed_seconds,
        "peak_rss_kib": peak_rss_kib,
        "peak_rss_bytes": peak_rss_kib * 1024,
        "swaps": swaps,
        "exit_status": exit_status,
    }


def _exact3_metrics_bytes(
    layers: list[dict[str, Any]],
    summary: dict[str, Any],
    timing: dict[str, Any],
) -> bytes:
    output = io.StringIO(newline="")
    fields = [
        "record_kind",
        "layer_id",
        "relative_l2",
        "cosine",
        "max_absolute",
        "inactive_max_abs",
        "inactive_sentinel_max_error",
        *OPERATION_COUNT_KEYS,
        "elapsed_seconds",
        "peak_rss_bytes",
        "passed",
    ]
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    for layer_id, layer in enumerate(layers):
        writer.writerow(
            {
                "record_kind": "layer",
                "layer_id": layer_id,
                **layer["output_quality"],
                "inactive_max_abs": layer["inactive_max_abs"],
                "inactive_sentinel_max_error": layer[
                    "inactive_sentinel_max_error"
                ],
                **layer["cumulative_operation_counts"],
                "passed": True,
            }
        )
    writer.writerow(
        {
            "record_kind": "final",
            "layer_id": "",
            "relative_l2": summary["relative_l2"],
            "cosine": summary["cosine"],
            "max_absolute": summary["max_absolute"],
            "inactive_max_abs": summary["inactive_max_abs"],
            "inactive_sentinel_max_error": summary[
                "inactive_sentinel_max_error"
            ],
            **summary["operation_counts"],
            "elapsed_seconds": timing["elapsed_seconds"],
            "peak_rss_bytes": timing["peak_rss_bytes"],
            "passed": True,
        }
    )
    return output.getvalue().encode("utf-8")


def _exact3_cross_validate_raw_bundle(
    manifest: dict[str, Any],
    files: dict[str, Path],
    repository_root: Path,
) -> None:
    stderr = files["stderr.log"].read_bytes()
    stderr_maximum = int(
        EXACT3_GATE_CONTRACT["raw_input_bounds"]["maximum_bytes"]["stderr.log"]
    )
    if len(stderr) > stderr_maximum or stderr:
        raise ValidationError("exact3 stderr.log must be byte-empty")
    layers, summary = _exact3_parse_stdout(files["stdout.log"])
    timing = _exact3_parse_time(files["time.log"], repository_root)
    expected_metrics = _exact3_metrics_bytes(layers, summary, timing)
    if files["metrics.csv"].read_bytes() != expected_metrics:
        raise ValidationError(
            "exact3 metrics.csv differs from independently derived raw evidence"
        )
    evidence = manifest["evidence"]
    _exact3_require_value(
        evidence["layers"],
        layers,
        "exact3 manifest.evidence.layers versus stdout.log",
    )
    _exact3_require_value(
        evidence["final_summary"],
        summary,
        "exact3 manifest.evidence.final_summary versus stdout.log",
    )
    manifest_timing = manifest["timing"]
    for key, expected in {
        "gnu_time_elapsed_text": timing["elapsed_text"],
        "elapsed_seconds": timing["elapsed_seconds"],
        "peak_rss_kib": timing["peak_rss_kib"],
        "peak_rss_bytes": timing["peak_rss_bytes"],
        "swaps": timing["swaps"],
        "exit_status": timing["exit_status"],
    }.items():
        _exact3_require_value(
            manifest_timing[key],
            expected,
            f"exact3 manifest.timing.{key} versus time.log",
        )
    _exact3_require_value(
        manifest["command"]["gnu_time_reported_command"],
        timing["command_text"],
        "exact3 manifest.command.gnu_time_reported_command versus time.log",
    )


def _exact3_validate_timestamp_semantics(value: Any, run_id: str) -> None:
    record = _require_exact_keys(
        value,
        {
            "run_id_label",
            "run_id_label_semantics",
            "execution_started_at",
            "execution_finished_at",
            "execution_timestamp_semantics",
            "raw_log_filesystem_timestamp_semantics",
            "sealed_at",
            "sealed_at_semantics",
        },
        "exact3 manifest.timestamp_semantics",
    )
    expected = {
        "run_id_label": run_id,
        "run_id_label_semantics": EXACT3_RUN_ID_LABEL_SEMANTICS,
        "execution_timestamp_semantics": EXACT3_EXECUTION_TIMESTAMP_SEMANTICS,
        "raw_log_filesystem_timestamp_semantics": EXACT3_RAW_TIMESTAMP_SEMANTICS,
        "sealed_at_semantics": EXACT3_SEALED_AT_SEMANTICS,
    }
    for key, expected_value in expected.items():
        _exact3_require_value(
            record[key],
            expected_value,
            f"exact3 manifest.timestamp_semantics.{key}",
        )
    execution_times: list[datetime] = []
    for key in ("execution_started_at", "execution_finished_at"):
        execution_value = record[key]
        if not isinstance(execution_value, str):
            raise ValidationError(
                f"exact3 manifest.timestamp_semantics.{key} must be a timestamp"
            )
        try:
            parsed_execution = datetime.fromisoformat(execution_value)
        except ValueError as error:
            raise ValidationError(
                f"exact3 manifest.timestamp_semantics.{key} is malformed"
            ) from error
        if (
            parsed_execution.tzinfo is None
            or parsed_execution.utcoffset() is None
        ):
            raise ValidationError(
                f"exact3 manifest.timestamp_semantics.{key} must be timezone-aware"
            )
        execution_times.append(parsed_execution)
    if execution_times[1] < execution_times[0]:
        raise ValidationError("exact3 execution_finished_at precedes started_at")
    sealed_at = record["sealed_at"]
    if not isinstance(sealed_at, str):
        raise ValidationError(
            "exact3 manifest.timestamp_semantics.sealed_at must be a timestamp"
        )
    try:
        parsed = datetime.fromisoformat(sealed_at)
    except ValueError as error:
        raise ValidationError(
            "exact3 manifest.timestamp_semantics.sealed_at is malformed"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond != 0:
        raise ValidationError(
            "exact3 sealed_at must be timezone-aware and rounded to whole seconds"
        )


def _exact3_validate_git(value: Any) -> None:
    record = _require_exact_keys(
        value,
        {
            "head",
            "branch",
            "detached",
            "clean",
            "snapshot_semantics",
            "status_porcelain_v1_z_sha256",
            "tracked_diff",
            "untracked_source_manifest",
        },
        "exact3 manifest.git",
    )
    head = record["head"]
    if not isinstance(head, str) or re.fullmatch(r"[0-9a-f]{40,64}", head) is None:
        raise ValidationError("exact3 manifest.git.head is malformed")
    _exact3_require_value(record["branch"], BRANCH, "exact3 manifest.git.branch")
    _exact3_require_value(record["detached"], False, "exact3 manifest.git.detached")
    if not isinstance(record["clean"], bool):
        raise ValidationError("exact3 manifest.git.clean must be boolean")
    _exact3_require_value(
        record["snapshot_semantics"],
        EXACT3_GIT_SNAPSHOT_SEMANTICS,
        "exact3 manifest.git.snapshot_semantics",
    )
    _exact3_require_sha256(
        record["status_porcelain_v1_z_sha256"],
        "exact3 manifest.git.status_porcelain_v1_z_sha256",
    )
    tracked = _require_exact_keys(
        record["tracked_diff"],
        {"command", "sha256", "size_bytes"},
        "exact3 manifest.git.tracked_diff",
    )
    _exact3_require_value(
        tracked["command"],
        "git diff --binary HEAD -- .",
        "exact3 manifest.git.tracked_diff.command",
    )
    _exact3_require_sha256(tracked["sha256"], "exact3 tracked diff sha256")
    if (
        not isinstance(tracked["size_bytes"], int)
        or isinstance(tracked["size_bytes"], bool)
        or tracked["size_bytes"] < 0
    ):
        raise ValidationError(
            "exact3 manifest.git.tracked_diff.size_bytes must be nonnegative"
        )
    untracked = _require_exact_keys(
        record["untracked_source_manifest"],
        {"selection", "canonicalization", "sha256", "entries"},
        "exact3 manifest.git.untracked_source_manifest",
    )
    _exact3_require_value(
        untracked["selection"],
        EXACT3_GIT_UNTRACKED_SELECTION,
        "exact3 untracked source selection",
    )
    _exact3_require_value(
        untracked["canonicalization"],
        EXACT3_GIT_CANONICALIZATION,
        "exact3 untracked source canonicalization",
    )
    entries_value = untracked["entries"]
    if not isinstance(entries_value, list):
        raise ValidationError("exact3 untracked source entries must be an array")
    entries = [
        _exact3_provenance_file_record(
            item,
            f"exact3 manifest.git.untracked_source_manifest.entries[{index}]",
        )
        for index, item in enumerate(entries_value)
    ]
    paths = [str(item["path"]) for item in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValidationError(
            "exact3 manifest.git untracked source paths must be sorted and unique"
        )
    expected_untracked_hash = hashlib.sha256(
        canonical_json_bytes(entries)
    ).hexdigest()
    _exact3_require_value(
        untracked["sha256"],
        expected_untracked_hash,
        "exact3 untracked source manifest sha256",
    )
    if record["clean"]:
        empty_hash = hashlib.sha256(b"").hexdigest()
        _exact3_require_value(
            record["status_porcelain_v1_z_sha256"],
            empty_hash,
            "clean exact3 git status hash",
        )
        _exact3_require_value(
            tracked["sha256"],
            empty_hash,
            "clean exact3 tracked diff hash",
        )
        _exact3_require_value(
            tracked["size_bytes"],
            0,
            "clean exact3 tracked diff size",
        )
        _exact3_require_value(entries, [], "clean exact3 untracked source entries")


def _exact3_validate_provenance(
    provenance: Any,
    repository_root: Path,
    evidence: dict[str, object],
) -> None:
    record = _require_exact_keys(
        provenance,
        {
            "snapshot_semantics",
            "source_files",
            "source_manifest_sha256",
            "config_files",
            "config_manifest_sha256",
            "executable",
            "trace_scale",
            "profile_transition",
        },
        "exact3 manifest.provenance",
    )
    _exact3_require_value(
        record["snapshot_semantics"],
        EXACT3_PROVENANCE_SNAPSHOT_SEMANTICS,
        "exact3 manifest.provenance.snapshot_semantics",
    )
    preseal_profile_record: dict[str, Any] | None = None
    for collection_name in ("source_files", "config_files"):
        collection = record[collection_name]
        if not isinstance(collection, list) or not collection:
            raise ValidationError(
                f"exact3 manifest.provenance.{collection_name} must be nonempty"
            )
        items = [
            _exact3_provenance_file_record(
                value,
                f"exact3 manifest.provenance.{collection_name}[{index}]",
            )
            for index, value in enumerate(collection)
        ]
        paths = [item["path"] for item in items]
        if len(paths) != len(set(paths)):
            raise ValidationError(
                f"exact3 manifest.provenance.{collection_name} paths duplicate"
            )
        expected_paths = (
            EXACT3_SOURCE_PATHS
            if collection_name == "source_files"
            else EXACT3_CONFIG_PATHS
        )
        if tuple(paths) != expected_paths:
            raise ValidationError(
                f"exact3 manifest.provenance.{collection_name} paths drifted"
            )
        for index, (item, expected_path) in enumerate(
            zip(items, expected_paths, strict=True)
        ):
            label = f"exact3 manifest.provenance.{collection_name}[{index}]"
            if collection_name == "config_files" and expected_path == PROFILE_PATH:
                preseal_profile_record = item
            else:
                _exact3_validate_live_file_record(
                    item,
                    expected_path,
                    repository_root,
                    label,
                )
        digest_key = collection_name.replace("_files", "_manifest_sha256")
        expected_digest = hashlib.sha256(canonical_json_bytes(items)).hexdigest()
        _exact3_require_value(
            record[digest_key],
            expected_digest,
            f"exact3 manifest.provenance.{digest_key}",
        )
    if preseal_profile_record is None:  # pragma: no cover - path tuple is frozen
        raise ValidationError("exact3 profile config provenance is missing")
    profile_transition = _validate_profile_transition(
        record["profile_transition"],
        preseal_profile_record,
    )
    _require_profile_schedule_sealed(
        repository_root,
        evidence,
        profile_transition,
    )
    executable = _exact3_provenance_file_record(
        record["executable"],
        "exact3 manifest.provenance.executable",
    )
    _exact3_require_value(
        executable["path"],
        EXACT3_EXECUTABLE_PATH,
        "exact3 manifest.provenance.executable.path",
    )
    if executable["mode"] & 0o111 == 0:
        raise ValidationError(
            "exact3 historical executable provenance lacks execute permission"
        )
    _exact3_require_value(
        record["trace_scale"],
        {
            "source_path": TRACE_SCALE_SOURCE_PATH,
            "json_locator": TRACE_SCALE_LOCATOR,
            "contract_id": TRACE_SCALE_CONTRACT_ID,
            "contract_sha256": TRACE_SCALE_CONTRACT_SHA256,
            "values_sha256": TRACE_SCALE_VALUES_SHA256,
            "raw_variance_sha256": TRACE_SCALE_RAW_VARIANCE_SHA256,
        },
        "exact3 manifest.provenance.trace_scale",
    )


def _exact3_validate_raw_and_derived(
    manifest: dict[str, Any],
    files: dict[str, Path],
    checksum_entries: dict[str, str],
) -> None:
    raw_logs = _require_exact_keys(
        manifest["raw_logs"],
        {"stdout.log", "stderr.log", "time.log"},
        "exact3 manifest.raw_logs",
    )
    for name in ("stdout.log", "stderr.log", "time.log"):
        record = _require_exact_keys(
            raw_logs[name],
            {
                "path",
                "sha256",
                "size_bytes",
                "filesystem_mtime_ns",
                "filesystem_ctime_ns",
            },
            f"exact3 manifest.raw_logs.{name}",
        )
        for key, expected in {
            "path": name,
            "sha256": checksum_entries[name],
            "size_bytes": files[name].stat().st_size,
        }.items():
            _exact3_require_value(
                record[key],
                expected,
                f"exact3 manifest.raw_logs.{name}.{key}",
            )
        for field in ("filesystem_mtime_ns", "filesystem_ctime_ns"):
            value = record[field]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValidationError(
                    f"exact3 manifest.raw_logs.{name}.{field} must be nonnegative int"
                )
        result = files[name].stat()
        for field, expected in {
            "filesystem_mtime_ns": result.st_mtime_ns,
            "filesystem_ctime_ns": result.st_ctime_ns,
        }.items():
            _exact3_require_value(
                record[field],
                expected,
                f"exact3 manifest.raw_logs.{name}.{field}",
            )
    derived = _require_exact_keys(
        manifest["derived_artifacts"],
        {"metrics.csv"},
        "exact3 manifest.derived_artifacts",
    )
    metrics = _require_exact_keys(
        derived["metrics.csv"],
        {"sha256", "size_bytes"},
        "exact3 manifest.derived_artifacts.metrics.csv",
    )
    for key, expected in {
        "sha256": checksum_entries["metrics.csv"],
        "size_bytes": files["metrics.csv"].stat().st_size,
    }.items():
        _exact3_require_value(
            metrics[key],
            expected,
            f"exact3 manifest.derived_artifacts.metrics.csv.{key}",
        )


def _exact3_parse_elapsed(value: Any) -> float:
    if not isinstance(value, str):
        raise ValidationError(
            "exact3 manifest.timing.gnu_time_elapsed_text must be a string"
        )
    parts = value.split(":")
    if len(parts) not in (2, 3) or any(not part for part in parts):
        raise ValidationError(
            "exact3 manifest.timing.gnu_time_elapsed_text is malformed"
        )
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[0]) if len(parts) == 3 else 0
    except ValueError as error:
        raise ValidationError(
            "exact3 manifest.timing.gnu_time_elapsed_text is malformed"
        ) from error
    if (
        not math.isfinite(seconds)
        or seconds < 0
        or seconds >= 60
        or minutes < 0
        or (len(parts) == 3 and minutes >= 60)
        or hours < 0
    ):
        raise ValidationError(
            "exact3 manifest.timing.gnu_time_elapsed_text is out of range"
        )
    elapsed = hours * 3600 + minutes * 60 + seconds
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise ValidationError(
            "exact3 manifest.timing elapsed duration must be positive and finite"
        )
    return elapsed


def _exact3_validate_command_timing_and_gate(
    manifest: dict[str, Any],
    repository_root: Path,
) -> None:
    resolved_repository = repository_root.resolve(strict=True)
    runtime_argv = [
        f"./{EXACT3_EXECUTABLE_PATH}",
        "--data-root",
        str(resolved_repository / "data"),
        "--diagnostic-layer-count",
        "3",
    ]
    command = _require_exact_keys(
        manifest["command"],
        {
            "argv",
            "runtime_argv",
            "gnu_time_reported_command",
            "cwd",
            "environment",
            "stdin",
            "attempt_count",
            "exit_code",
            "started_at",
            "finished_at",
        },
        "exact3 manifest.command",
    )
    for key, expected in {
        "runtime_argv": runtime_argv,
        "gnu_time_reported_command": shlex.join(runtime_argv),
        "cwd": str(resolved_repository),
        "environment": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
        "stdin": "DEVNULL",
        "attempt_count": 1,
        "exit_code": 0,
        "started_at": manifest["timestamp_semantics"]["execution_started_at"],
        "finished_at": manifest["timestamp_semantics"]["execution_finished_at"],
    }.items():
        _exact3_require_value(
            command[key],
            expected,
            f"exact3 manifest.command.{key}",
        )
    argv = command["argv"]
    if (
        not isinstance(argv, list)
        or len(argv) != len(runtime_argv) + 5
        or argv[:3] != ["/usr/bin/time", "-v", "-o"]
        or argv[4] != "--"
        or argv[5:] != runtime_argv
        or not isinstance(argv[3], str)
    ):
        raise ValidationError("exact3 manifest.command.argv drifted")
    time_path = Path(argv[3])
    stage_root = time_path.parent
    expected_stage_parent = resolved_repository / "results" / "openfhe"
    expected_stage_prefix = f".exact3-stage-{manifest['run_id']}-"
    if (
        not time_path.is_absolute()
        or time_path.name != "time.log"
        or stage_root.parent != expected_stage_parent
        or not stage_root.name.startswith(expected_stage_prefix)
        or len(stage_root.name) <= len(expected_stage_prefix)
    ):
        raise ValidationError(
            "exact3 manifest.command.argv does not identify its private staging time.log"
        )
    timing = _require_exact_keys(
        manifest["timing"],
        {
            "kind",
            "timing_claim",
            "gnu_time_elapsed_text",
            "elapsed_seconds",
            "peak_rss_kib",
            "peak_rss_bytes",
            "swaps",
            "exit_status",
        },
        "exact3 manifest.timing",
    )
    for key, expected in {
        "kind": "non_benchmark_diagnostic",
        "timing_claim": False,
        "swaps": 0,
        "exit_status": 0,
    }.items():
        _exact3_require_value(timing[key], expected, f"exact3 manifest.timing.{key}")
    elapsed = _exact3_parse_elapsed(timing["gnu_time_elapsed_text"])
    _exact3_require_value(
        timing["elapsed_seconds"],
        elapsed,
        "exact3 manifest.timing.elapsed_seconds",
    )
    peak_rss_kib = timing["peak_rss_kib"]
    if (
        not isinstance(peak_rss_kib, int)
        or isinstance(peak_rss_kib, bool)
        or peak_rss_kib <= 0
    ):
        raise ValidationError(
            "exact3 manifest.timing.peak_rss_kib must be a positive integer"
        )
    _exact3_require_value(
        timing["peak_rss_bytes"],
        peak_rss_kib * 1024,
        "exact3 manifest.timing.peak_rss_bytes",
    )
    _exact3_require_value(
        manifest["gate_contract"],
        EXACT3_GATE_CONTRACT,
        "exact3 manifest.gate_contract",
    )
    summary = manifest["evidence"]["final_summary"]
    _exact3_require_value(
        summary.get("peak_rss_bytes"),
        timing["peak_rss_bytes"],
        "exact3 manifest.evidence.final_summary.peak_rss_bytes",
    )


def _exact3_validate_execution_and_evidence(manifest: dict[str, Any]) -> None:
    _exact3_require_value(
        manifest["execution"],
        {
            "mode": "diagnostic_exact_prefix",
            "backend": "OpenFHE CKKS CPU",
            "encoder_layers": 3,
            "layer_ids": [0, 1, 2],
            "chain_mode": "ciphertext_output_to_next_input",
            "inter_layer_refreshes": 2,
            "checkpoint_decryption_owner": "client",
            "final_decryption_owner": "client",
            "server_private_key_present": False,
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "plaintext_activation_resets": 0,
            "artifact_eligible": False,
            "formal_schedule_sealed": True,
            "diagnostic_gates_passed": True,
        },
        "exact3 manifest.execution",
    )
    evidence = _require_exact_keys(
        manifest["evidence"],
        {
            "layers",
            "final_summary",
            "later_layer_steady_state_verified",
            "second_handoff_previous_raw_output_to_input_recovered",
        },
        "exact3 manifest.evidence",
    )
    layers = evidence["layers"]
    if not isinstance(layers, list) or len(layers) != 3:
        raise ValidationError("exact3 manifest.evidence.layers must contain 3 layers")
    for layer_id, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValidationError(
                f"exact3 manifest.evidence.layers[{layer_id}] must be an object"
            )
        for key, expected in {
            "layer_id": layer_id,
            "weights_layer_id": layer_id,
            "artifact_eligible": False,
            "formal_schedule_sealed": False,
            "checkpoint_decryption_owner": "client",
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "diagnostic_gates_passed": True,
        }.items():
            if key not in layer:
                raise ValidationError(
                    f"exact3 manifest.evidence.layers[{layer_id}].{key} is missing"
                )
            _exact3_require_value(
                layer[key],
                expected,
                f"exact3 manifest.evidence.layers[{layer_id}].{key}",
            )
    _exact3_require_value(
        evidence["later_layer_steady_state_verified"],
        True,
        "exact3 manifest.evidence.later_layer_steady_state_verified",
    )
    _exact3_require_value(
        evidence["second_handoff_previous_raw_output_to_input_recovered"],
        11,
        "exact3 manifest.evidence.second_handoff",
    )
    summary = evidence["final_summary"]
    if not isinstance(summary, dict):
        raise ValidationError(
            "exact3 manifest.evidence.final_summary must be an object"
        )
    for key, expected in {
        "security_claim": "none",
        "execution_mode": "server-only",
        "artifact_eligible": False,
        "formal_schedule_sealed": True,
        "encoder_layers": 3,
        "plaintext_activation_resets": 0,
        "checkpoint_decryption_owner": "client",
        "final_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
        "diagnostic_gates_passed": True,
    }.items():
        if key not in summary:
            raise ValidationError(
                f"exact3 manifest.evidence.final_summary.{key} is missing"
            )
        _exact3_require_value(
            summary[key],
            expected,
            f"exact3 manifest.evidence.final_summary.{key}",
        )
    _exact3_require_value(
        summary.get("operation_counts"),
        EXACT3_FINAL_COUNTS,
        "exact3 manifest.evidence.final_summary.operation_counts",
    )


def _validate_exact3_manifest(
    manifest: Any,
    evidence: dict[str, object],
    files: dict[str, Path],
    checksum_entries: dict[str, str],
    repository_root: Path,
) -> None:
    record = _require_exact_keys(
        manifest,
        {
            "schema",
            "schema_id",
            "schema_version",
            "milestone",
            "artifact_kind",
            "milestone_artifact_eligible",
            "schedule_evidence_eligible",
            "exact3_gate_passed",
            "formal_schedule_sealed",
            "timing_claim",
            "security_claim",
            "run_id",
            "timestamp_semantics",
            "git",
            "command",
            "profile",
            "provenance",
            "raw_logs",
            "timing",
            "gate_contract",
            "execution",
            "evidence",
            "derived_artifacts",
            "verdict",
        },
        "exact3 manifest",
    )
    for key, expected in {
        "schema": "diagnostic",
        "schema_id": EXACT3_MANIFEST_SCHEMA_ID,
        "schema_version": EXACT3_MANIFEST_SCHEMA_VERSION,
        "milestone": "M5",
        "artifact_kind": "exact3_schedule_evidence",
        "milestone_artifact_eligible": False,
        "schedule_evidence_eligible": True,
        "exact3_gate_passed": True,
        "formal_schedule_sealed": True,
        "timing_claim": False,
        "security_claim": "none",
        "run_id": evidence["run_id"],
    }.items():
        _exact3_require_value(record[key], expected, f"exact3 manifest.{key}")
    _exact3_validate_timestamp_semantics(
        record["timestamp_semantics"],
        str(evidence["run_id"]),
    )
    _exact3_validate_git(record["git"])
    profile = _require_exact_keys(
        record["profile"],
        {"id", "parameter_sha256", "security_claim", "warning"},
        "exact3 manifest.profile",
    )
    for key, expected in {
        "id": "paper_compat",
        "parameter_sha256": PROFILE_SHA256,
        "security_claim": "none",
        "warning": WARNING,
    }.items():
        _exact3_require_value(profile[key], expected, f"exact3 manifest.profile.{key}")
    _exact3_validate_provenance(record["provenance"], repository_root, evidence)
    _exact3_validate_raw_and_derived(record, files, checksum_entries)
    _exact3_validate_execution_and_evidence(record)
    _exact3_validate_command_timing_and_gate(record, repository_root)
    _exact3_cross_validate_raw_bundle(record, files, repository_root)
    _exact3_require_value(
        record["verdict"],
        {
            "exact3_gate_passed": True,
            "schedule_evidence_eligible": True,
            "formal_schedule_sealed": True,
            "milestone_artifact_eligible": False,
            "timing_claim": False,
            "security_claim": "none",
        },
        "exact3 manifest.verdict",
    )


def _verify_exact3_evidence_bundle(
    evidence: dict[str, object],
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    _, files = _exact3_bundle_files(evidence, repository_root)
    if sha256_file(files["manifest.json"]) != evidence["manifest_sha256"]:
        raise ValidationError(
            "exact3 manifest hash differs from the approved evidence object"
        )
    checksum_entries = _exact3_checksum_entries(
        files["SHA256SUMS"],
        str(evidence["sha256sums_sha256"]),
        files,
    )
    if checksum_entries["manifest.json"] != evidence["manifest_sha256"]:
        raise ValidationError(
            "exact3 manifest hash differs between SHA256SUMS and approved evidence"
        )
    manifest = load_json(files["manifest.json"])
    _validate_exact3_manifest(
        manifest,
        evidence,
        files,
        checksum_entries,
        repository_root,
    )
    return dict(evidence)


def _require_exact3_evidence(
    repository_root: Path = REPO_ROOT,
) -> dict[str, object]:
    if EXACT3_EVIDENCE is None:
        raise ValidationError(
            "approved exact-three-layer live OpenFHE evidence is missing"
        )
    evidence = _validate_exact3_evidence(EXACT3_EVIDENCE, "EXACT3_EVIDENCE")
    _require_profile_schedule_sealed(repository_root, evidence)
    return _verify_exact3_evidence_bundle(evidence, repository_root)


def checkpoint_metadata_payload(layers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(layers) != 12:
        raise ValidationError(
            "checkpoint metadata digest requires exactly 12 ordered layer records"
        )
    payload: list[dict[str, Any]] = []
    for layer_id, layer in enumerate(layers):
        if layer.get("layer_id") != layer_id:
            raise ValidationError(
                "checkpoint metadata digest requires layer order 0 through 11"
            )
        missing = set(CHECKPOINT_METADATA_FIELDS) - set(layer)
        if missing:
            raise ValidationError(
                f"checkpoint metadata digest is missing fields: {sorted(missing)}"
            )
        payload.append(
            {
                "layer_id": layer_id,
                **{field: layer[field] for field in CHECKPOINT_METADATA_FIELDS},
            }
        )
    return payload


def checkpoint_metadata_sha256(layers: list[dict[str, Any]]) -> str:
    return hashlib.sha256(
        canonical_json_bytes(checkpoint_metadata_payload(layers))
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_equal(left: Any, right: Any) -> bool:
    return canonical_json_bytes(left) == canonical_json_bytes(right)


def _resolve_ref(root: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValidationError(
            f"only local JSON Schema references are supported: {reference}"
        )
    current: Any = root
    for encoded in reference[2:].split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, dict) or part not in current:
            raise ValidationError(f"unresolvable JSON Schema reference: {reference}")
        current = current[part]
    if not isinstance(current, dict):
        raise ValidationError(f"JSON Schema reference is not an object: {reference}")
    return current


def _type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    raise ValidationError(f"unsupported JSON Schema type: {expected}")


def _parse_timestamp(value: str, label: str) -> datetime:
    if not (value.endswith("Z") or re.search(r"[+-][0-9]{2}:[0-9]{2}$", value)):
        raise ValidationError(f"{label}: date-time must include a timezone")
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValidationError(f"{label}: invalid RFC 3339 date-time") from error
    if result.tzinfo is None:
        raise ValidationError(f"{label}: date-time must include a timezone")
    return result


def validate_instance(
    instance: Any,
    schema: dict[str, Any],
    root: dict[str, Any],
    label: str = "$",
) -> None:
    if "$ref" in schema:
        validate_instance(instance, _resolve_ref(root, schema["$ref"]), root, label)
        return
    for subschema in schema.get("allOf", []):
        validate_instance(instance, subschema, root, label)
    expected_type = schema.get("type")
    if expected_type is not None and not _type_matches(instance, expected_type):
        raise ValidationError(
            f"{label}: expected {expected_type}, got {type(instance).__name__}"
        )
    if "const" in schema and not _json_equal(instance, schema["const"]):
        raise ValidationError(f"{label}: expected constant {schema['const']!r}")
    if "enum" in schema and not any(
        _json_equal(instance, candidate) for candidate in schema["enum"]
    ):
        raise ValidationError(f"{label}: value is not in the allowed enum")
    if isinstance(instance, dict):
        if len(instance) < schema.get("minProperties", 0):
            raise ValidationError(f"{label}: object has too few properties")
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in instance:
                raise ValidationError(f"{label}: missing required property {key!r}")
        for key, value in instance.items():
            if key in properties:
                validate_instance(value, properties[key], root, f"{label}.{key}")
            elif schema.get("additionalProperties", True) is False:
                raise ValidationError(f"{label}: unexpected property {key!r}")
    if isinstance(instance, list):
        if len(instance) < schema.get("minItems", 0):
            raise ValidationError(f"{label}: array is shorter than minItems")
        if "maxItems" in schema and len(instance) > schema["maxItems"]:
            raise ValidationError(f"{label}: array is longer than maxItems")
        if schema.get("uniqueItems", False):
            serialized = [canonical_json_bytes(item) for item in instance]
            if len(serialized) != len(set(serialized)):
                raise ValidationError(f"{label}: array items must be unique")
        prefix_schemas = schema.get("prefixItems", [])
        for index, prefix_schema in enumerate(prefix_schemas[: len(instance)]):
            validate_instance(instance[index], prefix_schema, root, f"{label}[{index}]")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(
                instance[len(prefix_schemas) :], len(prefix_schemas)
            ):
                validate_instance(item, item_schema, root, f"{label}[{index}]")
    if isinstance(instance, str):
        if len(instance) < schema.get("minLength", 0):
            raise ValidationError(f"{label}: string is shorter than minLength")
        if "maxLength" in schema and len(instance) > schema["maxLength"]:
            raise ValidationError(f"{label}: string is longer than maxLength")
        if "pattern" in schema and re.search(schema["pattern"], instance) is None:
            raise ValidationError(
                f"{label}: string does not match the required pattern"
            )
        if schema.get("format") == "date-time":
            _parse_timestamp(instance, label)
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if not math.isfinite(instance):
            raise ValidationError(f"{label}: non-finite number is forbidden")
        if "minimum" in schema and instance < schema["minimum"]:
            raise ValidationError(f"{label}: value is below minimum")
        if "maximum" in schema and instance > schema["maximum"]:
            raise ValidationError(f"{label}: value is above maximum")


def _walk_schema(node: Any, root: dict[str, Any], label: str = "$schema") -> None:
    if not isinstance(node, dict):
        raise ValidationError(f"{label}: every schema node must be an object")
    unknown = set(node) - SUPPORTED_SCHEMA_KEYWORDS
    if unknown:
        raise ValidationError(
            f"{label}: unsupported schema keywords: {sorted(unknown)}"
        )
    if "$ref" in node:
        if set(node) != {"$ref"}:
            raise ValidationError(f"{label}: $ref must not have sibling keywords")
        if not isinstance(node["$ref"], str):
            raise ValidationError(f"{label}.$ref must be a string")
        _resolve_ref(root, node["$ref"])
    if "type" in node:
        if not isinstance(node["type"], str):
            raise ValidationError(f"{label}.type must be a string")
        _type_matches(None, node["type"])
    if "enum" in node:
        choices = node["enum"]
        if not isinstance(choices, list) or not choices:
            raise ValidationError(f"{label}.enum must be a non-empty array")
        serialized = [canonical_json_bytes(choice) for choice in choices]
        if len(serialized) != len(set(serialized)):
            raise ValidationError(f"{label}.enum values must be unique")
    if "pattern" in node:
        if not isinstance(node["pattern"], str):
            raise ValidationError(f"{label}.pattern must be a string")
        try:
            re.compile(node["pattern"])
        except (TypeError, re.error) as error:
            raise ValidationError(f"{label}.pattern is invalid: {error}") from error
    if "format" in node and node["format"] != "date-time":
        raise ValidationError(f"{label}.format is unsupported")
    for keyword in (
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
    ):
        if keyword in node and (
            not isinstance(node[keyword], int)
            or isinstance(node[keyword], bool)
            or node[keyword] < 0
        ):
            raise ValidationError(f"{label}.{keyword} must be a nonnegative integer")
    for keyword in ("minimum", "maximum"):
        if keyword in node and (
            not isinstance(node[keyword], (int, float))
            or isinstance(node[keyword], bool)
            or not math.isfinite(node[keyword])
        ):
            raise ValidationError(f"{label}.{keyword} must be finite")
    for minimum_key, maximum_key in (
        ("minimum", "maximum"),
        ("minLength", "maxLength"),
        ("minItems", "maxItems"),
    ):
        if (
            minimum_key in node
            and maximum_key in node
            and node[minimum_key] > node[maximum_key]
        ):
            raise ValidationError(
                f"{label}.{minimum_key} must not exceed {maximum_key}"
            )
    if "uniqueItems" in node and not isinstance(node["uniqueItems"], bool):
        raise ValidationError(f"{label}.uniqueItems must be boolean")
    if "additionalProperties" in node and not isinstance(
        node["additionalProperties"], bool
    ):
        raise ValidationError(f"{label}.additionalProperties must be boolean")
    all_of = node.get("allOf", [])
    if not isinstance(all_of, list) or not all_of:
        if "allOf" in node:
            raise ValidationError(f"{label}.allOf must be a non-empty array")
    else:
        for index, child in enumerate(all_of):
            _walk_schema(child, root, f"{label}.allOf[{index}]")
    properties = node.get("properties", {})
    required = node.get("required", [])
    if not isinstance(properties, dict):
        raise ValidationError(f"{label}.properties must be an object")
    if not isinstance(required, list) or not all(
        isinstance(item, str) for item in required
    ):
        raise ValidationError(f"{label}.required must be an array of strings")
    if len(required) != len(set(required)) or set(required) - set(properties):
        raise ValidationError(
            f"{label}.required is duplicated or lacks property schemas"
        )
    for key, child in properties.items():
        _walk_schema(child, root, f"{label}.properties.{key}")
    definitions = node.get("$defs", {})
    if not isinstance(definitions, dict):
        raise ValidationError(f"{label}.$defs must be an object")
    for key, child in definitions.items():
        _walk_schema(child, root, f"{label}.$defs.{key}")
    item_schema = node.get("items")
    if item_schema is not None:
        _walk_schema(item_schema, root, f"{label}.items")
    prefix_items = node.get("prefixItems", [])
    if not isinstance(prefix_items, list) or not prefix_items:
        if "prefixItems" in node:
            raise ValidationError(f"{label}.prefixItems must be a non-empty array")
    else:
        if "maxItems" in node and len(prefix_items) > node["maxItems"]:
            raise ValidationError(f"{label}.prefixItems must not exceed maxItems")
        for index, child in enumerate(prefix_items):
            _walk_schema(child, root, f"{label}.prefixItems[{index}]")


def validate_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        raise ValidationError("schema root must be an object")
    if (
        schema.get("$schema") != SCHEMA_URI
        or schema.get("$id") != SCHEMA_ID
        or schema.get("title") != SCHEMA_TITLE
    ):
        raise ValidationError("schema identity drifted")
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
    ):
        raise ValidationError("schema root must be a closed object")
    if set(schema.get("required", [])) != REQUIRED_TOP_LEVEL:
        raise ValidationError("schema top-level required fields drifted")
    properties = schema.get("properties", {})
    if properties.get("schema_version", {}).get("const") != M5_SCHEMA_VERSION:
        raise ValidationError("M5 schema_version must be frozen at 6")
    if properties.get("milestone", {}).get("const") != "M5":
        raise ValidationError("schema-v6 milestone must be exactly M5")
    if properties.get("claim_boundary") != {
        "type": "array",
        "const": CLAIM_BOUNDARY,
    }:
        raise ValidationError("M5 schema claim_boundary drifted")
    exact3_schema = schema.get("$defs", {}).get("exact3_evidence", {})
    exact3_required = {
        "run_id",
        "relative_path",
        "manifest_sha256",
        "sha256sums_sha256",
        "layer_count",
        "artifact_eligible",
        "exact3_gate_passed",
        "schedule_evidence_eligible",
        "formal_schedule_sealed",
    }
    if (
        exact3_schema.get("additionalProperties") is not False
        or set(exact3_schema.get("required", [])) != exact3_required
        or set(exact3_schema.get("properties", {})) != exact3_required
    ):
        raise ValidationError("M5 schema exact3_evidence object drifted")
    inputs = properties.get("inputs", {})
    if (
        inputs.get("minItems") != INPUT_FILE_COUNT
        or inputs.get("maxItems") != INPUT_FILE_COUNT
    ):
        raise ValidationError("M5 schema must require exactly 448 inputs")
    layers = properties.get("metrics", {}).get("properties", {}).get("layers", {})
    prefix_items = layers.get("prefixItems") if isinstance(layers, dict) else None
    if (
        not isinstance(layers, dict)
        or layers.get("minItems") != len(LAYER_IDS)
        or layers.get("maxItems") != len(LAYER_IDS)
        or not isinstance(prefix_items, list)
        or len(prefix_items) != len(LAYER_IDS)
        or "items" in layers
    ):
        raise ValidationError(
            "M5 schema metrics.layers must use exactly 12 closed prefixItems"
        )
    _walk_schema(schema, schema)
    return schema


def _require_exact_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValidationError(f"{label} must be an object")
    if set(value) != expected:
        raise ValidationError(
            f"{label} keys differ: missing={sorted(expected - set(value))} "
            f"extra={sorted(set(value) - expected)}"
        )
    return value


def _require_finite(
    value: Any,
    label: str,
    minimum: float = -math.inf,
    maximum: float = math.inf,
) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value < minimum
        or value > maximum
    ):
        raise ValidationError(f"{label} violates [{minimum},{maximum}]")
    return float(value)


def _require_typed_equal(actual: Any, expected: Any, label: str) -> None:
    if not _json_equal(actual, expected):
        raise ValidationError(
            f"{label} mismatch: expected={expected!r} actual={actual!r}"
        )


def _resolve_confined_file(base: Path, relative: str, label: str) -> Path:
    raw = Path(relative)
    if (
        raw.is_absolute()
        or "\\" in relative
        or ".." in raw.parts
        or raw.as_posix() != relative
    ):
        raise ValidationError(
            f"{label}: path must be normalized and relative: {relative}"
        )
    resolved_base = base.resolve()
    resolved = (resolved_base / raw).resolve()
    try:
        resolved.relative_to(resolved_base)
    except ValueError as error:
        raise ValidationError(f"{label}: path escapes its evidence root") from error
    if not resolved.is_file():
        raise ValidationError(f"{label}: file does not exist: {resolved}")
    return resolved


def _verify_file_record(path: Path, record: dict[str, Any], label: str) -> None:
    if path.stat().st_size != record["bytes"]:
        raise ValidationError(f"{label}: byte count mismatch for {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValidationError(f"{label}: SHA-256 mismatch for {path}")


def _parse_cmake_cache(path: Path) -> dict[str, tuple[str, str]]:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ValidationError(
            f"build_configuration CMake cache is not canonical: {path}"
        )
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(
            f"cannot read build_configuration CMake cache: {error}"
        ) from error
    entries: dict[str, tuple[str, str]] = {}
    for line in lines:
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", maxsplit=1)
        if ":" not in key_and_type:
            continue
        key, entry_type = key_and_type.rsplit(":", maxsplit=1)
        if key in entries:
            raise ValidationError(f"CMake cache repeats key {key}")
        entries[key] = (entry_type, value)
    return entries


def _verify_cmake_cache(path: Path) -> None:
    entries = _parse_cmake_cache(path)
    expected = {
        "CMAKE_GENERATOR": ("INTERNAL", "Unix Makefiles"),
        "CMAKE_HOME_DIRECTORY": ("INTERNAL", str(REPO_ROOT)),
        "CMAKE_BUILD_TYPE": ("STRING", "Release"),
        "BUILD_TESTING": ("BOOL", "ON"),
        "OpenFHE_DIR": ("PATH", str(OPENFHE_PREFIX / "lib" / "OpenFHE")),
        "CMAKE_CXX_COMPILER": ("FILEPATH", SYSTEM_CXX),
        "CMAKE_MAKE_PROGRAM": ("FILEPATH", SYSTEM_MAKE),
        "CMAKE_CXX_FLAGS": ("STRING", ""),
        "CMAKE_CXX_FLAGS_RELEASE": ("STRING", "-O3 -DNDEBUG"),
        "CMAKE_EXE_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_EXE_LINKER_FLAGS_RELEASE": ("STRING", ""),
        "CMAKE_SHARED_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_MODULE_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_STATIC_LINKER_FLAGS": ("STRING", ""),
    }
    drift = {
        key: {"expected": value, "actual": entries.get(key)}
        for key, value in expected.items()
        if entries.get(key) != value
    }
    forbidden = {
        key: entries[key][1]
        for key in (
            "CMAKE_TOOLCHAIN_FILE",
            "CMAKE_CXX_COMPILER_LAUNCHER",
            "CMAKE_CXX_LINKER_LAUNCHER",
            "CMAKE_PROJECT_TOP_LEVEL_INCLUDES",
            "CMAKE_PROJECT_INCLUDE",
            "CMAKE_PROJECT_INCLUDE_BEFORE",
            "CMAKE_USER_MAKE_RULES_OVERRIDE",
            "CMAKE_USER_MAKE_RULES_OVERRIDE_CXX",
        )
        if key in entries and entries[key][1]
    }
    if drift or forbidden:
        raise ValidationError(
            f"CMake cache build configuration drifted: critical={drift} forbidden={forbidden}"
        )


def _absolute_regular_file_record(path: Path) -> dict[str, object]:
    if (
        not path.is_absolute()
        or path.is_symlink()
        or not path.is_file()
        or path.resolve() != path
    ):
        raise ValidationError(
            f"provenance path is not a canonical regular file: {path}"
        )
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _openfhe_include_tree_record() -> dict[str, object]:
    root = OPENFHE_PREFIX / "include" / "openfhe"
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ValidationError(f"OpenFHE include tree is not canonical: {root}")
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ValidationError(f"OpenFHE include tree contains symlink: {candidate}")
        if candidate.is_file():
            files.append(candidate)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValidationError("OpenFHE include tree is empty")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(files),
        "bytes": total_bytes,
        "algorithm": "SHA-256",
        "canonicalization": OPENFHE_INCLUDE_TREE_CANONICALIZATION,
        "sha256": digest.hexdigest(),
    }


def _live_openfhe_linked_libraries(executable: Path) -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            [SYSTEM_LDD, str(executable)],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
            env=_artifact_environment(),
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationError("cannot re-inspect OpenFHE linkage") from error
    linked: dict[str, dict[str, object]] = {}
    for line in completed.stdout.splitlines():
        if "libOPENFHE" not in line:
            continue
        match = re.fullmatch(
            r"\s*(libOPENFHE\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*",
            line,
        )
        if match is None:
            raise ValidationError(f"malformed OpenFHE ldd line: {line.strip()}")
        soname, target = match.groups()
        if soname not in OPENFHE_LINKED_LIBRARY_SONAMES or soname in linked:
            raise ValidationError(f"unexpected or duplicate OpenFHE SONAME: {soname}")
        target_path = Path(target)
        if not target_path.is_absolute():
            raise ValidationError(f"OpenFHE ldd target is not absolute: {target}")
        try:
            resolved = target_path.resolve(strict=True)
            resolved.relative_to(OPENFHE_PREFIX / "lib")
        except (OSError, ValueError) as error:
            raise ValidationError(
                f"OpenFHE ldd target is outside the frozen prefix: {target}"
            ) from error
        if resolved.name != OPENFHE_LINKED_LIBRARY_BASENAMES[soname]:
            raise ValidationError(f"OpenFHE versioned library drifted for {soname}")
        linked[soname] = {
            "soname": soname,
            **_absolute_regular_file_record(resolved),
        }
    if set(linked) != set(OPENFHE_LINKED_LIBRARY_SONAMES):
        raise ValidationError(
            "executable must resolve exactly the three frozen OpenFHE SONAMEs"
        )
    return [linked[soname] for soname in OPENFHE_LINKED_LIBRARY_SONAMES]


def _verify_build_configuration(manifest: dict[str, Any]) -> None:
    record = manifest["build_configuration"]
    expected_keys = {
        "generator",
        "source_root",
        "build_root",
        "openfhe_dir",
        "build_type",
        "build_testing",
        "cxx_compiler",
        "make_program",
        "cxx_flags",
        "cxx_flags_release",
        "exe_linker_flags",
        "exe_linker_flags_release",
        "shared_linker_flags",
        "module_linker_flags",
        "static_linker_flags",
        "environment_inheritance",
        "cleared_environment_variables",
        "forced_environment_variables",
        "cmake_cache",
        "openfhe_cmake_package_files",
        "openfhe_include_tree",
        "openfhe_linked_libraries",
    }
    if not isinstance(record, dict) or set(record) != expected_keys:
        raise ValidationError(
            "build_configuration keys differ from the frozen contract"
        )
    expected_core = {
        "generator": "Unix Makefiles",
        "source_root": str(REPO_ROOT),
        "build_root": str(BUILD_ROOT),
        "openfhe_dir": str(OPENFHE_PREFIX / "lib" / "OpenFHE"),
        "build_type": "Release",
        "build_testing": True,
        "cxx_compiler": SYSTEM_CXX,
        "make_program": SYSTEM_MAKE,
        "cxx_flags": "",
        "cxx_flags_release": "-O3 -DNDEBUG",
        "exe_linker_flags": "",
        "exe_linker_flags_release": "",
        "shared_linker_flags": "",
        "module_linker_flags": "",
        "static_linker_flags": "",
        "environment_inheritance": "ambient_minus_cleared_variables",
        "cleared_environment_variables": list(CONFIGURE_ENV_UNSET),
        "forced_environment_variables": dict(FORCED_SUBPROCESS_ENVIRONMENT),
    }
    for key, expected in expected_core.items():
        if not _json_equal(record[key], expected):
            raise ValidationError(f"build_configuration.{key} drifted")

    cache_record = record["cmake_cache"]
    if (
        not isinstance(cache_record, dict)
        or set(cache_record) != {"path", "bytes", "sha256"}
        or cache_record["path"] != "build-openfhe/CMakeCache.txt"
    ):
        raise ValidationError("build_configuration.cmake_cache shape drifted")
    cache_path = REPO_ROOT / cache_record["path"]
    _verify_file_record(cache_path, cache_record, "build_configuration.cmake_cache")
    _verify_cmake_cache(cache_path)

    package_root = OPENFHE_PREFIX / "lib" / "OpenFHE"
    expected_packages = [
        _absolute_regular_file_record(package_root / name)
        for name in OPENFHE_CMAKE_PACKAGE_FILES
    ]
    if record["openfhe_cmake_package_files"] != expected_packages:
        raise ValidationError("OpenFHE CMake package file provenance drifted")
    if record["openfhe_include_tree"] != _openfhe_include_tree_record():
        raise ValidationError("OpenFHE include tree provenance drifted")
    executable = REPO_ROOT / EXECUTABLE_PATH
    if record["openfhe_linked_libraries"] != _live_openfhe_linked_libraries(executable):
        raise ValidationError("OpenFHE linked-library provenance drifted")


def _decode_pointer(document: Any, pointer: str) -> Any:
    if not pointer.startswith("/"):
        raise ValidationError(
            "effective profile locator must be an absolute JSON pointer"
        )
    current = document
    for encoded in pointer[1:].split("/"):
        part = encoded.replace("~1", "/").replace("~0", "~")
        if isinstance(current, dict) and part in current:
            current = current[part]
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            raise ValidationError("effective profile locator does not resolve")
    return current


def _verify_profile(
    manifest: dict[str, Any], input_records: dict[str, dict[str, Any]]
) -> None:
    profile = manifest["profile"]
    record = input_records[PROFILE_PATH]
    if (
        profile["source_config_sha256"] != record["sha256"]
        or profile["source_config_bytes"] != record["bytes"]
    ):
        raise ValidationError("profile source record differs from the required input")
    source = load_json(REPO_ROOT / PROFILE_PATH)
    acceptance = source.get("m5_prototype_acceptance")
    if acceptance != M5_PROTOTYPE_ACCEPTANCE:
        raise ValidationError("M5 prototype acceptance contract drifted")
    effective = _decode_pointer(source, profile["effective_profile_locator"])
    if not _json_equal(effective, profile["effective_profile_payload"]):
        raise ValidationError(
            "effective profile payload differs from its source config"
        )
    digest = hashlib.sha256(canonical_json_bytes(effective)).hexdigest()
    if digest != PROFILE_SHA256 or profile["effective_profile_sha256"] != digest:
        raise ValidationError("effective profile SHA-256 differs from paper_compat")
    mismatches = {
        key: {"expected": expected, "actual": effective.get(key)}
        for key, expected in EXPECTED_PROFILE_FIELDS.items()
        if effective.get(key) != expected
    }
    if mismatches:
        raise ValidationError(
            f"effective paper_compat profile fields drifted: {mismatches}"
        )


def _bundle_identity(records: list[dict[str, Any]]) -> str:
    payload = [
        {"path": record["path"], "sha256": record["sha256"], "bytes": record["bytes"]}
        for record in sorted(records, key=lambda item: item["path"])
    ]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _expected_trace_inputs(
    trace_contract: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    try:
        required_files = trace_contract["required_files"]
        layers = trace_contract["layers"]
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"encoder trace contract is incomplete: {error}"
        ) from error
    if not isinstance(required_files, dict) or len(required_files) != 37:
        raise ValidationError(
            "encoder trace contract must map exactly 37 logical files"
        )
    if not isinstance(layers, list) or len(layers) != 12:
        raise ValidationError("encoder trace contract must contain exactly 12 layers")
    if any(not isinstance(layer, dict) for layer in layers):
        raise ValidationError("encoder trace layers must be objects")
    ordered_layer_ids = [layer.get("layer_id") for layer in layers]
    if ordered_layer_ids != list(LAYER_IDS):
        raise ValidationError(
            "encoder trace layer ids must be ordered identities 0 through 11"
        )
    by_id = {layer["layer_id"]: layer for layer in layers}

    expected: dict[str, dict[str, Any]] = {}
    groups: list[dict[str, Any]] = []
    for layer_id in LAYER_IDS:
        layer = by_id[layer_id]
        hashes = layer.get("sha256")
        if not isinstance(hashes, dict) or set(hashes) != set(required_files):
            raise ValidationError(f"trace layer {layer_id} does not bind all 37 files")
        group = {"weights": [], "trace": []}
        for logical_name in sorted(required_files):
            specification = required_files[logical_name]
            if not isinstance(specification, dict) or not isinstance(
                specification.get("path"), str
            ):
                raise ValidationError(f"trace mapping {logical_name!r} is malformed")
            contract_path = specification["path"]
            raw = Path(contract_path)
            if (
                raw.is_absolute()
                or "\\" in contract_path
                or ".." in raw.parts
                or raw.as_posix() != contract_path
            ):
                raise ValidationError(
                    f"trace mapping {logical_name!r} is not normalized"
                )
            path = (Path("data") / f"layer_{layer_id}" / raw).as_posix()
            if path in expected:
                raise ValidationError(f"encoder trace paths collide at {path}")
            role = "weights" if "/parms/" in f"/{contract_path}" else "trace"
            expected[path] = {
                "sha256": hashes[logical_name],
                "media_type": "text/csv",
                "role": role,
                "layer_id": layer_id,
            }
            group[role].append(path)
        if (
            len(group["weights"]) != WEIGHT_FILES_PER_LAYER
            or len(group["trace"]) != TRACE_FILES_PER_LAYER
        ):
            raise ValidationError(f"trace layer {layer_id} weight/trace split drifted")
        groups.append(group)
    if len(expected) != CSV_FILE_COUNT:
        raise ValidationError(
            "encoder trace contract does not resolve to exactly 444 CSVs"
        )
    return expected, groups


def _verify_inputs_and_contracts(
    manifest: dict[str, Any], repository_root: Path
) -> dict[str, dict[str, Any]]:
    _require_profile_schedule_sealed(repository_root)
    trace_scale_provenance = _trace_scale_contract_provenance(repository_root)
    approved_exact3_evidence = _require_exact3_evidence(repository_root)
    records: dict[str, dict[str, Any]] = {}
    paths = [record["path"] for record in manifest["inputs"]]
    if paths != sorted(paths):
        raise ValidationError("M5 inputs must be ordered lexicographically by path")
    for index, record in enumerate(manifest["inputs"]):
        path = record["path"]
        if path in records:
            raise ValidationError(f"duplicate input path: {path}")
        resolved = _resolve_confined_file(
            repository_root, path, f"inputs[{index}].path"
        )
        _verify_file_record(resolved, record, f"inputs[{index}]")
        records[path] = record

    trace_contract = load_json(repository_root / "config/moai_encoder_trace.json")
    try:
        if (
            trace_contract["profile_id"] != "paper_compat"
            or trace_contract["security_claim"] != "none"
            or trace_contract["dimensions"]["encoder_layers"] != 12
            or trace_contract["dimensions"]["trace_token_rows"] != 5
            or trace_contract["dimensions"]["hidden_size"] != 768
        ):
            raise ValidationError("encoder trace contract identity drifted")
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"encoder trace identity is incomplete: {error}"
        ) from error

    expected_trace, groups = _expected_trace_inputs(trace_contract)
    expected_paths = set(CONFIG_INPUTS) | set(expected_trace)
    if set(records) != expected_paths or len(records) != INPUT_FILE_COUNT:
        raise ValidationError(
            "M5 inputs differ from four configs plus 444 CSVs: "
            f"missing={sorted(expected_paths - set(records))} "
            f"extra={sorted(set(records) - expected_paths)}"
        )
    for path in CONFIG_INPUTS:
        record = records[path]
        if (
            record["media_type"] != "application/json"
            or record["role"] != "configuration"
        ):
            raise ValidationError(
                f"M5 config input {path} has the wrong media type or role"
            )
    for path, expected in expected_trace.items():
        record = records[path]
        for key in ("sha256", "media_type", "role"):
            if record[key] != expected[key]:
                raise ValidationError(
                    f"M5 trace input {path} has wrong {key}: "
                    f"expected={expected[key]!r} actual={record[key]!r}"
                )

    contracts = manifest["contracts"]
    bindings = {
        "approximation_config": "config/openfhe_approximations.json",
        "channel_scales_config": "config/moai_trace_channel_scales.json",
        "encoder_trace_contract": "config/moai_encoder_trace.json",
        "profile_config": PROFILE_PATH,
    }
    for prefix, expected_path in bindings.items():
        if contracts[f"{prefix}_path"] != expected_path:
            raise ValidationError(f"contracts.{prefix}_path drifted")
        if contracts[f"{prefix}_sha256"] != records[expected_path]["sha256"]:
            raise ValidationError(f"contracts.{prefix}_sha256 differs from inputs")

    try:
        sources = trace_contract["sources"]
        if set(sources) != {"approximations", "channel_scales"}:
            raise ValidationError("encoder trace source identities drifted")
        source_paths = {
            "approximations": "config/openfhe_approximations.json",
            "channel_scales": "config/moai_trace_channel_scales.json",
        }
        for name, path in source_paths.items():
            if (
                sources[name]["path"] != path
                or sources[name]["sha256"] != records[path]["sha256"]
            ):
                raise ValidationError(
                    f"encoder trace source {name} differs from inputs"
                )
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"encoder trace source binding is incomplete: {error}"
        ) from error

    expected_identities: list[dict[str, Any]] = []
    content_identities = {"weights": set(), "trace": set()}
    for layer_id, group in enumerate(groups):
        weight_records = [records[path] for path in group["weights"]]
        trace_records = [records[path] for path in group["trace"]]
        expected_identities.append(
            {
                "layer_id": layer_id,
                "weight_file_count": WEIGHT_FILES_PER_LAYER,
                "trace_file_count": TRACE_FILES_PER_LAYER,
                "weight_bundle_sha256": _bundle_identity(weight_records),
                "trace_bundle_sha256": _bundle_identity(trace_records),
            }
        )
        layer_prefix = Path("data") / f"layer_{layer_id}"
        for role, role_records in (
            ("weights", weight_records),
            ("trace", trace_records),
        ):
            content_payload = [
                {
                    "contract_path": Path(record["path"])
                    .relative_to(layer_prefix)
                    .as_posix(),
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
                for record in sorted(role_records, key=lambda item: item["path"])
            ]
            content_identities[role].add(
                hashlib.sha256(canonical_json_bytes(content_payload)).hexdigest()
            )
    if not _json_equal(contracts["layer_input_identities"], expected_identities):
        raise ValidationError(
            "contracts.layer_input_identities differs from 444 inputs"
        )
    weight_identities = {
        identity["weight_bundle_sha256"] for identity in expected_identities
    }
    trace_identities = {
        identity["trace_bundle_sha256"] for identity in expected_identities
    }
    if len(weight_identities) != 12 or len(trace_identities) != 12:
        raise ValidationError(
            "all 12 weight and trace bundle identities must be distinct"
        )
    if any(len(identities) != 12 for identities in content_identities.values()):
        raise ValidationError(
            "all 12 weight and trace bundles must also have distinct content identities"
        )

    if not _json_equal(contracts["execution"], EXPECTED_EXECUTION):
        raise ValidationError(
            "contracts.execution differs from the frozen ciphertext chain"
        )
    manifest_exact3_evidence = _validate_exact3_evidence(
        contracts["exact3_evidence"],
        "contracts.exact3_evidence",
    )
    if not _json_equal(manifest_exact3_evidence, approved_exact3_evidence):
        raise ValidationError(
            "contracts.exact3_evidence differs from the approved live evidence"
        )
    if not _json_equal(
        contracts["feature_packed_trace_scale_contract"],
        trace_scale_provenance,
    ):
        raise ValidationError(
            "contracts.feature_packed_trace_scale_contract differs from "
            "the independently recomputed source contract"
        )
    _require_schedule_preflight_record(
        contracts["schedule_preflight"],
        "contracts.schedule_preflight",
    )
    _require_crypto_preflight_record(
        contracts["crypto_preflight"],
        "contracts.crypto_preflight",
    )
    if not _json_equal(
        contracts["metadata_schedule"],
        _expected_metadata_schedule_contract(),
    ):
        raise ValidationError(
            "contracts.metadata_schedule differs from the frozen level schedule"
        )
    if not _json_equal(contracts["thresholds"], THRESHOLDS):
        raise ValidationError("contracts.thresholds differs from the frozen M5 gate")
    if not _json_equal(contracts["polynomial_intervals"], POLYNOMIAL_INTERVALS):
        raise ValidationError(
            "contracts.polynomial_intervals differs from the frozen ranges"
        )

    approximation = load_json(repository_root / "config/openfhe_approximations.json")
    try:
        operators = approximation["operators"]
        config_intervals = {
            "softmax_shifted_logits": [
                operators["softmax"]["exponential"]["interval"]["minimum"],
                operators["softmax"]["exponential"]["interval"]["maximum"],
            ],
            "softmax_denominator": [
                operators["softmax"]["reciprocal"]["interval"]["minimum"],
                operators["softmax"]["reciprocal"]["interval"]["maximum"],
            ],
            "ln1_normalized_variance": [
                operators["layernorm"]["inverse_sqrt"]["interval"]["minimum"],
                operators["layernorm"]["inverse_sqrt"]["interval"]["maximum"],
            ],
            "gelu_input": [
                operators["gelu"]["polynomial"]["interval"]["minimum"],
                operators["gelu"]["polynomial"]["interval"]["maximum"],
            ],
            "ln2_normalized_variance": [
                operators["layernorm"]["inverse_sqrt"]["interval"]["minimum"],
                operators["layernorm"]["inverse_sqrt"]["interval"]["maximum"],
            ],
        }
    except (KeyError, TypeError) as error:
        raise ValidationError(
            f"approximation range contract is incomplete: {error}"
        ) from error
    if not _json_equal(config_intervals, POLYNOMIAL_INTERVALS):
        raise ValidationError("checked-in approximation intervals drifted")

    _verify_profile(manifest, records)
    return records


def _command_tokens(record: Any, label: str, *, timed: bool) -> list[str]:
    keys = {"command", "cwd", "exit_code", "phase"}
    if timed:
        keys.update({"started_at", "finished_at"})
    value = _require_exact_keys(record, keys, label)
    if value["cwd"] != str(REPO_ROOT):
        raise ValidationError(f"{label}.cwd must be {REPO_ROOT}")
    if value["exit_code"] != 0 or value["phase"] != "artifact_generation":
        raise ValidationError(
            f"{label} must be a successful artifact-generation command"
        )
    if timed:
        started = _parse_timestamp(value["started_at"], f"{label}.started_at")
        finished = _parse_timestamp(value["finished_at"], f"{label}.finished_at")
        if finished < started:
            raise ValidationError(f"{label}.finished_at precedes started_at")
    try:
        tokens = shlex.split(value["command"])
    except ValueError as error:
        raise ValidationError(f"{label}.command is invalid shell syntax") from error
    if not tokens:
        raise ValidationError(f"{label}.command is empty")
    return tokens


def _expected_configure_command() -> list[str]:
    return [
        SYSTEM_ENV,
        *(f"--unset={name}" for name in CONFIGURE_ENV_UNSET),
        SYSTEM_CMAKE,
        "--fresh",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(BUILD_ROOT),
        "-G",
        "Unix Makefiles",
        f"-DOpenFHE_DIR:PATH={OPENFHE_PREFIX / 'lib' / 'OpenFHE'}",
        "-DCMAKE_BUILD_TYPE:STRING=Release",
        "-DBUILD_TESTING:BOOL=ON",
        f"-DCMAKE_CXX_COMPILER:FILEPATH={SYSTEM_CXX}",
        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={SYSTEM_MAKE}",
        "-DCMAKE_CXX_FLAGS:STRING=",
        "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
        "-DCMAKE_EXE_LINKER_FLAGS:STRING=",
        "-DCMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=",
        "-DCMAKE_SHARED_LINKER_FLAGS:STRING=",
        "-DCMAKE_MODULE_LINKER_FLAGS:STRING=",
        "-DCMAKE_STATIC_LINKER_FLAGS:STRING=",
    ]


def _verify_command_transcript(manifest: dict[str, Any], manifest_path: Path) -> None:
    commands = manifest["commands"]
    if len(commands) != 12:
        raise ValidationError("M5 commands transcript must contain exactly 12 records")
    expected_git = (
        [SYSTEM_GIT, "status", "--porcelain=v1", "--untracked-files=normal"],
        [SYSTEM_GIT, "branch", "--show-current"],
        [SYSTEM_GIT, "rev-parse", "HEAD"],
        [SYSTEM_GIT, "rev-parse", TRACKING_REF],
        [
            SYSTEM_GIT,
            "config",
            "--local",
            "--get-all",
            f"remote.{REMOTE_NAME}.url",
        ],
        [SYSTEM_GIT, "ls-remote", "--exit-code", REMOTE_URL, REMOTE_REF],
    )
    for index, expected in enumerate(expected_git):
        actual = _command_tokens(
            commands[index], f"commands[{index}] Git preflight", timed=False
        )
        if actual != expected:
            raise ValidationError(
                f"commands[{index}] differs from the frozen Git preflight"
            )

    build_root = REPO_ROOT / "build-openfhe"
    executable = REPO_ROOT / EXECUTABLE_PATH
    if (
        _command_tokens(
            commands[6],
            "commands[6] fresh fixed configure",
            timed=True,
        )
        != _expected_configure_command()
    ):
        raise ValidationError(
            "commands[6] differs from the frozen fresh fixed configure"
        )
    if _command_tokens(commands[7], "commands[7] clean-first build", timed=True) != [
        SYSTEM_CMAKE,
        "--build",
        str(build_root),
        "--clean-first",
        "-j",
        "4",
    ]:
        raise ValidationError("commands[7] differs from the frozen clean-first build")
    if _command_tokens(commands[8], "commands[8] OpenFHE linkage", timed=True) != [
        SYSTEM_LDD,
        str(executable),
    ]:
        raise ValidationError("commands[8] differs from the frozen ldd check")
    if _command_tokens(commands[9], "commands[9] narrow CTest", timed=True) != [
        SYSTEM_CTEST,
        "--test-dir",
        str(build_root),
        "--output-on-failure",
        "--verbose",
        "--no-tests=error",
        "-R",
        M5_CTEST_PATTERN,
    ]:
        raise ValidationError("commands[9] differs from the frozen M5 narrow CTest")

    workload = _command_tokens(commands[10], "commands[10] M5 workload", timed=True)
    if len(workload) != 6 or workload[:2] != ["/usr/bin/time", "--format=%M"]:
        raise ValidationError("commands[10] must use the frozen GNU time invocation")
    if (
        re.fullmatch(r"--output=/tmp/moai-m5-time-[A-Za-z0-9_.-]+\.txt", workload[2])
        is None
    ):
        raise ValidationError(
            "commands[10] GNU time output is not a fixed temporary path"
        )
    if workload[3:] != [str(executable), "--data-root", str(REPO_ROOT / "data")]:
        raise ValidationError("commands[10] M5 executable invocation drifted")

    validator = _command_tokens(commands[11], "commands[11] M5 validator", timed=False)
    if validator != [
        sys.executable,
        str(VALIDATOR_PATH),
        "--schema",
        str(DEFAULT_SCHEMA),
        "--manifest",
        str(manifest_path.resolve()),
        "--verify-git",
    ]:
        raise ValidationError("commands[11] differs from the frozen M5 validator")


def _run_git(repository_root: Path, arguments: list[str]) -> str:
    try:
        completed = subprocess.run(
            [SYSTEM_GIT, *arguments],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValidationError(
            f"Git verification failed: {SYSTEM_GIT} {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def _verify_schema_binding(
    manifest: dict[str, Any],
    repository_root: Path,
) -> None:
    binding = _require_exact_keys(
        manifest["schema_binding"],
        {"schema", "validator", "runner", "retired_unpublished_drafts"},
        "schema_binding",
    )
    _require_typed_equal(
        binding["retired_unpublished_drafts"],
        ["https://local.moai/openfhe-m5-artifact-schema-v4.json"],
        "schema_binding.retired_unpublished_drafts",
    )
    commit = manifest["git"]["local_commit"]
    for component, (component_id, relative_path) in SCHEMA_BINDING_SPECS.items():
        record = _require_exact_keys(
            binding[component],
            {"id", "path", "version", "sha256"},
            f"schema_binding.{component}",
        )
        expected = {
            "id": component_id,
            "path": relative_path,
            "version": M5_SCHEMA_VERSION,
        }
        for key, value in expected.items():
            _require_typed_equal(
                record[key],
                value,
                f"schema_binding.{component}.{key}",
            )
        binding_path = repository_root / relative_path
        if binding_path.is_symlink():
            raise ValidationError(
                f"schema_binding.{component}.path must not be a symbolic link"
            )
        resolved = _resolve_confined_file(
            repository_root,
            relative_path,
            f"schema_binding.{component}.path",
        )
        actual_sha256 = sha256_file(resolved)
        _require_typed_equal(
            record["sha256"],
            actual_sha256,
            f"schema_binding.{component}.sha256",
        )
        head_blob = _run_git(
            repository_root,
            ["rev-parse", f"{commit}:{relative_path}"],
        )
        worktree_blob = _run_git(
            repository_root,
            ["hash-object", "--", str(resolved)],
        )
        if (
            GIT_OBJECT_PATTERN.fullmatch(head_blob) is None
            or GIT_OBJECT_PATTERN.fullmatch(worktree_blob) is None
            or head_blob != worktree_blob
        ):
            raise ValidationError(
                f"schema_binding.{component} differs from the manifest HEAD blob"
            )


def _verify_git(manifest: dict[str, Any], repository_root: Path, live: bool) -> None:
    record = manifest["git"]
    commits = {
        record["local_commit"],
        record["tracking_commit"],
        record["remote_commit"],
    }
    if len(commits) != 1:
        raise ValidationError(
            "manifest local, tracking, and live-remote commits differ"
        )
    if not record["clean"]:
        raise ValidationError("M5 GO evidence requires git.clean=true")
    if not live:
        raise ValidationError("M5 GO verdict requires --verify-git live verification")
    if _run_git(repository_root, ["rev-parse", "HEAD"]) != record["local_commit"]:
        raise ValidationError("live local HEAD differs from the manifest")
    if _run_git(repository_root, ["branch", "--show-current"]) != BRANCH:
        raise ValidationError("live branch differs from refactor/openfhe-cpu")
    if _run_git(
        repository_root, ["status", "--porcelain=v1", "--untracked-files=normal"]
    ):
        raise ValidationError("live repository is not clean")
    if (
        _run_git(repository_root, ["rev-parse", TRACKING_REF])
        != record["tracking_commit"]
    ):
        raise ValidationError("live tracking ref differs from the manifest")
    origin_url = _run_git(
        repository_root,
        ["config", "--local", "--get-all", f"remote.{REMOTE_NAME}.url"],
    )
    if origin_url.splitlines() != [REMOTE_URL] or record["remote_url"] != REMOTE_URL:
        raise ValidationError("live origin URL differs from the frozen remote URL")
    remote_output = _run_git(
        repository_root,
        ["ls-remote", "--exit-code", REMOTE_URL, REMOTE_REF],
    )
    remote_lines = [line.split() for line in remote_output.splitlines() if line.strip()]
    if (
        len(remote_lines) != 1
        or len(remote_lines[0]) != 2
        or remote_lines[0] != [record["remote_commit"], REMOTE_REF]
    ):
        raise ValidationError("live remote ref differs from the manifest")


LAYER_STDOUT_KEYS = {
    "test",
    "profile",
    "security_claim",
    "layer_id",
    "weights_layer_id",
    "chain_input_source",
    "input_metadata",
    "raw_output_metadata",
    "softmax_denominator_metadata",
    "attention_output_metadata",
    "ln1_variance_metadata",
    "ln1_output_metadata",
    "ffn_output_metadata",
    "ln2_variance_metadata",
    "input_quality",
    "output_quality",
    "exact_trace_diagnostic",
    "inactive_max_abs",
    "inactive_zero_checkpoint_count",
    "inactive_sentinel_max_error",
    "inactive_polynomial_sentinel_ranges",
    "inactive_sentinel_range_status",
    "encrypted_polynomial_input_ranges",
    "handoff_refresh_performed",
    "refresh_operation_counts",
    "layer_operation_counts",
    "cumulative_operation_counts",
    "range_validation_owner",
    "checkpoint_decryption_owner",
    "server_decryptions",
    "server_plaintext_activations",
    "finite",
    "range_status",
}
SUMMARY_STDOUT_KEYS = {
    "test",
    "profile",
    "security_claim",
    "parameter_sha256",
    "execution_mode",
    "claim_scope",
    "encoder_layers",
    "chain_mode",
    "client_encrypt_calls",
    "encrypted_input_ciphertexts",
    "plaintext_activation_resets",
    "server_layer_evaluations",
    "inter_layer_refreshes",
    "checkpoint_decryption_owner",
    "final_decryption_owner",
    "server_private_key_present",
    "server_decryptions",
    "server_plaintext_activations",
    "approximation_range_status",
    "fixture_load_oracle_ms",
    "setup_keygen_ms",
    "client_encrypt_ms",
    "server_online_diagnostic_ms",
    "client_checkpoint_validate_ms",
    "relative_l2",
    "cosine",
    "max_absolute",
    "inactive_max_abs",
    "inactive_sentinel_max_error",
    "inactive_sentinel_range_status",
    "final_metadata",
    "operation_counts",
    "multiplicative_depth",
    "max_observed_level",
    "max_polynomial_depth",
    "peak_rss_bytes",
    "timing_claim",
    "latency_kind",
    "finite",
    "passed",
}
LAYER_METRIC_KEYS = {
    "layer_id",
    "weights_layer_id",
    "chain_input_source",
    "input_metadata",
    "raw_output_metadata",
    "softmax_denominator_metadata",
    "attention_output_metadata",
    "ln1_variance_metadata",
    "ln1_output_metadata",
    "ffn_output_metadata",
    "ln2_variance_metadata",
    "input_quality",
    "output_quality",
    "exact_trace_diagnostic",
    "inactive_max_abs",
    "inactive_zero_checkpoint_count",
    "inactive_sentinel_max_error",
    "inactive_polynomial_sentinel_ranges",
    "inactive_sentinel_range_status",
    "encrypted_polynomial_input_ranges",
    "handoff_refresh_performed",
    "refresh_operation_counts",
    "layer_operation_counts",
    "cumulative_operation_counts",
    "finite",
    "range_status",
}


def _parse_stdout_records(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read stdout.log: {error}") from error
    relevant: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        start = line.find("{")
        if start < 0:
            continue
        try:
            candidate = json.loads(
                line[start:],
                object_pairs_hook=_object_without_duplicates,
                parse_constant=_reject_non_finite,
            )
        except (json.JSONDecodeError, ValidationError) as error:
            raise ValidationError(
                f"stdout.log line {line_number} contains malformed JSON evidence"
            ) from error
        if not isinstance(candidate, dict):
            raise ValidationError(
                f"stdout.log line {line_number} contains non-object JSON evidence"
            )
        test_name = candidate.get("test")
        if test_name in {
            "openfhe_encoder_12_layer_layer",
            "openfhe_encoder_12_layer",
        }:
            relevant.append(candidate)
            continue
        if test_name == "openfhe_security_disclosure":
            disclosure = _require_exact_keys(
                candidate,
                {"test", "security_claim"},
                f"stdout.log security disclosure line {line_number}",
            )
            _require_typed_equal(
                disclosure["security_claim"],
                "none",
                f"stdout.log security disclosure line {line_number}.security_claim",
            )
            continue
        raise ValidationError(
            f"stdout.log line {line_number} contains unexpected JSON evidence "
            f"namespace {test_name!r}"
        )
    if len(relevant) != 13:
        raise ValidationError(
            "stdout must contain exactly 12 layer records then one summary"
        )
    layers = relevant[:12]
    summary = relevant[12]
    if any(record.get("test") != "openfhe_encoder_12_layer_layer" for record in layers):
        raise ValidationError(
            "stdout M5 layer records are not ordered before the summary"
        )
    if summary.get("test") != "openfhe_encoder_12_layer":
        raise ValidationError("stdout M5 final record is missing or out of order")
    return layers, summary


def _metadata_used_level(value: dict[str, Any], label: str) -> int:
    used = value["level"] + value["noise_scale_degree"] - 1
    depth = EXPECTED_PROFILE_FIELDS["multiplicative_depth"]
    if used < 0 or used >= depth:
        raise ValidationError(
            f"{label} used level {used} is outside the depth-{depth} schedule"
        )
    expected_remaining = depth - used
    if value["remaining_levels"] != expected_remaining:
        raise ValidationError(
            f"{label}.remaining_levels drifted: expected={expected_remaining} "
            f"from used level {used}, actual={value['remaining_levels']!r}"
        )
    return used


def _require_metadata(value: Any, expected: dict[str, Any], label: str) -> None:
    record = _require_exact_keys(value, set(expected), label)
    for key in (
        "level",
        "noise_scale_degree",
        "remaining_levels",
        "expected_scale_bits",
        "ciphertext_count",
    ):
        if not isinstance(record[key], int) or isinstance(record[key], bool):
            raise ValidationError(f"{label}.{key} must be an integer")
    _metadata_used_level(record, label)
    for key, expected_value in expected.items():
        if key == "scale_bits":
            target_scale = float(expected["expected_scale_bits"])
            actual_scale = _require_finite(
                record[key],
                f"{label}.scale_bits",
                target_scale - 1e-3,
                target_scale + 1e-3,
            )
            if abs(actual_scale - target_scale) > 1e-3:
                raise ValidationError(
                    f"{label}.scale_bits differs from expected_scale_bits "
                    "by more than 1e-3"
                )
            continue
        _require_typed_equal(record[key], expected_value, f"{label}.{key}")


def _require_used_level_delta(
    output: dict[str, Any],
    anchor: dict[str, Any],
    expected_delta: int,
    label: str,
) -> None:
    actual_delta = _metadata_used_level(
        output,
        f"{label} output",
    ) - _metadata_used_level(anchor, f"{label} anchor")
    if actual_delta != expected_delta:
        raise ValidationError(
            f"{label} used-level delta drifted: expected={expected_delta} "
            f"actual={actual_delta}"
        )


def _require_softmax_checkpoint_metadata(value: Any, label: str) -> None:
    if SOFTMAX_CHECKPOINT_METADATA is None:
        raise ValidationError(
            "M5 Softmax checkpoint metadata is not sealed by the live seam run"
        )
    _require_metadata(value, SOFTMAX_CHECKPOINT_METADATA, label)


def _require_layernorm_checkpoint_metadata(value: Any, label: str) -> None:
    if LAYERNORM_CHECKPOINT_METADATA is None:
        raise ValidationError(
            "M5 LayerNorm checkpoint metadata is not sealed by the live seam run"
        )
    _require_metadata(value, LAYERNORM_CHECKPOINT_METADATA, label)


def _require_layer0_input_metadata(value: Any, label: str) -> None:
    if LAYER0_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-0 input metadata is not sealed by the crypto preflight"
        )
    _require_metadata(value, LAYER0_INPUT_METADATA, label)


def _require_crypto_preflight_record(value: Any, label: str) -> dict[str, Any]:
    record = _require_exact_keys(value, CRYPTO_PREFLIGHT_KEYS, label)
    expected_scalars = {
        "test": "openfhe_encoder_12_layer_crypto_preflight",
        "profile": "paper_compat",
        "security_claim": "none",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "additive_he_operations": 0,
        "passed": True,
    }
    for key, expected in expected_scalars.items():
        _require_typed_equal(record[key], expected, f"{label}.{key}")
    _require_layer0_input_metadata(
        record["input_metadata"],
        f"{label}.input_metadata",
    )
    return record


def _require_schedule_preflight_record(value: Any, label: str) -> dict[str, Any]:
    record = _require_exact_keys(value, SCHEDULE_PREFLIGHT_KEYS, label)
    expected_scalars = {
        "test": "openfhe_encoder_12_layer_preflight",
        "profile": "paper_compat",
        "security_claim": "none",
        "encoder_layers": 12,
        "layer_ids": list(LAYER_IDS),
        "chain_mode": "plaintext_output_to_next_oracle_input",
        "expected_fixture_files": 444,
        "metadata_schedule_preflight": "synthetic_structure_only",
        "formal_metadata_schedule_source": (
            "two_layer_live_candidate_requires_exact_three"
        ),
        "formal_schedule_sealed": False,
        "he_metadata_verified_by_preflight": False,
        "trace_hash_gate": "separate_fail_closed_validator_required",
        "finite_and_in_range": True,
    }
    for key, expected in expected_scalars.items():
        _require_typed_equal(record[key], expected, f"{label}.{key}")
    _require_finite(
        record["final_relative_l2"], f"{label}.final_relative_l2", 0.0, 0.05
    )
    _require_finite(
        record["final_cosine"], f"{label}.final_cosine", 0.99, 1.000000000001
    )
    _require_finite(record["final_max_absolute"], f"{label}.final_max_absolute", 0.0)
    _require_finite(record["load_and_oracle_ms"], f"{label}.load_and_oracle_ms", 0.0)
    return record


def _require_handoff_input_metadata(value: Any, label: str) -> None:
    if LAYER_HANDOFF_INPUT_METADATA is None:
        raise ValidationError(
            "M5 layer-handoff input metadata is not sealed by the two-layer seam run"
        )
    _require_metadata(value, LAYER_HANDOFF_INPUT_METADATA, label)


def _require_quality(value: Any, label: str) -> dict[str, Any]:
    quality = _require_exact_keys(
        value,
        {"relative_l2", "cosine", "max_absolute"},
        label,
    )
    _require_finite(
        quality["relative_l2"],
        f"{label}.relative_l2",
        0.0,
        THRESHOLDS["per_layer_relative_l2_max"],
    )
    _require_finite(
        quality["cosine"],
        f"{label}.cosine",
        THRESHOLDS["per_layer_cosine_min"],
        1.000000000001,
    )
    _require_finite(quality["max_absolute"], f"{label}.max_absolute", 0.0)
    return quality


def _require_counts(value: Any, expected: dict[str, int], label: str) -> None:
    counts = _require_exact_keys(value, set(OPERATION_COUNT_KEYS), label)
    for key in OPERATION_COUNT_KEYS:
        count = counts[key]
        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count != expected[key]
        ):
            raise ValidationError(
                f"{label}.{key} must be {expected[key]}, got {count!r}"
            )


def _cumulative_counts(layer_id: int) -> dict[str, int]:
    layer_multiplier = layer_id + 1
    refresh_multiplier = min(layer_id + 1, 11)
    return {
        key: LAYER_COUNTS[key] * layer_multiplier
        + REFRESH_COUNTS[key] * refresh_multiplier
        for key in OPERATION_COUNT_KEYS
    }


def _verify_range_evidence(value: Any, label: str) -> None:
    ranges = _require_exact_keys(value, set(POLYNOMIAL_INTERVALS), label)
    for name, declared in POLYNOMIAL_INTERVALS.items():
        observed = ranges[name]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ValidationError(f"{label}.{name} must be a [minimum,maximum] array")
        minimum = _require_finite(observed[0], f"{label}.{name}[0]")
        maximum = _require_finite(observed[1], f"{label}.{name}[1]")
        if minimum > maximum:
            raise ValidationError(f"{label}.{name} minimum exceeds maximum")
        if minimum < declared[0] or maximum > declared[1]:
            raise ValidationError(
                f"{label}.{name} escaped pre-registered interval {declared}"
            )


def _verify_inactive_polynomial_sentinel_ranges(value: Any, label: str) -> None:
    ranges = _require_exact_keys(
        value,
        set(INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS),
        label,
    )
    for name, declared in INACTIVE_POLYNOMIAL_SENTINEL_INTERVALS.items():
        observed = ranges[name]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ValidationError(f"{label}.{name} must be a [minimum,maximum] array")
        minimum = _require_finite(observed[0], f"{label}.{name}[0]")
        maximum = _require_finite(observed[1], f"{label}.{name}[1]")
        if minimum > maximum:
            raise ValidationError(f"{label}.{name} minimum exceeds maximum")
        if minimum < declared[0] or maximum > declared[1]:
            raise ValidationError(
                f"{label}.{name} escaped pre-registered interval {declared}"
            )


def _verify_layer_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    manifest_layers: list[dict[str, Any]] = []
    for layer_id, record in enumerate(records):
        label = f"stdout layer record {layer_id}"
        regime = "layer_0" if layer_id == 0 else "layers_1_to_11"
        _require_exact_keys(record, LAYER_STDOUT_KEYS, label)
        expected_scalars = {
            "test": "openfhe_encoder_12_layer_layer",
            "profile": "paper_compat",
            "security_claim": "none",
            "layer_id": layer_id,
            "weights_layer_id": layer_id,
            "chain_input_source": (
                "client_encrypted_trace_input"
                if layer_id == 0
                else "previous_ciphertext_output_after_refresh"
            ),
            "handoff_refresh_performed": layer_id < 11,
            "range_validation_owner": "client",
            "checkpoint_decryption_owner": "client",
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "finite": True,
            "range_status": "passed",
            "inactive_sentinel_range_status": "passed",
            "inactive_zero_checkpoint_count": 28,
        }
        for key, expected in expected_scalars.items():
            _require_typed_equal(record[key], expected, f"{label}.{key}")
        if layer_id == 0:
            _require_layer0_input_metadata(
                record["input_metadata"], f"{label}.input_metadata"
            )
        else:
            _require_handoff_input_metadata(
                record["input_metadata"], f"{label}.input_metadata"
            )
        _require_metadata(
            record["raw_output_metadata"],
            RAW_OUTPUT_METADATA,
            f"{label}.raw_output_metadata",
        )
        _require_softmax_checkpoint_metadata(
            record["softmax_denominator_metadata"],
            f"{label}.softmax_denominator_metadata",
        )
        for field in ("ln1_variance_metadata", "ln2_variance_metadata"):
            _require_layernorm_checkpoint_metadata(record[field], f"{label}.{field}")
        _require_metadata(
            record["attention_output_metadata"],
            (
                LAYER0_ATTENTION_OUTPUT_METADATA
                if layer_id == 0
                else POST_REFRESH_ATTENTION_OUTPUT_METADATA
            ),
            f"{label}.attention_output_metadata",
        )
        _require_metadata(
            record["ln1_output_metadata"],
            (
                LAYER0_LN1_OUTPUT_METADATA
                if layer_id == 0
                else POST_REFRESH_LN1_OUTPUT_METADATA
            ),
            f"{label}.ln1_output_metadata",
        )
        _require_metadata(
            record["ffn_output_metadata"],
            (
                LAYER0_FFN_OUTPUT_METADATA
                if layer_id == 0
                else POST_REFRESH_FFN_OUTPUT_METADATA
            ),
            f"{label}.ffn_output_metadata",
        )
        _require_used_level_delta(
            record["input_metadata"],
            record["softmax_denominator_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["input_to_softmax_checkpoint_net_recovered"][
                regime
            ],
            f"{label}.input_to_softmax_checkpoint_net_recovered",
        )
        _require_used_level_delta(
            record["attention_output_metadata"],
            record["softmax_denominator_metadata"],
            RELATIVE_USED_LEVEL_DELTAS[
                "softmax_checkpoint_to_attention_output_consumed"
            ][regime],
            f"{label}.softmax_checkpoint_to_attention_output_consumed",
        )
        _require_used_level_delta(
            record["attention_output_metadata"],
            record["ln1_variance_metadata"],
            RELATIVE_USED_LEVEL_DELTAS[
                "attention_output_to_ln1_checkpoint_net_recovered"
            ][regime],
            f"{label}.attention_output_to_ln1_checkpoint_net_recovered",
        )
        _require_used_level_delta(
            record["ln1_output_metadata"],
            record["ln1_variance_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["ln1_checkpoint_to_ln1_output_consumed"][regime],
            f"{label}.ln1_checkpoint_to_ln1_output_consumed",
        )
        _require_used_level_delta(
            record["ffn_output_metadata"],
            record["ln1_output_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["ln1_output_to_ffn_output_consumed"][regime],
            f"{label}.ln1_output_to_ffn_output_consumed",
        )
        _require_used_level_delta(
            record["ffn_output_metadata"],
            record["ln2_variance_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["ffn_output_to_ln2_checkpoint_net_recovered"][
                regime
            ],
            f"{label}.ffn_output_to_ln2_checkpoint_net_recovered",
        )
        _require_used_level_delta(
            record["raw_output_metadata"],
            record["ln2_variance_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["ln2_checkpoint_to_raw_output_consumed"][regime],
            f"{label}.ln2_checkpoint_to_raw_output_consumed",
        )
        if layer_id > 0:
            previous_raw_output_metadata = records[layer_id - 1]["raw_output_metadata"]
            _require_metadata(
                previous_raw_output_metadata,
                RAW_OUTPUT_METADATA,
                f"{label}.previous_raw_output_metadata",
            )
            _require_used_level_delta(
                previous_raw_output_metadata,
                record["input_metadata"],
                RELATIVE_USED_LEVEL_DELTAS["previous_raw_output_to_input_recovered"][
                    regime
                ],
                f"{label}.previous_raw_output_to_input_recovered",
            )
        for field in ("input_quality", "output_quality", "exact_trace_diagnostic"):
            _require_quality(record[field], f"{label}.{field}")
        _require_finite(
            record["inactive_max_abs"],
            f"{label}.inactive_max_abs",
            0.0,
            THRESHOLDS["inactive_max_abs"],
        )
        _require_finite(
            record["inactive_sentinel_max_error"],
            f"{label}.inactive_sentinel_max_error",
            0.0,
        )
        _verify_inactive_polynomial_sentinel_ranges(
            record["inactive_polynomial_sentinel_ranges"],
            f"{label}.inactive_polynomial_sentinel_ranges",
        )
        _verify_range_evidence(
            record["encrypted_polynomial_input_ranges"],
            f"{label}.encrypted_polynomial_input_ranges",
        )
        _require_counts(
            record["refresh_operation_counts"],
            REFRESH_COUNTS if layer_id < 11 else ZERO_COUNTS,
            f"{label}.refresh_operation_counts",
        )
        _require_counts(
            record["layer_operation_counts"],
            LAYER_COUNTS,
            f"{label}.layer_operation_counts",
        )
        _require_counts(
            record["cumulative_operation_counts"],
            _cumulative_counts(layer_id),
            f"{label}.cumulative_operation_counts",
        )
        manifest_layers.append({key: record[key] for key in LAYER_METRIC_KEYS})
    return manifest_layers


def _verify_summary(
    summary: dict[str, Any], layers: list[dict[str, Any]]
) -> dict[str, Any]:
    _require_exact_keys(summary, SUMMARY_STDOUT_KEYS, "stdout M5 summary")
    expected_scalars = {
        "test": "openfhe_encoder_12_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": PROFILE_SHA256,
        "execution_mode": "server-only",
        "claim_scope": "m5_12_layer_correctness",
        "encoder_layers": 12,
        "chain_mode": "ciphertext_output_to_next_input",
        "client_encrypt_calls": 1,
        "encrypted_input_ciphertexts": 5,
        "plaintext_activation_resets": 0,
        "server_layer_evaluations": 12,
        "inter_layer_refreshes": 11,
        "checkpoint_decryption_owner": "client",
        "final_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "approximation_range_status": "all_client_validated",
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "inactive_sentinel_range_status": "all_client_validated",
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
    }
    for key, expected in expected_scalars.items():
        _require_typed_equal(summary[key], expected, f"stdout M5 summary.{key}")
    for key in (
        "fixture_load_oracle_ms",
        "setup_keygen_ms",
        "client_encrypt_ms",
        "client_checkpoint_validate_ms",
    ):
        _require_finite(summary[key], f"stdout M5 summary.{key}", 0.0)
    _require_finite(
        summary["server_online_diagnostic_ms"],
        "stdout M5 summary.server_online_diagnostic_ms",
        1e-9,
    )
    _require_finite(
        summary["relative_l2"],
        "stdout M5 summary.relative_l2",
        0.0,
        THRESHOLDS["final_relative_l2_max"],
    )
    _require_finite(
        summary["cosine"],
        "stdout M5 summary.cosine",
        THRESHOLDS["final_cosine_min"],
        1.000000000001,
    )
    _require_finite(summary["max_absolute"], "stdout M5 summary.max_absolute", 0.0)
    _require_finite(
        summary["inactive_max_abs"],
        "stdout M5 summary.inactive_max_abs",
        0.0,
        THRESHOLDS["inactive_max_abs"],
    )
    _require_finite(
        summary["inactive_sentinel_max_error"],
        "stdout M5 summary.inactive_sentinel_max_error",
        0.0,
    )
    peak_rss_bytes = summary["peak_rss_bytes"]
    if (
        not isinstance(peak_rss_bytes, int)
        or isinstance(peak_rss_bytes, bool)
        or peak_rss_bytes <= 0
    ):
        raise ValidationError("stdout M5 summary.peak_rss_bytes must be positive")
    _require_metadata(
        summary["final_metadata"], RAW_OUTPUT_METADATA, "stdout final_metadata"
    )
    _require_counts(
        summary["operation_counts"], TOTAL_COUNTS, "stdout operation_counts"
    )

    final_layer = layers[-1]
    for summary_key, quality_key in (
        ("relative_l2", "relative_l2"),
        ("cosine", "cosine"),
        ("max_absolute", "max_absolute"),
    ):
        if not math.isclose(
            float(summary[summary_key]),
            float(final_layer["output_quality"][quality_key]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValidationError(
                f"stdout final {summary_key} differs from layer-11 output_quality"
            )
    for field in ("inactive_max_abs", "inactive_sentinel_max_error"):
        expected_maximum = max(float(layer[field]) for layer in layers)
        if not math.isclose(
            float(summary[field]),
            expected_maximum,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValidationError(
                f"stdout final {field} differs from the 12-layer maximum"
            )
    if not _json_equal(summary["final_metadata"], final_layer["raw_output_metadata"]):
        raise ValidationError("stdout final metadata differs from layer-11 raw output")
    if not _json_equal(
        summary["operation_counts"], final_layer["cumulative_operation_counts"]
    ):
        raise ValidationError(
            "stdout final counts differ from layer-11 cumulative counts"
        )
    return summary


def _close(actual: Any, expected: Any, label: str) -> None:
    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except (TypeError, ValueError) as error:
        raise ValidationError(f"{label} is not numeric") from error
    if not math.isfinite(actual_number) or not math.isclose(
        actual_number,
        expected_number,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        raise ValidationError(
            f"{label} differs: expected={expected!r} actual={actual!r}"
        )


def _csv_integer(value: Any, expected: int, label: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"0|[1-9][0-9]*", value) is None:
        raise ValidationError(
            f"{label} must be a canonical nonnegative decimal integer"
        )
    actual = int(value)
    if actual != expected:
        raise ValidationError(f"{label} must be exactly {expected}, got {actual}")
    return actual


def _read_metrics_csv(
    path: Path,
    summary: dict[str, Any],
    layers: list[dict[str, Any]],
) -> tuple[float, int, dict[str, float]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != CSV_FIELDS:
                raise ValidationError(
                    "metrics.csv header differs from the frozen M5 field order: "
                    f"expected={CSV_FIELDS!r} actual={tuple(reader.fieldnames or ())!r}"
                )
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValidationError(f"cannot read metrics.csv: {error}") from error
    if len(rows) != 1:
        raise ValidationError("metrics.csv must contain exactly one correctness row")
    row = rows[0]
    if set(row) != set(CSV_FIELDS) or None in row:
        raise ValidationError("metrics.csv row columns differ from the frozen header")
    _csv_integer(row["run"], 1, "metrics.csv run")
    _csv_integer(row["exit_code"], 0, "metrics.csv exit_code")
    expected_checkpoint_hash = checkpoint_metadata_sha256(layers)
    if row["checkpoint_metadata_sha256"] != expected_checkpoint_hash:
        raise ValidationError(
            "metrics.csv checkpoint_metadata_sha256 differs from stdout"
        )
    try:
        elapsed = float(row["elapsed_seconds"])
    except (TypeError, ValueError) as error:
        raise ValidationError(
            f"metrics.csv elapsed_seconds is invalid: {error}"
        ) from error
    peak_text = row["peak_rss_kib"]
    if (
        not isinstance(peak_text, str)
        or re.fullmatch(r"[1-9][0-9]*", peak_text) is None
    ):
        raise ValidationError(
            "metrics.csv peak_rss_kib must be a canonical positive integer"
        )
    peak_rss_kib = int(peak_text)
    if not math.isfinite(elapsed) or elapsed <= 0.0 or peak_rss_kib <= 0:
        raise ValidationError("metrics.csv resource fields must be finite and positive")

    phase_fields = {
        key: summary[key]
        for key in (
            "fixture_load_oracle_ms",
            "setup_keygen_ms",
            "client_encrypt_ms",
            "server_online_diagnostic_ms",
            "client_checkpoint_validate_ms",
        )
    }
    for key, expected in phase_fields.items():
        _close(row[key], expected, f"metrics.csv {key}")
    for column, expected in {
        "final_rel_l2": summary["relative_l2"],
        "final_cosine": summary["cosine"],
        "final_max_absolute": summary["max_absolute"],
        "inactive_max_abs": summary["inactive_max_abs"],
        "inactive_sentinel_max_error": summary["inactive_sentinel_max_error"],
        "final_scale_bits": summary["final_metadata"]["scale_bits"],
    }.items():
        _close(row[column], expected, f"metrics.csv {column}")
    discrete = {
        "final_level": summary["final_metadata"]["level"],
        "final_noise_scale_degree": summary["final_metadata"]["noise_scale_degree"],
        "final_remaining_levels": summary["final_metadata"]["remaining_levels"],
        "final_ciphertext_count": summary["final_metadata"]["ciphertext_count"],
        **summary["operation_counts"],
    }
    for column, expected in discrete.items():
        _csv_integer(row[column], expected, f"metrics.csv {column}")
    expected_literals = {
        "timing_claim": "false",
        "latency_kind": "non_benchmark_diagnostic",
        "finite": "true",
        "passed": "true",
    }
    for key, expected in expected_literals.items():
        if row[key] != expected:
            raise ValidationError(
                f"metrics.csv {key} must be {expected!r}, got {row[key]!r}"
            )
    if summary["peak_rss_bytes"] > peak_rss_kib * 1024:
        raise ValidationError("stdout peak RSS exceeds GNU time metrics.csv peak RSS")
    return (
        elapsed,
        peak_rss_kib,
        {key: float(value) for key, value in phase_fields.items()},
    )


def _verify_manifest_metrics(
    manifest: dict[str, Any],
    layer_metrics: list[dict[str, Any]],
    summary: dict[str, Any],
    elapsed: float,
    peak_rss_kib: int,
    phases: dict[str, float],
) -> None:
    metrics = manifest["metrics"]
    expected_keys = {
        "repeat_count",
        "successful_repeats",
        "warmup_count",
        "timing_claim",
        "latency_kind",
        "elapsed_seconds",
        "peak_rss_kib",
        "checkpoint_metadata_sha256",
        "phase_latency_ms",
        "layers",
        "final_quality",
        "final_metadata",
        "operation_counts",
        "finite",
    }
    _require_exact_keys(metrics, expected_keys, "metrics")
    expected_scalars = {
        "repeat_count": 1,
        "successful_repeats": 1,
        "warmup_count": 0,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "peak_rss_kib": peak_rss_kib,
        "finite": True,
    }
    for key, expected in expected_scalars.items():
        _require_typed_equal(metrics[key], expected, f"metrics.{key}")
    _close(metrics["elapsed_seconds"], elapsed, "metrics.elapsed_seconds")
    expected_checkpoint_hash = checkpoint_metadata_sha256(layer_metrics)
    if metrics["checkpoint_metadata_sha256"] != expected_checkpoint_hash:
        raise ValidationError("metrics.checkpoint_metadata_sha256 differs from stdout")
    for key, expected in phases.items():
        _close(
            metrics["phase_latency_ms"][key],
            expected,
            f"metrics.phase_latency_ms.{key}",
        )
    if not _json_equal(metrics["layers"], layer_metrics):
        raise ValidationError("metrics.layers differs from the 12 stdout layer records")
    expected_final_quality = {
        "relative_l2": summary["relative_l2"],
        "cosine": summary["cosine"],
        "max_absolute": summary["max_absolute"],
        "inactive_max_abs": summary["inactive_max_abs"],
        "inactive_sentinel_max_error": summary["inactive_sentinel_max_error"],
        "inactive_sentinel_range_status": summary["inactive_sentinel_range_status"],
    }
    if not _json_equal(metrics["final_quality"], expected_final_quality):
        raise ValidationError("metrics.final_quality differs from stdout")
    if not _json_equal(metrics["final_metadata"], summary["final_metadata"]):
        raise ValidationError("metrics.final_metadata differs from stdout")
    if not _json_equal(metrics["operation_counts"], summary["operation_counts"]):
        raise ValidationError("metrics.operation_counts differs from stdout")


def _verify_checksum_evidence(
    artifact_root: Path, artifact_records: dict[str, dict[str, Any]]
) -> None:
    try:
        lines = (artifact_root / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ValidationError(f"cannot read SHA256SUMS: {error}") from error
    entries: dict[str, str] = {}
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)", line)
        if match is None or match.group(2) in entries:
            raise ValidationError(f"invalid or duplicate SHA256SUMS line: {line!r}")
        entries[match.group(2)] = match.group(1)
    if set(entries) != {"stdout.log", "metrics.csv"}:
        raise ValidationError(
            "SHA256SUMS must cover exactly stdout.log and metrics.csv"
        )
    for path, digest in entries.items():
        if artifact_records[path]["sha256"] != digest:
            raise ValidationError(f"SHA256SUMS digest disagrees for {path}")


def _verify_artifacts_and_runtime(
    manifest_path: Path,
    manifest: dict[str, Any],
    repository_root: Path,
) -> None:
    executable = _resolve_confined_file(
        repository_root,
        manifest["workload"]["executable_path"],
        "workload.executable_path",
    )
    _verify_file_record(
        executable,
        {
            "bytes": manifest["workload"]["executable_bytes"],
            "sha256": manifest["workload"]["executable_sha256"],
        },
        "workload executable",
    )

    artifact_root = manifest_path.resolve().parent
    expected_inventory = {
        "manifest.json",
        "stdout.log",
        "metrics.csv",
        "SHA256SUMS",
    }
    try:
        actual_inventory = {entry.name for entry in artifact_root.iterdir()}
    except OSError as error:
        raise ValidationError(
            f"cannot inspect M5 artifact directory: {error}"
        ) from error
    if actual_inventory != expected_inventory:
        raise ValidationError(
            "M5 artifact directory must contain exactly the sealed four-file bundle: "
            f"missing={sorted(expected_inventory - actual_inventory)} "
            f"extra={sorted(actual_inventory - expected_inventory)}"
        )
    required = {
        "stdout.log": ("stdout", "text/plain"),
        "metrics.csv": ("metrics", "text/csv"),
        "SHA256SUMS": ("checksum", "text/plain"),
    }
    if [record["path"] for record in manifest["artifacts"]] != list(required):
        raise ValidationError(
            "artifacts must be ordered stdout.log, metrics.csv, SHA256SUMS"
        )
    records: dict[str, dict[str, Any]] = {}
    for index, record in enumerate(manifest["artifacts"]):
        path = record["path"]
        if path in records:
            raise ValidationError(f"duplicate artifact path: {path}")
        expected_role, expected_media_type = required[path]
        if (
            record["role"] != expected_role
            or record["media_type"] != expected_media_type
        ):
            raise ValidationError(f"artifact {path} role or media type drifted")
        resolved = _resolve_confined_file(
            artifact_root, path, f"artifacts[{index}].path"
        )
        if resolved == manifest_path.resolve():
            raise ValidationError("manifest.json cannot hash itself")
        _verify_file_record(resolved, record, f"artifacts[{index}]")
        records[path] = record

    layers, summary = _parse_stdout_records(artifact_root / "stdout.log")
    layer_metrics = _verify_layer_records(layers)
    _verify_summary(summary, layers)
    elapsed, peak_rss_kib, phases = _read_metrics_csv(
        artifact_root / "metrics.csv",
        summary,
        layers,
    )
    _verify_manifest_metrics(
        manifest,
        layer_metrics,
        summary,
        elapsed,
        peak_rss_kib,
        phases,
    )
    _verify_checksum_evidence(artifact_root, records)


def validate_manifest(
    manifest_path: Path,
    manifest: Any,
    schema: dict[str, Any],
    verify_git: bool,
) -> None:
    if (
        isinstance(manifest, dict)
        and manifest.get("schema_version") != M5_SCHEMA_VERSION
    ):
        raise ValidationError("M5 validator accepts only schema_version=6")
    validate_instance(manifest, schema, schema)
    assert isinstance(manifest, dict)
    started = _parse_timestamp(manifest["started_at"], "$.started_at")
    finished = _parse_timestamp(manifest["finished_at"], "$.finished_at")
    if finished < started:
        raise ValidationError("finished_at precedes started_at")
    repository_root = Path(manifest["git"]["repository_root"])
    if repository_root.resolve() != REPO_ROOT.resolve():
        raise ValidationError(f"manifest repository_root must be {REPO_ROOT}")
    resolved_manifest = manifest_path.resolve()
    artifact_root = resolved_manifest.parent
    if resolved_manifest.name != "manifest.json":
        raise ValidationError("M5 manifest filename must be manifest.json")
    if artifact_root.name != manifest["run_id"]:
        raise ValidationError("M5 artifact directory name must equal manifest.run_id")
    if artifact_root.parent.resolve() != OUTPUT_ROOT.resolve():
        raise ValidationError(
            f"M5 artifact directory must be a direct child of {OUTPUT_ROOT.resolve()}"
        )
    if manifest["claim_boundary"] != CLAIM_BOUNDARY:
        raise ValidationError("claim_boundary differs from the frozen M5 scope")
    _verify_build_configuration(manifest)
    _verify_inputs_and_contracts(manifest, repository_root)
    _verify_command_transcript(manifest, manifest_path)
    _verify_artifacts_and_runtime(manifest_path, manifest, repository_root)
    _verify_git(manifest, repository_root, verify_git)
    _verify_schema_binding(manifest, repository_root)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--verify-git",
        action="store_true",
        help="verify clean local HEAD, tracking ref, and live remote ref",
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    if arguments.verify_git and arguments.manifest is None:
        raise ValidationError("--verify-git requires --manifest")
    if arguments.schema.resolve() != DEFAULT_SCHEMA.resolve():
        raise ValidationError(f"M5 v6 validator requires --schema {DEFAULT_SCHEMA}")
    schema = validate_schema(load_json(arguments.schema))
    result: dict[str, Any] = {
        "test": "validate_openfhe_m5_artifact",
        "schema": str(arguments.schema.resolve()),
        "schema_version": M5_SCHEMA_VERSION,
        "schema_valid": True,
    }
    if arguments.manifest is not None:
        manifest = load_json(arguments.manifest)
        validate_manifest(arguments.manifest, manifest, schema, arguments.verify_git)
        result.update(
            {
                "manifest": str(arguments.manifest.resolve()),
                "run_id": manifest["run_id"],
                "manifest_valid": True,
                "git_verified_live": arguments.verify_git,
            }
        )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValidationError, OSError) as error:
        print(f"validate_openfhe_m5_artifact failed: {error}", file=sys.stderr)
        sys.exit(1)
