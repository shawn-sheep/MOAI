#!/usr/bin/env python3
"""Run and seal the frozen M5 OpenFHE 12-layer correctness evidence bundle.

This runner is intentionally not a benchmark harness.  It builds and validates
the clean M5 target, runs the frozen plaintext/contract preflight, then performs
exactly one complete server-only ciphertext chain through all 12 encoder layers.
There is no warm-up and no repeat: timing remains a non-benchmark diagnostic
until M6.  Prefix diagnostics are deliberately not exposed by this interface.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import re
import shlex
import shutil
import stat
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXECUTABLE = REPO_ROOT / "build-openfhe" / "openfhe_encoder_12_layer_smoke"
DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "results" / "openfhe"
DEFAULT_OPENFHE_PREFIX = Path("/home/shawnsheep/opt/openfhe_v1_5_1")
SYSTEM_CMAKE = Path("/usr/bin/cmake")
SYSTEM_CTEST = Path("/usr/bin/ctest")
SYSTEM_CXX = Path("/usr/bin/c++")
SYSTEM_GIT = Path("/usr/bin/git")
SYSTEM_LDD = Path("/usr/bin/ldd")
SYSTEM_MAKE = Path("/usr/bin/gmake")
SYSTEM_ENV = Path("/usr/bin/env")
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
SCHEMA_PATH = REPO_ROOT / "docs" / "openfhe-m5-artifact-schema-v6.json"
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m5_artifact_v6.py"
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_encoder12_artifact_v6.py"
SCHEMA_ID = "https://local.moai/openfhe-m5-artifact-schema-v6.json"
VALIDATOR_ID = "moai.openfhe.m5.artifact-validator.v6"
RUNNER_ID = "moai.openfhe.m5.artifact-runner.v6"
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
TIME_EXECUTABLE = Path("/usr/bin/time")
REMOTE_NAME = "origin"
REMOTE_URL = "https://github.com/shawn-sheep/MOAI.git"
BRANCH = "refactor/openfhe-cpu"
TRACKING_REF = "refs/remotes/origin/refactor/openfhe-cpu"
REMOTE_REF = "refs/heads/refactor/openfhe-cpu"
PROFILE_PATH = "config/paper_compat_feature_packed.json"
PROFILE_LOCATOR = "/effective_profile"
PROFILE_SHA256 = "94f30e628e21f02146ce7ed9820194eabba3820f6e1e17176a31f8c5acf8b0be"
M5_SCHEMA_VERSION = 6
WARNING = "Research reproduction parameters only. Do not claim 128-bit security."
CLAIM_BOUNDARY = (
    "Five-token M5 12-layer encoder trace replay only; not task-level inference.",
    "server-only is an API/target trust boundary, not operating-system process "
    "isolation; correctness executable links client observer for checkpoints",
    "paper_compat OpenFHE CKKS CPU parameters with security_claim=none.",
    "M5 correctness is prototype-only with inactive/cross-lane <=1e-3; legacy "
    "MOAI defined no such threshold and strict numerical parity is not claimed.",
    "Timing is a non-benchmark diagnostic; no speedup claim.",
    "GPU, Discrete CKKS/FBT, QDQ, tokenizer, and classifier are excluded.",
)
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
LAYERNORM_REGISTERED_INTERVAL = (0.5, 1536.0)
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
REQUIRED_INPUTS = (
    "config/moai_encoder_trace.json",
    "config/moai_trace_channel_scales.json",
    "config/openfhe_approximations.json",
    PROFILE_PATH,
)
ENCODER_LAYERS = 12
FILES_PER_LAYER = 37
WEIGHT_FILES_PER_LAYER = 16
TRACE_FILES_PER_LAYER = 21
EXPECTED_INPUT_COUNT = len(REQUIRED_INPUTS) + ENCODER_LAYERS * FILES_PER_LAYER
EXPECTED_MULTIPLICATIVE_DEPTH = 47
EXPECTED_MAX_OBSERVED_LEVEL = 45
EXPECTED_MAX_POLYNOMIAL_DEPTH = 10
EXPECTED_LAYER_COUNTS = {
    "rotations": 6300,
    "ct_pt_multiplications": 51885,
    "ct_ct_multiplications": 95,
    "explicit_rescale_requests": 810,
    "chebyshev_evaluations": 55,
    "estimated_polynomial_multiplications": 1150,
    "bootstraps": 25,
    "bootstrap_iterations": 50,
}
EXPECTED_REFRESH_COUNTS = {
    "rotations": 0,
    "ct_pt_multiplications": 5,
    "ct_ct_multiplications": 0,
    "explicit_rescale_requests": 5,
    "chebyshev_evaluations": 0,
    "estimated_polynomial_multiplications": 0,
    "bootstraps": 5,
    "bootstrap_iterations": 10,
}
ZERO_COUNTS = {key: 0 for key in EXPECTED_LAYER_COUNTS}
EXPECTED_TOTAL_COUNTS = {
    key: EXPECTED_LAYER_COUNTS[key] * ENCODER_LAYERS
    + EXPECTED_REFRESH_COUNTS[key] * (ENCODER_LAYERS - 1)
    for key in EXPECTED_LAYER_COUNTS
}
EXACT3_FINAL_COUNTS = {
    key: EXPECTED_LAYER_COUNTS[key] * 3 + EXPECTED_REFRESH_COUNTS[key] * 2
    for key in EXPECTED_LAYER_COUNTS
}
# These tuples remain static candidates only.  Formal execution additionally
# requires a sealed profile schedule and approved exact-three-layer evidence
# before the HE child process can start.
LAYER0_INPUT_METADATA: dict[str, object] | None = {
    "level": 29,
    "noise_scale_degree": 1,
    "remaining_levels": 18,
    "scale_bits": 50,
    "expected_scale_bits": 50,
    "ciphertext_count": 5,
}
LAYER_HANDOFF_INPUT_METADATA: dict[str, object] | None = {
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
SOFTMAX_CHECKPOINT_METADATA: dict[str, object] | None = {
    "level": 18,
    "noise_scale_degree": 2,
    "remaining_levels": 28,
    "scale_bits": 100,
    "expected_scale_bits": 100,
    "ciphertext_count": 5,
}
LAYERNORM_CHECKPOINT_METADATA: dict[str, object] | None = {
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
COUNT_KEYS = set(EXPECTED_LAYER_COUNTS)
METADATA_KEYS = {
    "level",
    "noise_scale_degree",
    "remaining_levels",
    "scale_bits",
    "expected_scale_bits",
    "ciphertext_count",
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
QUALITY_KEYS = {"relative_l2", "cosine", "max_absolute"}
RANGE_KEYS = {
    "softmax_shifted_logits",
    "softmax_denominator",
    "ln1_normalized_variance",
    "gelu_input",
    "ln2_normalized_variance",
}
INACTIVE_SENTINEL_RANGE_KEYS = {
    "softmax_denominator",
    "ln1_normalized_variance",
    "ln2_normalized_variance",
}
FROZEN_RANGES = {
    "softmax_shifted_logits": (-16.0, 5.0),
    "softmax_denominator": (0.01, 80.0),
    "ln1_normalized_variance": (0.5, 1536.0),
    "gelu_input": (-80.0, 128.0),
    "ln2_normalized_variance": (0.5, 1536.0),
}
INACTIVE_SENTINEL_INTERVALS = {
    "softmax_denominator": FROZEN_RANGES["softmax_denominator"],
    "ln1_normalized_variance": LAYERNORM_REGISTERED_INTERVAL,
    "ln2_normalized_variance": LAYERNORM_REGISTERED_INTERVAL,
}
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
LAYER_RECORD_KEYS = {
    "test",
    "profile",
    "security_claim",
    "layer_id",
    "weights_layer_id",
    "chain_input_source",
    "handoff_refresh_performed",
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
FINAL_RECORD_KEYS = {
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
FORMAL_RUNTIME_TEST_NAMES = frozenset(
    {
        "openfhe_encoder_12_layer_layer",
        "openfhe_encoder_12_layer",
    }
)
FORMAL_NON_EVIDENCE_JSON_ALLOWLIST = {
    "openfhe_security_disclosure": frozenset({"test", "security_claim"}),
}
DIAGNOSTIC_EVIDENCE_TEST_PREFIXES = (
    "openfhe_encoder_metadata_calibration",
    "openfhe_encoder_exact_prefix",
)
DIAGNOSTIC_ONLY_EVIDENCE_FIELDS = frozenset(
    {
        "artifact_eligible",
        "formal_schedule_sealed",
        "metadata_validation_mode",
        "metadata_used_levels",
        "metadata_used_level_deltas",
        "diagnostic_gates_passed",
    }
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
    "rotations",
    "ct_pt_multiplications",
    "ct_ct_multiplications",
    "explicit_rescale_requests",
    "chebyshev_evaluations",
    "estimated_polynomial_multiplications",
    "bootstraps",
    "bootstrap_iterations",
    "timing_claim",
    "latency_kind",
    "finite",
    "passed",
)
# This narrow set intentionally includes the executable's --preflight-only
# CTest and the fixed layer-1/input-level-29 M4 regression, while excluding the
# full openfhe_encoder_12_layer_smoke workload.
M5_CTEST_PATTERN = (
    "^(moai_trace_contract|openfhe_(server_trust_boundary|"
    "server_trust_boundary_negative|profile_contract|"
    "profile_validator_contract|artifact_schema_contract|artifact_validator_contract|"
    "encoder_artifact_runner_contract|m4_v5_artifact_schema_contract|"
    "m4_v5_artifact_validator_contract|m4_v5_encoder_artifact_runner_contract|"
    "m5_artifact_schema_contract|m5_artifact_validator_contract|"
    "encoder12_artifact_runner_contract|m5_v6_artifact_schema_contract|"
    "m5_v6_artifact_validator_contract|m5_v6_encoder12_artifact_runner_contract|"
    "evaluation_key_bundle_smoke|feature_packed_smoke|"
    "feature_packed_attention_smoke|feature_bootstrap_smoke|"
    "feature_layernorm_smoke|"
    "encoder_fixture_contract|encoder_plaintext_oracle_smoke|"
    "encoder_layer_smoke|"
    "encoder_12_layer_preflight|encoder_12_layer_crypto_preflight|"
    "encoder_trace_contract|"
    "encoder_trace_validator_contract))$"
)
M5_CTEST_EXPECTED_TESTS = (
    "openfhe_server_trust_boundary",
    "openfhe_server_trust_boundary_negative",
    "openfhe_profile_contract",
    "openfhe_profile_validator_contract",
    "openfhe_artifact_schema_contract",
    "openfhe_artifact_validator_contract",
    "openfhe_encoder_artifact_runner_contract",
    "openfhe_m4_v5_artifact_schema_contract",
    "openfhe_m4_v5_artifact_validator_contract",
    "openfhe_m4_v5_encoder_artifact_runner_contract",
    "openfhe_m5_artifact_schema_contract",
    "openfhe_m5_artifact_validator_contract",
    "openfhe_encoder12_artifact_runner_contract",
    "openfhe_m5_v6_artifact_schema_contract",
    "openfhe_m5_v6_artifact_validator_contract",
    "openfhe_m5_v6_encoder12_artifact_runner_contract",
    "moai_trace_contract",
    "openfhe_encoder_trace_contract",
    "openfhe_encoder_trace_validator_contract",
    "openfhe_feature_packed_smoke",
    "openfhe_feature_packed_attention_smoke",
    "openfhe_encoder_layer_smoke",
    "openfhe_encoder_fixture_contract",
    "openfhe_encoder_12_layer_preflight",
    "openfhe_encoder_12_layer_crypto_preflight",
    "openfhe_encoder_plaintext_oracle_smoke",
    "openfhe_evaluation_key_bundle_smoke",
    "openfhe_feature_layernorm_smoke",
    "openfhe_feature_bootstrap_smoke",
)
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:-]{0,127}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_OBJECT_PATTERN = re.compile(r"^[0-9a-f]{40,64}$")


class ArtifactRunnerError(RuntimeError):
    """Raised when an M5 artifact cannot be produced without weakening a gate."""


def _metadata_schedule_contract() -> dict[str, object]:
    if (
        LAYER0_INPUT_METADATA is None
        or LAYER_HANDOFF_INPUT_METADATA is None
        or SOFTMAX_CHECKPOINT_METADATA is None
        or LAYERNORM_CHECKPOINT_METADATA is None
    ):
        raise ArtifactRunnerError(
            "M5 metadata_schedule cannot be emitted before all seam tuples are sealed"
        )
    schedule = {
        "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
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
        _validate_metadata(schedule[name], schedule[name], f"metadata_schedule.{name}")
    regime_fields = (
        "attention_output_by_layer_regime",
        "ln1_output_by_layer_regime",
        "ffn_output_by_layer_regime",
    )
    for field in regime_fields:
        for regime, metadata in schedule[field].items():
            _validate_metadata(
                metadata,
                metadata,
                f"metadata_schedule.{field}.{regime}",
            )
    for regime, input_field in (
        ("layer_0", "initial_layer_input"),
        ("layers_1_to_11", "post_refresh_layer_input"),
    ):
        _validate_used_level_delta(
            schedule[input_field],
            schedule["post_bootstrap_softmax_denominator_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["input_to_softmax_checkpoint_net_recovered"][
                regime
            ],
            f"metadata_schedule.{regime}_input_to_softmax_checkpoint_recovery",
        )
        _validate_used_level_delta(
            schedule["attention_output_by_layer_regime"][regime],
            schedule["post_bootstrap_softmax_denominator_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS[
                "softmax_checkpoint_to_attention_output_consumed"
            ][regime],
            f"metadata_schedule.{regime}_softmax_to_attention_consumption",
        )
        _validate_used_level_delta(
            schedule["attention_output_by_layer_regime"][regime],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS[
                "attention_output_to_ln1_checkpoint_net_recovered"
            ][regime],
            f"metadata_schedule.{regime}_attention_to_ln1_checkpoint_recovery",
        )
        _validate_used_level_delta(
            schedule["ln1_output_by_layer_regime"][regime],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["ln1_checkpoint_to_ln1_output_consumed"][regime],
            f"metadata_schedule.ln1_output_by_layer_regime.{regime}",
        )
        _validate_used_level_delta(
            schedule["ffn_output_by_layer_regime"][regime],
            schedule["ln1_output_by_layer_regime"][regime],
            RELATIVE_USED_LEVEL_DELTAS["ln1_output_to_ffn_output_consumed"][regime],
            f"metadata_schedule.ffn_output_by_layer_regime.{regime}",
        )
        _validate_used_level_delta(
            schedule["ffn_output_by_layer_regime"][regime],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["ffn_output_to_ln2_checkpoint_net_recovered"][
                regime
            ],
            f"metadata_schedule.{regime}_ffn_to_ln2_checkpoint_recovery",
        )
        _validate_used_level_delta(
            schedule["raw_layer_output"],
            schedule["post_bootstrap_canonicalized_layernorm_variance_checkpoint"],
            RELATIVE_USED_LEVEL_DELTAS["ln2_checkpoint_to_raw_output_consumed"][regime],
            f"metadata_schedule.{regime}_ln2_to_raw_output_consumption",
        )
    _validate_used_level_delta(
        schedule["raw_layer_output"],
        schedule["post_refresh_layer_input"],
        RELATIVE_USED_LEVEL_DELTAS["previous_raw_output_to_input_recovered"][
            "layers_1_to_11"
        ],
        "metadata_schedule.previous_raw_output_to_input_recovered",
    )
    return schedule


@dataclass(frozen=True)
class GitState:
    head: str
    tracking_head: str
    remote_url: str
    remote_head: str
    commands: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class M5Preflight:
    commands: tuple[dict[str, object], ...]
    crypto_preflight: dict[str, Any]
    build_configuration: dict[str, object]
    schedule_preflight: dict[str, Any]


@dataclass(frozen=True)
class RunnerConfig:
    executable: Path
    data_root: Path
    output_root: Path
    run_id: str
    openfhe_prefix: Path = DEFAULT_OPENFHE_PREFIX


@dataclass(frozen=True)
class RunSample:
    stdout: str
    layers: tuple[dict[str, Any], ...]
    final: dict[str, Any]
    elapsed_seconds: float
    peak_rss_kib: int
    command: dict[str, object]


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_binding_snapshot(head: str, phase: str) -> dict[str, object]:
    binding: dict[str, object] = {}
    for component, (component_id, relative_path) in SCHEMA_BINDING_SPECS.items():
        resolved = REPO_ROOT / relative_path
        if not resolved.is_file() or resolved.is_symlink():
            raise ArtifactRunnerError(
                f"M5 v6 {component} binding is not a regular repository file {phase}: "
                f"{relative_path}"
            )
        head_blob, _ = _run_git(["rev-parse", f"{head}:{relative_path}"])
        worktree_blob, _ = _run_git(["hash-object", "--", str(resolved)])
        if (
            GIT_OBJECT_PATTERN.fullmatch(head_blob) is None
            or GIT_OBJECT_PATTERN.fullmatch(worktree_blob) is None
            or head_blob != worktree_blob
        ):
            raise ArtifactRunnerError(
                f"M5 v6 {component} differs from the HEAD blob {phase}: "
                f"head={head_blob!r} worktree={worktree_blob!r}"
            )
        binding[component] = {
            "id": component_id,
            "path": relative_path,
            "version": M5_SCHEMA_VERSION,
            "sha256": _sha256(resolved),
        }
    binding["retired_unpublished_drafts"] = [
        "https://local.moai/openfhe-m5-artifact-schema-v4.json"
    ]
    return binding


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactRunnerError(f"value is not canonical JSON: {error}") from error


def _checkpoint_metadata_payload(
    layers: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if len(layers) != ENCODER_LAYERS:
        raise ArtifactRunnerError(
            "checkpoint metadata digest requires exactly 12 ordered layer records"
        )
    payload: list[dict[str, Any]] = []
    for layer_id, layer in enumerate(layers):
        if layer.get("layer_id") != layer_id:
            raise ArtifactRunnerError(
                "checkpoint metadata digest requires layer order 0 through 11"
            )
        missing = set(CHECKPOINT_METADATA_FIELDS) - set(layer)
        if missing:
            raise ArtifactRunnerError(
                f"checkpoint metadata digest is missing fields: {sorted(missing)}"
            )
        payload.append(
            {
                "layer_id": layer_id,
                **{field: layer[field] for field in CHECKPOINT_METADATA_FIELDS},
            }
        )
    return payload


def _checkpoint_metadata_sha256(
    layers: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(_checkpoint_metadata_payload(layers))
    ).hexdigest()


def _load_json(path: Path) -> Any:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactRunnerError(f"duplicate JSON key in {path}: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ArtifactRunnerError(f"non-finite JSON number in {path}: {token}")

    def parse_finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ArtifactRunnerError(f"non-finite JSON number in {path}: {token}")
        return value

    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(
                stream,
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
                parse_float=parse_finite_float,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ArtifactRunnerError(f"cannot read JSON {path}: {error}") from error


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
                raise ArtifactRunnerError(
                    f"{label}{coordinate} must be a finite binary64 value"
                )
            digest.update(struct.pack("<d", float(node)))
            return
        if not isinstance(node, list) or len(node) != dimensions[0]:
            raise ArtifactRunnerError(f"{label}{coordinate} shape differs from {shape}")
        for index, child in enumerate(node):
            visit(child, dimensions[1:], (*coordinate, index))

    visit(value, shape, ())
    return digest.hexdigest()


def _trace_scale_contract_provenance(
    repository_root: Path = REPO_ROOT,
) -> dict[str, str]:
    approximation_path = repository_root / TRACE_SCALE_SOURCE_PATH
    approximation = _load_json(approximation_path)
    try:
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
    except (KeyError, TypeError) as error:
        raise ArtifactRunnerError(
            f"trace-scale contract locator is missing: {TRACE_SCALE_LOCATOR}"
        ) from error
    if not isinstance(contract, dict):
        raise ArtifactRunnerError("trace-scale contract must be an object")

    declared_contract_hash = contract.get("contract_sha256")
    contract_payload = {
        key: value for key, value in contract.items() if key != "contract_sha256"
    }
    computed_contract_hash = hashlib.sha256(
        _canonical_json_bytes(contract_payload)
    ).hexdigest()
    computed_values_hash = _binary64_tensor_sha256(
        contract.get("values"),
        TRACE_SCALE_SHAPE,
        "trace-scale values",
    )
    declared_hashes = {
        "contract_sha256": declared_contract_hash,
        "values_sha256": contract.get("values_sha256"),
        "raw_variance_sha256": contract.get("raw_variance_sha256"),
    }
    if any(
        not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None
        for value in declared_hashes.values()
    ):
        raise ArtifactRunnerError(
            "trace-scale contract hashes must be 64 lowercase hex"
        )
    if (
        contract.get("contract_id") != TRACE_SCALE_CONTRACT_ID
        or tuple(contract.get("shape", ())) != TRACE_SCALE_SHAPE
        or computed_contract_hash != TRACE_SCALE_CONTRACT_SHA256
        or declared_contract_hash != computed_contract_hash
        or computed_values_hash != TRACE_SCALE_VALUES_SHA256
        or declared_hashes["values_sha256"] != computed_values_hash
        or declared_hashes["raw_variance_sha256"] != TRACE_SCALE_RAW_VARIANCE_SHA256
    ):
        raise ArtifactRunnerError("trace-scale contract identity or hash drifted")
    try:
        hard_interval = contract["normalized_variance_range"]["hard_interval"]
        runtime_dependency = contract["scope"]["runtime_activation_dependency"]
        exact_identity_gate = contract["inactive_guard"]["exact_identity_gate"]
    except (KeyError, TypeError) as error:
        raise ArtifactRunnerError(
            f"trace-scale semantic gate is incomplete: {error}"
        ) from error
    if (
        hard_interval != list(LAYERNORM_REGISTERED_INTERVAL)
        or runtime_dependency != "none"
        or exact_identity_gate is not False
    ):
        raise ArtifactRunnerError("trace-scale semantic gate drifted")

    profile = _load_json(repository_root / PROFILE_PATH)
    try:
        profile_binding = profile["feature_packed_layernorm_override"][
            "trace_scale_contract"
        ]
    except (KeyError, TypeError) as error:
        raise ArtifactRunnerError(
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
        raise ArtifactRunnerError(
            "feature profile trace-scale cross-binding differs from the source contract"
        )
    return provenance


def _profile_pointer_parent(
    document: dict[str, Any],
    pointer: str,
) -> tuple[dict[str, Any], str]:
    if not pointer.startswith("/") or pointer == "/":
        raise ArtifactRunnerError(
            f"profile transition JSON pointer is malformed: {pointer!r}"
        )
    tokens = [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    ]
    if any(not token for token in tokens):
        raise ArtifactRunnerError(
            f"profile transition JSON pointer is malformed: {pointer!r}"
        )
    current: Any = document
    for token in tokens[:-1]:
        if not isinstance(current, dict) or token not in current:
            raise ArtifactRunnerError(
                f"profile transition JSON pointer is missing: {pointer}"
            )
        current = current[token]
    if not isinstance(current, dict):
        raise ArtifactRunnerError(
            f"profile transition JSON pointer parent is not an object: {pointer}"
        )
    return current, tokens[-1]


def _profile_immutable_projection_sha256(profile: Any) -> str:
    if not isinstance(profile, dict):
        raise ArtifactRunnerError("feature profile must be a JSON object")
    projection = json.loads(_canonical_json_bytes(profile).decode("utf-8"))
    for pointer in PROFILE_TRANSITION_ALLOWED_JSON_POINTERS:
        parent, key = _profile_pointer_parent(projection, pointer)
        parent.pop(key, None)
    return hashlib.sha256(_canonical_json_bytes(projection)).hexdigest()


def _validate_profile_transition(
    value: Any,
    preseal_profile_record: dict[str, Any],
) -> dict[str, Any]:
    transition = _exact3_require_keys(
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
        raise ArtifactRunnerError(
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
    profile = _load_json(repository_root / PROFILE_PATH)
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
        raise ArtifactRunnerError(
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
            raise ArtifactRunnerError(
                "feature profile metadata schedule is not sealed: "
                f"{label} expected={expected!r} actual={actual!r}"
            )
    seal = _exact3_require_keys(
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
        raise ArtifactRunnerError(
            "feature profile exact3 evidence differs from the approved evidence"
        )
    preseal = _exact3_require_keys(
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
        raise ArtifactRunnerError(
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
    keys = {
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
    if not isinstance(value, dict) or set(value) != keys:
        raise ArtifactRunnerError(f"{label} keys differ from the exact-three contract")
    run_id = value["run_id"]
    if (
        not isinstance(run_id, str)
        or RUN_ID_PATTERN.fullmatch(run_id) is None
        or run_id == LEGACY_R16_CALIBRATION_EVIDENCE["run_id"]
    ):
        raise ArtifactRunnerError(
            f"{label}.run_id is missing, malformed, or retired r16"
        )
    expected_relative_path = f"results/openfhe/{run_id}"
    if value["relative_path"] != expected_relative_path:
        raise ArtifactRunnerError(
            f"{label}.relative_path must be exactly {expected_relative_path!r}"
        )
    for key in ("manifest_sha256", "sha256sums_sha256"):
        digest = value[key]
        if not isinstance(digest, str) or SHA256_PATTERN.fullmatch(digest) is None:
            raise ArtifactRunnerError(f"{label}.{key} must be 64 lowercase hex")
        if digest == LEGACY_R16_CALIBRATION_EVIDENCE[key]:
            raise ArtifactRunnerError(f"{label}.{key} reuses retired r16 evidence")
    expected = {
        "layer_count": 3,
        "artifact_eligible": False,
        "exact3_gate_passed": True,
        "schedule_evidence_eligible": True,
        "formal_schedule_sealed": True,
    }
    for key, expected_value in expected.items():
        if not _typed_equal(value[key], expected_value):
            raise ArtifactRunnerError(
                f"{label}.{key} mismatch: expected={expected_value!r} "
                f"actual={value[key]!r}"
            )
    return dict(value)


def _exact3_require_keys(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ArtifactRunnerError(
            f"{label} keys differ: missing={sorted(expected - actual)} "
            f"extra={sorted(actual - expected)}"
        )
    return value


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


def _exact3_require_value(value: Any, expected: Any, label: str) -> None:
    if not _exact3_typed_equal(value, expected):
        raise ArtifactRunnerError(
            f"{label} mismatch: expected={expected!r} actual={value!r}"
        )


def _exact3_require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise ArtifactRunnerError(f"{label} must be 64 lowercase hex")
    return value


def _exact3_require_relative_file_path(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ArtifactRunnerError(f"{label} must be a relative POSIX path")
    raw = Path(value)
    if (
        not value
        or raw.is_absolute()
        or "\\" in value
        or ".." in raw.parts
        or raw.as_posix() != value
    ):
        raise ArtifactRunnerError(f"{label} must be a canonical relative POSIX path")
    return value


def _exact3_provenance_file_record(value: Any, label: str) -> dict[str, Any]:
    record = _exact3_require_keys(
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
            raise ArtifactRunnerError(f"{label}.{name} must be a nonnegative integer")
    if (
        record["inode"] == 0
        or record["link_count"] != 1
        or not stat.S_ISREG(record["mode"])
        or stat.S_ISLNK(record["mode"])
    ):
        raise ArtifactRunnerError(
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
        raise ArtifactRunnerError(
            f"cannot inspect live exact3 provenance file {expected_path}: {error}"
        ) from error
    if (
        resolved != path
        or not stat.S_ISREG(result.st_mode)
        or stat.S_ISLNK(result.st_mode)
        or result.st_nlink != 1
    ):
        raise ArtifactRunnerError(
            "live exact3 provenance file must be ordinary, non-symlink, and "
            f"singly linked: {expected_path}"
        )
    _exact3_require_value(result.st_size, record["size_bytes"], f"{label}.size_bytes")
    _exact3_require_value(_sha256(path), record["sha256"], f"{label}.sha256")
    if require_executable and not os.access(path, os.X_OK):
        raise ArtifactRunnerError(
            f"live exact3 provenance executable is not executable: {expected_path}"
        )


def _exact3_bundle_files(
    evidence: dict[str, object],
    repository_root: Path,
) -> tuple[Path, dict[str, Path]]:
    try:
        resolved_repository = repository_root.resolve(strict=True)
    except OSError as error:
        raise ArtifactRunnerError(
            f"cannot resolve exact3 repository root: {error}"
        ) from error
    bundle_root = resolved_repository / str(evidence["relative_path"])
    expected_parent = resolved_repository / "results" / "openfhe"
    try:
        root_stat = bundle_root.lstat()
        resolved_bundle = bundle_root.resolve(strict=True)
    except OSError as error:
        raise ArtifactRunnerError(
            f"cannot open approved exact3 evidence directory: {error}"
        ) from error
    if (
        bundle_root.parent != expected_parent
        or bundle_root.name != evidence["run_id"]
        or resolved_bundle != bundle_root
        or not stat.S_ISDIR(root_stat.st_mode)
        or stat.S_ISLNK(root_stat.st_mode)
    ):
        raise ArtifactRunnerError(
            "approved exact3 evidence path must be one canonical non-symlink "
            "results/openfhe/<run_id> directory"
        )
    try:
        entries = {entry.name: entry for entry in bundle_root.iterdir()}
    except OSError as error:
        raise ArtifactRunnerError(
            f"cannot inspect approved exact3 evidence directory: {error}"
        ) from error
    expected_inventory = set(EXACT3_BUNDLE_PATHS)
    if set(entries) != expected_inventory:
        raise ArtifactRunnerError(
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
            raise ArtifactRunnerError(
                f"cannot inspect exact3 evidence file {name}: {error}"
            ) from error
        if (
            resolved != path
            or not stat.S_ISREG(result.st_mode)
            or stat.S_ISLNK(result.st_mode)
            or result.st_nlink != 1
        ):
            raise ArtifactRunnerError(
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
    if _sha256(checksum_path) != approved_sha256:
        raise ArtifactRunnerError(
            "exact3 SHA256SUMS hash differs from the approved evidence object"
        )
    try:
        data = checksum_path.read_bytes()
        text = data.decode("ascii")
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(f"cannot read exact3 SHA256SUMS: {error}") from error
    if not data.endswith(b"\n") or b"\r" in data or b"\0" in data:
        raise ArtifactRunnerError(
            "exact3 SHA256SUMS must be NUL-free ASCII with final LF"
        )
    lines = text[:-1].split("\n")
    if len(lines) != len(EXACT3_CHECKSUM_PATHS) or any(not line for line in lines):
        raise ArtifactRunnerError("exact3 SHA256SUMS must contain exactly five lines")
    entries: dict[str, str] = {}
    ordered_names: list[str] = []
    for line in lines:
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9._+-]+)", line)
        if match is None:
            raise ArtifactRunnerError(f"malformed exact3 SHA256SUMS line: {line!r}")
        digest, name = match.groups()
        if name in entries:
            raise ArtifactRunnerError(f"duplicate exact3 SHA256SUMS entry: {name}")
        entries[name] = digest
        ordered_names.append(name)
    if tuple(ordered_names) != EXACT3_CHECKSUM_PATHS:
        raise ArtifactRunnerError(
            "exact3 SHA256SUMS paths/order must be stdout.log, stderr.log, "
            "time.log, metrics.csv, manifest.json"
        )
    for name, digest in entries.items():
        if _sha256(files[name]) != digest:
            raise ArtifactRunnerError(f"exact3 SHA256SUMS digest mismatch for {name}")
    return entries


def _exact3_load_json_line(text: str, label: str) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ArtifactRunnerError(f"{label} repeats JSON key {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ArtifactRunnerError(f"{label} contains non-finite JSON number {token}")

    def parse_finite_float(token: str) -> float:
        value = float(token)
        if not math.isfinite(value):
            raise ArtifactRunnerError(
                f"{label} contains non-finite JSON number {token}"
            )
        return value

    try:
        value = json.loads(
            text,
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
            parse_float=parse_finite_float,
        )
    except json.JSONDecodeError as error:
        raise ArtifactRunnerError(f"{label} is malformed JSON: {error}") from error
    if not isinstance(value, dict):
        raise ArtifactRunnerError(f"{label} must be a JSON object")
    return value


def _exact3_validate_range_map(
    value: Any,
    intervals: dict[str, tuple[float, float]],
    label: str,
) -> dict[str, list[float]]:
    record = _exact3_require_keys(value, set(intervals), label)
    validated: dict[str, list[float]] = {}
    for name, interval in intervals.items():
        observed = record[name]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ArtifactRunnerError(f"{label}.{name} must be [minimum,maximum]")
        minimum = _finite_number(
            observed[0],
            f"{label}.{name}[0]",
            interval[0],
            interval[1],
        )
        maximum = _finite_number(
            observed[1],
            f"{label}.{name}[1]",
            interval[0],
            interval[1],
        )
        if minimum > maximum:
            raise ArtifactRunnerError(f"{label}.{name} minimum exceeds maximum")
        validated[name] = [minimum, maximum]
    return validated


def _exact3_expected_counts(layer_id: int) -> dict[str, int]:
    return {
        key: EXPECTED_LAYER_COUNTS[key] * (layer_id + 1)
        + EXPECTED_REFRESH_COUNTS[key] * min(layer_id + 1, 2)
        for key in EXPECTED_LAYER_COUNTS
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
    layer = _exact3_require_keys(
        value,
        LAYER_RECORD_KEYS | diagnostic_keys,
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
        raise ArtifactRunnerError("exact3 metadata schedule contains an unsealed tuple")
    for name, expected in metadata_expected.items():
        assert expected is not None
        _validate_metadata(
            layer[name],
            expected,
            f"exact3 stdout layer {layer_id}.{name}",
        )
    used = {
        "input": _used_level(layer["input_metadata"], "exact3 input metadata"),
        "softmax_denominator": _used_level(
            layer["softmax_denominator_metadata"],
            "exact3 softmax metadata",
        ),
        "attention_output": _used_level(
            layer["attention_output_metadata"],
            "exact3 attention metadata",
        ),
        "ln1_variance": _used_level(
            layer["ln1_variance_metadata"],
            "exact3 ln1 variance metadata",
        ),
        "ln1_output": _used_level(
            layer["ln1_output_metadata"],
            "exact3 ln1 output metadata",
        ),
        "ffn_output": _used_level(
            layer["ffn_output_metadata"],
            "exact3 ffn metadata",
        ),
        "ln2_variance": _used_level(
            layer["ln2_variance_metadata"],
            "exact3 ln2 variance metadata",
        ),
        "raw_output": _used_level(
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
        else _used_level(
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
        _validate_quality(layer[name], f"exact3 stdout layer {layer_id}.{name}")
    _finite_number(
        layer["inactive_max_abs"],
        f"exact3 stdout layer {layer_id}.inactive_max_abs",
        0.0,
        M5_PROTOTYPE_INACTIVE_MAX_ABS,
    )
    sentinel_maximum = _finite_number(
        layer["inactive_sentinel_max_error"],
        f"exact3 stdout layer {layer_id}.inactive_sentinel_max_error",
        0.0,
        math.inf,
    )
    sentinel_ranges = _exact3_validate_range_map(
        layer["inactive_polynomial_sentinel_ranges"],
        INACTIVE_SENTINEL_INTERVALS,
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
        raise ArtifactRunnerError(
            f"exact3 stdout layer {layer_id} inactive sentinel maximum is inconsistent"
        )
    _exact3_validate_range_map(
        layer["encrypted_polynomial_input_ranges"],
        FROZEN_RANGES,
        f"exact3 stdout layer {layer_id}.encrypted_polynomial_input_ranges",
    )
    _validate_counts(
        layer["refresh_operation_counts"],
        EXPECTED_REFRESH_COUNTS if layer_id < 2 else ZERO_COUNTS,
        f"exact3 stdout layer {layer_id}.refresh_operation_counts",
    )
    _validate_counts(
        layer["layer_operation_counts"],
        EXPECTED_LAYER_COUNTS,
        f"exact3 stdout layer {layer_id}.layer_operation_counts",
    )
    _validate_counts(
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
    summary = _exact3_require_keys(
        value,
        FINAL_RECORD_KEYS | diagnostic_keys,
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
        "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
        "max_observed_level": EXPECTED_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": EXPECTED_MAX_POLYNOMIAL_DEPTH,
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
        _finite_number(
            summary[name],
            f"exact3 stdout final summary.{name}",
            0.0,
            math.inf,
        )
    _finite_number(
        summary["server_online_diagnostic_ms"],
        "exact3 stdout final summary.server_online_diagnostic_ms",
        1e-12,
        math.inf,
    )
    quality = {
        "relative_l2": summary["relative_l2"],
        "cosine": summary["cosine"],
        "max_absolute": summary["max_absolute"],
    }
    _validate_quality(quality, "exact3 stdout final summary quality")
    _exact3_require_value(
        quality,
        layers[-1]["output_quality"],
        "exact3 stdout final summary quality",
    )
    _finite_number(
        summary["inactive_max_abs"],
        "exact3 stdout final summary.inactive_max_abs",
        0.0,
        M5_PROTOTYPE_INACTIVE_MAX_ABS,
    )
    _exact3_require_value(
        summary["inactive_max_abs"],
        max(layer["inactive_max_abs"] for layer in layers),
        "exact3 stdout final summary.inactive_max_abs",
    )
    _finite_number(
        summary["inactive_sentinel_max_error"],
        "exact3 stdout final summary.inactive_sentinel_max_error",
        0.0,
        math.inf,
    )
    _exact3_require_value(
        summary["inactive_sentinel_max_error"],
        max(layer["inactive_sentinel_max_error"] for layer in layers),
        "exact3 stdout final summary.inactive_sentinel_max_error",
    )
    _validate_metadata(
        summary["final_metadata"],
        RAW_OUTPUT_METADATA,
        "exact3 stdout final summary.final_metadata",
    )
    _exact3_require_value(
        summary["final_metadata"],
        layers[-1]["raw_output_metadata"],
        "exact3 stdout final summary.final_metadata",
    )
    _validate_counts(
        summary["operation_counts"],
        EXACT3_FINAL_COUNTS,
        "exact3 stdout final summary.operation_counts",
    )
    peak = summary["peak_rss_bytes"]
    if not isinstance(peak, int) or isinstance(peak, bool) or peak <= 0:
        raise ArtifactRunnerError(
            "exact3 stdout final summary.peak_rss_bytes must be a positive integer"
        )
    return summary


def _exact3_parse_stdout(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data = path.read_bytes()
    maximum = int(EXACT3_GATE_CONTRACT["raw_input_bounds"]["maximum_bytes"]["stdout.log"])
    if not data or len(data) > maximum:
        raise ArtifactRunnerError("exact3 stdout.log size is outside the frozen bound")
    if not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise ArtifactRunnerError(
            "exact3 stdout.log must be NUL-free UTF-8 with LF line endings"
        )
    lines_bytes = data[:-1].split(b"\n")
    if len(lines_bytes) != 5 or any(not line for line in lines_bytes):
        raise ArtifactRunnerError(
            "exact3 stdout.log must contain one warning and exactly four JSON lines"
        )
    maximum_line = int(
        EXACT3_GATE_CONTRACT["raw_input_bounds"]["stdout_maximum_line_bytes"]
    )
    if any(len(line) > maximum_line for line in lines_bytes):
        raise ArtifactRunnerError("exact3 stdout.log contains an over-sized line")
    try:
        lines = [line.decode("utf-8") for line in lines_bytes]
    except UnicodeDecodeError as error:
        raise ArtifactRunnerError("exact3 stdout.log is not UTF-8") from error
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
        raise ArtifactRunnerError("exact3 time.log size is outside the frozen bound")
    if not data.endswith(b"\n") or b"\r" in data or b"\x00" in data:
        raise ArtifactRunnerError(
            "exact3 time.log must be NUL-free UTF-8 with LF line endings"
        )
    try:
        lines = data.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise ArtifactRunnerError("exact3 time.log is not UTF-8") from error
    if len(lines) > 64:
        raise ArtifactRunnerError("exact3 time.log contains too many lines")
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
                    raise ArtifactRunnerError(
                        f"exact3 time.log repeats GNU time field {key!r}"
                    )
                fields[key] = stripped[len(prefix) :]
    missing = [key for key, value in fields.items() if value is None]
    if missing:
        raise ArtifactRunnerError(
            f"exact3 time.log is missing GNU time fields: {missing!r}"
        )
    command_field = fields["Command being timed"]
    assert command_field is not None
    if len(command_field) < 2 or not (
        command_field.startswith('"') and command_field.endswith('"')
    ):
        raise ArtifactRunnerError(
            "exact3 GNU time command must use its quoted verbose format"
        )
    command_text = command_field[1:-1]
    try:
        command_argv = shlex.split(command_text)
    except ValueError as error:
        raise ArtifactRunnerError(
            "exact3 GNU time command is not shell-parseable"
        ) from error
    expected_argv = [
        f"./{EXACT3_EXECUTABLE_PATH}",
        "--data-root",
        str(repository_root.resolve(strict=True) / "data"),
        "--diagnostic-layer-count",
        "3",
    ]
    _exact3_require_value(
        command_argv,
        expected_argv,
        "exact3 GNU time command",
    )
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
        raise ArtifactRunnerError(
            "exact3 GNU time integer field is malformed"
        ) from error
    if peak_rss_kib <= 0 or swaps != 0 or exit_status != 0:
        raise ArtifactRunnerError(
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
        *EXPECTED_LAYER_COUNTS,
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
        raise ArtifactRunnerError("exact3 stderr.log must be byte-empty")
    layers, summary = _exact3_parse_stdout(files["stdout.log"])
    timing = _exact3_parse_time(files["time.log"], repository_root)
    expected_metrics = _exact3_metrics_bytes(layers, summary, timing)
    if files["metrics.csv"].read_bytes() != expected_metrics:
        raise ArtifactRunnerError(
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


def _exact3_validate_timestamp_semantics(
    value: Any,
    run_id: str,
) -> None:
    record = _exact3_require_keys(
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
        value = record[key]
        if not isinstance(value, str):
            raise ArtifactRunnerError(
                f"exact3 manifest.timestamp_semantics.{key} must be a timestamp"
            )
        try:
            parsed_execution = datetime.fromisoformat(value)
        except ValueError as error:
            raise ArtifactRunnerError(
                f"exact3 manifest.timestamp_semantics.{key} is malformed"
            ) from error
        if (
            parsed_execution.tzinfo is None
            or parsed_execution.utcoffset() is None
        ):
            raise ArtifactRunnerError(
                f"exact3 manifest.timestamp_semantics.{key} must be timezone-aware"
            )
        execution_times.append(parsed_execution)
    if execution_times[1] < execution_times[0]:
        raise ArtifactRunnerError("exact3 execution_finished_at precedes started_at")
    sealed_at = record["sealed_at"]
    if not isinstance(sealed_at, str):
        raise ArtifactRunnerError(
            "exact3 manifest.timestamp_semantics.sealed_at must be a timestamp"
        )
    try:
        parsed = datetime.fromisoformat(sealed_at)
    except ValueError as error:
        raise ArtifactRunnerError(
            "exact3 manifest.timestamp_semantics.sealed_at is malformed"
        ) from error
    if parsed.tzinfo is None or parsed.utcoffset() is None or parsed.microsecond != 0:
        raise ArtifactRunnerError(
            "exact3 sealed_at must be timezone-aware and rounded to whole seconds"
        )


def _exact3_validate_git(value: Any) -> None:
    record = _exact3_require_keys(
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
        raise ArtifactRunnerError("exact3 manifest.git.head is malformed")
    _exact3_require_value(
        record["branch"],
        BRANCH,
        "exact3 manifest.git.branch",
    )
    _exact3_require_value(
        record["detached"],
        False,
        "exact3 manifest.git.detached",
    )
    if not isinstance(record["clean"], bool):
        raise ArtifactRunnerError("exact3 manifest.git.clean must be boolean")
    _exact3_require_value(
        record["snapshot_semantics"],
        EXACT3_GIT_SNAPSHOT_SEMANTICS,
        "exact3 manifest.git.snapshot_semantics",
    )
    _exact3_require_sha256(
        record["status_porcelain_v1_z_sha256"],
        "exact3 manifest.git.status_porcelain_v1_z_sha256",
    )
    tracked = _exact3_require_keys(
        record["tracked_diff"],
        {"command", "sha256", "size_bytes"},
        "exact3 manifest.git.tracked_diff",
    )
    _exact3_require_value(
        tracked["command"],
        "git diff --binary HEAD -- .",
        "exact3 manifest.git.tracked_diff.command",
    )
    _exact3_require_sha256(
        tracked["sha256"],
        "exact3 manifest.git.tracked_diff.sha256",
    )
    if (
        not isinstance(tracked["size_bytes"], int)
        or isinstance(tracked["size_bytes"], bool)
        or tracked["size_bytes"] < 0
    ):
        raise ArtifactRunnerError(
            "exact3 manifest.git.tracked_diff.size_bytes must be nonnegative"
        )
    untracked = _exact3_require_keys(
        record["untracked_source_manifest"],
        {"selection", "canonicalization", "sha256", "entries"},
        "exact3 manifest.git.untracked_source_manifest",
    )
    _exact3_require_value(
        untracked["selection"],
        EXACT3_GIT_UNTRACKED_SELECTION,
        "exact3 manifest.git.untracked_source_manifest.selection",
    )
    _exact3_require_value(
        untracked["canonicalization"],
        EXACT3_GIT_CANONICALIZATION,
        "exact3 manifest.git.untracked_source_manifest.canonicalization",
    )
    entries_value = untracked["entries"]
    if not isinstance(entries_value, list):
        raise ArtifactRunnerError(
            "exact3 manifest.git.untracked_source_manifest.entries must be an array"
        )
    entries = [
        _exact3_provenance_file_record(
            item,
            f"exact3 manifest.git.untracked_source_manifest.entries[{index}]",
        )
        for index, item in enumerate(entries_value)
    ]
    paths = [str(item["path"]) for item in entries]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ArtifactRunnerError(
            "exact3 manifest.git untracked source paths must be sorted and unique"
        )
    expected_untracked_hash = hashlib.sha256(
        _canonical_json_bytes(entries)
    ).hexdigest()
    _exact3_require_value(
        untracked["sha256"],
        expected_untracked_hash,
        "exact3 manifest.git.untracked_source_manifest.sha256",
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
        _exact3_require_value(
            entries,
            [],
            "clean exact3 untracked source entries",
        )


def _exact3_validate_provenance(
    provenance: Any,
    repository_root: Path,
    evidence: dict[str, object],
) -> None:
    record = _exact3_require_keys(
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
            raise ArtifactRunnerError(
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
            raise ArtifactRunnerError(
                f"exact3 manifest.provenance.{collection_name} paths duplicate"
            )
        expected_paths = (
            EXACT3_SOURCE_PATHS
            if collection_name == "source_files"
            else EXACT3_CONFIG_PATHS
        )
        if tuple(paths) != expected_paths:
            raise ArtifactRunnerError(
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
        expected_digest = hashlib.sha256(_canonical_json_bytes(items)).hexdigest()
        _exact3_require_value(
            record[digest_key],
            expected_digest,
            f"exact3 manifest.provenance.{digest_key}",
        )
    if preseal_profile_record is None:  # pragma: no cover - path tuple is frozen
        raise ArtifactRunnerError("exact3 profile config provenance is missing")
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
        raise ArtifactRunnerError(
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
    raw_logs = _exact3_require_keys(
        manifest["raw_logs"],
        {"stdout.log", "stderr.log", "time.log"},
        "exact3 manifest.raw_logs",
    )
    for name in ("stdout.log", "stderr.log", "time.log"):
        record = _exact3_require_keys(
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
                raise ArtifactRunnerError(
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
    derived = _exact3_require_keys(
        manifest["derived_artifacts"],
        {"metrics.csv"},
        "exact3 manifest.derived_artifacts",
    )
    metrics = _exact3_require_keys(
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
        raise ArtifactRunnerError(
            "exact3 manifest.timing.gnu_time_elapsed_text must be a string"
        )
    parts = value.split(":")
    if len(parts) not in (2, 3) or any(not part for part in parts):
        raise ArtifactRunnerError(
            "exact3 manifest.timing.gnu_time_elapsed_text is malformed"
        )
    try:
        seconds = float(parts[-1])
        minutes = int(parts[-2])
        hours = int(parts[0]) if len(parts) == 3 else 0
    except ValueError as error:
        raise ArtifactRunnerError(
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
        raise ArtifactRunnerError(
            "exact3 manifest.timing.gnu_time_elapsed_text is out of range"
        )
    elapsed = hours * 3600 + minutes * 60 + seconds
    if elapsed <= 0 or not math.isfinite(elapsed):
        raise ArtifactRunnerError(
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
    command = _exact3_require_keys(
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
        raise ArtifactRunnerError("exact3 manifest.command.argv drifted")
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
        raise ArtifactRunnerError(
            "exact3 manifest.command.argv does not identify its private staging time.log"
        )
    timing = _exact3_require_keys(
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
        raise ArtifactRunnerError(
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
    evidence = _exact3_require_keys(
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
        raise ArtifactRunnerError(
            "exact3 manifest.evidence.layers must contain 3 layers"
        )
    for layer_id, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ArtifactRunnerError(
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
                raise ArtifactRunnerError(
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
        raise ArtifactRunnerError(
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
            raise ArtifactRunnerError(
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
    record = _exact3_require_keys(
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
    profile = _exact3_require_keys(
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
    if _sha256(files["manifest.json"]) != evidence["manifest_sha256"]:
        raise ArtifactRunnerError(
            "exact3 manifest hash differs from the approved evidence object"
        )
    checksum_entries = _exact3_checksum_entries(
        files["SHA256SUMS"],
        str(evidence["sha256sums_sha256"]),
        files,
    )
    if checksum_entries["manifest.json"] != evidence["manifest_sha256"]:
        raise ArtifactRunnerError(
            "exact3 manifest hash differs between SHA256SUMS and approved evidence"
        )
    manifest = _load_json(files["manifest.json"])
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
        raise ArtifactRunnerError(
            "approved exact-three-layer live OpenFHE evidence is missing"
        )
    evidence = _validate_exact3_evidence(EXACT3_EVIDENCE, "EXACT3_EVIDENCE")
    _require_profile_schedule_sealed(repository_root, evidence)
    return _verify_exact3_evidence_bundle(evidence, repository_root)


def _sealed_static_contracts() -> tuple[dict[str, str], dict[str, object]]:
    evidence = _require_exact3_evidence()
    provenance = _trace_scale_contract_provenance()
    return provenance, evidence


def _resolve_repository_file(path: Path, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(REPO_ROOT.resolve())
    except ValueError as error:
        raise ArtifactRunnerError(
            f"{label} must stay inside {REPO_ROOT}: {path}"
        ) from error
    if not resolved.is_file():
        raise ArtifactRunnerError(f"{label} is not a file: {resolved}")
    return resolved


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _resolve_m5_executable(path: Path, *, require_exists: bool = True) -> Path:
    candidate = _lexical_absolute(path)
    expected = _lexical_absolute(DEFAULT_EXECUTABLE)
    if candidate != expected:
        raise ArtifactRunnerError(
            f"M5 executable must be the fixed build target {DEFAULT_EXECUTABLE}"
        )
    try:
        relative_parts = candidate.relative_to(REPO_ROOT).parts
    except ValueError as error:  # pragma: no cover - fixed target is repository-local
        raise ArtifactRunnerError(
            f"M5 executable must stay inside {REPO_ROOT}"
        ) from error
    cursor = REPO_ROOT
    for part in relative_parts:
        cursor /= part
        if cursor.is_symlink():
            raise ArtifactRunnerError(
                f"M5 executable path must not contain symlinks: {cursor}"
            )
    if not require_exists:
        if candidate.exists() and not candidate.is_file():
            raise ArtifactRunnerError(
                f"M5 executable is not a regular file: {candidate}"
            )
        return candidate
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise ArtifactRunnerError(
            f"M5 executable is not an executable file: {candidate}"
        )
    return candidate


def _resolve_output_root(path: Path) -> Path:
    resolved = path.resolve()
    allowed = DEFAULT_OUTPUT_ROOT.resolve()
    if resolved != allowed:
        raise ArtifactRunnerError(f"M5 output root must be exactly {allowed}: {path}")
    return resolved


def _command_record(
    command: list[str],
    returncode: int,
    started: str | None = None,
    finished: str | None = None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "command": shlex.join(command),
        "cwd": str(REPO_ROOT),
        "exit_code": returncode,
        "phase": "artifact_generation",
    }
    if started is not None and finished is not None:
        record["started_at"] = started
        record["finished_at"] = finished
    return record


def _run_git(arguments: list[str]) -> tuple[str, dict[str, object]]:
    command = [str(SYSTEM_GIT), *arguments]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_git_environment(),
        stdin=subprocess.DEVNULL,
    )
    record = _command_record(command, completed.returncode)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"{shlex.join(command)} failed: {detail}")
    return completed.stdout.strip(), record


def _preflight_git() -> GitState:
    commands: list[dict[str, object]] = []
    status, record = _run_git(["status", "--porcelain=v1", "--untracked-files=normal"])
    commands.append(record)
    if status:
        raise ArtifactRunnerError(
            "repository is not clean; refusing to run M5 evidence"
        )

    branch, record = _run_git(["branch", "--show-current"])
    commands.append(record)
    if branch != BRANCH:
        raise ArtifactRunnerError(f"branch must be {BRANCH}, got {branch!r}")

    head, record = _run_git(["rev-parse", "HEAD"])
    commands.append(record)
    if SHA_PATTERN.fullmatch(head) is None:
        raise ArtifactRunnerError(f"local HEAD is not a full Git SHA: {head!r}")

    tracking_head, record = _run_git(["rev-parse", TRACKING_REF])
    commands.append(record)
    if SHA_PATTERN.fullmatch(tracking_head) is None:
        raise ArtifactRunnerError(
            f"tracking ref is not a full Git SHA: {tracking_head!r}"
        )

    remote_url, record = _run_git(
        ["config", "--local", "--get-all", f"remote.{REMOTE_NAME}.url"]
    )
    commands.append(record)
    if remote_url.splitlines() != [REMOTE_URL]:
        raise ArtifactRunnerError(
            f"{REMOTE_NAME} URL must be exactly {REMOTE_URL!r}, got {remote_url!r}"
        )

    remote_output, record = _run_git(
        ["ls-remote", "--exit-code", REMOTE_URL, REMOTE_REF]
    )
    commands.append(record)
    remote_lines = [line.split() for line in remote_output.splitlines() if line.strip()]
    if len(remote_lines) != 1 or len(remote_lines[0]) < 2:
        raise ArtifactRunnerError(
            f"remote ref lookup was not unique: {remote_output!r}"
        )
    remote_head, remote_ref = remote_lines[0][0], remote_lines[0][1]
    if SHA_PATTERN.fullmatch(remote_head) is None or remote_ref != REMOTE_REF:
        raise ArtifactRunnerError(f"remote ref lookup was malformed: {remote_output!r}")
    if len({head, tracking_head, remote_head}) != 1:
        raise ArtifactRunnerError(
            "local, tracking, and live remote commits differ: "
            f"local={head} tracking={tracking_head} remote={remote_head}"
        )
    return GitState(head, tracking_head, remote_url, remote_head, tuple(commands))


def _read_openfhe_version(openfhe_prefix: Path) -> str:
    version_file = openfhe_prefix / "lib" / "OpenFHE" / "OpenFHEConfigVersion.cmake"
    try:
        contents = version_file.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(
            f"cannot read OpenFHE version file: {error}"
        ) from error
    match = re.search(
        r'^set\(PACKAGE_VERSION "([0-9]+\.[0-9]+\.[0-9]+)"\)$', contents, re.M
    )
    if match is None or match.group(1) != "1.5.1":
        raise ArtifactRunnerError(
            f"OpenFHE package version must be 1.5.1: {version_file}"
        )
    return match.group(1)


def _require_frozen_openfhe_prefix(openfhe_prefix: Path) -> Path:
    prefix = Path(os.path.abspath(os.fspath(openfhe_prefix)))
    expected = DEFAULT_OPENFHE_PREFIX
    if prefix != expected or prefix.resolve() != expected:
        raise ArtifactRunnerError(
            f"OpenFHE prefix must be the canonical path {expected}, got {openfhe_prefix}"
        )
    return prefix


def _configure_command(build_root: Path, openfhe_prefix: Path) -> list[str]:
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    if Path(os.path.abspath(os.fspath(build_root))) != DEFAULT_EXECUTABLE.parent:
        raise ArtifactRunnerError(
            f"M5 build root must be {DEFAULT_EXECUTABLE.parent}, got {build_root}"
        )
    return [
        str(SYSTEM_ENV),
        *(f"--unset={name}" for name in CONFIGURE_ENV_UNSET),
        str(SYSTEM_CMAKE),
        "--fresh",
        "-S",
        str(REPO_ROOT),
        "-B",
        str(DEFAULT_EXECUTABLE.parent),
        "-G",
        "Unix Makefiles",
        f"-DOpenFHE_DIR:PATH={prefix / 'lib' / 'OpenFHE'}",
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


def _parse_cmake_cache(cache_path: Path) -> dict[str, tuple[str, str]]:
    if (
        cache_path.is_symlink()
        or not cache_path.is_file()
        or cache_path.resolve() != cache_path
    ):
        raise ArtifactRunnerError(
            f"M5 CMake cache must be a regular file: {cache_path}"
        )
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as error:
        raise ArtifactRunnerError(f"cannot read M5 CMake cache: {error}") from error
    entries: dict[str, tuple[str, str]] = {}
    for line in lines:
        if not line or line.startswith(("//", "#")) or "=" not in line:
            continue
        key_and_type, value = line.split("=", maxsplit=1)
        if ":" not in key_and_type:
            continue
        key, entry_type = key_and_type.rsplit(":", maxsplit=1)
        if key in entries:
            raise ArtifactRunnerError(f"M5 CMake cache repeats key {key}")
        entries[key] = (entry_type, value)
    return entries


def _verify_build_configuration(
    build_root: Path,
    openfhe_prefix: Path,
) -> dict[str, object]:
    if Path(os.path.abspath(os.fspath(build_root))) != DEFAULT_EXECUTABLE.parent:
        raise ArtifactRunnerError(
            f"M5 build root must be {DEFAULT_EXECUTABLE.parent}, got {build_root}"
        )
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    cache_path = build_root / "CMakeCache.txt"
    entries = _parse_cmake_cache(cache_path)
    expected_entries = {
        "CMAKE_GENERATOR": ("INTERNAL", "Unix Makefiles"),
        "CMAKE_HOME_DIRECTORY": ("INTERNAL", str(REPO_ROOT)),
        "CMAKE_BUILD_TYPE": ("STRING", "Release"),
        "BUILD_TESTING": ("BOOL", "ON"),
        "OpenFHE_DIR": ("PATH", str(prefix / "lib" / "OpenFHE")),
        # CMake normalizes this known cache entry from the command-line
        # FILEPATH hint to STRING while project(... LANGUAGES CXX) initializes.
        "CMAKE_CXX_COMPILER": ("STRING", str(SYSTEM_CXX)),
        "CMAKE_MAKE_PROGRAM": ("FILEPATH", str(SYSTEM_MAKE)),
        "CMAKE_CXX_FLAGS": ("STRING", ""),
        "CMAKE_CXX_FLAGS_RELEASE": ("STRING", "-O3 -DNDEBUG"),
        "CMAKE_EXE_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_EXE_LINKER_FLAGS_RELEASE": ("STRING", ""),
        "CMAKE_SHARED_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_MODULE_LINKER_FLAGS": ("STRING", ""),
        "CMAKE_STATIC_LINKER_FLAGS": ("STRING", ""),
    }
    drift = {
        key: {"expected": expected, "actual": entries.get(key)}
        for key, expected in expected_entries.items()
        if entries.get(key) != expected
    }
    forbidden_nonempty = {
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
    if drift or forbidden_nonempty:
        raise ArtifactRunnerError(
            "M5 CMake cache binding drifted: "
            f"critical={drift} forbidden={forbidden_nonempty}"
        )
    _read_openfhe_version(prefix)
    return {
        "generator": "Unix Makefiles",
        "source_root": str(REPO_ROOT),
        "build_root": str(DEFAULT_EXECUTABLE.parent),
        "openfhe_dir": str(prefix / "lib" / "OpenFHE"),
        "build_type": "Release",
        "build_testing": True,
        "cxx_compiler": str(SYSTEM_CXX),
        "make_program": str(SYSTEM_MAKE),
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
        "cmake_cache": {
            "path": "build-openfhe/CMakeCache.txt",
            "bytes": cache_path.stat().st_size,
            "sha256": _sha256(cache_path),
        },
    }


def _regular_file_record(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file() or path.resolve() != path:
        raise ArtifactRunnerError(f"provenance input must be a regular file: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _openfhe_package_records(openfhe_prefix: Path) -> list[dict[str, object]]:
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    package_root = prefix / "lib" / "OpenFHE"
    return [
        _regular_file_record(package_root / name)
        for name in OPENFHE_CMAKE_PACKAGE_FILES
    ]


def _openfhe_include_tree_record(openfhe_prefix: Path) -> dict[str, object]:
    prefix = _require_frozen_openfhe_prefix(openfhe_prefix)
    root = prefix / "include" / "openfhe"
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        raise ArtifactRunnerError(
            f"OpenFHE include tree must be a canonical directory: {root}"
        )
    files: list[Path] = []
    for candidate in root.rglob("*"):
        if candidate.is_symlink():
            raise ArtifactRunnerError(
                f"OpenFHE include tree must not contain symlinks: {candidate}"
            )
        if candidate.is_file():
            files.append(candidate)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ArtifactRunnerError(f"OpenFHE include tree is empty: {root}")
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
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return {
        "root": str(root),
        "file_count": len(files),
        "bytes": total_bytes,
        "algorithm": "SHA-256",
        "canonicalization": OPENFHE_INCLUDE_TREE_CANONICALIZATION,
        "sha256": digest.hexdigest(),
    }


def _run_checked_command(
    command: list[str], label: str
) -> tuple[dict[str, object], str]:
    started = _timestamp()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_artifact_environment(),
    )
    finished = _timestamp()
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(
            f"{label} failed with {completed.returncode}: {detail[-4000:]}"
        )
    return (
        _command_record(command, completed.returncode, started, finished),
        completed.stdout,
    )


def _require_ctest_passed_tests(
    stdout: str,
    expected_tests: tuple[str, ...],
    label: str,
) -> None:
    summary_pattern = re.compile(
        r"^\s*(\d+)/(\d+)\s+Test\s+#(\d+):\s+([A-Za-z0-9_.+-]+)"
        r"\s+\.+\s+(.+?)\s+[0-9]+(?:\.[0-9]+)?\s+sec\s*$",
        re.MULTILINE,
    )
    summaries = summary_pattern.findall(stdout)
    observed_tests = tuple(name for _, _, _, name, _ in summaries)
    if len(set(observed_tests)) != len(observed_tests):
        raise ArtifactRunnerError(
            f"{label} CTest summary contains duplicate test-name records: "
            f"{observed_tests}"
        )
    test_ids = tuple(int(test_id) for _, _, test_id, _, _ in summaries)
    if len(set(test_ids)) != len(test_ids):
        raise ArtifactRunnerError(
            f"{label} CTest summary contains duplicate test-id records: {test_ids}"
        )
    expected_count = len(expected_tests)
    ordinals = {int(ordinal) for ordinal, _, _, _, _ in summaries}
    totals = {int(total) for _, total, _, _, _ in summaries}
    statuses = tuple(status.strip() for _, _, _, _, status in summaries)
    footer_pattern = re.compile(
        rf"^100% tests passed, 0 tests failed out of {expected_count}$",
        re.MULTILINE,
    )
    footer_count = len(footer_pattern.findall(stdout))
    if (
        set(observed_tests) != set(expected_tests)
        or len(observed_tests) != expected_count
        or ordinals != set(range(1, expected_count + 1))
        or totals != {expected_count}
        or statuses != ("Passed",) * expected_count
        or footer_count != 1
    ):
        missing = [name for name in expected_tests if name not in observed_tests]
        unexpected = [name for name in observed_tests if name not in expected_tests]
        raise ArtifactRunnerError(
            f"{label} CTest did not enumerate and pass the exact frozen test set: "
            f"missing={missing} unexpected={unexpected} "
            f"ordinals={sorted(ordinals)} totals={sorted(totals)} "
            f"statuses={statuses} footer_count={footer_count} "
            f"actual={observed_tests} expected={expected_tests}"
        )


def _inspect_openfhe_linkage(
    executable: Path,
    openfhe_prefix: Path,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    command = [str(SYSTEM_LDD), str(executable)]
    started = _timestamp()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_artifact_environment(),
    )
    finished = _timestamp()
    if completed.returncode != 0:
        raise ArtifactRunnerError(
            f"cannot inspect M5 OpenFHE linkage: {completed.stderr.strip()}"
        )
    linked: dict[str, dict[str, object]] = {}
    for line in completed.stdout.splitlines():
        if "libOPENFHE" not in line:
            continue
        match = re.fullmatch(
            r"\s*(libOPENFHE\S+)\s+=>\s+(\S+)\s+\(0x[0-9a-fA-F]+\)\s*",
            line,
        )
        if match is None:
            raise ArtifactRunnerError(f"malformed M5 OpenFHE ldd line: {line.strip()}")
        soname, target = match.groups()
        if target == "not" or soname not in OPENFHE_LINKED_LIBRARY_SONAMES:
            raise ArtifactRunnerError(
                f"M5 OpenFHE library is unresolved: {line.strip()}"
            )
        if soname in linked:
            raise ArtifactRunnerError(f"M5 OpenFHE SONAME is duplicated: {soname}")
        target_path = Path(target)
        if not target_path.is_absolute():
            raise ArtifactRunnerError(f"M5 OpenFHE ldd path is not absolute: {target}")
        try:
            resolved = target_path.resolve(strict=True)
        except OSError as error:
            raise ArtifactRunnerError(
                f"M5 OpenFHE library cannot be resolved: {target}"
            ) from error
        library_root = _require_frozen_openfhe_prefix(openfhe_prefix) / "lib"
        try:
            resolved.relative_to(library_root)
        except ValueError as error:
            raise ArtifactRunnerError(
                f"M5 executable links OpenFHE outside {library_root}: {resolved}"
            ) from error
        if resolved.is_symlink() or not resolved.is_file():
            raise ArtifactRunnerError(
                f"M5 resolved OpenFHE library is not a regular file: {resolved}"
            )
        if resolved.name != OPENFHE_LINKED_LIBRARY_BASENAMES[soname]:
            raise ArtifactRunnerError(
                f"M5 OpenFHE versioned path drifted for {soname}: {resolved}"
            )
        linked[soname] = {
            "soname": soname,
            **_regular_file_record(resolved),
        }
    if set(linked) != set(OPENFHE_LINKED_LIBRARY_SONAMES):
        raise ArtifactRunnerError(
            "M5 executable must resolve exactly three OpenFHE SONAMEs: "
            f"expected={list(OPENFHE_LINKED_LIBRARY_SONAMES)} "
            f"actual={sorted(linked)}"
        )
    records = [linked[soname] for soname in OPENFHE_LINKED_LIBRARY_SONAMES]
    return _command_record(command, completed.returncode, started, finished), records


def _build_configuration_snapshot(
    build_root: Path,
    executable: Path,
    openfhe_prefix: Path,
    linked_libraries: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    result = _verify_build_configuration(build_root, openfhe_prefix)
    if linked_libraries is None:
        _, linked_libraries = _inspect_openfhe_linkage(executable, openfhe_prefix)
    return {
        **result,
        "openfhe_cmake_package_files": _openfhe_package_records(openfhe_prefix),
        "openfhe_include_tree": _openfhe_include_tree_record(openfhe_prefix),
        "openfhe_linked_libraries": linked_libraries,
    }


def _require_build_configuration(
    expected: dict[str, object],
    build_root: Path,
    executable: Path,
    openfhe_prefix: Path,
    phase: str,
) -> None:
    actual = _build_configuration_snapshot(build_root, executable, openfhe_prefix)
    if actual != expected:
        raise ArtifactRunnerError(
            f"M5 build/OpenFHE provenance changed {phase}: "
            f"expected={expected} actual={actual}"
        )


def _run_m5_preflight(
    build_root: Path, executable: Path, openfhe_prefix: Path
) -> M5Preflight:
    if re.fullmatch(M5_CTEST_PATTERN, "openfhe_encoder_layer_smoke") is None:
        raise ArtifactRunnerError(
            "M5 narrow CTest pattern omits openfhe_encoder_layer_smoke"
        )
    configure_command = _configure_command(build_root, openfhe_prefix)
    build_command = [
        str(SYSTEM_CMAKE),
        "--build",
        str(build_root),
        "--clean-first",
        "-j",
        "4",
    ]
    test_command = [
        str(SYSTEM_CTEST),
        "--test-dir",
        str(build_root),
        "--output-on-failure",
        "--verbose",
        "--no-tests=error",
        "-R",
        M5_CTEST_PATTERN,
    ]
    configure_record, _ = _run_checked_command(
        configure_command,
        "fresh fixed-configuration M5 configure",
    )
    configured = _verify_build_configuration(build_root, openfhe_prefix)
    build_record, _ = _run_checked_command(build_command, "clean-commit M5 build")
    if _verify_build_configuration(build_root, openfhe_prefix) != configured:
        raise ArtifactRunnerError("M5 CMake configuration changed during the build")
    rebuilt = _resolve_m5_executable(executable)
    linkage_record, linked_libraries = _inspect_openfhe_linkage(
        rebuilt,
        openfhe_prefix,
    )
    build_configuration = _build_configuration_snapshot(
        build_root,
        rebuilt,
        openfhe_prefix,
        linked_libraries,
    )
    test_record, test_stdout = _run_checked_command(
        test_command, "clean-commit M5 narrow gates"
    )
    _require_ctest_passed_tests(
        test_stdout,
        M5_CTEST_EXPECTED_TESTS,
        "M5 narrow gates",
    )
    schedule_preflight = _parse_schedule_preflight_output(test_stdout)
    crypto_preflight = _parse_crypto_preflight_output(test_stdout)
    _require_build_configuration(
        build_configuration,
        build_root,
        rebuilt,
        openfhe_prefix,
        "after M5 narrow gates",
    )
    return M5Preflight(
        commands=(configure_record, build_record, linkage_record, test_record),
        crypto_preflight=crypto_preflight,
        build_configuration=build_configuration,
        schedule_preflight=schedule_preflight,
    )


def _version_line(command: list[str], label: str) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_artifact_environment(),
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"cannot identify {label}: {detail}")
    return completed.stdout.strip().splitlines()[0]


def _environment(openfhe_prefix: Path) -> dict[str, object]:
    prefix = openfhe_prefix.resolve()
    if not prefix.is_dir():
        raise ArtifactRunnerError(f"OpenFHE prefix is not a directory: {prefix}")
    release = platform.release()
    return {
        "os": f"{platform.system()} {release}",
        "architecture": platform.machine(),
        "compiler": _version_line([str(SYSTEM_CXX), "--version"], "C++ compiler"),
        "cmake": _version_line([str(SYSTEM_CMAKE), "--version"], "CMake"),
        "python": platform.python_version(),
        "openfhe_version": _read_openfhe_version(prefix),
        "openfhe_prefix": str(prefix),
        "wsl": "microsoft" in release.lower() or "WSL_DISTRO_NAME" in os.environ,
    }


def _repository_record(path: str) -> dict[str, object]:
    resolved = _resolve_repository_file(REPO_ROOT / path, f"input {path}")
    return {
        "path": path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "media_type": "application/json",
        "role": "configuration",
    }


def _trace_input_records(
    data_root: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if data_root.resolve() != DEFAULT_DATA_ROOT.resolve():
        raise ArtifactRunnerError(f"M5 data root must be {DEFAULT_DATA_ROOT.resolve()}")
    contract = _load_json(REPO_ROOT / "config" / "moai_encoder_trace.json")
    try:
        required_files = contract["required_files"]
        layers = contract["layers"]
    except (KeyError, TypeError) as error:
        raise ArtifactRunnerError(
            "encoder trace contract has an invalid shape"
        ) from error
    if not isinstance(required_files, dict) or len(required_files) != FILES_PER_LAYER:
        raise ArtifactRunnerError(
            "M5 trace contract must bind exactly 37 files per layer"
        )
    if not isinstance(layers, list) or [
        item.get("layer_id") for item in layers
    ] != list(range(12)):
        raise ArtifactRunnerError(
            "M5 trace contract must contain ordered layer ids 0..11"
        )

    records: list[dict[str, object]] = []
    identities: list[dict[str, object]] = []
    for layer in layers:
        layer_id = layer["layer_id"]
        expected_hashes = layer.get("sha256")
        if not isinstance(expected_hashes, dict) or set(expected_hashes) != set(
            required_files
        ):
            raise ArtifactRunnerError(f"layer {layer_id} trace hash map drifted")
        weight_bundle: list[dict[str, object]] = []
        trace_bundle: list[dict[str, object]] = []
        for logical_name in sorted(required_files):
            specification = required_files[logical_name]
            if not isinstance(specification, dict) or not isinstance(
                specification.get("path"), str
            ):
                raise ArtifactRunnerError(
                    f"trace file mapping is invalid: {logical_name}"
                )
            contract_path = specification["path"]
            relative = Path("data") / f"layer_{layer_id}" / contract_path
            resolved = _resolve_repository_file(
                REPO_ROOT / relative, f"layer-{layer_id} trace input {logical_name}"
            )
            actual_hash = _sha256(resolved)
            if actual_hash != expected_hashes[logical_name]:
                raise ArtifactRunnerError(
                    f"layer-{layer_id} trace hash mismatch for {logical_name}: "
                    f"expected={expected_hashes[logical_name]} actual={actual_hash}"
                )
            is_weight = "/parms/" in "/" + contract_path
            record = {
                "path": relative.as_posix(),
                "sha256": actual_hash,
                "bytes": resolved.stat().st_size,
                "media_type": "text/csv",
                "role": "weights" if is_weight else "trace",
            }
            records.append(record)
            bundle_record = {key: record[key] for key in ("path", "sha256", "bytes")}
            (weight_bundle if is_weight else trace_bundle).append(bundle_record)
        weight_bundle.sort(key=lambda item: str(item["path"]))
        trace_bundle.sort(key=lambda item: str(item["path"]))
        if (
            len(weight_bundle) != WEIGHT_FILES_PER_LAYER
            or len(trace_bundle) != TRACE_FILES_PER_LAYER
        ):
            raise ArtifactRunnerError(
                f"layer {layer_id} must bind 16 weight and 21 trace files"
            )
        identities.append(
            {
                "layer_id": layer_id,
                "weight_file_count": len(weight_bundle),
                "trace_file_count": len(trace_bundle),
                "weight_bundle_sha256": hashlib.sha256(
                    _canonical_json_bytes(weight_bundle)
                ).hexdigest(),
                "trace_bundle_sha256": hashlib.sha256(
                    _canonical_json_bytes(trace_bundle)
                ).hexdigest(),
            }
        )
    if len(records) != ENCODER_LAYERS * FILES_PER_LAYER:
        raise ArtifactRunnerError("M5 trace input count must be exactly 444")
    return records, identities


def _profile_record() -> dict[str, object]:
    path = _resolve_repository_file(REPO_ROOT / PROFILE_PATH, "feature profile")
    config = _load_json(path)
    if not isinstance(config, dict) or not isinstance(
        config.get("effective_profile"), dict
    ):
        raise ArtifactRunnerError("feature profile lacks /effective_profile")
    acceptance = config.get("m5_prototype_acceptance")
    if acceptance != M5_PROTOTYPE_ACCEPTANCE:
        raise ArtifactRunnerError("M5 prototype acceptance contract drifted")
    effective = config["effective_profile"]
    effective_hash = hashlib.sha256(_canonical_json_bytes(effective)).hexdigest()
    if (
        effective_hash != PROFILE_SHA256
        or config.get("effective_profile_sha256") != PROFILE_SHA256
    ):
        raise ArtifactRunnerError(
            f"feature profile hash drifted: expected={PROFILE_SHA256} actual={effective_hash}"
        )
    return {
        "id": "paper_compat",
        "security_claim": "none",
        "warning": WARNING,
        "source_config_path": PROFILE_PATH,
        "source_config_sha256": _sha256(path),
        "source_config_bytes": path.stat().st_size,
        "effective_profile_locator": PROFILE_LOCATOR,
        "canonicalization": CANONICALIZATION,
        "effective_profile_payload": effective,
        "effective_profile_sha256": effective_hash,
    }


def _require_executable_hash(
    executable: Path, expected_sha256: str, phase: str
) -> None:
    actual = _sha256(executable)
    if actual != expected_sha256:
        raise ArtifactRunnerError(
            f"M5 executable changed {phase}: expected={expected_sha256} actual={actual}"
        )


def _require_input_hashes(inputs: list[dict[str, object]], phase: str) -> None:
    for record in inputs:
        path = record.get("path")
        expected_sha256 = record.get("sha256")
        expected_bytes = record.get("bytes")
        if (
            not isinstance(path, str)
            or not isinstance(expected_sha256, str)
            or not isinstance(expected_bytes, int)
        ):
            raise ArtifactRunnerError(
                f"invalid frozen input record {phase}: {record!r}"
            )
        resolved = _resolve_repository_file(REPO_ROOT / path, f"frozen input {path}")
        actual_sha256 = _sha256(resolved)
        actual_bytes = resolved.stat().st_size
        if actual_sha256 != expected_sha256 or actual_bytes != expected_bytes:
            raise ArtifactRunnerError(
                f"frozen input changed {phase}: {path} expected_sha256={expected_sha256} "
                f"actual_sha256={actual_sha256} expected_bytes={expected_bytes} "
                f"actual_bytes={actual_bytes}"
            )


def _extract_records(stdout: str, test_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    formal_stdout = test_name in FORMAL_RUNTIME_TEST_NAMES

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ValueError(f"non-finite JSON number: {token}")

    for line in stdout.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            candidate = json.loads(
                line[start:],
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
            )
        except (json.JSONDecodeError, ValueError) as error:
            if formal_stdout:
                raise ArtifactRunnerError(
                    "formal M5 stdout contains a malformed JSON-bearing line: "
                    f"{line[:400]!r} ({error})"
                ) from error
            continue
        if not isinstance(candidate, dict):
            if formal_stdout:
                raise ArtifactRunnerError("formal M5 stdout JSON lines must be objects")
            continue
        candidate_test = candidate.get("test")
        if not isinstance(candidate_test, str):
            if formal_stdout:
                raise ArtifactRunnerError(
                    "formal M5 stdout JSON object lacks a string test discriminator"
                )
            continue
        diagnostic_fields = DIAGNOSTIC_ONLY_EVIDENCE_FIELDS.intersection(candidate)
        diagnostic_namespace = candidate_test.startswith(
            DIAGNOSTIC_EVIDENCE_TEST_PREFIXES
        )
        diagnostic_claim_scope = "claim_scope" in candidate and not (
            candidate_test == "openfhe_encoder_12_layer"
            and candidate["claim_scope"] == "m5_12_layer_correctness"
        )
        if diagnostic_namespace or diagnostic_fields or diagnostic_claim_scope:
            raise ArtifactRunnerError(
                "formal evidence stdout contains diagnostic-only JSON: "
                f"test={candidate_test!r} fields={sorted(diagnostic_fields)}"
            )
        allowed_non_evidence_keys = FORMAL_NON_EVIDENCE_JSON_ALLOWLIST.get(
            candidate_test
        )
        if allowed_non_evidence_keys is not None:
            if (
                set(candidate) != allowed_non_evidence_keys
                or candidate.get("security_claim") != "none"
            ):
                raise ArtifactRunnerError(
                    f"allowed non-evidence JSON {candidate_test!r} drifted"
                )
            continue
        if formal_stdout and candidate_test not in FORMAL_RUNTIME_TEST_NAMES:
            raise ArtifactRunnerError(
                "formal M5 stdout contains an unexpected JSON evidence record: "
                f"{candidate_test!r}"
            )
        if candidate_test == test_name:
            records.append(candidate)
    return records


def _finite_number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or not minimum <= value <= maximum
    ):
        raise ArtifactRunnerError(f"{label} violates [{minimum}, {maximum}]")
    return float(value)


def _typed_equal(actual: Any, expected: Any) -> bool:
    """Compare frozen JSON scalars without Python's bool/int equivalence."""
    return type(actual) is type(expected) and actual == expected


def _used_level(value: dict[str, Any], label: str) -> int:
    used = value["level"] + value["noise_scale_degree"] - 1
    if used < 0 or used >= EXPECTED_MULTIPLICATIVE_DEPTH:
        raise ArtifactRunnerError(
            f"{label} used level {used} is outside the depth-47 schedule"
        )
    expected_remaining = EXPECTED_MULTIPLICATIVE_DEPTH - used
    if value["remaining_levels"] != expected_remaining:
        raise ArtifactRunnerError(
            f"{label}.remaining_levels drifted: expected={expected_remaining} "
            f"from used level {used}, actual={value['remaining_levels']!r}"
        )
    return used


def _validate_metadata(value: Any, expected: dict[str, Any], label: str) -> None:
    if not isinstance(value, dict) or set(value) != METADATA_KEYS:
        raise ArtifactRunnerError(f"{label} metadata shape drifted")
    for key in (
        "level",
        "noise_scale_degree",
        "remaining_levels",
        "expected_scale_bits",
        "ciphertext_count",
    ):
        if not isinstance(value[key], int) or isinstance(value[key], bool):
            raise ArtifactRunnerError(f"{label}.{key} must be an integer")
    _used_level(value, label)
    for key in (
        "level",
        "noise_scale_degree",
        "remaining_levels",
        "expected_scale_bits",
        "ciphertext_count",
    ):
        if not _typed_equal(value[key], expected[key]):
            raise ArtifactRunnerError(
                f"{label}.{key} drifted: expected={expected[key]!r} "
                f"actual={value[key]!r}"
            )
    scale_bits = _finite_number(
        value["scale_bits"], f"{label}.scale_bits", 1.0, math.inf
    )
    if abs(scale_bits - float(value["expected_scale_bits"])) > 1e-3:
        raise ArtifactRunnerError(
            f"{label}.scale_bits differs from expected_scale_bits by more than 1e-3"
        )


def _validate_used_level_delta(
    output: dict[str, Any],
    anchor: dict[str, Any],
    expected_delta: int,
    label: str,
) -> None:
    actual_delta = _used_level(output, f"{label} output") - _used_level(
        anchor,
        f"{label} anchor",
    )
    if actual_delta != expected_delta:
        raise ArtifactRunnerError(
            f"{label} used-level delta drifted: expected={expected_delta} "
            f"actual={actual_delta}"
        )


def _validate_softmax_checkpoint_metadata(value: Any, label: str) -> None:
    if SOFTMAX_CHECKPOINT_METADATA is None:
        raise ArtifactRunnerError(
            "M5 Softmax checkpoint metadata is not sealed by the live seam run"
        )
    _validate_metadata(value, SOFTMAX_CHECKPOINT_METADATA, label)


def _validate_layernorm_checkpoint_metadata(value: Any, label: str) -> None:
    if LAYERNORM_CHECKPOINT_METADATA is None:
        raise ArtifactRunnerError(
            "M5 LayerNorm checkpoint metadata is not sealed by the live seam run"
        )
    _validate_metadata(value, LAYERNORM_CHECKPOINT_METADATA, label)


def _validate_layer0_input_metadata(value: Any, label: str) -> None:
    if LAYER0_INPUT_METADATA is None:
        raise ArtifactRunnerError(
            "M5 layer-0 input metadata is not sealed by the crypto preflight"
        )
    _validate_metadata(value, LAYER0_INPUT_METADATA, label)


def _validate_crypto_preflight_record(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != CRYPTO_PREFLIGHT_KEYS:
        raise ArtifactRunnerError(f"{label} keys differ from the frozen contract")
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
        if not _typed_equal(value[key], expected):
            raise ArtifactRunnerError(
                f"{label}.{key} mismatch: expected={expected!r} actual={value[key]!r}"
            )
    _validate_layer0_input_metadata(value["input_metadata"], f"{label}.input_metadata")
    return value


def _extract_ctest_records(stdout: str, test_name: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ValueError(f"non-finite JSON number: {token}")

    for line in stdout.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            candidate = json.loads(
                line[start:],
                object_pairs_hook=reject_duplicate_keys,
                parse_constant=reject_non_finite,
            )
        except (json.JSONDecodeError, ValueError) as error:
            raise ArtifactRunnerError(
                "M5 narrow CTest contains a malformed JSON-bearing line: "
                f"{line[:400]!r} ({error})"
            ) from error
        if isinstance(candidate, dict) and candidate.get("test") == test_name:
            records.append(candidate)
    return records


def _validate_schedule_preflight_record(
    value: Any,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != SCHEDULE_PREFLIGHT_KEYS:
        raise ArtifactRunnerError(f"{label} keys differ from the frozen contract")
    expected_scalars = {
        "test": "openfhe_encoder_12_layer_preflight",
        "profile": "paper_compat",
        "security_claim": "none",
        "encoder_layers": 12,
        "layer_ids": list(range(12)),
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
        if not _typed_equal(value[key], expected):
            raise ArtifactRunnerError(
                f"{label}.{key} mismatch: expected={expected!r} actual={value[key]!r}"
            )
    _finite_number(
        value["final_relative_l2"],
        f"{label}.final_relative_l2",
        0.0,
        0.05,
    )
    _finite_number(
        value["final_cosine"],
        f"{label}.final_cosine",
        0.99,
        1.000000000001,
    )
    _finite_number(
        value["final_max_absolute"],
        f"{label}.final_max_absolute",
        0.0,
        math.inf,
    )
    _finite_number(
        value["load_and_oracle_ms"],
        f"{label}.load_and_oracle_ms",
        0.0,
        math.inf,
    )
    return value


def _parse_schedule_preflight_output(stdout: str) -> dict[str, Any]:
    records = _extract_ctest_records(
        stdout,
        "openfhe_encoder_12_layer_preflight",
    )
    if len(records) != 1:
        raise ArtifactRunnerError(
            "M5 narrow CTest must emit exactly one schedule-preflight JSON record, "
            f"got {len(records)}"
        )
    return _validate_schedule_preflight_record(
        records[0],
        "M5 narrow CTest schedule preflight",
    )


def _parse_crypto_preflight_output(stdout: str) -> dict[str, Any]:
    records = _extract_ctest_records(
        stdout,
        "openfhe_encoder_12_layer_crypto_preflight",
    )
    if len(records) != 1:
        raise ArtifactRunnerError(
            "M5 narrow CTest must emit exactly one crypto-preflight JSON record, "
            f"got {len(records)}"
        )
    return _validate_crypto_preflight_record(
        records[0],
        "M5 narrow CTest crypto preflight",
    )


def _validate_handoff_input_metadata(value: Any, label: str) -> None:
    if LAYER_HANDOFF_INPUT_METADATA is None:
        raise ArtifactRunnerError(
            "M5 handoff input metadata is not sealed by the two-layer seam run"
        )
    _validate_metadata(value, LAYER_HANDOFF_INPUT_METADATA, label)


def _validate_quality(value: Any, label: str) -> None:
    if not isinstance(value, dict) or set(value) != QUALITY_KEYS:
        raise ArtifactRunnerError(f"{label} quality shape drifted")
    _finite_number(value["relative_l2"], f"{label}.relative_l2", 0.0, 5e-2)
    _finite_number(value["cosine"], f"{label}.cosine", 0.99, 1.000000000001)
    _finite_number(value["max_absolute"], f"{label}.max_absolute", 0.0, math.inf)


def _expected_cumulative(layer_id: int) -> dict[str, int]:
    completed_layers = layer_id + 1
    completed_refreshes = min(completed_layers, ENCODER_LAYERS - 1)
    return {
        key: EXPECTED_LAYER_COUNTS[key] * completed_layers
        + EXPECTED_REFRESH_COUNTS[key] * completed_refreshes
        for key in EXPECTED_LAYER_COUNTS
    }


def _validate_counts(value: Any, expected: dict[str, int], label: str) -> None:
    if not isinstance(value, dict) or set(value) != COUNT_KEYS:
        raise ArtifactRunnerError(f"{label} operation-count shape drifted")
    if any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0
        for item in value.values()
    ):
        raise ArtifactRunnerError(
            f"{label} operation counts must be non-negative integers"
        )
    if value != expected:
        raise ArtifactRunnerError(f"{label} operation schedule drifted")


def _validate_layer_record(
    record: dict[str, Any],
    layer_id: int,
    previous_raw_output_metadata: dict[str, Any] | None = None,
) -> None:
    label = f"layer[{layer_id}]"
    regime = "layer_0" if layer_id == 0 else "layers_1_to_11"
    if set(record) != LAYER_RECORD_KEYS:
        raise ArtifactRunnerError(
            f"{label} keys differ: missing={sorted(LAYER_RECORD_KEYS - set(record))} "
            f"extra={sorted(set(record) - LAYER_RECORD_KEYS)}"
        )
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
        "handoff_refresh_performed": layer_id < ENCODER_LAYERS - 1,
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
        if not _typed_equal(record.get(key), expected):
            raise ArtifactRunnerError(
                f"{label}.{key} mismatch: expected={expected!r} actual={record.get(key)!r}"
            )
    output_metadata = RAW_OUTPUT_METADATA
    if layer_id == 0:
        _validate_layer0_input_metadata(
            record["input_metadata"], f"{label}.input_metadata"
        )
    else:
        _validate_handoff_input_metadata(
            record["input_metadata"], f"{label}.input_metadata"
        )
    _validate_metadata(
        record["raw_output_metadata"], output_metadata, f"{label}.raw_output_metadata"
    )
    _validate_softmax_checkpoint_metadata(
        record["softmax_denominator_metadata"],
        f"{label}.softmax_denominator_metadata",
    )
    _validate_metadata(
        record["attention_output_metadata"],
        (
            LAYER0_ATTENTION_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_ATTENTION_OUTPUT_METADATA
        ),
        f"{label}.attention_output_metadata",
    )
    _validate_layernorm_checkpoint_metadata(
        record["ln1_variance_metadata"],
        f"{label}.ln1_variance_metadata",
    )
    _validate_metadata(
        record["ln1_output_metadata"],
        (
            LAYER0_LN1_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_LN1_OUTPUT_METADATA
        ),
        f"{label}.ln1_output_metadata",
    )
    _validate_metadata(
        record["ffn_output_metadata"],
        (
            LAYER0_FFN_OUTPUT_METADATA
            if layer_id == 0
            else POST_REFRESH_FFN_OUTPUT_METADATA
        ),
        f"{label}.ffn_output_metadata",
    )
    _validate_layernorm_checkpoint_metadata(
        record["ln2_variance_metadata"],
        f"{label}.ln2_variance_metadata",
    )
    _validate_used_level_delta(
        record["input_metadata"],
        record["softmax_denominator_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["input_to_softmax_checkpoint_net_recovered"][regime],
        f"{label}.input_to_softmax_checkpoint_net_recovered",
    )
    _validate_used_level_delta(
        record["attention_output_metadata"],
        record["softmax_denominator_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["softmax_checkpoint_to_attention_output_consumed"][
            regime
        ],
        f"{label}.softmax_checkpoint_to_attention_output_consumed",
    )
    _validate_used_level_delta(
        record["attention_output_metadata"],
        record["ln1_variance_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["attention_output_to_ln1_checkpoint_net_recovered"][
            regime
        ],
        f"{label}.attention_output_to_ln1_checkpoint_net_recovered",
    )
    _validate_used_level_delta(
        record["ln1_output_metadata"],
        record["ln1_variance_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["ln1_checkpoint_to_ln1_output_consumed"][regime],
        f"{label}.ln1_checkpoint_to_ln1_output_consumed",
    )
    _validate_used_level_delta(
        record["ffn_output_metadata"],
        record["ln1_output_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["ln1_output_to_ffn_output_consumed"][regime],
        f"{label}.ln1_output_to_ffn_output_consumed",
    )
    _validate_used_level_delta(
        record["ffn_output_metadata"],
        record["ln2_variance_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["ffn_output_to_ln2_checkpoint_net_recovered"][
            regime
        ],
        f"{label}.ffn_output_to_ln2_checkpoint_net_recovered",
    )
    _validate_used_level_delta(
        record["raw_output_metadata"],
        record["ln2_variance_metadata"],
        RELATIVE_USED_LEVEL_DELTAS["ln2_checkpoint_to_raw_output_consumed"][regime],
        f"{label}.ln2_checkpoint_to_raw_output_consumed",
    )
    if layer_id > 0:
        if previous_raw_output_metadata is None:
            raise ArtifactRunnerError(
                f"{label}.previous_raw_output_metadata is required"
            )
        _validate_metadata(
            previous_raw_output_metadata,
            RAW_OUTPUT_METADATA,
            f"{label}.previous_raw_output_metadata",
        )
        _validate_used_level_delta(
            previous_raw_output_metadata,
            record["input_metadata"],
            RELATIVE_USED_LEVEL_DELTAS["previous_raw_output_to_input_recovered"][
                regime
            ],
            f"{label}.previous_raw_output_to_input_recovered",
        )
    for key in ("input_quality", "output_quality", "exact_trace_diagnostic"):
        _validate_quality(record[key], f"{label}.{key}")
    _finite_number(
        record["inactive_max_abs"],
        f"{label}.inactive_max_abs",
        0.0,
        M5_PROTOTYPE_INACTIVE_MAX_ABS,
    )
    _finite_number(
        record["inactive_sentinel_max_error"],
        f"{label}.inactive_sentinel_max_error",
        0.0,
        math.inf,
    )
    sentinel_ranges = record["inactive_polynomial_sentinel_ranges"]
    if (
        not isinstance(sentinel_ranges, dict)
        or set(sentinel_ranges) != INACTIVE_SENTINEL_RANGE_KEYS
    ):
        raise ArtifactRunnerError(
            f"{label}.inactive_polynomial_sentinel_ranges drifted"
        )
    for key in sorted(INACTIVE_SENTINEL_RANGE_KEYS):
        observed = sentinel_ranges[key]
        frozen = INACTIVE_SENTINEL_INTERVALS[key]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ArtifactRunnerError(
                f"{label}.inactive_polynomial_sentinel_ranges.{key} "
                "must contain [min,max]"
            )
        minimum = _finite_number(
            observed[0],
            f"{label}.inactive_polynomial_sentinel_ranges.{key}[0]",
            frozen[0],
            frozen[1],
        )
        maximum = _finite_number(
            observed[1],
            f"{label}.inactive_polynomial_sentinel_ranges.{key}[1]",
            frozen[0],
            frozen[1],
        )
        if minimum > maximum:
            raise ArtifactRunnerError(
                f"{label}.inactive_polynomial_sentinel_ranges.{key} is reversed"
            )
    ranges = record["encrypted_polynomial_input_ranges"]
    if not isinstance(ranges, dict) or set(ranges) != RANGE_KEYS:
        raise ArtifactRunnerError(f"{label}.encrypted_polynomial_input_ranges drifted")
    for key, frozen in FROZEN_RANGES.items():
        observed = ranges[key]
        if not isinstance(observed, list) or len(observed) != 2:
            raise ArtifactRunnerError(f"{label}.ranges.{key} must contain [min,max]")
        minimum = _finite_number(
            observed[0], f"{label}.ranges.{key}[0]", frozen[0], frozen[1]
        )
        maximum = _finite_number(
            observed[1], f"{label}.ranges.{key}[1]", frozen[0], frozen[1]
        )
        if minimum > maximum:
            raise ArtifactRunnerError(f"{label}.ranges.{key} is reversed")
    _validate_counts(
        record["refresh_operation_counts"],
        EXPECTED_REFRESH_COUNTS if layer_id < ENCODER_LAYERS - 1 else ZERO_COUNTS,
        f"{label}.refresh_operation_counts",
    )
    _validate_counts(
        record["layer_operation_counts"],
        EXPECTED_LAYER_COUNTS,
        f"{label}.layer_operation_counts",
    )
    _validate_counts(
        record["cumulative_operation_counts"],
        _expected_cumulative(layer_id),
        f"{label}.cumulative_operation_counts",
    )


def _validate_final_record(final: dict[str, Any], layers: list[dict[str, Any]]) -> None:
    if set(final) != FINAL_RECORD_KEYS:
        raise ArtifactRunnerError(
            f"final record keys differ: missing={sorted(FINAL_RECORD_KEYS - set(final))} "
            f"extra={sorted(set(final) - FINAL_RECORD_KEYS)}"
        )
    expected = {
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
        "inactive_sentinel_range_status": "all_client_validated",
        "multiplicative_depth": EXPECTED_MULTIPLICATIVE_DEPTH,
        "max_observed_level": EXPECTED_MAX_OBSERVED_LEVEL,
        "max_polynomial_depth": EXPECTED_MAX_POLYNOMIAL_DEPTH,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
    }
    for key, value in expected.items():
        if not _typed_equal(final.get(key), value):
            raise ArtifactRunnerError(
                f"final.{key} mismatch: expected={value!r} actual={final.get(key)!r}"
            )
    for field in (
        "fixture_load_oracle_ms",
        "setup_keygen_ms",
        "client_encrypt_ms",
        "server_online_diagnostic_ms",
        "client_checkpoint_validate_ms",
    ):
        _finite_number(final[field], f"final.{field}", 0.0, math.inf)
    if final["server_online_diagnostic_ms"] <= 0.0:
        raise ArtifactRunnerError("final.server_online_diagnostic_ms must be positive")
    _finite_number(final["relative_l2"], "final.relative_l2", 0.0, 5e-2)
    _finite_number(final["cosine"], "final.cosine", 0.99, 1.000000000001)
    _finite_number(final["max_absolute"], "final.max_absolute", 0.0, math.inf)
    _finite_number(
        final["inactive_max_abs"],
        "final.inactive_max_abs",
        0.0,
        M5_PROTOTYPE_INACTIVE_MAX_ABS,
    )
    _finite_number(
        final["inactive_sentinel_max_error"],
        "final.inactive_sentinel_max_error",
        0.0,
        math.inf,
    )
    _validate_metadata(
        final["final_metadata"], RAW_OUTPUT_METADATA, "final.final_metadata"
    )
    if final["final_metadata"] != layers[-1]["raw_output_metadata"]:
        raise ArtifactRunnerError(
            "final.final_metadata differs from layer 11 raw output"
        )
    _validate_counts(
        final["operation_counts"], EXPECTED_TOTAL_COUNTS, "final.operation_counts"
    )
    peak_rss = final["peak_rss_bytes"]
    if not isinstance(peak_rss, int) or isinstance(peak_rss, bool) or peak_rss <= 0:
        raise ArtifactRunnerError("final.peak_rss_bytes must be a positive integer")
    bindings = {
        "relative_l2": "relative_l2",
        "cosine": "cosine",
        "max_absolute": "max_absolute",
    }
    for final_key, layer_key in bindings.items():
        if not math.isclose(
            float(final[final_key]),
            float(layers[-1]["output_quality"][layer_key]),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ArtifactRunnerError(f"final.{final_key} differs from layer 11 output")
    for field in ("inactive_max_abs", "inactive_sentinel_max_error"):
        layer_maximum = max(float(layer[field]) for layer in layers)
        if not math.isclose(
            float(final[field]),
            layer_maximum,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ArtifactRunnerError(f"final.{field} differs from 12-layer maximum")


def _runtime_command(config: RunnerConfig, resource_file: Path) -> list[str]:
    command = [
        str(TIME_EXECUTABLE),
        "--format=%M",
        f"--output={resource_file}",
        str(config.executable),
        "--data-root",
        str(config.data_root),
    ]
    forbidden = {
        "--diagnostic-layer-count",
        "--metadata-calibration",
        "--preflight-only",
        "--crypto-preflight-only",
        "--layer",
        "--input-level",
    }
    if forbidden.intersection(command):  # pragma: no cover - immutable command guard
        raise ArtifactRunnerError(
            "M5 evidence command contains a diagnostic/prefix option"
        )
    return command


def _run_once(config: RunnerConfig) -> RunSample:
    if (
        LAYER0_INPUT_METADATA is None
        or LAYER_HANDOFF_INPUT_METADATA is None
        or SOFTMAX_CHECKPOINT_METADATA is None
        or LAYERNORM_CHECKPOINT_METADATA is None
    ):
        raise ArtifactRunnerError(
            "M5 layer-0/handoff/checkpoint metadata is not sealed by the crypto "
            "preflight and two-layer seam run"
        )
    descriptor, resource_name = tempfile.mkstemp(
        prefix="moai-m5-time-", suffix=".txt", dir="/tmp"
    )
    os.close(descriptor)
    resource_file = Path(resource_name)
    command = _runtime_command(config, resource_file)
    started = _timestamp()
    monotonic_start = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            env=_artifact_environment(),
        )
        elapsed_seconds = time.perf_counter() - monotonic_start
        finished = _timestamp()
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise ArtifactRunnerError(
                f"M5 correctness run exited with {completed.returncode}: {detail[:4000]}"
            )
        if completed.stderr.strip():
            raise ArtifactRunnerError(
                "M5 correctness run emitted stderr despite exit_code=0: "
                f"{completed.stderr.strip()[:4000]}"
            )
        try:
            resource_lines = resource_file.read_text(encoding="utf-8").splitlines()
            if len(resource_lines) != 1:
                raise ValueError("expected one GNU time output line")
            peak_rss_kib = int(resource_lines[0])
        except (OSError, UnicodeError, ValueError) as error:
            raise ArtifactRunnerError(f"cannot parse M5 peak RSS: {error}") from error
        if elapsed_seconds <= 0.0 or peak_rss_kib <= 0:
            raise ArtifactRunnerError("M5 run produced invalid resource metrics")
        layers = _extract_records(completed.stdout, "openfhe_encoder_12_layer_layer")
        finals = _extract_records(completed.stdout, "openfhe_encoder_12_layer")
        if len(layers) != ENCODER_LAYERS:
            raise ArtifactRunnerError(
                f"M5 run must emit exactly 12 layer records, got {len(layers)}"
            )
        if len(finals) != 1:
            raise ArtifactRunnerError(
                f"M5 run must emit exactly one final record, got {len(finals)}"
            )
        for layer_id, record in enumerate(layers):
            _validate_layer_record(
                record,
                layer_id,
                (
                    None
                    if layer_id == 0
                    else layers[layer_id - 1]["raw_output_metadata"]
                ),
            )
        _validate_final_record(finals[0], layers)
        if finals[0]["peak_rss_bytes"] > peak_rss_kib * 1024:
            raise ArtifactRunnerError("runtime peak RSS exceeds GNU time peak RSS")
        return RunSample(
            completed.stdout,
            tuple(layers),
            finals[0],
            elapsed_seconds,
            peak_rss_kib,
            _command_record(command, completed.returncode, started, finished),
        )
    finally:
        resource_file.unlink(missing_ok=True)


def _write_metrics(path: Path, sample: RunSample) -> None:
    final = sample.final
    metadata = final["final_metadata"]
    counts = final["operation_counts"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "run": 1,
                "exit_code": 0,
                "elapsed_seconds": format(sample.elapsed_seconds, ".17g"),
                "peak_rss_kib": sample.peak_rss_kib,
                **{
                    field: format(float(final[field]), ".17g")
                    for field in (
                        "fixture_load_oracle_ms",
                        "setup_keygen_ms",
                        "client_encrypt_ms",
                        "server_online_diagnostic_ms",
                        "client_checkpoint_validate_ms",
                    )
                },
                "final_rel_l2": format(float(final["relative_l2"]), ".17g"),
                "final_cosine": format(float(final["cosine"]), ".17g"),
                "final_max_absolute": format(float(final["max_absolute"]), ".17g"),
                "inactive_max_abs": format(float(final["inactive_max_abs"]), ".17g"),
                "inactive_sentinel_max_error": format(
                    float(final["inactive_sentinel_max_error"]), ".17g"
                ),
                "checkpoint_metadata_sha256": _checkpoint_metadata_sha256(
                    sample.layers
                ),
                "final_level": metadata["level"],
                "final_noise_scale_degree": metadata["noise_scale_degree"],
                "final_remaining_levels": metadata["remaining_levels"],
                "final_scale_bits": format(float(metadata["scale_bits"]), ".17g"),
                "final_ciphertext_count": metadata["ciphertext_count"],
                **counts,
                "timing_claim": "false",
                "latency_kind": "non_benchmark_diagnostic",
                "finite": "true",
                "passed": "true",
            }
        )


def _artifact_record(
    root: Path, path: str, role: str, media_type: str
) -> dict[str, object]:
    resolved = root / path
    return {
        "path": path,
        "sha256": _sha256(resolved),
        "bytes": resolved.stat().st_size,
        "media_type": media_type,
        "role": role,
    }


def _validator_command(manifest_path: Path) -> list[str]:
    return [
        sys.executable,
        str(VALIDATOR_PATH),
        "--schema",
        str(SCHEMA_PATH),
        "--manifest",
        str(manifest_path),
        "--verify-git",
    ]


def _validate_artifact(manifest_path: Path) -> None:
    command = _validator_command(manifest_path)
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        env=_artifact_environment(),
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ArtifactRunnerError(f"artifact validator rejected the M5 run: {detail}")


def _metrics_manifest(sample: RunSample) -> dict[str, object]:
    final = sample.final
    omitted_layer_fields = {
        "test",
        "profile",
        "security_claim",
        "range_validation_owner",
        "checkpoint_decryption_owner",
        "server_decryptions",
        "server_plaintext_activations",
    }
    return {
        "repeat_count": 1,
        "successful_repeats": 1,
        "warmup_count": 0,
        "elapsed_seconds": sample.elapsed_seconds,
        "peak_rss_kib": sample.peak_rss_kib,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "checkpoint_metadata_sha256": _checkpoint_metadata_sha256(sample.layers),
        "phase_latency_ms": {
            key: final[key]
            for key in (
                "fixture_load_oracle_ms",
                "setup_keygen_ms",
                "client_encrypt_ms",
                "server_online_diagnostic_ms",
                "client_checkpoint_validate_ms",
            )
        },
        "layers": [
            {
                key: value
                for key, value in layer.items()
                if key not in omitted_layer_fields
            }
            for layer in sample.layers
        ],
        "final_quality": {
            "relative_l2": final["relative_l2"],
            "cosine": final["cosine"],
            "max_absolute": final["max_absolute"],
            "inactive_max_abs": final["inactive_max_abs"],
            "inactive_sentinel_max_error": final["inactive_sentinel_max_error"],
            "inactive_sentinel_range_status": final["inactive_sentinel_range_status"],
        },
        "final_metadata": final["final_metadata"],
        "operation_counts": final["operation_counts"],
        "finite": True,
    }


def generate_m5_artifact(config: RunnerConfig) -> Path:
    if RUN_ID_PATTERN.fullmatch(config.run_id) is None:
        raise ArtifactRunnerError(f"invalid run id: {config.run_id!r}")
    trace_scale_provenance, exact3_evidence = _sealed_static_contracts()
    started_at = _timestamp()
    executable = _resolve_m5_executable(config.executable, require_exists=False)
    data_root = config.data_root.resolve()
    if not data_root.is_dir() or data_root != DEFAULT_DATA_ROOT.resolve():
        raise ArtifactRunnerError(f"M5 data root must be {DEFAULT_DATA_ROOT.resolve()}")
    if not TIME_EXECUTABLE.is_file() or not os.access(TIME_EXECUTABLE, os.X_OK):
        raise ArtifactRunnerError(f"GNU time is unavailable: {TIME_EXECUTABLE}")

    output_root = _resolve_output_root(config.output_root)
    git_state = _preflight_git()
    schema_binding = _schema_binding_snapshot(git_state.head, "before correctness run")
    preflight = _run_m5_preflight(
        executable.parent, executable, config.openfhe_prefix.resolve()
    )
    executable = _resolve_m5_executable(DEFAULT_EXECUTABLE)
    executable_sha256 = _sha256(executable)
    environment = _environment(config.openfhe_prefix)
    profile = _profile_record()
    trace_inputs, layer_identities = _trace_input_records(data_root)
    inputs = sorted(
        [*(_repository_record(path) for path in REQUIRED_INPUTS), *trace_inputs],
        key=lambda record: str(record["path"]),
    )
    if (
        len(inputs) != EXPECTED_INPUT_COUNT
        or len({item["path"] for item in inputs}) != EXPECTED_INPUT_COUNT
    ):
        raise ArtifactRunnerError("M5 inputs must contain exactly 448 unique records")
    input_by_path = {record["path"]: record for record in inputs}

    output_root.mkdir(parents=True, exist_ok=True)
    run_root = output_root / config.run_id
    try:
        run_root.mkdir(exist_ok=False)
    except FileExistsError as error:
        raise ArtifactRunnerError(
            f"run directory already exists: {run_root}"
        ) from error

    remove_unvalidated = True
    try:
        _require_build_configuration(
            preflight.build_configuration,
            executable.parent,
            executable,
            config.openfhe_prefix,
            "before correctness run",
        )
        _require_executable_hash(
            executable, executable_sha256, "before correctness run"
        )
        _require_input_hashes(inputs, "before correctness run")
        sample = _run_once(
            RunnerConfig(
                executable, data_root, output_root, config.run_id, config.openfhe_prefix
            )
        )
        _require_executable_hash(executable, executable_sha256, "after correctness run")
        _require_input_hashes(inputs, "after correctness run")
        _require_build_configuration(
            preflight.build_configuration,
            executable.parent,
            executable,
            config.openfhe_prefix,
            "after correctness run",
        )
        if (
            _schema_binding_snapshot(
                git_state.head,
                "after correctness run",
            )
            != schema_binding
        ):
            raise ArtifactRunnerError("M5 v6 schema binding changed during the run")

        stdout_path = run_root / "stdout.log"
        metrics_path = run_root / "metrics.csv"
        checksum_path = run_root / "SHA256SUMS"
        manifest_path = run_root / "manifest.json"
        stdout_path.write_text(
            sample.stdout if sample.stdout.endswith("\n") else sample.stdout + "\n",
            encoding="utf-8",
        )
        _write_metrics(metrics_path, sample)
        checksum_path.write_text(
            f"{_sha256(stdout_path)}  stdout.log\n{_sha256(metrics_path)}  metrics.csv\n",
            encoding="utf-8",
        )

        validator_command = _validator_command(manifest_path)
        commands = [
            *git_state.commands,
            *preflight.commands,
            sample.command,
            _command_record(validator_command, 0),
        ]
        if len(commands) != 12:
            raise ArtifactRunnerError(
                f"M5 command contract must contain 12 records, got {len(commands)}"
            )
        executable_path = executable.relative_to(REPO_ROOT.resolve()).as_posix()
        manifest: dict[str, object] = {
            "schema_version": M5_SCHEMA_VERSION,
            "schema_binding": schema_binding,
            "run_id": config.run_id,
            "milestone": "M5",
            "started_at": started_at,
            "finished_at": _timestamp(),
            "git": {
                "repository_root": str(REPO_ROOT),
                "branch": BRANCH,
                "local_commit": git_state.head,
                "clean": True,
                "tracking_ref": TRACKING_REF,
                "tracking_commit": git_state.tracking_head,
                "remote_name": REMOTE_NAME,
                "remote_url": git_state.remote_url,
                "remote_ref": REMOTE_REF,
                "remote_commit": git_state.remote_head,
            },
            "commands": commands,
            "build_configuration": preflight.build_configuration,
            "environment": environment,
            "profile": profile,
            "inputs": inputs,
            "workload": {
                "scope": (
                    "M5 server-only five-token BERT-base 12-layer encoder "
                    "ciphertext-chain trace replay"
                ),
                "backend": "OpenFHE CKKS CPU",
                "executable_path": executable_path,
                "executable_sha256": executable_sha256,
                "executable_bytes": executable.stat().st_size,
                "repeat_count": 1,
                "warmup_count": 0,
                "timing_claim": False,
                "latency_kind": "non_benchmark_diagnostic",
            },
            "contracts": {
                "approximation_config_path": "config/openfhe_approximations.json",
                "approximation_config_sha256": input_by_path[
                    "config/openfhe_approximations.json"
                ]["sha256"],
                "channel_scales_config_path": "config/moai_trace_channel_scales.json",
                "channel_scales_config_sha256": input_by_path[
                    "config/moai_trace_channel_scales.json"
                ]["sha256"],
                "encoder_trace_contract_path": "config/moai_encoder_trace.json",
                "encoder_trace_contract_sha256": input_by_path[
                    "config/moai_encoder_trace.json"
                ]["sha256"],
                "profile_config_path": PROFILE_PATH,
                "profile_config_sha256": input_by_path[PROFILE_PATH]["sha256"],
                "execution": {
                    "mode": "server-only",
                    "encoder_layers": 12,
                    "layer_ids": list(range(12)),
                    "chain_mode": "ciphertext_output_to_next_input",
                    "client_encrypt_calls": 1,
                    "encrypted_input_ciphertexts": 5,
                    "plaintext_activation_resets": 0,
                    "server_layer_evaluations": 12,
                    "inter_layer_refreshes": 11,
                    "trace_shape": [5, 768],
                    "feature_block_size": 1024,
                    "checkpoint_decryption_owner": "client",
                    "final_decryption_owner": "client",
                    "server_private_key_present": False,
                    "server_decryptions": 0,
                    "server_plaintext_activations": False,
                    "approximation_range_status": "all_client_validated",
                    "quality_reference": (
                        "12-layer chained frozen polynomial oracle from "
                        "config/moai_encoder_trace.json"
                    ),
                },
                "feature_packed_trace_scale_contract": trace_scale_provenance,
                "exact3_evidence": exact3_evidence,
                "metadata_schedule": _metadata_schedule_contract(),
                "schedule_preflight": preflight.schedule_preflight,
                "crypto_preflight": preflight.crypto_preflight,
                "layer_input_identities": layer_identities,
                "thresholds": {
                    "per_layer_relative_l2_max": 5e-2,
                    "per_layer_cosine_min": 0.99,
                    "final_relative_l2_max": 5e-2,
                    "final_cosine_min": 0.99,
                    "inactive_max_abs": M5_PROTOTYPE_INACTIVE_MAX_ABS,
                    "inactive_sentinel_in_interval_required": True,
                    "finite_required": True,
                },
                "polynomial_intervals": {
                    key: [minimum, maximum]
                    for key, (minimum, maximum) in FROZEN_RANGES.items()
                },
            },
            "metrics": _metrics_manifest(sample),
            "gate": {
                "passed": True,
                "decision": "PASS_M5_RUNNABLE_PROTOTYPE",
                "checks": {
                    "build": "PASS",
                    "trust_boundary": "PASS",
                    "input_integrity": "PASS",
                    "ciphertext_chain": "PASS",
                    "approximation_ranges": "PASS",
                    "correctness": "PASS",
                    "artifact_integrity": "PASS",
                    "remote_sha": "PASS",
                },
            },
            "artifacts": [
                _artifact_record(run_root, "stdout.log", "stdout", "text/plain"),
                _artifact_record(run_root, "metrics.csv", "metrics", "text/csv"),
                _artifact_record(run_root, "SHA256SUMS", "checksum", "text/plain"),
            ],
            "claim_boundary": list(CLAIM_BOUNDARY),
            "verdict": "GO_PROTOTYPE",
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        _validate_artifact(manifest_path)
        remove_unvalidated = False
        return run_root
    finally:
        if remove_unvalidated:
            try:
                shutil.rmtree(run_root)
            except OSError as cleanup_error:
                raise ArtifactRunnerError(
                    f"failed to remove unvalidated run directory {run_root}: {cleanup_error}"
                ) from cleanup_error


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    git_sha_hint = "unknown"
    try:
        if arguments.schema.resolve() != SCHEMA_PATH.resolve():
            raise ArtifactRunnerError(f"M5 v6 runner requires --schema {SCHEMA_PATH}")
        # Fail closed before even a read-only git subprocess.  The generator
        # repeats this check so direct API callers receive the same guarantee.
        _sealed_static_contracts()
        git_sha_hint = (
            subprocess.run(
                [str(SYSTEM_GIT), "rev-parse", "--short=7", "HEAD"],
                cwd=REPO_ROOT,
                check=False,
                capture_output=True,
                text=True,
                env=_git_environment(),
                stdin=subprocess.DEVNULL,
            ).stdout.strip()
            or "unknown"
        )
        run_id = arguments.run_id or (
            f"{datetime.now().astimezone().strftime('%Y%m%dT%H%M%S%z')}"
            f"-m5-{git_sha_hint}"
        )
        run_root = generate_m5_artifact(
            RunnerConfig(
                arguments.executable,
                arguments.data_root,
                arguments.output_root,
                run_id,
            )
        )
    except (ArtifactRunnerError, OSError, ValueError) as error:
        print(f"run_openfhe_encoder12_artifact failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "artifact_root": str(run_root),
                "manifest": str(run_root / "manifest.json"),
                "milestone": "M5",
                "run_id": run_root.name,
                "git_sha_hint": git_sha_hint,
                "validated": True,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
