#!/usr/bin/env python3
"""Positive and tamper-negative tests for the schema-v2 artifact validator."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import shlex
import statistics
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_openfhe_artifact as validator  # noqa: E402


ORIGINAL_SHA256_FILE = validator.sha256_file
ENCODER_TRACE_CONTRACT = validator.load_json(REPO_ROOT / "config/moai_encoder_trace.json")
TRACE_SHA256_BY_PATH: dict[str, str] = {}
for _layer in ENCODER_TRACE_CONTRACT["layers"]:
    for _logical_name, _specification in ENCODER_TRACE_CONTRACT[
        "required_files"
    ].items():
        _path = (
            Path("data")
            / f"layer_{_layer['layer_id']}"
            / _specification["path"]
        ).as_posix()
        TRACE_SHA256_BY_PATH[_path] = _layer["sha256"][_logical_name]


def _sha256_without_large_trace_reads(path: Path) -> str:
    try:
        relative_path = path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        relative_path = ""
    if relative_path in TRACE_SHA256_BY_PATH:
        return TRACE_SHA256_BY_PATH[relative_path]
    return ORIGINAL_SHA256_FILE(path)


class ArtifactValidatorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="moai-artifact-validator-"
        )
        self.artifact_root = Path(self.temporary_directory.name)
        self.manifest_path = self.artifact_root / "manifest.json"
        self.schema = validator.validate_schema(
            validator.load_json(REPO_ROOT / "docs/openfhe-artifact-schema.json")
        )
        self.runtime_records = self._make_runtime_records()
        self._write_stdout()
        self._write_metrics()
        self.manifest = self._make_manifest()
        self._reseal_artifacts()
        self._write_manifest()

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _repository_record(path: str, role: str) -> dict[str, object]:
        resolved = REPO_ROOT / path
        return {
            "path": path,
            "sha256": validator.sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": "application/json",
            "role": role,
        }

    @staticmethod
    def _artifact_record(root: Path, path: str, role: str) -> dict[str, object]:
        resolved = root / path
        media_types = {
            "stdout.log": "text/plain",
            "metrics.csv": "text/csv",
            "SHA256SUMS": "text/plain",
        }
        return {
            "path": path,
            "sha256": validator.sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": media_types[path],
            "role": role,
        }

    def _make_runtime_records(self) -> list[dict[str, object]]:
        profile_config = validator.load_json(REPO_ROOT / "config/paper_compat.json")
        effective = profile_config["openfhe_translation"][
            "m3_smoke_effective_runtime"
        ]["parameters"]
        effective_hash = profile_config["openfhe_translation"][
            "m3_smoke_effective_runtime"
        ]["effective_profile_sha256"]
        records: list[dict[str, object]] = []
        for index in range(3):
            record = {
                "test": "openfhe_nonlinear_smoke",
                "profile": "paper_compat",
                "security_claim": "none",
                "parameter_sha256": effective_hash,
                "effective_profile_schema_version": effective[
                    "effective_profile_schema_version"
                ],
                "ring_dimension": effective["ring_dimension"],
                "slot_count": effective["slot_count"],
                "multiplicative_depth": effective["multiplicative_depth"],
                "levels_available_after_bootstrap": effective[
                    "levels_available_after_bootstrap"
                ],
                "encoded_slots": effective["bootstrap_slots"],
                "packing_active_slots": effective["bootstrap_slots"],
                "logical_active_slots": 4,
                "bootstrap_iterations_per_call": effective[
                    "bootstrap_iterations"
                ],
                "bootstrap_precision": effective["bootstrap_precision"],
                "gelu_degree": 319,
                "softmax_exp_degree": 27,
                "softmax_reciprocal_degree": 383,
                "softmax_shift_layer": 1,
                "softmax_shift_head": 2,
                "softmax_shift_sha256": (
                    "41ecf6ade53f674096c9afc752ccfe99834876ea4768bcf4bf3ea25a8f502432"
                ),
                "layernorm_invsqrt_degree": 159,
                "gelu_rel_l2": 0.0010 + index * 0.0001,
                "gelu_cosine": 0.99990 - index * 0.00001,
                "ffn_rel_l2": 0.0020 + index * 0.0001,
                "ffn_cosine": 0.99991 - index * 0.00001,
                "softmax_rel_l2": 0.0030 + index * 0.0001,
                "softmax_cosine": 0.99992 - index * 0.00001,
                "denominator_post_max_abs": 0.00000010 + index * 0.00000001,
                "layernorm_rel_l2": 0.0040 + index * 0.0001,
                "layernorm_cosine": 0.99993 - index * 0.00001,
                "inactive_max_abs": 0.00000040 + index * 0.00000001,
                "max_observed_level": 25 + index,
                **validator.M3_EXPECTED_OPERATION_COUNTS,
            }
            for name, prefix in validator.M3_RUNTIME_POLYNOMIAL_PREFIXES.items():
                polynomial = validator.M3_EXPECTED_POLYNOMIALS[name]
                record.update(
                    {
                        f"{prefix}_interval_min": polynomial["interval"]["minimum"],
                        f"{prefix}_interval_max": polynomial["interval"]["maximum"],
                        f"{prefix}_required_depth": polynomial["required_depth"],
                        f"{prefix}_estimated_multiplications": polynomial[
                            "estimated_multiplications"
                        ],
                    }
                )
            records.append(record)
        return records

    def _write_stdout(self) -> None:
        lines = [json.dumps(record, sort_keys=True) for record in self.runtime_records]
        (self.artifact_root / "stdout.log").write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    def _write_metrics(self) -> None:
        fieldnames = [
            "run",
            "exit_code",
            "elapsed_seconds",
            "peak_rss_kib",
            "inactive_max_abs",
            "softmax_rel_l2",
            "softmax_cosine",
            "bootstraps",
            "bootstrap_iterations",
        ]
        with (self.artifact_root / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, record in enumerate(self.runtime_records, start=1):
                writer.writerow(
                    {
                        "run": index,
                        "exit_code": 0,
                        "elapsed_seconds": (1.0, 1.5, 2.0)[index - 1],
                        "peak_rss_kib": (1000, 1200, 1100)[index - 1],
                        "inactive_max_abs": record["inactive_max_abs"],
                        "softmax_rel_l2": record["softmax_rel_l2"],
                        "softmax_cosine": record["softmax_cosine"],
                        "bootstraps": record["bootstraps"],
                        "bootstrap_iterations": record["bootstrap_iterations"],
                    }
                )

    def _make_manifest(self) -> dict[str, object]:
        profile_path = REPO_ROOT / "config/paper_compat.json"
        profile_config = validator.load_json(profile_path)
        effective = profile_config["openfhe_translation"][
            "m3_smoke_effective_runtime"
        ]["parameters"]
        effective_hash = validator.hashlib.sha256(
            validator.canonical_json_bytes(effective)
        ).hexdigest()
        approximation = validator.load_json(
            REPO_ROOT / "config/openfhe_approximations.json"
        )
        operators = approximation["operators"]
        inputs = [
            self._repository_record(path, "configuration")
            for path in sorted(validator.REQUIRED_M3_INPUTS)
        ]
        input_by_path = {record["path"]: record for record in inputs}
        executable_path = "scripts/validate_openfhe_artifact.py"
        executable = REPO_ROOT / executable_path
        quality_fields = {
            "gelu_rel_l2_max": ("gelu_rel_l2", max),
            "ffn_rel_l2_max": ("ffn_rel_l2", max),
            "softmax_rel_l2_max": ("softmax_rel_l2", max),
            "layernorm_rel_l2_max": ("layernorm_rel_l2", max),
            "gelu_cosine_min": ("gelu_cosine", min),
            "ffn_cosine_min": ("ffn_cosine", min),
            "softmax_cosine_min": ("softmax_cosine", min),
            "layernorm_cosine_min": ("layernorm_cosine", min),
            "inactive_max_abs_max": ("inactive_max_abs", max),
            "denominator_post_max_abs_max": (
                "denominator_post_max_abs",
                max,
            ),
        }
        quality = {
            summary: reducer(record[runtime] for record in self.runtime_records)
            for summary, (runtime, reducer) in quality_fields.items()
        }
        return {
            "schema_version": 2,
            "run_id": "synthetic-m3-artifact-validator",
            "milestone": "M3",
            "started_at": "2026-07-29T10:00:00+09:00",
            "finished_at": "2026-07-29T10:01:00+09:00",
            "git": {
                "repository_root": str(REPO_ROOT),
                "branch": "refactor/openfhe-cpu",
                "local_commit": "0" * 40,
                "clean": False,
            },
            "commands": [
                {
                    "command": "synthetic fixture generation",
                    "cwd": str(REPO_ROOT),
                    "exit_code": 0,
                    "phase": "artifact_generation",
                }
            ],
            "environment": {
                "os": "synthetic",
                "architecture": "x86_64",
                "compiler": "synthetic",
                "cmake": "synthetic",
                "python": sys.version.split()[0],
                "openfhe_version": "1.5.1",
                "openfhe_prefix": "/synthetic/openfhe",
                "wsl": True,
            },
            "profile": {
                "id": "paper_compat",
                "security_claim": "none",
                "warning": (
                    "Research reproduction parameters only. "
                    "Do not claim 128-bit security."
                ),
                "source_config_path": "config/paper_compat.json",
                "source_config_sha256": validator.sha256_file(profile_path),
                "source_config_bytes": profile_path.stat().st_size,
                "effective_profile_locator": (
                    "/openfhe_translation/m3_smoke_effective_runtime/parameters"
                ),
                "canonicalization": validator.CANONICALIZATION,
                "effective_profile_payload": effective,
                "effective_profile_sha256": effective_hash,
            },
            "inputs": inputs,
            "workload": {
                "scope": "synthetic validator fixture",
                "backend": "OpenFHE CKKS CPU",
                "executable_path": executable_path,
                "executable_sha256": validator.sha256_file(executable),
                "executable_bytes": executable.stat().st_size,
                "repeat_count": 3,
            },
            "contracts": {
                "approximation_config_path": "config/openfhe_approximations.json",
                "approximation_config_sha256": input_by_path[
                    "config/openfhe_approximations.json"
                ]["sha256"],
                "trace_contract_path": "config/moai_trace_channel_scales.json",
                "trace_contract_sha256": input_by_path[
                    "config/moai_trace_channel_scales.json"
                ]["sha256"],
                "profile_config_path": "config/paper_compat.json",
                "profile_config_sha256": input_by_path[
                    "config/paper_compat.json"
                ]["sha256"],
                "softmax_shift": {
                    "layer": 1,
                    "head": 2,
                    "values_sha256": operators["softmax"]["shift_contract"][
                        "values_sha256"
                    ],
                },
                "polynomials": {
                    "gelu": self._polynomial_record(
                        operators["gelu"]["polynomial"]
                    ),
                    "softmax_exponential": self._polynomial_record(
                        operators["softmax"]["exponential"]
                    ),
                    "softmax_reciprocal": self._polynomial_record(
                        operators["softmax"]["reciprocal"]
                    ),
                    "layernorm_inverse_sqrt": self._polynomial_record(
                        operators["layernorm"]["inverse_sqrt"]
                    ),
                },
                "thresholds": copy.deepcopy(validator.M3_EXPECTED_THRESHOLDS),
            },
            "metrics": {
                "repeat_count": 3,
                "successful_repeats": 3,
                "elapsed_seconds": {
                    "minimum": 1.0,
                    "median": statistics.median((1.0, 1.5, 2.0)),
                    "maximum": 2.0,
                },
                "peak_rss_kib_max": 1200,
                "quality": quality,
                "operation_counts": copy.deepcopy(
                    validator.M3_EXPECTED_OPERATION_COUNTS
                ),
                "max_observed_level_max": 27,
            },
            "gate": {
                "passed": False,
                "decision": "synthetic validator evidence only",
                "checks": {
                    "build": "SKIPPED",
                    "trust_boundary": "SKIPPED",
                    "correctness": "SKIPPED",
                    "repeatability": "SKIPPED",
                    "artifact_integrity": "PASS",
                    "remote_sha": "SKIPPED",
                },
            },
            "artifacts": [],
            "claim_boundary": ["synthetic artifact validator fixture only"],
            "verdict": "SOURCE_AUDIT_ONLY",
        }

    @staticmethod
    def _polynomial_record(contract: dict[str, object]) -> dict[str, object]:
        return {
            "degree": contract["degree"],
            "interval": copy.deepcopy(contract["interval"]),
            "required_depth": contract["openfhe_ps"]["required_depth"],
            "estimated_multiplications": contract["openfhe_ps"][
                "estimated_multiplications"
            ],
            "coefficient_sha256": contract["coefficient_sha256"],
        }

    def _reseal_artifacts(self) -> None:
        stdout_hash = validator.sha256_file(self.artifact_root / "stdout.log")
        metrics_hash = validator.sha256_file(self.artifact_root / "metrics.csv")
        (self.artifact_root / "SHA256SUMS").write_text(
            f"{stdout_hash}  stdout.log\n{metrics_hash}  metrics.csv\n",
            encoding="utf-8",
        )
        self.manifest["artifacts"] = [
            self._artifact_record(self.artifact_root, "stdout.log", "stdout"),
            self._artifact_record(self.artifact_root, "metrics.csv", "metrics"),
            self._artifact_record(self.artifact_root, "SHA256SUMS", "checksum"),
        ]

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _validate_current_manifest(self) -> None:
        loaded = validator.load_json(self.manifest_path)
        validator.validate_manifest(
            self.manifest_path,
            loaded,
            self.schema,
            verify_git=False,
        )

    def test_positive_source_audit_manifest(self) -> None:
        self._validate_current_manifest()

    def test_synthetic_m3_artifact_retains_live_git_dispatch(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        commit = "1" * 40
        manifest["git"]["local_commit"] = commit
        manifest["git"]["clean"] = False

        with tempfile.TemporaryDirectory(prefix="moai-m3-synthetic-snapshot-") as root:
            snapshot_root = Path(root) / "repo"
            snapshot_root.mkdir()
            (snapshot_root / ".git").mkdir()
            for record in manifest["inputs"]:
                path = record["path"]
                destination = snapshot_root / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes((REPO_ROOT / path).read_bytes())

            executable = snapshot_root / manifest["workload"]["executable_path"]
            executable.parent.mkdir(parents=True, exist_ok=True)
            executable.write_bytes(
                (REPO_ROOT / manifest["workload"]["executable_path"]).read_bytes()
            )

            snapshot_artifact_root = Path(root) / "artifact"
            snapshot_artifact_root.mkdir()
            for artifact in manifest["artifacts"]:
                path = artifact["path"]
                (snapshot_artifact_root / path).write_bytes(
                    (self.artifact_root / path).read_bytes()
                )
            snapshot_manifest = copy.deepcopy(manifest)
            snapshot_manifest["git"]["repository_root"] = str(snapshot_root)
            snapshot_manifest_path = snapshot_artifact_root / "manifest.json"
            snapshot_manifest_path.write_text(
                json.dumps(snapshot_manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            def fake_git(_repository_root: Path, arguments: list[str]) -> str:
                if arguments == ["rev-parse", "HEAD"]:
                    return commit
                if arguments == ["branch", "--show-current"]:
                    return "refactor/openfhe-cpu"
                if arguments == ["status", "--porcelain=v1"]:
                    return " M synthetic-fixture"
                raise AssertionError(f"unexpected git invocation: {arguments}")

            with (
                mock.patch.object(validator, "REPO_ROOT", snapshot_root),
                mock.patch.object(validator, "_run_git", side_effect=fake_git),
            ):
                validator.validate_manifest(
                    snapshot_manifest_path,
                    snapshot_manifest,
                    self.schema,
                    verify_git=True,
                )

    def test_resealed_runtime_counter_tamper_is_rejected(self) -> None:
        self.runtime_records[0]["rotations"] = 1
        self._write_stdout()
        self._reseal_artifacts()
        self._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "rotations mismatch",
        ):
            self._validate_current_manifest()

    def _assert_resealed_runtime_field_tamper_is_rejected(
        self,
        field: str,
        value: object,
    ) -> None:
        self.runtime_records[0][field] = value
        self._write_stdout()
        self._reseal_artifacts()
        self._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            f"{field} mismatch",
        ):
            self._validate_current_manifest()

    def test_resealed_runtime_interval_tamper_is_rejected(self) -> None:
        self._assert_resealed_runtime_field_tamper_is_rejected(
            "gelu_interval_min",
            -79.0,
        )

    def test_resealed_runtime_depth_tamper_is_rejected(self) -> None:
        self._assert_resealed_runtime_field_tamper_is_rejected(
            "softmax_exp_required_depth",
            5,
        )

    def test_resealed_runtime_multiplication_tamper_is_rejected(self) -> None:
        self._assert_resealed_runtime_field_tamper_is_rejected(
            "layernorm_invsqrt_estimated_multiplications",
            22,
        )

    def test_non_finite_elapsed_seconds_is_rejected(self) -> None:
        metrics_path = self.artifact_root / "metrics.csv"
        metrics_path.write_text(
            metrics_path.read_text(encoding="utf-8").replace("1.0,1000", "nan,1000", 1),
            encoding="utf-8",
        )
        self._reseal_artifacts()
        self._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "resource metrics",
        ):
            self._validate_current_manifest()

    def test_tampered_checksum_entry_is_rejected(self) -> None:
        checksum_path = self.artifact_root / "SHA256SUMS"
        checksum_path.write_text(
            checksum_path.read_text(encoding="utf-8").replace(
                validator.sha256_file(self.artifact_root / "stdout.log"),
                "0" * 64,
                1,
            ),
            encoding="utf-8",
        )
        for index, record in enumerate(self.manifest["artifacts"]):
            if record["path"] == "SHA256SUMS":
                self.manifest["artifacts"][index] = self._artifact_record(
                    self.artifact_root,
                    "SHA256SUMS",
                    "checksum",
                )
        self._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "SHA256SUMS digest disagrees",
        ):
            self._validate_current_manifest()

    def test_go_without_live_git_verification_is_rejected(self) -> None:
        self.manifest["verdict"] = "GO"
        self.manifest["gate"]["passed"] = True
        self.manifest["gate"]["checks"] = {
            key: "PASS" for key in self.manifest["gate"]["checks"]
        }
        self.manifest["git"].update(
            {
                "clean": True,
                "remote_name": "origin",
                "remote_ref": "refs/heads/refactor/openfhe-cpu",
                "remote_commit": "0" * 40,
            }
        )
        self._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "requires --verify-git live verification",
        ):
            self._validate_current_manifest()


class EncoderArtifactFixture:
    """Create complete synthetic M4 evidence without weakening source checks."""

    def __init__(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory(
            prefix="moai-m4-artifact-validator-"
        )
        self.artifact_root = Path(self.temporary_directory.name)
        self.manifest_path = self.artifact_root / "manifest.json"
        self.schema = validator.validate_schema(
            validator.load_json(REPO_ROOT / "docs/openfhe-artifact-schema.json")
        )
        profile_config = validator.load_json(
            REPO_ROOT / "config/paper_compat_feature_packed.json"
        )
        self.effective = profile_config["effective_profile"]
        self.effective_hash = validator.hashlib.sha256(
            validator.canonical_json_bytes(self.effective)
        ).hexdigest()
        self.runtime_records = self._make_m4_runtime_records()
        self.diagnostic_records = self._make_m4_diagnostic_records()
        self._write_stdout()
        self._write_metrics()
        self.manifest = self._make_manifest()
        self._reseal_artifacts()
        self._write_manifest()

    def cleanup(self) -> None:
        self.temporary_directory.cleanup()

    @staticmethod
    def _repository_record(path: str) -> dict[str, object]:
        resolved = REPO_ROOT / path
        return {
            "path": path,
            "sha256": validator.sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": "application/json",
            "role": "configuration",
        }

    def _trace_repository_records(self) -> list[dict[str, object]]:
        layer_ids = [validator.M4_LAYER_ID]
        records: list[dict[str, object]] = []
        required_files = ENCODER_TRACE_CONTRACT["required_files"]
        layer_by_id = {
            layer["layer_id"]: layer for layer in ENCODER_TRACE_CONTRACT["layers"]
        }
        for layer_id in layer_ids:
            for logical_name in sorted(required_files):
                contract_path = required_files[logical_name]["path"]
                path = (
                    Path("data") / f"layer_{layer_id}" / contract_path
                ).as_posix()
                resolved = REPO_ROOT / path
                records.append(
                    {
                        "path": path,
                        "sha256": layer_by_id[layer_id]["sha256"][logical_name],
                        "bytes": resolved.stat().st_size,
                        "media_type": "text/csv",
                        "role": (
                            "weights" if "/parms/" in f"/{contract_path}" else "trace"
                        ),
                    }
                )
        return records

    @staticmethod
    def _operation_counts() -> dict[str, int]:
        bootstraps = 25
        return {
            "rotations": validator.ENCODER_ROTATIONS_PER_LAYER,
            "ct_pt_multiplications": (
                validator.ENCODER_CT_PT_MULTIPLICATIONS_PER_LAYER
            ),
            "ct_ct_multiplications": (
                validator.ENCODER_CT_CT_MULTIPLICATIONS_PER_LAYER
            ),
            "explicit_rescale_requests": (
                validator.ENCODER_EXPLICIT_RESCALE_REQUESTS_PER_LAYER
            ),
            "chebyshev_evaluations": (
                validator.ENCODER_CHEBYSHEV_EVALUATIONS_PER_LAYER
            ),
            "estimated_polynomial_multiplications": (
                validator.ENCODER_ESTIMATED_POLYNOMIAL_MULTIPLICATIONS_PER_LAYER
            ),
            "bootstraps": bootstraps,
            "bootstrap_iterations": (
                bootstraps * validator.ENCODER_BOOTSTRAP_ITERATIONS_PER_CALL
            ),
        }

    def _identity(self) -> dict[str, object]:
        return {
            "profile": "paper_compat",
            "security_claim": "none",
            "parameter_sha256": self.effective_hash,
            "execution_mode": "server-only",
            "actual_trace_shape": [5, 768],
            "actual_trace_value_count": 3840,
            "feature_block_size": 1024,
            "checkpoint_decryption_owner": "client",
            "server_private_key_present": False,
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "multiplicative_depth": validator.M4_MULTIPLICATIVE_DEPTH,
            "max_observed_level": validator.M4_MAX_OBSERVED_LEVEL,
            "max_polynomial_depth": validator.M4_MAX_POLYNOMIAL_DEPTH,
        }

    def _make_m4_runtime_records(self) -> list[dict[str, object]]:
        records = []
        for repeat in range(5):
            checkpoints = [
                {
                    "name": name,
                    "level": (40, 32, 45, 29)[index],
                    "noise_scale_degree": 2,
                    "remaining_levels": (6, 14, 1, 17)[index],
                    "scale_bits": 100,
                    "ciphertext_count": 5,
                    "decryption_owner": "client",
                }
                for index, name in enumerate(validator.M4_CHECKPOINT_NAMES)
            ]
            records.append(
                {
                    "test": "openfhe_encoder_layer",
                    **self._identity(),
                    "encoder_layers": 1,
                    "layer_id": 1,
                    "input_level": 29,
                    "output_level": 29,
                    "remaining_levels": 17,
                    "relative_l2": 0.005 + repeat * 0.0001,
                    "cosine": 0.9998 - repeat * 0.00001,
                    "inactive_max_abs": 4e-7 + repeat * 1e-8,
                    "checkpoints": checkpoints,
                    "operation_counts": self._operation_counts(),
                }
            )
        return records

    def _make_m4_diagnostic_records(self) -> list[dict[str, object]]:
        diagnostics: list[dict[str, object]] = []
        for repeat, record in enumerate(self.runtime_records):
            counts = record["operation_counts"]
            diagnostics.append(
                {
                    "test": "openfhe_encoder_layer_smoke",
                    "profile": record["profile"],
                    "security_claim": record["security_claim"],
                    "parameter_sha256": record["parameter_sha256"],
                    "multiplicative_depth": record["multiplicative_depth"],
                    "max_observed_level": record["max_observed_level"],
                    "max_polynomial_depth": record["max_polynomial_depth"],
                    "layer": record["layer_id"],
                    "input_level": record["input_level"],
                    "tokens": 5,
                    "hidden_size": 768,
                    "intermediate_size": 3072,
                    "feature_block": 1024,
                    "fixture_load_ms": 10.0 + repeat,
                    "setup_keygen_ms": 100.0 + repeat,
                    "client_encrypt_ms": 20.0 + repeat,
                    "server_online_ms": 500.0 + repeat * 10.0,
                    "client_decrypt_validate_ms": 30.0 + repeat,
                    "relative_l2": record["relative_l2"],
                    "cosine": record["cosine"],
                    "max_absolute": 0.01 + repeat * 0.001,
                    "exact_trace_relative_l2": 0.006 + repeat * 0.0001,
                    "exact_trace_cosine": 0.9997 - repeat * 0.00001,
                    "exact_trace_max_absolute": 0.02 + repeat * 0.001,
                    "inactive_max_absolute": record["inactive_max_abs"],
                    "peak_rss_bytes": (1001 + repeat) * 1024,
                    "rotations": counts["rotations"],
                    "ct_pt_multiplications": counts["ct_pt_multiplications"],
                    "ct_ct_multiplications": counts["ct_ct_multiplications"],
                    "explicit_rescale_requests": counts[
                        "explicit_rescale_requests"
                    ],
                    "chebyshev_evaluations": counts[
                        "chebyshev_evaluations"
                    ],
                    "estimated_polynomial_multiplications": counts[
                        "estimated_polynomial_multiplications"
                    ],
                    "bootstraps": counts["bootstraps"],
                    "bootstrap_iterations": counts["bootstrap_iterations"],
                    "final_level": record["output_level"],
                    "final_remaining_levels": record["remaining_levels"],
                }
            )
        return diagnostics

    def _write_stdout(self) -> None:
        records: list[dict[str, object]] = []
        for diagnostic, target in zip(
            self.diagnostic_records, self.runtime_records
        ):
            records.extend((diagnostic, target))
        (self.artifact_root / "stdout.log").write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )

    def _write_metrics(self) -> None:
        fieldnames = [
            "run",
            "exit_code",
            "elapsed_seconds",
            "peak_rss_kib",
            "final_rel_l2",
            "final_cosine",
            "inactive_max_abs",
            "multiplicative_depth",
            "max_observed_level",
            "max_polynomial_depth",
            "checkpoint_metadata_sha256",
            "bootstraps",
            "bootstrap_iterations",
        ]
        fieldnames[4:4] = [
            *validator.M4_PHASE_LATENCY_FIELDS,
            "batch_total_ms",
            "batch_amortized_ms_per_token",
            "server_amortized_ms_per_token",
        ]
        with (self.artifact_root / "metrics.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index, record in enumerate(self.runtime_records, start=1):
                counts = record["operation_counts"]
                row = {
                    "run": index,
                    "exit_code": 0,
                    "elapsed_seconds": float(index),
                    "peak_rss_kib": 1000 + index,
                    "final_rel_l2": record["relative_l2"],
                    "final_cosine": record["cosine"],
                    "inactive_max_abs": record["inactive_max_abs"],
                    "multiplicative_depth": record["multiplicative_depth"],
                    "max_observed_level": record["max_observed_level"],
                    "max_polynomial_depth": record["max_polynomial_depth"],
                    "checkpoint_metadata_sha256": (
                        validator.checkpoint_metadata_sha256(record["checkpoints"])
                    ),
                    "bootstraps": counts["bootstraps"],
                    "bootstrap_iterations": counts["bootstrap_iterations"],
                }
                diagnostic = self.diagnostic_records[index - 1]
                batch_total_ms = float(index) * 1000.0
                row.update(
                    {
                        **{
                            field: diagnostic[field]
                            for field in validator.M4_PHASE_LATENCY_FIELDS
                        },
                        "batch_total_ms": batch_total_ms,
                        "batch_amortized_ms_per_token": (
                            batch_total_ms / validator.M4_TOKENS_PER_BATCH
                        ),
                        "server_amortized_ms_per_token": (
                            diagnostic["server_online_ms"]
                            / validator.M4_TOKENS_PER_BATCH
                        ),
                    }
                )
                writer.writerow(row)

    def _profile(self) -> dict[str, object]:
        path = REPO_ROOT / "config/paper_compat_feature_packed.json"
        return {
            "id": "paper_compat",
            "security_claim": "none",
            "warning": (
                "Research reproduction parameters only. "
                "Do not claim 128-bit security."
            ),
            "source_config_path": "config/paper_compat_feature_packed.json",
            "source_config_sha256": validator.sha256_file(path),
            "source_config_bytes": path.stat().st_size,
            "effective_profile_locator": "/effective_profile",
            "canonicalization": validator.CANONICALIZATION,
            "effective_profile_payload": copy.deepcopy(self.effective),
            "effective_profile_sha256": self.effective_hash,
        }

    def _contracts(self, input_by_path: dict[str, dict[str, object]]) -> dict[str, object]:
        common = {
            "approximation_config_path": "config/openfhe_approximations.json",
            "approximation_config_sha256": input_by_path[
                "config/openfhe_approximations.json"
            ]["sha256"],
            "encoder_trace_contract_path": "config/moai_encoder_trace.json",
            "encoder_trace_contract_sha256": input_by_path[
                "config/moai_encoder_trace.json"
            ]["sha256"],
            "profile_config_path": "config/paper_compat_feature_packed.json",
            "profile_config_sha256": input_by_path[
                "config/paper_compat_feature_packed.json"
            ]["sha256"],
        }
        trust = {
            "mode": "server-only",
            "trace_shape": [5, 768],
            "feature_block_size": 1024,
            "checkpoint_decryption_owner": "client",
            "server_private_key_present": False,
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "multiplicative_depth": validator.M4_MULTIPLICATIVE_DEPTH,
            "max_observed_level": validator.M4_MAX_OBSERVED_LEVEL,
            "max_polynomial_depth": validator.M4_MAX_POLYNOMIAL_DEPTH,
        }
        execution = {
            **trust,
            "quality_reference": (
                "config/moai_encoder_trace.json single-layer frozen polynomial oracle"
            ),
            "encoder_layers": 1,
            "layer_id": 1,
            "required_checkpoints": list(validator.M4_CHECKPOINT_NAMES),
        }
        return {
            **common,
            "execution": execution,
            "thresholds": copy.deepcopy(validator.M4_EXPECTED_THRESHOLDS),
        }

    def _timing_metrics(self) -> dict[str, object]:
        return {
            "repeat_count": 5,
            "successful_repeats": 5,
            "warmup_count": 1,
            "elapsed_seconds": {"minimum": 1.0, "median": 3.0, "maximum": 5.0},
            "peak_rss_kib_max": 1005,
            "actual_trace_shape": [5, 768],
            "actual_trace_value_count": 3840,
            "feature_block_size": 1024,
        }

    def _m4_metrics(self) -> dict[str, object]:
        records = self.runtime_records
        diagnostics = self.diagnostic_records

        def summarize(values: list[float]) -> dict[str, float]:
            return {
                "minimum": min(values),
                "median": statistics.median(values),
                "maximum": max(values),
            }

        checkpoints = []
        for index, name in enumerate(validator.M4_CHECKPOINT_NAMES):
            values = [record["checkpoints"][index] for record in records]
            checkpoints.append(
                {
                    "name": name,
                    "level_min": min(item["level"] for item in values),
                    "level_max": max(item["level"] for item in values),
                    "noise_scale_degree_min": min(
                        item["noise_scale_degree"] for item in values
                    ),
                    "noise_scale_degree_max": max(
                        item["noise_scale_degree"] for item in values
                    ),
                    "remaining_levels_min": min(
                        item["remaining_levels"] for item in values
                    ),
                    "remaining_levels_max": max(
                        item["remaining_levels"] for item in values
                    ),
                    "scale_bits_min": min(item["scale_bits"] for item in values),
                    "scale_bits_max": max(item["scale_bits"] for item in values),
                    "ciphertext_count": values[0]["ciphertext_count"],
                    "decryption_owner": "client",
                }
            )
        return {
            **self._timing_metrics(),
            "tokens_per_batch": validator.M4_TOKENS_PER_BATCH,
            "phase_latency_ms": {
                field: summarize(
                    [float(diagnostic[field]) for diagnostic in diagnostics]
                )
                for field in validator.M4_PHASE_LATENCY_FIELDS
            },
            "batch_total_ms": summarize(
                [float(index) * 1000.0 for index in range(1, 6)]
            ),
            "batch_amortized_ms_per_token": summarize(
                [
                    float(index) * 1000.0 / validator.M4_TOKENS_PER_BATCH
                    for index in range(1, 6)
                ]
            ),
            "server_amortized_ms_per_token": summarize(
                [
                    float(diagnostic["server_online_ms"])
                    / validator.M4_TOKENS_PER_BATCH
                    for diagnostic in diagnostics
                ]
            ),
            "input_level": 29,
            "output_level": 29,
            "remaining_levels": 17,
            "multiplicative_depth": validator.M4_MULTIPLICATIVE_DEPTH,
            "max_observed_level": validator.M4_MAX_OBSERVED_LEVEL,
            "max_polynomial_depth": validator.M4_MAX_POLYNOMIAL_DEPTH,
            "quality": {
                "relative_l2_max": max(item["relative_l2"] for item in records),
                "cosine_min": min(item["cosine"] for item in records),
                "inactive_max_abs_max": max(
                    item["inactive_max_abs"] for item in records
                ),
            },
            "checkpoints": checkpoints,
            "checkpoint_metadata_sha256": validator.M4_CHECKPOINT_METADATA_SHA256,
            "operation_counts": copy.deepcopy(records[0]["operation_counts"]),
        }

    def _make_manifest(self) -> dict[str, object]:
        inputs = [
            self._repository_record(path)
            for path in sorted(validator.REQUIRED_ENCODER_INPUTS)
        ]
        inputs.extend(self._trace_repository_records())
        input_by_path = {item["path"]: item for item in inputs}
        executable_path = validator.M4_WORKLOAD_EXECUTABLE_PATH
        executable = REPO_ROOT / executable_path
        return {
            "schema_version": 2,
            "run_id": "synthetic-m4-artifact-validator",
            "milestone": "M4",
            "started_at": "2026-07-29T10:00:00+09:00",
            "finished_at": "2026-07-29T10:10:00+09:00",
            "git": {
                "repository_root": str(REPO_ROOT),
                "branch": "refactor/openfhe-cpu",
                "local_commit": "0" * 40,
                "clean": False,
            },
            "commands": [
                {
                    "command": "synthetic encoder artifact generation",
                    "cwd": str(REPO_ROOT),
                    "exit_code": 0,
                    "phase": "artifact_generation",
                }
            ],
            "environment": {
                "os": "synthetic",
                "architecture": "x86_64",
                "compiler": "synthetic",
                "cmake": "synthetic",
                "python": sys.version.split()[0],
                "openfhe_version": "1.5.1",
                "openfhe_prefix": "/synthetic/openfhe",
                "wsl": True,
            },
            "profile": self._profile(),
            "inputs": inputs,
            "workload": {
                "scope": "synthetic M4 encoder validator fixture",
                "backend": "OpenFHE CKKS CPU",
                "executable_path": executable_path,
                "executable_sha256": validator.sha256_file(executable),
                "executable_bytes": executable.stat().st_size,
                "repeat_count": 5,
            },
            "contracts": self._contracts(input_by_path),
            "metrics": self._m4_metrics(),
            "gate": {
                "passed": False,
                "decision": "synthetic validator evidence only",
                "checks": {
                    "build": "SKIPPED",
                    "trust_boundary": "SKIPPED",
                    "correctness": "SKIPPED",
                    "repeatability": "SKIPPED",
                    "artifact_integrity": "PASS",
                    "remote_sha": "SKIPPED",
                },
            },
            "artifacts": [],
            "claim_boundary": ["synthetic encoder artifact validator fixture only"],
            "verdict": "SOURCE_AUDIT_ONLY",
        }

    def _artifact_record(self, path: str, role: str) -> dict[str, object]:
        resolved = self.artifact_root / path
        media_types = {
            "stdout.log": "text/plain",
            "metrics.csv": "text/csv",
            "SHA256SUMS": "text/plain",
        }
        return {
            "path": path,
            "sha256": validator.sha256_file(resolved),
            "bytes": resolved.stat().st_size,
            "media_type": media_types[path],
            "role": role,
        }

    def _reseal_artifacts(self) -> None:
        stdout_hash = validator.sha256_file(self.artifact_root / "stdout.log")
        metrics_hash = validator.sha256_file(self.artifact_root / "metrics.csv")
        (self.artifact_root / "SHA256SUMS").write_text(
            f"{stdout_hash}  stdout.log\n{metrics_hash}  metrics.csv\n",
            encoding="utf-8",
        )
        self.manifest["artifacts"] = [
            self._artifact_record("stdout.log", "stdout"),
            self._artifact_record("metrics.csv", "metrics"),
            self._artifact_record("SHA256SUMS", "checksum"),
        ]

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def rewrite_runtime(self) -> None:
        self._write_stdout()
        self._reseal_artifacts()
        self._write_manifest()

    def validate(self) -> None:
        validator.validate_manifest(
            self.manifest_path,
            validator.load_json(self.manifest_path),
            self.schema,
            verify_git=False,
        )


class M4ArtifactValidatorContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.sha256_patch = mock.patch.object(
            validator,
            "sha256_file",
            side_effect=_sha256_without_large_trace_reads,
        )
        self.sha256_patch.start()
        self.addCleanup(self.sha256_patch.stop)
        self.fixture = EncoderArtifactFixture()
        self.addCleanup(self.fixture.cleanup)

    def _promote_to_go_with_frozen_commands(self) -> None:
        timestamped = {
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "phase": "artifact_generation",
            "started_at": "2026-07-29T10:00:00+09:00",
            "finished_at": "2026-07-29T10:01:00+09:00",
        }
        untimed = {
            "cwd": str(REPO_ROOT),
            "exit_code": 0,
            "phase": "artifact_generation",
        }
        build_root = REPO_ROOT / "build-openfhe"
        executable = REPO_ROOT / validator.M4_WORKLOAD_EXECUTABLE_PATH
        manifest_path = self.fixture.manifest_path.resolve()
        commands = [
            {
                **untimed,
                "command": "git status --porcelain=v1 --untracked-files=normal",
            },
            {**untimed, "command": "git branch --show-current"},
            {**untimed, "command": "git rev-parse HEAD"},
            {
                **untimed,
                "command": (
                    "git ls-remote --exit-code origin "
                    "refs/heads/refactor/openfhe-cpu"
                ),
            },
            {
                **timestamped,
                "command": shlex.join(
                    [
                        "cmake",
                        "--build",
                        str(build_root),
                        "--clean-first",
                        "-j",
                        "4",
                    ]
                ),
            },
            {
                **timestamped,
                "command": shlex.join(
                    [
                        "ctest",
                        "--test-dir",
                        str(build_root),
                        "--output-on-failure",
                        "--no-tests=error",
                        "-R",
                        validator.M4_CTEST_PATTERN,
                    ]
                ),
            },
            {
                **timestamped,
                "command": shlex.join(["ldd", str(executable)]),
            },
        ]
        for index in range(6):
            commands.append(
                {
                    **timestamped,
                    "command": shlex.join(
                        [
                            "/usr/bin/time",
                            "--format=%M",
                            f"--output=/tmp/moai-m4-time-synthetic-{index}.txt",
                            str(executable),
                            "--data-root",
                            str(REPO_ROOT / "data"),
                            "--layer",
                            "1",
                            "--input-level",
                            "29",
                        ]
                    ),
                }
            )
        commands.append(
            {
                **untimed,
                "command": shlex.join(
                    [
                        sys.executable,
                        str(REPO_ROOT / "scripts" / "validate_openfhe_artifact.py"),
                        "--manifest",
                        str(manifest_path),
                        "--verify-git",
                    ]
                ),
            }
        )
        self.fixture.manifest["commands"] = commands
        self.fixture.manifest["verdict"] = "GO"
        self.fixture.manifest["gate"]["passed"] = True
        self.fixture.manifest["gate"]["checks"] = {
            key: "PASS" for key in self.fixture.manifest["gate"]["checks"]
        }
        self.fixture.manifest["git"].update(
            {
                "clean": True,
                "remote_name": "origin",
                "remote_ref": "refs/heads/refactor/openfhe-cpu",
                "remote_commit": "0" * 40,
            }
        )
        self.fixture._write_manifest()

    def test_positive_complete_server_only_layer_evidence(self) -> None:
        self.assertEqual(len(self.fixture.manifest["inputs"]), 4 + 37)
        self.fixture.validate()

    def test_m4_go_frozen_command_transcript_is_accepted(self) -> None:
        self._promote_to_go_with_frozen_commands()
        validator._verify_m4_go_command_transcript(
            self.fixture.manifest, self.fixture.manifest_path
        )

    def test_m4_go_command_transcript_rejects_missing_extra_reordered_or_arbitrary(
        self,
    ) -> None:
        self._promote_to_go_with_frozen_commands()
        baseline = copy.deepcopy(self.fixture.manifest["commands"])
        mutations = {
            "missing": lambda commands: commands.pop(),
            "extra": lambda commands: commands.append(copy.deepcopy(commands[-1])),
            "reordered": lambda commands: commands.__setitem__(
                slice(0, 2), [commands[1], commands[0]]
            ),
            "arbitrary": lambda commands: commands[4].update({"command": "true"}),
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name):
                commands = copy.deepcopy(baseline)
                mutate(commands)
                self.fixture.manifest["commands"] = commands
                with self.assertRaisesRegex(
                    validator.ValidationError, "commands|transcript"
                ):
                    validator._verify_m4_go_command_transcript(
                        self.fixture.manifest, self.fixture.manifest_path
                    )

    def test_m4_go_command_transcript_freezes_ctest_and_workload_arguments(self) -> None:
        self._promote_to_go_with_frozen_commands()
        baseline = copy.deepcopy(self.fixture.manifest["commands"])
        mutations = {
            "ctest": (5, " --no-tests=error", ""),
            "workload": (7, "--layer 1", "--layer 2"),
            "validator": (13, "--verify-git", "--schema"),
        }
        for name, (index, old, new) in mutations.items():
            with self.subTest(name=name):
                commands = copy.deepcopy(baseline)
                commands[index]["command"] = commands[index]["command"].replace(
                    old, new, 1
                )
                self.fixture.manifest["commands"] = commands
                with self.assertRaisesRegex(validator.ValidationError, "commands"):
                    validator._verify_m4_go_command_transcript(
                        self.fixture.manifest, self.fixture.manifest_path
                    )

    def test_missing_channel_scale_config_is_rejected(self) -> None:
        self.fixture.manifest["inputs"] = [
            record
            for record in self.fixture.manifest["inputs"]
            if record["path"] != "config/moai_trace_channel_scales.json"
        ]
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "missing frozen config"):
            self.fixture.validate()

    def test_missing_layer_one_trace_input_is_rejected(self) -> None:
        trace_path = next(
            record["path"]
            for record in self.fixture.manifest["inputs"]
            if record["path"].startswith("data/layer_1/")
        )
        self.fixture.manifest["inputs"] = [
            record
            for record in self.fixture.manifest["inputs"]
            if record["path"] != trace_path
        ]
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "frozen encoder trace set"):
            self.fixture.validate()

    def test_extra_encoder_trace_input_is_rejected(self) -> None:
        layer_one = next(
            record
            for record in self.fixture.manifest["inputs"]
            if record["path"].startswith("data/layer_1/")
        )
        layer_zero_path = layer_one["path"].replace("data/layer_1/", "data/layer_0/", 1)
        layer_zero = REPO_ROOT / layer_zero_path
        self.fixture.manifest["inputs"].append(
            {
                **layer_one,
                "path": layer_zero_path,
                "sha256": TRACE_SHA256_BY_PATH[layer_zero_path],
                "bytes": layer_zero.stat().st_size,
            }
        )
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "frozen encoder trace set"):
            self.fixture.validate()

    def test_trace_input_media_type_is_frozen(self) -> None:
        trace_record = next(
            record
            for record in self.fixture.manifest["inputs"]
            if record["role"] == "trace"
        )
        trace_record["media_type"] = "text/plain"
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "wrong media_type"):
            self.fixture.validate()

    def test_trace_input_role_is_frozen(self) -> None:
        weight_record = next(
            record
            for record in self.fixture.manifest["inputs"]
            if record["role"] == "weights"
        )
        weight_record["role"] = "trace"
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "wrong role"):
            self.fixture.validate()

    def test_workload_executable_is_frozen(self) -> None:
        script_path = "scripts/validate_openfhe_artifact.py"
        script = REPO_ROOT / script_path
        self.fixture.manifest["workload"].update(
            {
                "executable_path": script_path,
                "executable_sha256": validator.sha256_file(script),
                "executable_bytes": script.stat().st_size,
            }
        )
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "executable_path must be"):
            self.fixture.validate()

    def test_milestone_payload_dispatch_rejects_m5_label(self) -> None:
        self.fixture.manifest["milestone"] = "M5"
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "schema alternative"):
            self.fixture.validate()

    def test_m6_is_not_accepted_by_schema_v2(self) -> None:
        self.fixture.manifest["milestone"] = "M6"
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "schema alternative"):
            self.fixture.validate()

    def test_relaxed_m4_threshold_is_rejected(self) -> None:
        self.fixture.manifest["contracts"]["thresholds"]["relative_l2_max"] = 0.011
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "schema alternative"):
            self.fixture.validate()

    def test_server_private_key_is_rejected(self) -> None:
        self.fixture.runtime_records[0]["server_private_key_present"] = True
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "server_private_key_present"):
            self.fixture.validate()

    def test_effective_profile_hash_mismatch_is_rejected(self) -> None:
        self.fixture.runtime_records[0]["parameter_sha256"] = "0" * 64
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "parameter_sha256"):
            self.fixture.validate()

    def test_actual_trace_shape_mismatch_is_rejected(self) -> None:
        self.fixture.runtime_records[0]["actual_trace_shape"] = [1, 768]
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "actual_trace_shape"):
            self.fixture.validate()

    def test_actual_trace_value_count_placeholder_is_rejected(self) -> None:
        self.fixture.runtime_records[0]["actual_trace_value_count"] = 0
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(
            validator.ValidationError, "actual_trace_value_count"
        ):
            self.fixture.validate()

    def test_m4_rejects_the_m3_profile_source_binding(self) -> None:
        profile_config = validator.load_json(REPO_ROOT / "config/paper_compat.json")
        m3_effective = profile_config["openfhe_translation"][
            "m3_smoke_effective_runtime"
        ]["parameters"]
        m3_path = REPO_ROOT / "config/paper_compat.json"
        self.fixture.manifest["profile"].update(
            {
                "source_config_path": "config/paper_compat.json",
                "source_config_sha256": validator.sha256_file(m3_path),
                "source_config_bytes": m3_path.stat().st_size,
                "effective_profile_locator": (
                    "/openfhe_translation/m3_smoke_effective_runtime/parameters"
                ),
                "effective_profile_payload": m3_effective,
                "effective_profile_sha256": validator.hashlib.sha256(
                    validator.canonical_json_bytes(m3_effective)
                ).hexdigest(),
            }
        )
        self.fixture._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError, "schema alternative"
        ):
            self.fixture.validate()

    def test_m4_layer_one_is_frozen(self) -> None:
        self.fixture.runtime_records[0]["layer_id"] = 10
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "layer identity"):
            self.fixture.validate()

    def test_m4_level_schedule_is_frozen(self) -> None:
        self.fixture.runtime_records[0]["input_level"] = 0
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "input_level must be 29"):
            self.fixture.validate()

    def test_m4_target_depth_schedule_is_exact(self) -> None:
        expected = {
            "multiplicative_depth": validator.M4_MULTIPLICATIVE_DEPTH,
            "max_observed_level": validator.M4_MAX_OBSERVED_LEVEL,
            "max_polynomial_depth": validator.M4_MAX_POLYNOMIAL_DEPTH,
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.fixture.runtime_records[0][field] = value - 1
                self.fixture.rewrite_runtime()
                with self.assertRaisesRegex(validator.ValidationError, field):
                    self.fixture.validate()
                self.fixture.runtime_records[0][field] = value
                self.fixture.rewrite_runtime()

    def test_m4_diagnostic_depth_schedule_is_exact(self) -> None:
        expected = {
            "multiplicative_depth": validator.M4_MULTIPLICATIVE_DEPTH,
            "max_observed_level": validator.M4_MAX_OBSERVED_LEVEL,
            "max_polynomial_depth": validator.M4_MAX_POLYNOMIAL_DEPTH,
        }
        for field, value in expected.items():
            with self.subTest(field=field):
                self.fixture.diagnostic_records[0][field] = value - 1
                self.fixture.rewrite_runtime()
                with self.assertRaisesRegex(validator.ValidationError, field):
                    self.fixture.validate()
                self.fixture.diagnostic_records[0][field] = value
                self.fixture.rewrite_runtime()

    def test_m4_manifest_level_summary_is_required(self) -> None:
        self.fixture.manifest["metrics"]["remaining_levels"] = 16
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "schema alternative"):
            self.fixture.validate()

    def test_m4_single_layer_oracle_is_frozen(self) -> None:
        self.fixture.manifest["contracts"]["execution"]["quality_reference"] = (
            "config/moai_encoder_trace.json chained_polynomial_oracle"
        )
        self.fixture._write_manifest()
        with self.assertRaisesRegex(validator.ValidationError, "schema alternative"):
            self.fixture.validate()

    def test_missing_client_checkpoint_is_rejected(self) -> None:
        self.fixture.runtime_records[0]["checkpoints"].pop()
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "checkpoints"):
            self.fixture.validate()

    def test_checkpoint_scale_bits_is_strict_bounded_and_exact(self) -> None:
        expected = validator.M4_EXPECTED_CHECKPOINTS[0]
        for replacement in (0.0, float("nan"), 101.0, 99.0):
            with self.subTest(replacement=replacement):
                checkpoint = copy.deepcopy(expected)
                checkpoint["scale_bits"] = replacement
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    "scale_bits|metadata",
                ):
                    validator._verify_checkpoint(checkpoint, expected, "checkpoint")

    def test_every_checkpoint_metadata_field_is_exact(self) -> None:
        mutations = {
            "name": "encoder_output",
            "level": 41,
            "noise_scale_degree": 1,
            "remaining_levels": 4,
            "scale_bits": 99.0,
            "ciphertext_count": 4,
            "decryption_owner": "server",
        }
        expected = validator.M4_EXPECTED_CHECKPOINTS[0]
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                checkpoint = copy.deepcopy(expected)
                checkpoint[field] = replacement
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    field + "|metadata",
                ):
                    validator._verify_checkpoint(checkpoint, expected, "checkpoint")

    def test_checkpoint_metadata_hash_is_canonical_and_frozen(self) -> None:
        checkpoints = copy.deepcopy(self.fixture.runtime_records[0]["checkpoints"])
        self.assertEqual(
            validator.checkpoint_metadata_sha256(checkpoints),
            validator.M4_CHECKPOINT_METADATA_SHA256,
        )
        checkpoints[0]["scale_bits"] = 99.0
        self.assertNotEqual(
            validator.checkpoint_metadata_sha256(checkpoints),
            validator.M4_CHECKPOINT_METADATA_SHA256,
        )

    def test_checkpoint_noise_scale_degree_is_strict_and_positive(self) -> None:
        for replacement in (0, 1.0):
            with self.subTest(replacement=replacement):
                self.fixture.runtime_records[0]["checkpoints"][0][
                    "noise_scale_degree"
                ] = replacement
                self.fixture.rewrite_runtime()
                with self.assertRaisesRegex(
                    validator.ValidationError, "noise_scale_degree"
                ):
                    self.fixture.validate()
                self.fixture.runtime_records = (
                    self.fixture._make_m4_runtime_records()
                )
                self.fixture.diagnostic_records = (
                    self.fixture._make_m4_diagnostic_records()
                )
        checkpoint = copy.deepcopy(self.fixture.runtime_records[0]["checkpoints"][0])
        checkpoint["noise_scale_degree"] = float("nan")
        with self.assertRaisesRegex(validator.ValidationError, "noise_scale_degree"):
            validator._verify_checkpoint(
                checkpoint, validator.M4_EXPECTED_CHECKPOINTS[0], "checkpoint"
            )

    def test_checkpoint_remaining_levels_is_strict_and_positive(self) -> None:
        for replacement in (0, 1.0):
            with self.subTest(replacement=replacement):
                self.fixture.runtime_records[0]["checkpoints"][0][
                    "remaining_levels"
                ] = replacement
                self.fixture.rewrite_runtime()
                with self.assertRaisesRegex(
                    validator.ValidationError, "remaining_levels"
                ):
                    self.fixture.validate()
                self.fixture.runtime_records = (
                    self.fixture._make_m4_runtime_records()
                )
                self.fixture.diagnostic_records = (
                    self.fixture._make_m4_diagnostic_records()
                )
        checkpoint = copy.deepcopy(self.fixture.runtime_records[0]["checkpoints"][0])
        checkpoint["remaining_levels"] = float("nan")
        with self.assertRaisesRegex(validator.ValidationError, "remaining_levels"):
            validator._verify_checkpoint(
                checkpoint, validator.M4_EXPECTED_CHECKPOINTS[0], "checkpoint"
            )

    def test_manifest_checkpoint_metadata_must_match_stdout_summary(self) -> None:
        checkpoint = self.fixture.manifest["metrics"]["checkpoints"][0]
        checkpoint["noise_scale_degree_max"] += 1
        self.fixture._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "schema alternative|metrics.checkpoints differs from stdout",
        ):
            self.fixture.validate()

    def test_checkpoint_ciphertext_count_is_frozen(self) -> None:
        self.fixture.runtime_records[0]["checkpoints"][0]["ciphertext_count"] = 4
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "ciphertext_count must be 5"):
            self.fixture.validate()

    def test_rotation_count_is_frozen(self) -> None:
        self.fixture.runtime_records[0]["operation_counts"]["rotations"] = 6299
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(validator.ValidationError, "rotations must be 6300"):
            self.fixture.validate()

    def test_every_encoder_operation_count_is_frozen(self) -> None:
        expected = {
            "ct_pt_multiplications": 51865,
            "ct_ct_multiplications": 95,
            "explicit_rescale_requests": 800,
            "chebyshev_evaluations": 55,
            "estimated_polynomial_multiplications": 1150,
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.fixture.runtime_records[0]["operation_counts"][key] = value - 1
                self.fixture.rewrite_runtime()
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    rf"{key} must be {value}",
                ):
                    self.fixture.validate()
                self.fixture.runtime_records[0]["operation_counts"][key] = value
                self.fixture.rewrite_runtime()

    def test_wrong_bootstrap_count_is_rejected(self) -> None:
        self.fixture.runtime_records[0]["operation_counts"]["bootstraps"] = 24
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "bootstraps must be 25",
        ):
            self.fixture.validate()

    def test_wrong_bootstrap_iteration_count_is_rejected(self) -> None:
        self.fixture.runtime_records[0]["operation_counts"][
            "bootstrap_iterations"
        ] = 49
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(
            validator.ValidationError, "bootstrap_iterations must be 50"
        ):
            self.fixture.validate()

    def test_missing_or_nonfinite_phase_diagnostic_is_rejected(self) -> None:
        for replacement, message in (
            ("wrong_diagnostic", "diagnostic count differs"),
            ("openfhe_encoder_layer_smoke", "diagnostic count differs"),
        ):
            with self.subTest(replacement=replacement):
                if replacement == "wrong_diagnostic":
                    self.fixture.diagnostic_records[0]["test"] = replacement
                else:
                    self.fixture.diagnostic_records[0]["setup_keygen_ms"] = float("nan")
                self.fixture.rewrite_runtime()
                with self.assertRaisesRegex(validator.ValidationError, message):
                    self.fixture.validate()
                self.fixture.diagnostic_records = (
                    self.fixture._make_m4_diagnostic_records()
                )
                self.fixture.rewrite_runtime()

    def test_diagnostic_target_quality_mismatch_is_rejected(self) -> None:
        self.fixture.diagnostic_records[0]["relative_l2"] = 0.006
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(
            validator.ValidationError, "differs from the target runtime record"
        ):
            self.fixture.validate()

    def test_diagnostic_peak_rss_mismatch_is_rejected(self) -> None:
        self.fixture.diagnostic_records[0]["peak_rss_bytes"] += 1024
        self.fixture.rewrite_runtime()
        with self.assertRaisesRegex(
            validator.ValidationError, "diagnostic peak RSS exceeds peak_rss_kib"
        ):
            self.fixture.validate()

    def test_diagnostic_peak_rss_may_be_below_sealed_gnu_time_peak(self) -> None:
        self.fixture.diagnostic_records[0]["peak_rss_bytes"] -= 512 * 1024
        self.fixture.rewrite_runtime()
        self.fixture.validate()

    def test_csv_phase_and_amortized_latency_tampering_is_rejected(self) -> None:
        metrics_path = self.fixture.artifact_root / "metrics.csv"
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0]["server_online_ms"] = "501.5"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.fixture._reseal_artifacts()
        self.fixture._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError, "server_online_ms differs from stdout diagnostic"
        ):
            self.fixture.validate()

        self.fixture._write_metrics()
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
            fieldnames = list(rows[0])
        rows[0]["batch_amortized_ms_per_token"] = "201"
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        self.fixture._reseal_artifacts()
        self.fixture._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "batch_amortized_ms_per_token differs from its source metric",
        ):
            self.fixture.validate()

    def test_csv_depth_and_checkpoint_hash_tampering_is_rejected(self) -> None:
        metrics_path = self.fixture.artifact_root / "metrics.csv"
        mutations = {
            "multiplicative_depth": "46",
            "max_observed_level": "44",
            "max_polynomial_depth": "9",
            "checkpoint_metadata_sha256": "0" * 64,
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                self.fixture._write_metrics()
                with metrics_path.open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
                    fieldnames = list(rows[0])
                rows[0][field] = replacement
                with metrics_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                self.fixture._reseal_artifacts()
                self.fixture._write_manifest()
                with self.assertRaisesRegex(validator.ValidationError, field):
                    self.fixture.validate()

    def test_manifest_depth_and_checkpoint_hash_are_exact(self) -> None:
        mutations = {
            "multiplicative_depth": 46,
            "max_observed_level": 44,
            "max_polynomial_depth": 9,
            "checkpoint_metadata_sha256": "0" * 64,
        }
        baseline = copy.deepcopy(self.fixture.manifest["metrics"])
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                self.fixture.manifest["metrics"] = copy.deepcopy(baseline)
                self.fixture.manifest["metrics"][field] = replacement
                self.fixture._write_manifest()
                with self.assertRaisesRegex(
                    validator.ValidationError, "schema alternative|" + field
                ):
                    self.fixture.validate()

    def test_csv_missing_or_nonfinite_phase_latency_is_rejected(self) -> None:
        metrics_path = self.fixture.artifact_root / "metrics.csv"
        for mode, message in (
            ("missing", "metrics.csv columns differ"),
            ("nonfinite", "fixture_load_ms is not finite"),
        ):
            with self.subTest(mode=mode):
                self.fixture._write_metrics()
                with metrics_path.open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    rows = list(csv.DictReader(handle))
                    fieldnames = list(rows[0])
                if mode == "missing":
                    fieldnames.remove("fixture_load_ms")
                    for row in rows:
                        row.pop("fixture_load_ms")
                else:
                    rows[0]["fixture_load_ms"] = "nan"
                with metrics_path.open(
                    "w", encoding="utf-8", newline=""
                ) as handle:
                    writer = csv.DictWriter(handle, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                self.fixture._reseal_artifacts()
                self.fixture._write_manifest()
                with self.assertRaisesRegex(validator.ValidationError, message):
                    self.fixture.validate()

    def test_manifest_phase_latency_summary_tampering_is_rejected(self) -> None:
        self.fixture.manifest["metrics"]["phase_latency_ms"]["server_online_ms"][
            "median"
        ] += 1.0
        self.fixture._write_manifest()
        with self.assertRaisesRegex(
            validator.ValidationError,
            "phase_latency_ms.server_online_ms.median differs from metrics.csv",
        ):
            self.fixture.validate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
