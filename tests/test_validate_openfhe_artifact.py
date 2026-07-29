#!/usr/bin/env python3
"""Positive and tamper-negative tests for the schema-v2 artifact validator."""

from __future__ import annotations

import copy
import csv
import json
import statistics
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import validate_openfhe_artifact as validator  # noqa: E402


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
