#!/usr/bin/env python3
"""Fail-closed tests for the complete MOAI encoder trace contract."""

from __future__ import annotations

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "validate_openfhe_encoder_trace.py"
MANIFEST_PATH = REPO_ROOT / "config" / "moai_encoder_trace.json"

SPEC = importlib.util.spec_from_file_location("validate_openfhe_encoder_trace", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import machinery failure
    raise RuntimeError(f"cannot import validator: {SCRIPT_PATH}")
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


class EncoderTraceContractTest(unittest.TestCase):
    """Run the expensive trace once, then exercise narrow failure paths."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = VALIDATOR.load_json(MANIFEST_PATH)
        cls.report = VALIDATOR.validate_contract(REPO_ROOT / "data", cls.manifest)

    def test_full_444_file_contract_passes(self) -> None:
        self.assertEqual(self.report["static"]["file_count"], 444)
        self.assertEqual(len(self.report["static"]["per_layer"]), 12)
        self.assertEqual(len(self.report["chained"]["per_layer"]), 12)
        final = self.report["chained"]["final"]
        self.assertLessEqual(final["relative_l2"], 0.05)
        self.assertGreaterEqual(final["cosine"], 0.99)

    def test_security_claim_is_locked_to_none(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["security_claim"] = "128-bit"
        with self.assertRaisesRegex(VALIDATOR.ContractError, "security_claim"):
            self._validate_structure(manifest)

    def test_required_shape_and_orientation_are_locked(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["required_files"]["qkt_output"]["shape"] = [5, 12, 5]
        with self.assertRaisesRegex(VALIDATOR.ContractError, "required_files"):
            self._validate_structure(manifest)

        manifest = copy.deepcopy(self.manifest)
        manifest["required_files"]["query_weight"]["orientation"] = "input_by_output"
        with self.assertRaisesRegex(VALIDATOR.ContractError, "required_files"):
            self._validate_structure(manifest)

    def test_sparse_channel_scales_are_bound_to_source(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["layers"][10]["channel_scale_overrides"]["1046"] = 14
        with self.assertRaisesRegex(VALIDATOR.ContractError, "channel_scale_overrides"):
            self._validate_structure(manifest)

    def test_all_37_hash_ids_are_required_for_every_layer(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        del manifest["layers"][4]["sha256"]["attention_output"]
        with self.assertRaisesRegex(VALIDATOR.ContractError, r"layers\[4\]\.sha256"):
            self._validate_structure(manifest)

    def test_frozen_thresholds_cannot_be_relaxed(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["thresholds"]["chained_polynomial_oracle_final"]["relative_l2"] = 0.051
        with self.assertRaisesRegex(VALIDATOR.ContractError, "thresholds"):
            self._validate_structure(manifest)

    def test_polynomial_hash_binding_cannot_be_replaced(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["approximation_binding"]["polynomials"]["gelu"][
            "coefficient_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(VALIDATOR.ContractError, "approximation_binding"):
            self._validate_structure(manifest)

    def test_trace_file_hash_mismatch_fails_before_numeric_validation(self) -> None:
        original_sha256 = VALIDATOR.sha256_file

        def changed_trace_digest(path: Path) -> str:
            if path.name == "K.csv" and "layer_0" in path.parts:
                return "0" * 64
            return original_sha256(path)

        with mock.patch.object(VALIDATOR, "sha256_file", side_effect=changed_trace_digest):
            with self.assertRaisesRegex(VALIDATOR.ContractError, "SHA-256 mismatch"):
                VALIDATOR.validate_contract(REPO_ROOT / "data", self.manifest)

    def test_frozen_oracle_summary_tamper_is_detected(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["chained_oracle_summary"]["final"]["relative_l2"] += 1e-4
        cached_result = (
            self.report["static"],
            self.report["chained"],
            [{} for _ in range(12)],
        )
        with mock.patch.object(VALIDATOR, "run_trace", return_value=cached_result):
            with self.assertRaisesRegex(VALIDATOR.ContractError, "chained_oracle_summary"):
                VALIDATOR.validate_contract(REPO_ROOT / "data", manifest)

    def test_duplicate_json_keys_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "duplicate.json"
            path.write_text('{"schema_version": 1, "schema_version": 1}\n', encoding="utf-8")
            with self.assertRaisesRegex(VALIDATOR.ContractError, "duplicate JSON key"):
                VALIDATOR.load_json(path)

    def test_nonfinite_summary_is_rejected(self) -> None:
        with self.assertRaisesRegex(VALIDATOR.ContractError, "changed"):
            VALIDATOR.compare_frozen(float("nan"), 0.0, "summary.metric")

    @staticmethod
    def _validate_structure(manifest: dict) -> None:
        scale_source, approximation_source = VALIDATOR.source_contracts()
        VALIDATOR.validate_manifest_structure(manifest, scale_source, approximation_source)


if __name__ == "__main__":
    unittest.main()
