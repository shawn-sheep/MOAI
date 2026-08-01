#!/usr/bin/env python3
"""Fail-closed tests for the frozen OpenFHE profile validator."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_openfhe_profile.py"
BASE_CONFIG = REPO_ROOT / "config" / "paper_compat.json"
FEATURE_CONFIG = REPO_ROOT / "config" / "paper_compat_feature_packed.json"
APPROXIMATION_CONFIG = REPO_ROOT / "config" / "openfhe_approximations.json"


class OpenFHEProfileValidatorTests(unittest.TestCase):
    def run_validator(
        self,
        feature_config: dict[str, object],
        approximation_config: dict[str, object] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_path = (
                Path(temporary_directory) / "paper_compat_feature_packed.json"
            )
            approximation_path = (
                Path(temporary_directory) / "openfhe_approximations.json"
            )
            feature_path.write_text(
                json.dumps(feature_config, ensure_ascii=False),
                encoding="utf-8",
            )
            approximation_path.write_text(
                json.dumps(
                    approximation_config or self.load_approximation_config(),
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            return subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR),
                    "--config",
                    str(BASE_CONFIG),
                    "--feature-config",
                    str(feature_path),
                    "--approximation-config",
                    str(approximation_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

    def load_feature_config(self) -> dict[str, object]:
        return json.loads(FEATURE_CONFIG.read_text(encoding="utf-8"))

    def load_approximation_config(self) -> dict[str, object]:
        return json.loads(APPROXIMATION_CONFIG.read_text(encoding="utf-8"))

    def sealed_feature_config(self) -> dict[str, object]:
        feature_config = self.load_feature_config()
        feature_config["validation_status"] = (
            "exact-three live M5 metadata schedule sealed; live M4 regression and full "
            "12-layer MOAI-observability-compatible prototype gates remain required"
        )
        candidate = feature_config["m5_schedule_candidate"]
        self.assertIsInstance(candidate, dict)
        candidate["status"] = "sealed_exact3"
        candidate["source"] = (
            "approved exact-three-layer live OpenFHE schedule evidence"
        )
        candidate["formal_schedule_sealed"] = True
        candidate["calibrated_layers"] = [0, 1, 2]
        candidate["exact3_seal"] = {
            "source": "approved exact-three-layer live OpenFHE schedule evidence",
            "evidence": {
                "run_id": "exact3-profile-validator-fixture",
                "relative_path": (
                    "results/openfhe/exact3-profile-validator-fixture"
                ),
                "manifest_sha256": "1" * 64,
                "sha256sums_sha256": "2" * 64,
                "layer_count": 3,
                "artifact_eligible": False,
                "exact3_gate_passed": True,
                "schedule_evidence_eligible": True,
                "formal_schedule_sealed": True,
            },
            "pre_seal_profile": {
                "path": "config/paper_compat_feature_packed.json",
                "sha256": "3" * 64,
                "size_bytes": 4096,
                "transition": (
                    "candidate profile captured by the exact3 sealer before the "
                    "one-way schedule-seal transition"
                ),
            },
        }
        override = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(override, dict)
        override["schedule_status"] = "sealed"
        return feature_config

    def test_frozen_feature_profile_passes(self) -> None:
        result = self.run_validator(self.load_feature_config())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"passed":true', result.stdout)

    def test_frozen_sealed_feature_profile_passes(self) -> None:
        result = self.run_validator(self.sealed_feature_config())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"passed":true', result.stdout)

    def test_mixed_candidate_and_sealed_state_fails_closed(self) -> None:
        feature_config = self.sealed_feature_config()
        feature_config["validation_status"] = (
            "two-layer live M5 schedule recorded as candidate; formal schedule unsealed "
            "until exact-three succeeds; live M4 regression and 12-layer "
            "MOAI-observability-compatible prototype gates remain required"
        )
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("internally inconsistent", result.stderr)

    def test_sealed_exact3_evidence_and_extra_keys_fail_closed(self) -> None:
        for label, mutate in (
            (
                "bad digest",
                lambda seal: seal["evidence"].__setitem__("manifest_sha256", "0"),
            ),
            ("extra key", lambda seal: seal.__setitem__("unexpected", True)),
        ):
            with self.subTest(label=label):
                feature_config = self.sealed_feature_config()
                candidate = feature_config["m5_schedule_candidate"]
                self.assertIsInstance(candidate, dict)
                seal = candidate["exact3_seal"]
                self.assertIsInstance(seal, dict)
                mutate(seal)
                result = self.run_validator(feature_config)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("exact3", result.stderr)

    def test_validation_status_drift_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        feature_config["validation_status"] = "five-token encoder gate required"
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed paper_compat identity/security contract drifted",
            result.stderr,
        )

    def test_m5_prototype_acceptance_threshold_is_frozen(self) -> None:
        feature_config = self.load_feature_config()
        contract = feature_config["m5_prototype_acceptance"]
        self.assertIsInstance(contract, dict)
        thresholds = contract["thresholds"]
        self.assertIsInstance(thresholds, dict)
        thresholds["inactive_or_cross_lane_max_absolute"] = 1.1e-3
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("M5 prototype acceptance contract drifted", result.stderr)

    def test_m5_prototype_acceptance_cannot_claim_legacy_threshold(self) -> None:
        feature_config = self.load_feature_config()
        contract = feature_config["m5_prototype_acceptance"]
        self.assertIsInstance(contract, dict)
        contract["legacy_basis"] = "legacy MOAI used an inactive 1e-3 threshold"
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("M5 prototype acceptance contract drifted", result.stderr)

    def test_m5_prototype_acceptance_rejects_scope_policy_and_extra_key_drift(self) -> None:
        for label, key, value in (
            ("scope", "scope", "all M5 diagnostics"),
            ("active policy", "active_quality_policy", "relaxed"),
            ("sentinel policy", "sentinel_range_policy", "disabled"),
            ("extra key", "unexpected", True),
        ):
            with self.subTest(label=label):
                feature_config = self.load_feature_config()
                contract = feature_config["m5_prototype_acceptance"]
                self.assertIsInstance(contract, dict)
                contract[key] = value
                result = self.run_validator(feature_config)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("M5 prototype acceptance contract drifted", result.stderr)

    def test_m5_candidate_cannot_claim_sealed_before_exact_three(self) -> None:
        feature_config = self.load_feature_config()
        feature_config["validation_status"] = (
            "two-layer live M5 schedule recorded as candidate; formal schedule "
            "unsealed until exact-three succeeds; live M4 regression and 12-layer "
            "MOAI-observability-compatible prototype gates remain required"
        )
        candidate = feature_config["m5_schedule_candidate"]
        self.assertIsInstance(candidate, dict)
        candidate["status"] = "candidate_unsealed"
        candidate["source"] = (
            "reviewed live two-layer metadata calibration output; no sealed artifact "
            "identity claimed"
        )
        candidate["formal_schedule_sealed"] = True
        candidate["calibrated_layers"] = [0, 1]
        candidate.pop("exact3_seal", None)
        layernorm = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(layernorm, dict)
        layernorm["schedule_status"] = (
            "two-layer live M5 candidate recorded but unsealed; exact-three, live M4 "
            "regression, and 12-layer MOAI-observability-compatible prototype gates "
            "remain required"
        )
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unvalidated candidate claims sealed", result.stderr)

    def test_m5_candidate_layer0_raw_tuple_tamper_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        candidate = feature_config["m5_schedule_candidate"]
        self.assertIsInstance(candidate, dict)
        regimes = candidate["layer_regimes"]
        self.assertIsInstance(regimes, dict)
        layer0 = regimes["layer0"]
        self.assertIsInstance(layer0, dict)
        checkpoints = layer0["checkpoint_metadata"]
        self.assertIsInstance(checkpoints, dict)
        checkpoints["raw_output"] = [29, 2, 17]
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("M5 schedule candidate contract drifted", result.stderr)

    def test_m5_candidate_later_delta_tamper_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        candidate = feature_config["m5_schedule_candidate"]
        self.assertIsInstance(candidate, dict)
        regimes = candidate["layer_regimes"]
        self.assertIsInstance(regimes, dict)
        later_layers = regimes["later_layers"]
        self.assertIsInstance(later_layers, dict)
        later_layers["used_level_deltas"][7] = 10
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("M5 schedule candidate contract drifted", result.stderr)

    def test_m5_candidate_cumulative_count_tamper_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        candidate = feature_config["m5_schedule_candidate"]
        self.assertIsInstance(candidate, dict)
        cumulative = candidate["two_layer_cumulative_operation_counts"]
        self.assertIsInstance(cumulative, dict)
        cumulative["ct_pt_multiplications"] = 103774
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("M5 schedule candidate contract drifted", result.stderr)

    def test_layernorm_override_field_drift_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        override = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(override, dict)
        override["post_bootstrap_required_depth"] = 11
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed LayerNorm override contract drifted",
            result.stderr,
        )

    def test_layernorm_preconditioner_drift_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        override = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(override, dict)
        override["bootstrap_preconditioner"] = 1024
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed LayerNorm override contract drifted",
            result.stderr,
        )

    def test_layernorm_scale_binding_drift_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        override = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(override, dict)
        binding = override["trace_scale_contract"]
        self.assertIsInstance(binding, dict)
        binding["values_sha256"] = "0" * 64
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed LayerNorm override contract drifted",
            result.stderr,
        )

    def test_layernorm_output_inactive_gate_relaxation_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        override = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(override, dict)
        gate = override["gate"]
        self.assertIsInstance(gate, dict)
        gate["output_inactive_max_absolute"] = math.nextafter(1e-6, math.inf)
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed LayerNorm override contract drifted",
            result.stderr,
        )

    def test_layernorm_output_inactive_gate_cannot_tighten_config_only(self) -> None:
        feature_config = self.load_feature_config()
        override = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(override, dict)
        gate = override["gate"]
        self.assertIsInstance(gate, dict)
        gate["output_inactive_max_absolute"] = 5e-7
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed LayerNorm override contract drifted",
            result.stderr,
        )

    def test_layernorm_output_inactive_gate_rejects_non_finite(self) -> None:
        feature_config = self.load_feature_config()
        override = feature_config["feature_packed_layernorm_override"]
        self.assertIsInstance(override, dict)
        gate = override["gate"]
        self.assertIsInstance(gate, dict)
        gate["output_inactive_max_absolute"] = math.nan
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed LayerNorm override contract drifted",
            result.stderr,
        )

    def test_layernorm_scale_tensor_tamper_fails_closed(self) -> None:
        approximation = self.load_approximation_config()
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
        self.assertIsInstance(contract, dict)
        contract["values"][0][0][0] = 96.0
        result = self.run_validator(self.load_feature_config(), approximation)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("positive power of two", result.stderr)

    def test_layernorm_raw_variance_hash_tamper_fails_closed(self) -> None:
        approximation = self.load_approximation_config()
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
        self.assertIsInstance(contract, dict)
        contract["raw_variance_sha256"] = "0" * 64
        result = self.run_validator(self.load_feature_config(), approximation)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale provenance drifted", result.stderr)

    def test_layernorm_runtime_dependency_tamper_fails_closed(self) -> None:
        approximation = self.load_approximation_config()
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
        self.assertIsInstance(contract, dict)
        contract["scope"]["runtime_activation_dependency"] = "plaintext"
        result = self.run_validator(self.load_feature_config(), approximation)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("scale scope drifted", result.stderr)

    def test_layernorm_contract_hash_tamper_fails_closed(self) -> None:
        approximation = self.load_approximation_config()
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
        self.assertIsInstance(contract, dict)
        contract["contract_sha256"] = "0" * 64
        result = self.run_validator(self.load_feature_config(), approximation)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("contract hash drifted", result.stderr)


if __name__ == "__main__":
    unittest.main()
