#!/usr/bin/env python3
"""Focused unit tests for the M6 benchmark runner contract."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_openfhe_m6_benchmark.py"
spec = importlib.util.spec_from_file_location("run_openfhe_m6_benchmark", RUNNER_PATH)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {RUNNER_PATH}")
runner = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = runner
spec.loader.exec_module(runner)


def sample(seed: int = 0) -> dict[str, object]:
    setup = 100.0 + seed
    encrypt = 10.0 + seed
    server = 1000.0 + seed
    decrypt = 5.0 + seed
    online = encrypt + server + decrypt
    end_to_end = setup + online
    keys = {
        "serialization_format": runner.SERIALIZATION_FORMAT,
        "context_bytes": 100,
        "public_key_bytes": 200,
        "private_key_bytes": 150,
        "evaluation_multiplication_key_bytes": 300,
        "evaluation_automorphism_key_bytes": 400,
        "server_key_bundle_component_sum_bytes": 1000,
    }
    tensor = {
        "serialization_format": runner.SERIALIZATION_FORMAT,
        "ciphertext_count": 5,
        "ciphertext_component_sum_bytes": 500,
    }
    return {
        "test": "openfhe_encoder_12_layer_benchmark_sample",
        "backend": "OpenFHE CKKS CPU",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": runner.PROFILE_SHA256,
        "benchmark_sample": True,
        "timing_claim": True,
        "latency_kind": "benchmark",
        "claim_scope": "m6_12_layer_benchmark_sample",
        "execution_mode": "server-only",
        "encoder_layers": 12,
        "token_count": 5,
        "chain_mode": "ciphertext_output_to_next_input",
        "client_encrypt_calls": 1,
        "plaintext_activation_resets": 0,
        "server_layer_evaluations": 12,
        "inter_layer_refreshes": 11,
        "checkpoint_decryption_owner": "client",
        "observer_present_during_server_online": False,
        "checkpoint_decryptions": 0,
        "final_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "approximation_range_status": (
            "prevalidated_by_bound_m5_artifact_not_observed_in_sample"
        ),
        "timing_ms": {
            "fixture_load_oracle": 2.0 + seed,
            "setup_keygen": setup,
            "client_encrypt": encrypt,
            "server_online": server,
            "client_decrypt": decrypt,
            "client_validate": 3.0 + seed,
            "serialized_size_measurement": 4.0 + seed,
            "online_batch": online,
            "online_batch_amortized_per_token": online / 5.0,
            "end_to_end_batch": end_to_end,
            "end_to_end_amortized_per_token": end_to_end / 5.0,
            "server_online_amortized_per_token": server / 5.0,
        },
        "correctness": {
            "relative_l2": 0.001 + seed * 1e-5,
            "cosine": 0.99999,
            "max_absolute": 0.01,
            "inactive_max_abs": 1e-6,
            "finite": True,
            "passed": True,
        },
        "serialized_sizes": {
            "keys": keys,
            "encrypted_input": copy.deepcopy(tensor),
            "final_output": copy.deepcopy(tensor),
        },
        "final_metadata": {
            "level": 30,
            "noise_scale_degree": 2,
            "remaining_levels": 16,
            "scale_bits": 100.0000001,
            "expected_scale_bits": 100,
            "ciphertext_count": 5,
        },
        "operation_counts": copy.deepcopy(runner.EXPECTED_COUNTS),
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "peak_rss_bytes": 1024 * 1024,
        "peak_rss_scope": "process_high_water_mark",
        "passed": True,
    }


def executed(index: int) -> runner.ExecutedSample:
    value = sample(index)
    return runner.ExecutedSample(
        phase="measured",
        index=index,
        sample=value,
        stdout=json.dumps(value) + "\n",
        stderr="",
        time_line=(
            f'{{"phase":"measured","index":{index},"elapsed_seconds":2.0,'
            '"max_rss_kib":2048,"exit_status":0}'
        ),
        external_wall_seconds=2.0 + index,
        external_peak_rss_bytes=2 * 1024 * 1024,
        command={"argv": ["synthetic"], "cwd": str(REPO_ROOT), "returncode": 0},
    )


class BenchmarkSampleTests(unittest.TestCase):
    def test_frozen_hash_constants_are_sha256(self) -> None:
        for value in (
            runner.PROFILE_SHA256,
            runner.M5_MANIFEST_SHA256,
            runner.M5_SHA256SUMS_SHA256,
        ):
            self.assertIsNotNone(runner.SHA256_PATTERN.fullmatch(value))

    def test_valid_sample_and_online_contract(self) -> None:
        value = sample()
        self.assertIs(runner._validate_benchmark_sample(value), value)

    def test_rejects_online_time_that_excludes_client_decrypt(self) -> None:
        value = sample()
        value["timing_ms"]["online_batch"] -= 1.0  # type: ignore[index]
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "online batch"):
            runner._validate_benchmark_sample(value)

    def test_rejects_end_to_end_time_with_uncontracted_phases(self) -> None:
        value = sample()
        value["timing_ms"]["end_to_end_batch"] += 1.0  # type: ignore[index]
        value["timing_ms"]["end_to_end_amortized_per_token"] += 0.2  # type: ignore[index]
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "end-to-end batch"):
            runner._validate_benchmark_sample(value)

    def test_rejects_failed_correctness(self) -> None:
        value = sample()
        value["correctness"]["relative_l2"] = 0.051  # type: ignore[index]
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "correctness"):
            runner._validate_benchmark_sample(value)

    def test_rejects_server_bundle_sum_that_includes_private_key(self) -> None:
        value = sample()
        value["serialized_sizes"]["keys"][  # type: ignore[index]
            "server_key_bundle_component_sum_bytes"
        ] += 150
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "component sum"):
            runner._validate_benchmark_sample(value)

    def test_extracts_exactly_one_sample_json(self) -> None:
        encoded = json.dumps(sample())
        self.assertEqual(runner._extract_unique_sample(encoded + "\n", "unit"), sample())
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "exactly one"):
            runner._extract_unique_sample(encoded + "\n" + encoded + "\n", "unit")

    def test_rejects_duplicate_sample_json_keys(self) -> None:
        encoded = json.dumps(sample())
        duplicated = encoded[:-1] + ',"test":"openfhe_encoder_12_layer_benchmark_sample"}'
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "duplicate JSON key"):
            runner._extract_unique_sample(duplicated + "\n", "unit")


class StatisticsAndRepeatTests(unittest.TestCase):
    def test_median_mad_and_token_amortization(self) -> None:
        result = runner._timing_summary([1.0, 2.0, 3.0, 4.0, 100.0])
        self.assertEqual(result["batch"]["median"], 3.0)
        self.assertEqual(result["batch"]["mad"], 1.0)
        self.assertEqual(result["amortized_per_token"]["median"], 0.6)

    def test_metrics_require_five_measured_samples(self) -> None:
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "exactly five"):
            runner._metrics([executed(index) for index in range(1, 5)])

    def test_metrics_freeze_bytes_and_counts_across_five(self) -> None:
        records = [executed(index) for index in range(1, 6)]
        metrics = runner._metrics(records)
        self.assertEqual(metrics["measured_count"], 5)
        self.assertEqual(metrics["operation_counts"], runner.EXPECTED_COUNTS)
        records[4].sample["serialized_sizes"]["final_output"][  # type: ignore[index]
            "ciphertext_component_sum_bytes"
        ] += 1
        with self.assertRaisesRegex(runner.BenchmarkRunnerError, "serialized-size"):
            runner._metrics(records)

    def test_runtime_commands_are_independent_benchmark_processes(self) -> None:
        config = runner.RunnerConfig(run_id="unit")
        warmup = runner._runtime_command(config, Path("/tmp/warmup.time"), "warmup", 0)
        measured = [
            runner._runtime_command(
                config,
                Path(f"/tmp/measured-{index}.time"),
                "measured",
                index,
            )
            for index in range(1, 6)
        ]
        self.assertEqual(warmup[-1], "--benchmark-sample")
        self.assertEqual(len(measured), 5)
        self.assertEqual(len({tuple(command) for command in measured}), 5)
        self.assertTrue(all(command[0] == "/usr/bin/time" for command in measured))

    def test_failure_artifact_is_preserved_and_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging = root / ".unit.staging"
            failed = root / "unit-failed"
            staging.mkdir()
            runner._atomic_write(staging / "stdout.log", b"partial evidence\n")
            runner._preserve_failure_artifact(
                staging,
                failed,
                RuntimeError("synthetic failure"),
                [executed(1)],
            )
            self.assertFalse(staging.exists())
            self.assertEqual((failed / "stdout.log").read_text(), "partial evidence\n")
            failure = (failed / "failure.txt").read_text()
            self.assertIn("artifact_eligible=false", failure)
            self.assertIn("status=FAILED_INCOMPLETE", failure)

    def test_interrupted_sample_preserves_streamed_partial_raw_logs(self) -> None:
        def interrupting_run(*args: object, **kwargs: object) -> object:
            stdout = kwargs["stdout"]
            stderr = kwargs["stderr"]
            stdout.write(b"partial stdout before interruption\n")
            stderr.write(b"partial stderr before interruption\n")
            raise KeyboardInterrupt("synthetic interruption")

        with tempfile.TemporaryDirectory() as temporary:
            staging = Path(temporary)
            config = runner.RunnerConfig(run_id="unit")
            with mock.patch.object(runner.subprocess, "run", side_effect=interrupting_run):
                with self.assertRaises(KeyboardInterrupt):
                    runner._run_sample(config, staging, "warmup", 0)
            stdout_path, stderr_path, _ = runner._phase_raw_paths(staging, "warmup", 0)
            self.assertEqual(
                stdout_path.read_text(),
                "partial stdout before interruption\n",
            )
            self.assertEqual(
                stderr_path.read_text(),
                "partial stderr before interruption\n",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
