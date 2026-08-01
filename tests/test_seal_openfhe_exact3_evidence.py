#!/usr/bin/env python3
"""Unit tests for the bounded exact-three diagnostic evidence sealer."""

from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
import uuid
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(REPO_ROOT / "scripts"))

import seal_openfhe_exact3_evidence as sealer  # noqa: E402


DERIVED_NAMES = ("metrics.csv", "manifest.json", "SHA256SUMS")
PEAK_RSS_KIB = 123456


def _quality(layer_id: int, offset: float = 0.0) -> dict[str, float]:
    return {
        "relative_l2": 0.001 + layer_id * 0.0001 + offset,
        "cosine": 0.99999 - layer_id * 0.000001,
        "max_absolute": 0.01 + layer_id * 0.001 + offset,
    }


def _expected_cumulative(layer_id: int) -> dict[str, int]:
    return sealer._added_counts(
        sealer._scaled_counts(sealer.LAYER_COUNTS, layer_id + 1),
        sealer._scaled_counts(sealer.REFRESH_COUNTS, min(layer_id + 1, 2)),
    )


def _layer_record(layer_id: int) -> dict[str, object]:
    metadata = sealer.LAYER0_METADATA if layer_id == 0 else sealer.LATER_LAYER_METADATA
    used_levels = (
        sealer.LAYER0_USED_LEVELS if layer_id == 0 else sealer.LATER_USED_LEVELS
    )
    used_deltas = (
        sealer.LAYER0_USED_LEVEL_DELTAS
        if layer_id == 0
        else sealer.LATER_USED_LEVEL_DELTAS
    )
    inactive_maximum = (layer_id + 1) * 1e-8
    return {
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
        **copy.deepcopy(metadata),
        "metadata_used_levels": copy.deepcopy(used_levels),
        "metadata_used_level_deltas": copy.deepcopy(used_deltas),
        "input_quality": _quality(layer_id),
        "output_quality": _quality(layer_id, 0.00001),
        "exact_trace_diagnostic": _quality(layer_id, 0.00002),
        "inactive_max_abs": inactive_maximum,
        "inactive_zero_checkpoint_count": 28,
        "inactive_sentinel_max_error": 0.0,
        "inactive_polynomial_sentinel_ranges": {
            "softmax_denominator": [1.0, 1.0],
            "ln1_normalized_variance": [1.0, 1.0],
            "ln2_normalized_variance": [1.0, 1.0],
        },
        "inactive_sentinel_range_status": "passed",
        "encrypted_polynomial_input_ranges": {
            "softmax_shifted_logits": [-12.0, 4.0],
            "softmax_denominator": [0.1, 70.0],
            "ln1_normalized_variance": [0.6, 1400.0],
            "gelu_input": [-70.0, 120.0],
            "ln2_normalized_variance": [0.7, 1300.0],
        },
        "refresh_operation_counts": copy.deepcopy(
            sealer.REFRESH_COUNTS if layer_id < 2 else sealer.ZERO_COUNTS
        ),
        "layer_operation_counts": copy.deepcopy(sealer.LAYER_COUNTS),
        "cumulative_operation_counts": _expected_cumulative(layer_id),
        "range_validation_owner": "client",
        "checkpoint_decryption_owner": "client",
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "finite": True,
        "range_status": "passed",
        "diagnostic_gates_passed": True,
    }


def _summary(layers: list[dict[str, object]]) -> dict[str, object]:
    final = layers[-1]
    quality = final["output_quality"]
    assert isinstance(quality, dict)
    return {
        "test": "openfhe_encoder_exact_prefix",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": sealer.PROFILE_SHA256,
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
        "fixture_load_oracle_ms": 10.0,
        "setup_keygen_ms": 20.0,
        "client_encrypt_ms": 30.0,
        "server_online_diagnostic_ms": 100.0,
        "client_checkpoint_validate_ms": 40.0,
        "relative_l2": quality["relative_l2"],
        "cosine": quality["cosine"],
        "max_absolute": quality["max_absolute"],
        "inactive_max_abs": max(layer["inactive_max_abs"] for layer in layers),
        "inactive_sentinel_max_error": 0.0,
        "inactive_sentinel_range_status": "all_client_validated",
        "final_metadata": copy.deepcopy(final["raw_output_metadata"]),
        "operation_counts": copy.deepcopy(sealer.FINAL_COUNTS),
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "peak_rss_bytes": PEAK_RSS_KIB * 1024,
        "timing_claim": False,
        "latency_kind": "non_benchmark_diagnostic",
        "finite": True,
        "passed": True,
        "diagnostic_gates_passed": True,
    }


def _records() -> list[dict[str, object]]:
    layers = [_layer_record(layer_id) for layer_id in range(3)]
    return [*layers, _summary(layers)]


def _stdout_bytes(
    records: list[dict[str, object]],
    *,
    allow_nan: bool = False,
) -> bytes:
    lines = [sealer.PROFILE_WARNING]
    lines.extend(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=allow_nan,
        )
        for record in records
    )
    return ("\n".join(lines) + "\n").encode("utf-8")


def _time_bytes(*, exit_status: int = 0, swaps: int = 0) -> bytes:
    command = " ".join(sealer.EXPECTED_RUNTIME_ARGV)
    return (
        f'\tCommand being timed: "{command}"\n'
        "\tUser time (seconds): 100.00\n"
        "\tSystem time (seconds): 2.00\n"
        "\tPercent of CPU this job got: 800%\n"
        "\tElapsed (wall clock) time (h:mm:ss or m:ss): 2:03.50\n"
        "\tMaximum resident set size (kbytes): "
        f"{PEAK_RSS_KIB}\n"
        "\tAverage resident set size (kbytes): 0\n"
        "\tMajor (requiring I/O) page faults: 0\n"
        f"\tSwaps: {swaps}\n"
        f"\tExit status: {exit_status}\n"
    ).encode("utf-8")


def _write_raw_files(
    root: Path,
    records: list[dict[str, object]] | None = None,
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "stdout.log").write_bytes(_stdout_bytes(records or _records()))
    (root / "stderr.log").write_bytes(b"")
    (root / "time.log").write_bytes(_time_bytes())


def _candidate_profile() -> dict[str, object]:
    return {
        "profile_id": "paper_compat",
        "m5_prototype_acceptance": copy.deepcopy(sealer.M5_PROTOTYPE_ACCEPTANCE),
        "validation_status": sealer.PROFILE_TRANSITION_PRE_STATE_VALUES[
            "/validation_status"
        ],
        "m5_schedule_candidate": {
            "status": sealer.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/status"
            ],
            "source": sealer.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/m5_schedule_candidate/source"
            ],
            "formal_schedule_sealed": False,
            "calibrated_layers": [0, 1],
            "immutable_note": "must survive byte-semantically",
        },
        "feature_packed_layernorm_override": {
            "schedule_status": sealer.PROFILE_TRANSITION_PRE_STATE_VALUES[
                "/feature_packed_layernorm_override/schedule_status"
            ],
            "trace_scale_contract": {"contract_id": "fixture"},
        },
        "unrelated": {"typed": True, "value": 7},
    }


def _candidate_profile_bytes() -> bytes:
    text = json.dumps(
        _candidate_profile(),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    )
    text = text.replace(
        '    "calibrated_layers": [\n      0,\n      1\n    ],',
        '    "calibrated_layers": [0, 1],',
    )
    return (text + "\n").encode("utf-8")


def _file_record(path: str, payload: bytes, inode: int) -> dict[str, object]:
    return {
        "path": path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "device": 1,
        "inode": inode,
        "mode": 0o100644,
        "link_count": 1,
        "mtime_ns": 1000 + inode,
        "ctime_ns": 2000 + inode,
    }


def _profile_evidence(run_id: str = "exact3-fixture") -> dict[str, object]:
    return {
        "run_id": run_id,
        "relative_path": f"results/openfhe/{run_id}",
        "manifest_sha256": "1" * 64,
        "sha256sums_sha256": "2" * 64,
        "layer_count": 3,
        "artifact_eligible": False,
        "exact3_gate_passed": True,
        "schedule_evidence_eligible": True,
        "formal_schedule_sealed": True,
    }


class Exact3EvidenceSealerTest(unittest.TestCase):
    def setUp(self) -> None:
        sealer.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        self.run_id = f"exact3-sealer-test-{uuid.uuid4().hex}"
        self.run_root = sealer.OUTPUT_ROOT / self.run_id
        self.stdout_data = _stdout_bytes(_records())
        self.stderr_data = b""
        self.time_data = _time_bytes()
        self.child_returncode = 0
        self.profile_transition_calls: list[tuple[object, object]] = []

        profile_payload = _candidate_profile_bytes()
        profile_record = _file_record(sealer.PROFILE_PATH, profile_payload, 12)
        source_record = _file_record("src/openfhe/fixture.cpp", b"source\n", 10)
        config_record = _file_record("config/fixture.json", b"{}\n", 11)
        executable_record = _file_record(
            sealer.EXECUTABLE_PATH,
            b"ELF-fixture",
            13,
        )
        executable_record["mode"] = 0o100755
        transition = sealer._profile_transition_record(
            _candidate_profile(),
            profile_record,
        )
        self.provenance = sealer.ProvenanceSnapshot(
            source_files=(source_record,),
            config_files=(config_record, profile_record),
            source_manifest_sha256="3" * 64,
            config_manifest_sha256="4" * 64,
            executable=executable_record,
            trace_scale={
                "source_path": sealer.TRACE_SCALE_SOURCE_PATH,
                "json_locator": sealer.TRACE_SCALE_LOCATOR,
                "contract_id": sealer.TRACE_SCALE_CONTRACT_ID,
                "contract_sha256": sealer.TRACE_SCALE_CONTRACT_SHA256,
                "values_sha256": sealer.TRACE_SCALE_VALUES_SHA256,
                "raw_variance_sha256": sealer.TRACE_SCALE_RAW_VARIANCE_SHA256,
            },
            profile_transition=transition,
        )
        self.git = sealer.GitSnapshot(
            manifest={
                "head": "a" * 40,
                "branch": "refactor/openfhe-cpu",
                "detached": False,
                "clean": False,
                "snapshot_semantics": sealer.GIT_SNAPSHOT_SEMANTICS,
                "status_porcelain_v1_z_sha256": "5" * 64,
                "tracked_diff": {
                    "command": "git diff --binary HEAD -- .",
                    "sha256": "6" * 64,
                    "size_bytes": 1,
                },
                "untracked_source_manifest": {
                    "selection": "fixture",
                    "canonicalization": "sorted-key compact UTF-8 JSON",
                    "sha256": "7" * 64,
                    "entries": [],
                },
            },
            fingerprint=("a" * 40, "refactor/openfhe-cpu", "5" * 64, "6" * 64, "7" * 64),
        )

    def tearDown(self) -> None:
        if self.run_root.exists() and not self.run_root.is_symlink():
            shutil.rmtree(self.run_root)
        elif self.run_root.is_symlink():
            self.run_root.unlink()
        for staging in sealer.OUTPUT_ROOT.glob(
            f".exact3-stage-{self.run_id}-*"
        ):
            if staging.is_dir() and not staging.is_symlink():
                shutil.rmtree(staging)
            else:
                staging.unlink()

    def _fake_execute(self, staging_root: Path) -> sealer.LaunchEvidence:
        (staging_root / "stdout.log").write_bytes(self.stdout_data)
        (staging_root / "stderr.log").write_bytes(self.stderr_data)
        (staging_root / "time.log").write_bytes(self.time_data)
        return sealer.LaunchEvidence(
            argv=(
                sealer.TIME_EXECUTABLE,
                "-v",
                "-o",
                str(staging_root / "time.log"),
                "--",
                *sealer.EXPECTED_RUNTIME_ARGV,
            ),
            started_at="2026-07-31T10:00:00.000001+09:00",
            finished_at="2026-07-31T10:02:03.500001+09:00",
            returncode=self.child_returncode,
        )

    def _fake_profile_transition(
        self,
        provenance: sealer.ProvenanceSnapshot,
        evidence: dict[str, object],
    ) -> dict[str, object]:
        self.profile_transition_calls.append((provenance, copy.deepcopy(evidence)))
        return {"path": sealer.PROFILE_PATH}

    def _invoke(
        self,
        *,
        executor: object | None = None,
        git_sequence: list[sealer.GitSnapshot] | None = None,
        provenance_sequence: list[sealer.ProvenanceSnapshot] | None = None,
        profile_transition: object | None = None,
    ) -> dict[str, object]:
        execute_effect = executor or self._fake_execute
        git_effect = git_sequence or [self.git, self.git, self.git, self.git]
        provenance_effect = provenance_sequence or [
            self.provenance,
            self.provenance,
            self.provenance,
            self.provenance,
        ]
        profile_effect = profile_transition or self._fake_profile_transition
        result: Path | None = None
        error: Exception | None = None
        with (
            patch.object(
                sealer,
                "_execute_exact3_once",
                side_effect=execute_effect,
            ) as execute_mock,
            patch.object(sealer, "_git_snapshot", side_effect=git_effect) as git_mock,
            patch.object(
                sealer,
                "_provenance_snapshot",
                side_effect=provenance_effect,
            ) as provenance_mock,
            patch.object(
                sealer,
                "_apply_profile_transition",
                side_effect=profile_effect,
            ) as profile_mock,
        ):
            try:
                result = sealer.run_and_seal(self.run_root)
            except Exception as caught:  # test helper intentionally captures failures
                error = caught
        return {
            "result": result,
            "error": error,
            "execute_calls": execute_mock.call_count,
            "git_calls": git_mock.call_count,
            "provenance_calls": provenance_mock.call_count,
            "profile_calls": profile_mock.call_count,
        }

    def _records(self) -> list[dict[str, object]]:
        lines = self.stdout_data.decode("utf-8").splitlines()
        return [json.loads(line) for line in lines[1:]]

    def _set_records(
        self,
        records: list[dict[str, object]],
        *,
        allow_nan: bool = False,
    ) -> None:
        self.stdout_data = _stdout_bytes(records, allow_nan=allow_nan)

    def _assert_no_staging(self) -> None:
        self.assertEqual(
            list(sealer.OUTPUT_ROOT.glob(f".exact3-stage-{self.run_id}-*")),
            [],
        )

    def _assert_rejected_without_publication(self) -> dict[str, object]:
        outcome = self._invoke()
        self.assertIsInstance(outcome["error"], sealer.SealError)
        self.assertFalse(self.run_root.exists())
        self._assert_no_staging()
        self.assertEqual(outcome["execute_calls"], 1)
        self.assertEqual(outcome["profile_calls"], 0)
        return outcome

    def test_happy_fixture_runs_once_and_publishes_complete_bundle(self) -> None:
        outcome = self._invoke()
        self.assertIsNone(outcome["error"])
        self.assertEqual(outcome["result"], self.run_root / "manifest.json")
        self.assertEqual(outcome["execute_calls"], 1)
        self.assertEqual(outcome["git_calls"], 4)
        self.assertEqual(outcome["provenance_calls"], 4)
        self.assertEqual(outcome["profile_calls"], 1)
        self.assertEqual(len(self.profile_transition_calls), 1)
        self.assertEqual(
            {path.name for path in self.run_root.iterdir()},
            set(sealer.RAW_NAMES) | set(sealer.OUTPUT_NAMES),
        )
        self._assert_no_staging()

        manifest_path = self.run_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "diagnostic")
        self.assertEqual(manifest["schema_id"], sealer.EXACT3_MANIFEST_SCHEMA_ID)
        self.assertEqual(
            manifest["schema_version"],
            sealer.EXACT3_MANIFEST_SCHEMA_VERSION,
        )
        self.assertEqual(manifest["artifact_kind"], "exact3_schedule_evidence")
        self.assertFalse(manifest["milestone_artifact_eligible"])
        self.assertTrue(manifest["schedule_evidence_eligible"])
        self.assertTrue(manifest["formal_schedule_sealed"])
        self.assertFalse(manifest["timing_claim"])
        self.assertEqual(manifest["security_claim"], "none")
        self.assertEqual(
            manifest["timestamp_semantics"]["execution_started_at"],
            "2026-07-31T10:00:00.000001+09:00",
        )
        self.assertEqual(
            manifest["timestamp_semantics"]["execution_finished_at"],
            "2026-07-31T10:02:03.500001+09:00",
        )
        self.assertEqual(manifest["command"]["attempt_count"], 1)
        self.assertEqual(manifest["command"]["exit_code"], 0)
        self.assertEqual(
            manifest["command"]["runtime_argv"],
            list(sealer.EXPECTED_RUNTIME_ARGV),
        )
        self.assertEqual(
            manifest["command"]["environment"],
            sealer.RUNTIME_ENVIRONMENT,
        )
        self.assertEqual(manifest["command"]["stdin"], "DEVNULL")
        self.assertEqual(manifest["timing"]["elapsed_seconds"], 123.5)
        self.assertEqual(manifest["timing"]["peak_rss_kib"], PEAK_RSS_KIB)
        source = manifest["provenance"]["source_files"][0]
        for field in (
            "device",
            "inode",
            "mode",
            "link_count",
            "mtime_ns",
            "ctime_ns",
        ):
            self.assertIn(field, source)
        transition = manifest["provenance"]["profile_transition"]
        self.assertEqual(
            transition["allowed_json_pointers"],
            list(sealer.PROFILE_TRANSITION_ALLOWED_JSON_POINTERS),
        )
        self.assertIn("published before", transition["application_order"])
        self.assertEqual(
            transition["post_state_contract"]["values"],
            sealer.PROFILE_TRANSITION_POST_STATE_VALUES,
        )
        self.assertEqual(
            manifest["evidence"]["final_summary"]["operation_counts"],
            sealer.FINAL_COUNTS,
        )

        sums: dict[str, str] = {}
        for line in (
            (self.run_root / "SHA256SUMS").read_text(encoding="ascii").splitlines()
        ):
            digest, name = line.split("  ", 1)
            sums[name] = digest
        self.assertEqual(
            set(sums),
            {"stdout.log", "stderr.log", "time.log", "metrics.csv", "manifest.json"},
        )
        for name, digest in sums.items():
            self.assertEqual(
                hashlib.sha256((self.run_root / name).read_bytes()).hexdigest(),
                digest,
            )
        approved = self.profile_transition_calls[0][1]
        self.assertEqual(approved["run_id"], self.run_id)
        self.assertEqual(
            approved["manifest_sha256"],
            hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        self.assertEqual(
            approved["sha256sums_sha256"],
            hashlib.sha256((self.run_root / "SHA256SUMS").read_bytes()).hexdigest(),
        )

        metrics = list(
            csv.DictReader(
                io.StringIO((self.run_root / "metrics.csv").read_text(encoding="utf-8"))
            )
        )
        self.assertEqual(len(metrics), 4)
        self.assertEqual(
            [row["record_kind"] for row in metrics],
            ["layer", "layer", "layer", "final"],
        )
        self.assertEqual(metrics[-1]["rotations"], "18900")
        self.assertEqual(metrics[-1]["peak_rss_bytes"], str(PEAK_RSS_KIB * 1024))

    def test_post_run_only_legacy_seal_is_unconditionally_retired(self) -> None:
        _write_raw_files(self.run_root)
        with self.assertRaisesRegex(sealer.SealError, "post-run-only"):
            sealer.seal_run(self.run_root)
        self.assertEqual(
            {path.name for path in self.run_root.iterdir()},
            set(sealer.RAW_NAMES),
        )

    def test_nonzero_child_return_is_rejected_without_retry(self) -> None:
        self.child_returncode = 9
        outcome = self._assert_rejected_without_publication()
        self.assertIn("sole exact3 child", str(outcome["error"]))
        self.assertEqual(outcome["git_calls"], 2)
        self.assertEqual(outcome["provenance_calls"], 2)

    def test_launcher_argv_drift_is_rejected(self) -> None:
        def drifted(staging_root: Path) -> sealer.LaunchEvidence:
            launch = self._fake_execute(staging_root)
            return replace(launch, argv=(*launch.argv[:-1], "2"))

        outcome = self._invoke(executor=drifted)
        self.assertIsInstance(outcome["error"], sealer.SealError)
        self.assertIn("frozen command", str(outcome["error"]))
        self.assertEqual(outcome["execute_calls"], 1)
        self.assertFalse(self.run_root.exists())
        self._assert_no_staging()

    def test_low_level_launcher_invokes_one_frozen_subprocess(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="exact3-launcher-",
            dir="/tmp",
        ) as directory:
            staging_root = Path(directory)
            completed = sealer.subprocess.CompletedProcess(
                args=(),
                returncode=0,
            )
            with patch.object(
                sealer.subprocess,
                "run",
                return_value=completed,
            ) as run_mock:
                launch = sealer._execute_exact3_once(staging_root)
            self.assertEqual(run_mock.call_count, 1)
            expected_argv = (
                sealer.TIME_EXECUTABLE,
                "-v",
                "-o",
                str(staging_root / "time.log"),
                "--",
                *sealer.EXPECTED_RUNTIME_ARGV,
            )
            self.assertEqual(launch.argv, expected_argv)
            self.assertEqual(launch.returncode, 0)
            positional, keyword = run_mock.call_args
            self.assertEqual(positional, (expected_argv,))
            self.assertEqual(keyword["cwd"], sealer.REPO_ROOT)
            self.assertEqual(keyword["env"], sealer.RUNTIME_ENVIRONMENT)
            self.assertIs(keyword["stdin"], sealer.subprocess.DEVNULL)
            self.assertFalse(keyword["check"])
            self.assertEqual(
                {path.name for path in staging_root.iterdir()},
                set(sealer.RAW_NAMES),
            )

    def test_git_mutation_between_launch_brackets_is_rejected(self) -> None:
        changed = replace(self.git, fingerprint=(*self.git.fingerprint[:-1], "8" * 64))
        outcome = self._invoke(git_sequence=[self.git, changed])
        self.assertIsInstance(outcome["error"], sealer.SealError)
        self.assertIn("git/worktree changed", str(outcome["error"]))
        self.assertEqual(outcome["execute_calls"], 1)
        self.assertFalse(self.run_root.exists())
        self._assert_no_staging()

    def test_source_config_and_executable_mutations_are_each_rejected(self) -> None:
        mutations: dict[str, sealer.ProvenanceSnapshot] = {}
        changed_source = dict(self.provenance.source_files[0])
        changed_source["sha256"] = "8" * 64
        mutations["source"] = replace(
            self.provenance,
            source_files=(changed_source,),
        )
        changed_config = dict(self.provenance.config_files[0])
        changed_config["mtime_ns"] += 1
        mutations["config"] = replace(
            self.provenance,
            config_files=(changed_config, self.provenance.config_files[1]),
        )
        changed_executable = dict(self.provenance.executable)
        changed_executable["inode"] += 1
        mutations["executable"] = replace(
            self.provenance,
            executable=changed_executable,
        )
        for label, changed in mutations.items():
            with self.subTest(label=label):
                outcome = self._invoke(
                    provenance_sequence=[self.provenance, changed],
                )
                self.assertIsInstance(outcome["error"], sealer.SealError)
                self.assertIn("file identity changed", str(outcome["error"]))
                self.assertEqual(outcome["execute_calls"], 1)
                self.assertFalse(self.run_root.exists())
                self._assert_no_staging()

    def test_profile_transition_failure_keeps_orphan_bundle(self) -> None:
        def fail_transition(*_: object) -> None:
            raise sealer.SealError("simulated profile replace failure")

        outcome = self._invoke(profile_transition=fail_transition)
        self.assertIsInstance(outcome["error"], sealer.SealError)
        self.assertEqual(outcome["execute_calls"], 1)
        self.assertEqual(outcome["profile_calls"], 1)
        self.assertTrue(self.run_root.is_dir())
        self.assertEqual(
            {path.name for path in self.run_root.iterdir()},
            set(sealer.RAW_NAMES) | set(sealer.OUTPUT_NAMES),
        )
        self._assert_no_staging()

    def test_late_provenance_drift_keeps_bundle_unapproved(self) -> None:
        changed_executable = dict(self.provenance.executable)
        changed_executable["mtime_ns"] += 1
        changed = replace(
            self.provenance,
            executable=changed_executable,
        )
        outcome = self._invoke(
            provenance_sequence=[
                self.provenance,
                self.provenance,
                self.provenance,
                changed,
            ],
        )
        self.assertIsInstance(outcome["error"], sealer.SealError)
        self.assertEqual(outcome["execute_calls"], 1)
        self.assertEqual(outcome["profile_calls"], 0)
        self.assertTrue(self.run_root.is_dir())
        self.assertEqual(
            {path.name for path in self.run_root.iterdir()},
            set(sealer.RAW_NAMES) | set(sealer.OUTPUT_NAMES),
        )
        self._assert_no_staging()

    def test_destination_existing_is_rejected_before_child(self) -> None:
        self.run_root.mkdir()
        sentinel = self.run_root / "sentinel"
        sentinel.write_text("do not replace\n", encoding="utf-8")
        outcome = self._invoke()
        self.assertIsInstance(outcome["error"], sealer.SealError)
        self.assertEqual(outcome["execute_calls"], 0)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "do not replace\n")

    def test_rename_noreplace_never_replaces_destination(self) -> None:
        source = sealer.OUTPUT_ROOT / f".exact3-stage-rename-{uuid.uuid4().hex}"
        destination = sealer.OUTPUT_ROOT / f"exact3-rename-{uuid.uuid4().hex}"
        source.mkdir()
        destination.mkdir()
        sentinel = destination / "sentinel"
        sentinel.write_text("preserve\n", encoding="utf-8")
        try:
            with self.assertRaises(sealer.SealError):
                sealer._rename_noreplace(source, destination)
            self.assertTrue(source.is_dir())
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve\n")
        finally:
            shutil.rmtree(source)
            shutil.rmtree(destination)

    def test_raw_validation_failures_do_not_publish(self) -> None:
        cases: list[tuple[str, object]] = []

        duplicate_layer = _records()
        duplicate_layer[1]["layer_id"] = 0
        duplicate_layer[1]["weights_layer_id"] = 0
        cases.append(("duplicate layer", _stdout_bytes(duplicate_layer)))

        wrong_metadata = _records()
        wrong_metadata[2]["ffn_output_metadata"]["level"] = 42
        cases.append(("metadata", _stdout_bytes(wrong_metadata)))

        wrong_count = _records()
        wrong_count[2]["layer_operation_counts"]["ct_pt_multiplications"] -= 1
        cases.append(("operation count", _stdout_bytes(wrong_count)))

        trust_violation = _records()
        trust_violation[3]["server_decryptions"] = 1
        cases.append(("trust boundary", _stdout_bytes(trust_violation)))

        quality = _records()
        quality[1]["output_quality"]["relative_l2"] = 0.051
        cases.append(("quality", _stdout_bytes(quality)))

        inactive = _records()
        inactive[2]["inactive_max_abs"] = 1.1e-3
        inactive[3]["inactive_max_abs"] = 1.1e-3
        cases.append(("M5 prototype inactive", _stdout_bytes(inactive)))

        nonfinite = _records()
        nonfinite[0]["output_quality"]["relative_l2"] = float("nan")
        cases.append(("non-finite", _stdout_bytes(nonfinite, allow_nan=True)))

        malformed = _stdout_bytes(_records()).replace(b"}\n", b"\n", 1)
        cases.append(("malformed", malformed))

        duplicate_key = _stdout_bytes(_records()).replace(
            b'{"artifact_eligible":',
            b'{"artifact_eligible":false,"artifact_eligible":',
            1,
        )
        cases.append(("duplicate JSON key", duplicate_key))

        for label, payload in cases:
            with self.subTest(label=label):
                self.stdout_data = payload
                self._assert_rejected_without_publication()

    def test_prototype_inactive_value_between_old_and_new_bounds_is_accepted(self) -> None:
        records = _records()
        records[2]["inactive_max_abs"] = 5e-4
        records[3]["inactive_max_abs"] = 5e-4
        parsed = sealer._parse_stdout(_stdout_bytes(records))
        self.assertEqual(parsed.layers[2]["inactive_max_abs"], 5e-4)
        self.assertEqual(parsed.summary["inactive_max_abs"], 5e-4)

    def test_stderr_and_gnu_time_failures_do_not_publish(self) -> None:
        cases = (
            ("stderr", b"unexpected diagnostic\n", _time_bytes()),
            ("time exit", b"", _time_bytes(exit_status=1)),
            ("swaps", b"", _time_bytes(swaps=1)),
        )
        for label, stderr_data, time_data in cases:
            with self.subTest(label=label):
                self.stderr_data = stderr_data
                self.time_data = time_data
                self._assert_rejected_without_publication()
                self.stderr_data = b""
                self.time_data = _time_bytes()

    def test_raw_symlink_is_rejected(self) -> None:
        def symlinked(staging_root: Path) -> sealer.LaunchEvidence:
            launch = self._fake_execute(staging_root)
            stdout = staging_root / "stdout.log"
            stdout.unlink()
            stdout.symlink_to(staging_root / "time.log")
            return launch

        outcome = self._invoke(executor=symlinked)
        self.assertIsInstance(outcome["error"], sealer.SealError)
        self.assertFalse(self.run_root.exists())
        self._assert_no_staging()

    def test_candidate_profile_pre_state_and_exact3_absence_are_strict(self) -> None:
        profile = _candidate_profile()
        record = _file_record(
            sealer.PROFILE_PATH,
            sealer._pretty_json_bytes(profile),
            20,
        )
        drifted = copy.deepcopy(profile)
        drifted["m5_schedule_candidate"]["formal_schedule_sealed"] = True
        with self.assertRaises(sealer.SealError):
            sealer._profile_transition_record(drifted, record)
        already_sealed = copy.deepcopy(profile)
        already_sealed["m5_schedule_candidate"]["exact3_seal"] = {}
        with self.assertRaises(sealer.SealError):
            sealer._profile_transition_record(already_sealed, record)

    def test_candidate_profile_rejects_every_prototype_boundary_drift(self) -> None:
        base = _candidate_profile()
        record = _file_record(
            sealer.PROFILE_PATH,
            _candidate_profile_bytes(),
            20,
        )
        for label, key, value in (
            ("legacy", "legacy_basis", "legacy MOAI used 1e-3"),
            ("scope", "scope", "all M5 diagnostics"),
            ("active", "active_quality_policy", "relaxed"),
            ("sentinel", "sentinel_range_policy", "disabled"),
            ("extra", "unexpected", True),
        ):
            with self.subTest(label=label):
                profile = copy.deepcopy(base)
                profile["m5_prototype_acceptance"][key] = value
                with self.assertRaisesRegex(
                    sealer.SealError,
                    "prototype acceptance contract drifted",
                ):
                    sealer._profile_transition_record(profile, record)

    def test_sealed_profile_changes_exactly_seven_pointers(self) -> None:
        candidate = _candidate_profile()
        payload = _candidate_profile_bytes()
        record = _file_record(sealer.PROFILE_PATH, payload, 21)
        transition = sealer._profile_transition_record(candidate, record)
        evidence = _profile_evidence()
        sealed = sealer._sealed_profile_value(candidate, transition, evidence)
        self.assertEqual(
            sealer._profile_changed_pointers(candidate, sealed),
            set(sealer.PROFILE_TRANSITION_ALLOWED_JSON_POINTERS),
        )
        self.assertEqual(
            sealer._profile_immutable_projection_sha256(candidate),
            sealer._profile_immutable_projection_sha256(sealed),
        )
        self.assertEqual(sealed["unrelated"], candidate["unrelated"])
        self.assertEqual(
            sealed["m5_schedule_candidate"]["exact3_seal"]["evidence"],
            evidence,
        )

    def test_disallowed_profile_change_alters_immutable_projection(self) -> None:
        candidate = _candidate_profile()
        changed = copy.deepcopy(candidate)
        changed["unrelated"]["value"] = 8
        self.assertNotEqual(
            sealer._profile_immutable_projection_sha256(candidate),
            sealer._profile_immutable_projection_sha256(changed),
        )

    def _real_profile_fixture(
        self,
        repository_root: Path,
    ) -> tuple[bytes, sealer.ProvenanceSnapshot]:
        profile_path = repository_root / sealer.PROFILE_PATH
        profile_path.parent.mkdir(parents=True)
        candidate_bytes = _candidate_profile_bytes()
        profile_path.write_bytes(candidate_bytes)
        os.chmod(profile_path, 0o640)
        with patch.object(sealer, "REPO_ROOT", repository_root):
            record = sealer._repository_file_record(sealer.PROFILE_PATH)
        transition = sealer._profile_transition_record(_candidate_profile(), record)
        provenance = sealer.ProvenanceSnapshot(
            source_files=(),
            config_files=(record,),
            source_manifest_sha256="3" * 64,
            config_manifest_sha256="4" * 64,
            executable={},
            trace_scale={},
            profile_transition=transition,
        )
        return candidate_bytes, provenance

    def test_profile_transition_uses_atomic_replace_and_preserves_mode(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="exact3-profile-",
            dir="/tmp",
        ) as directory:
            repository_root = Path(directory)
            _, provenance = self._real_profile_fixture(repository_root)
            with patch.object(sealer, "REPO_ROOT", repository_root):
                result = sealer._apply_profile_transition(
                    provenance,
                    _profile_evidence(),
                )
            profile_path = repository_root / sealer.PROFILE_PATH
            sealed = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(os.stat(profile_path).st_mode & 0o777, 0o640)
            self.assertEqual(
                sealed["m5_schedule_candidate"]["status"],
                sealer.PROFILE_CANDIDATE_SEALED_STATUS,
            )
            self.assertEqual(
                sealed["m5_schedule_candidate"]["exact3_seal"]["evidence"],
                _profile_evidence(),
            )
            self.assertEqual(
                result["changed_json_pointers"],
                sorted(sealer.PROFILE_TRANSITION_ALLOWED_JSON_POINTERS),
            )
            self.assertEqual(
                list(profile_path.parent.glob(".*.exact3-seal-*.tmp")),
                [],
            )
            self.assertEqual(
                list(profile_path.parent.glob(".*.exact3-rollback-*.tmp")),
                [],
            )

    def test_profile_replace_failure_retains_candidate_and_removes_temp(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="exact3-profile-fail-",
            dir="/tmp",
        ) as directory:
            repository_root = Path(directory)
            candidate_bytes, provenance = self._real_profile_fixture(repository_root)
            with (
                patch.object(sealer, "REPO_ROOT", repository_root),
                patch.object(sealer.os, "replace", side_effect=OSError("simulated")),
            ):
                with self.assertRaises(sealer.SealError):
                    sealer._apply_profile_transition(
                        provenance,
                        _profile_evidence(),
                    )
            profile_path = repository_root / sealer.PROFILE_PATH
            self.assertEqual(profile_path.read_bytes(), candidate_bytes)
            self.assertEqual(
                list(profile_path.parent.glob(".*.exact3-seal-*.tmp")),
                [],
            )
            self.assertEqual(
                list(profile_path.parent.glob(".*.exact3-rollback-*.tmp")),
                [],
            )

    def test_post_replace_fsync_failure_restores_candidate(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="exact3-profile-fsync-fail-",
            dir="/tmp",
        ) as directory:
            repository_root = Path(directory)
            candidate_bytes, provenance = self._real_profile_fixture(repository_root)
            real_fsync = os.fsync
            directory_failure_injected = False

            def fail_first_directory_fsync(descriptor: int) -> None:
                nonlocal directory_failure_injected
                if (
                    sealer.stat.S_ISDIR(os.fstat(descriptor).st_mode)
                    and not directory_failure_injected
                ):
                    directory_failure_injected = True
                    raise OSError("simulated directory fsync failure")
                real_fsync(descriptor)

            with (
                patch.object(sealer, "REPO_ROOT", repository_root),
                patch.object(sealer.os, "fsync", side_effect=fail_first_directory_fsync),
            ):
                with self.assertRaisesRegex(sealer.SealError, "candidate profile was restored"):
                    sealer._apply_profile_transition(
                        provenance,
                        _profile_evidence(),
                    )
            profile_path = repository_root / sealer.PROFILE_PATH
            self.assertTrue(directory_failure_injected)
            self.assertEqual(profile_path.read_bytes(), candidate_bytes)
            self.assertEqual(
                list(profile_path.parent.glob(".*.exact3-seal-*.tmp")),
                [],
            )
            self.assertEqual(
                list(profile_path.parent.glob(".*.exact3-rollback-*.tmp")),
                [],
            )

    def test_profile_symlink_hardlink_and_identity_drift_are_rejected(self) -> None:
        cases = ("symlink", "hardlink", "identity")
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory(
                    prefix=f"exact3-profile-{case}-",
                    dir="/tmp",
                ) as directory:
                    repository_root = Path(directory)
                    _, provenance = self._real_profile_fixture(repository_root)
                    profile_path = repository_root / sealer.PROFILE_PATH
                    if case == "symlink":
                        target = profile_path.with_name("target.json")
                        target.write_bytes(profile_path.read_bytes())
                        profile_path.unlink()
                        profile_path.symlink_to(target)
                    elif case == "hardlink":
                        os.link(profile_path, profile_path.with_name("second-link.json"))
                    else:
                        replacement = profile_path.with_name("replacement.json")
                        replacement.write_bytes(profile_path.read_bytes())
                        os.chmod(replacement, 0o640)
                        os.replace(replacement, profile_path)
                    with patch.object(sealer, "REPO_ROOT", repository_root):
                        with self.assertRaises(sealer.SealError):
                            sealer._apply_profile_transition(
                                provenance,
                                _profile_evidence(),
                            )
                    self.assertEqual(
                        list(profile_path.parent.glob(".*.exact3-seal-*.tmp")),
                        [],
                    )
                    self.assertEqual(
                        list(profile_path.parent.glob(".*.exact3-rollback-*.tmp")),
                        [],
                    )

    def test_path_escape_and_cli_missing_run_root_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="exact3-path-escape-") as directory:
            escaped = Path(directory) / "run"
            with self.assertRaises(sealer.SealError):
                sealer.run_and_seal(escaped)
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                sealer._parse_arguments([])


if __name__ == "__main__":
    unittest.main()
