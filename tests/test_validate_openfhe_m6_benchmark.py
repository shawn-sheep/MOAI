#!/usr/bin/env python3
"""Focused positive and tamper-negative tests for the M6 validator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_openfhe_m6_benchmark as runner  # noqa: E402
import validate_openfhe_m6_benchmark as validator  # noqa: E402


RUNNER_TEST_PATH = REPO_ROOT / "tests" / "test_run_openfhe_m6_benchmark.py"
spec = importlib.util.spec_from_file_location("m6_runner_test_fixture", RUNNER_TEST_PATH)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {RUNNER_TEST_PATH}")
fixture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_commands(manifest: dict[str, object]) -> list[dict[str, object]]:
    command_argv: list[list[str]] = [
        ["/usr/bin/git", "status", "--porcelain=v1", "--untracked-files=normal"],
        ["/usr/bin/git", "branch", "--show-current"],
        ["/usr/bin/git", "rev-parse", "HEAD"],
        ["/usr/bin/git", "rev-parse", validator.TRACKING_REF],
        ["/usr/bin/git", "config", "--local", "--get-all", "remote.origin.url"],
        [
            "/usr/bin/git",
            "ls-remote",
            "--exit-code",
            validator.REMOTE_URL,
            validator.REMOTE_REF,
        ],
    ]
    cleared = manifest["build_configuration"][  # type: ignore[index]
        "cleared_environment_variables"
    ]
    command_argv.extend(
        [
            [
                "/usr/bin/env",
                *(f"--unset={name}" for name in cleared),
                "/usr/bin/cmake",
                "--fresh",
                "-S",
                str(REPO_ROOT),
                "-B",
                str(REPO_ROOT / "build-openfhe"),
                "-G",
                "Unix Makefiles",
                "-DOpenFHE_DIR:PATH=/home/shawnsheep/opt/openfhe_v1_5_1/lib/OpenFHE",
                "-DCMAKE_BUILD_TYPE:STRING=Release",
                "-DBUILD_TESTING:BOOL=ON",
                "-DCMAKE_CXX_COMPILER:FILEPATH=/usr/bin/c++",
                "-DCMAKE_MAKE_PROGRAM:FILEPATH=/usr/bin/gmake",
                "-DCMAKE_CXX_FLAGS:STRING=",
                "-DCMAKE_CXX_FLAGS_RELEASE:STRING=-O3 -DNDEBUG",
                "-DCMAKE_EXE_LINKER_FLAGS:STRING=",
                "-DCMAKE_EXE_LINKER_FLAGS_RELEASE:STRING=",
                "-DCMAKE_SHARED_LINKER_FLAGS:STRING=",
                "-DCMAKE_MODULE_LINKER_FLAGS:STRING=",
                "-DCMAKE_STATIC_LINKER_FLAGS:STRING=",
            ],
            [
                "/usr/bin/cmake",
                "--build",
                str(REPO_ROOT / "build-openfhe"),
                "--clean-first",
                "-j",
                "4",
            ],
            [
                "/usr/bin/ldd",
                str(REPO_ROOT / "build-openfhe/openfhe_encoder_12_layer_smoke"),
            ],
            [
                "/usr/bin/ctest",
                "--test-dir",
                str(REPO_ROOT / "build-openfhe"),
                "--output-on-failure",
                "--verbose",
                "--no-tests=error",
                "-R",
                validator.m5_validator.M5_CTEST_PATTERN,
            ],
            [
                "/usr/bin/ctest",
                "--test-dir",
                str(REPO_ROOT / "build-openfhe"),
                "--output-on-failure",
                "--verbose",
                "--no-tests=error",
                "-R",
                validator.M6_CTEST_PATTERN,
            ],
        ]
    )
    config = runner.RunnerConfig(run_id=str(manifest["run_id"]))
    command_argv.extend(
        runner._runtime_command(config, Path(f"/tmp/{phase}-{index}.time"), phase, index)
        for phase, index in validator._expected_phases()
    )
    command_argv.append(
        [
            sys.executable,
            str(REPO_ROOT / "scripts/validate_openfhe_m6_benchmark.py"),
            "--schema",
            str(validator.DEFAULT_SCHEMA),
            "--manifest",
            str(validator.OUTPUT_ROOT / manifest["run_id"] / "manifest.json"),
            "--verify-git",
        ]
    )
    return [
        {"argv": argv, "cwd": str(REPO_ROOT), "returncode": 0}
        for argv in command_argv
    ]


def _write_bundle() -> tuple[tempfile.TemporaryDirectory[str], Path, dict[str, object]]:
    validator.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    run_id = "synthetic-m6-benchmark"
    temporary = tempfile.TemporaryDirectory(
        prefix=f".{run_id}.staging-",
        dir=validator.OUTPUT_ROOT,
    )
    root = Path(temporary.name)
    measured = [fixture.executed(index) for index in range(1, 6)]
    warmup_sample = fixture.sample(0)
    warmup = runner.ExecutedSample(
        phase="warmup",
        index=0,
        sample=warmup_sample,
        stdout=json.dumps(warmup_sample) + "\n",
        stderr="",
        time_line="",
        external_wall_seconds=2.0,
        external_peak_rss_bytes=2 * 1024 * 1024,
        command={"argv": ["synthetic"], "cwd": str(REPO_ROOT), "returncode": 0},
    )
    samples = [warmup, *measured]
    (root / "stdout.log").write_text(
        runner._log_blocks(samples, "stdout"), encoding="utf-8"
    )
    (root / "stderr.log").write_text(
        runner._log_blocks(samples, "stderr"), encoding="utf-8"
    )
    time_lines = []
    for item in samples:
        time_lines.append(
            json.dumps(
                {
                    "phase": item.phase,
                    "index": item.index,
                    "elapsed_seconds": item.external_wall_seconds,
                    "max_rss_kib": item.external_peak_rss_bytes // 1024,
                    "exit_status": 0,
                },
                separators=(",", ":"),
            )
        )
    (root / "time.log").write_text("\n".join(time_lines) + "\n", encoding="utf-8")
    (root / "metrics.csv").write_text(runner._metrics_csv(measured), encoding="utf-8")
    checksums = "".join(
        f"{_sha256(root / name)}  {name}\n" for name in validator.CHECKSUM_PATHS
    )
    (root / "SHA256SUMS").write_text(checksums, encoding="ascii")

    artifact_records = []
    for name, role, media_type in validator.ARTIFACT_LAYOUT:
        path = root / name
        artifact_records.append(
            {
                "path": name,
                "role": role,
                "media_type": media_type,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    zero_sha = "0" * 40
    m5_manifest = json.loads(
        (
            REPO_ROOT
            / validator.M5_RELATIVE_PATH
            / "manifest.json"
        ).read_text(encoding="utf-8")
    )
    build_configuration = copy.deepcopy(m5_manifest["build_configuration"])
    cleared = sorted(set(runner.m5.CONFIGURE_ENV_UNSET))
    build_configuration["cleared_environment_variables"] = cleared
    build_configuration["forced_environment_variables"] = {
        "LANG": "C",
        "LC_ALL": "C",
        **validator.FORCED_THREAD_ENVIRONMENT,
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "schema_binding": {
            "canonicalization": "git-blob-and-working-tree-sha256-v1",
            "files": [
                {
                    "path": path,
                    "bytes": 1,
                    "sha256": "a" * 64,
                    "git_blob_sha256": "a" * 64,
                }
                for path in validator.SCHEMA_BINDING_PATHS
            ],
        },
        "run_id": run_id,
        "milestone": "M6",
        "started_at": "2026-08-02T00:00:00+09:00",
        "finished_at": "2026-08-02T00:00:01+09:00",
        "git": {
            "repository_root": str(REPO_ROOT),
            "branch": validator.BRANCH,
            "local_commit": zero_sha,
            "clean": True,
            "tracking_ref": validator.TRACKING_REF,
            "tracking_commit": zero_sha,
            "remote_name": "origin",
            "remote_url": validator.REMOTE_URL,
            "remote_ref": validator.REMOTE_REF,
            "remote_commit": zero_sha,
        },
        "m5_prerequisite": {
            "run_id": validator.M5_RUN_ID,
            "relative_path": validator.M5_RELATIVE_PATH,
            "manifest_sha256": validator.M5_MANIFEST_SHA256,
            "sha256sums_sha256": validator.M5_SHA256SUMS_SHA256,
            "decision": validator.M5_DECISION,
            "verdict": validator.M5_VERDICT,
            "validated": True,
        },
        "profile": {
            "id": "paper_compat",
            "security_claim": "none",
            "effective_profile_sha256": validator.PROFILE_SHA256,
            "warning": "Research reproduction parameters only. Do not claim 128-bit security.",
        },
        "build_configuration": build_configuration,
        "workload": {
            "backend": "OpenFHE CKKS CPU",
            "scope": "server-only five-token BERT-base 12-layer encoder trace replay",
            "executable_path": "build-openfhe/openfhe_encoder_12_layer_smoke",
            "executable_sha256": "b" * 64,
            "executable_bytes": 1,
            "warmup_count": 1,
            "measured_count": 5,
            "independent_processes": True,
            "token_count": 5,
            "encoder_layers": 12,
            "timing_kind": "benchmark",
        },
        "preflight": {
            "fresh_configure": True,
            "clean_build": True,
            "narrow_ctest": True,
            "passed_tests": list(validator.M6_CTEST_EXPECTED_TESTS),
            "schedule_preflight": copy.deepcopy(
                m5_manifest["contracts"]["schedule_preflight"]
            ),
            "crypto_preflight": copy.deepcopy(
                m5_manifest["contracts"]["crypto_preflight"]
            ),
        },
        "commands": [
            {"argv": [f"synthetic-{index}"], "cwd": str(REPO_ROOT), "returncode": 0}
            for index in range(18)
        ],
        "environment": {
            "architecture": "x86_64",
            "cmake": "cmake version synthetic",
            "compiler": "c++ synthetic",
            "openfhe_prefix": "/home/shawnsheep/opt/openfhe_v1_5_1",
            "openfhe_version": "1.5.1",
            "os": "Linux synthetic WSL2",
            "python": "3.10.20",
            "wsl": True,
            "logical_cpu_count": 16,
            "threading": {
                "forced": {"OMP_NUM_THREADS": "16", "OMP_DYNAMIC": "FALSE"},
                "cleared_inherited_variables": cleared,
                "applies_to": "configure, build, CTest, warm-up, measured samples, validator",
            },
        },
        "inputs": [{"path": "synthetic", "bytes": 1, "sha256": "c" * 64}],
        "samples": {
            "warmup": warmup.manifest_record(),
            "measured": [item.manifest_record() for item in measured],
        },
        "metrics": runner._metrics(measured),
        "comparison": runner._seal_comparison(),
        "gate": {
            "passed": True,
            "decision": "PASS_M6_OPENFHE_BENCHMARK",
            "checks": {
                "git": "PASS",
                "m5_prerequisite": "PASS",
                "build": "PASS",
                "narrow_ctest": "PASS",
                "repeat_contract": "PASS",
                "correctness": "PASS",
                "bytes_and_counts": "PASS",
                "statistics": "PASS",
                "artifact_integrity": "PASS",
                "seal_comparability": "PASS_NONCOMPARABLE",
            },
        },
        "artifacts": artifact_records,
        "claim_boundary": copy.deepcopy(validator.CLAIM_BOUNDARY),
        "verdict": "GO_M6_OPENFHE_BENCHMARK",
    }
    manifest["commands"] = _valid_commands(manifest)
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return temporary, root / "manifest.json", manifest


class SchemaAndSemanticTests(unittest.TestCase):
    def test_frozen_hash_constants_are_sha256(self) -> None:
        for value in (
            validator.PROFILE_SHA256,
            validator.M5_MANIFEST_SHA256,
            validator.M5_SHA256SUMS_SHA256,
        ):
            self.assertEqual(len(value), 64)
            self.assertRegex(value, r"^[0-9a-f]{64}$")

    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = validator.validate_schema(validator.load_json(validator.DEFAULT_SCHEMA))

    def test_schema_static_self_check(self) -> None:
        self.assertEqual(self.schema["$id"], validator.SCHEMA_ID)

    def test_full_synthetic_bundle_passes_semantic_and_hash_checks(self) -> None:
        temporary, manifest_path, manifest = _write_bundle()
        self.addCleanup(temporary.cleanup)
        with (
            mock.patch.object(validator, "_verify_m5_prerequisite"),
            mock.patch.object(validator, "_verify_build_configuration"),
            mock.patch.object(validator, "_verify_workload"),
            mock.patch.object(validator, "_verify_inputs"),
            mock.patch.object(validator, "_verify_schema_binding"),
        ):
            validator.validate_bundle(
                manifest_path,
                manifest,
                self.schema,
                verify_git=False,
                expected_run_id=manifest["run_id"],
                allow_staging_root=True,
            )

    def test_rejects_tampered_raw_log(self) -> None:
        temporary, manifest_path, manifest = _write_bundle()
        self.addCleanup(temporary.cleanup)
        (manifest_path.parent / "stdout.log").write_text("tampered\n", encoding="utf-8")
        with (
            mock.patch.object(validator, "_verify_m5_prerequisite"),
            mock.patch.object(validator, "_verify_build_configuration"),
            mock.patch.object(validator, "_verify_workload"),
            mock.patch.object(validator, "_verify_inputs"),
            mock.patch.object(validator, "_verify_schema_binding"),
            self.assertRaisesRegex(validator.ValidationError, "artifact record"),
        ):
            validator.validate_bundle(
                manifest_path,
                manifest,
                self.schema,
                verify_git=False,
                expected_run_id=manifest["run_id"],
                allow_staging_root=True,
            )

    def test_rejects_repeat_count_or_order_drift(self) -> None:
        temporary, _, manifest = _write_bundle()
        self.addCleanup(temporary.cleanup)
        manifest["samples"]["measured"][0]["index"] = 2  # type: ignore[index]
        with self.assertRaisesRegex(validator.ValidationError, "repeat/order"):
            validator._manifest_samples(manifest)

    def test_rejects_duplicate_keys_in_raw_sample_json(self) -> None:
        encoded = json.dumps(fixture.sample())
        duplicated = encoded[:-1] + ',"passed":true}'
        with self.assertRaisesRegex(validator.ValidationError, "malformed JSON"):
            validator._sample_json_from_block(duplicated + "\n", "unit")

    def test_rejects_fake_preflight_names_and_commands(self) -> None:
        temporary, _, manifest = _write_bundle()
        self.addCleanup(temporary.cleanup)
        manifest["preflight"]["passed_tests"] = [  # type: ignore[index]
            f"fake-{index}" for index in range(9)
        ]
        with self.assertRaisesRegex(validator.ValidationError, "passed-test"):
            validator._verify_preflight(manifest)
        manifest["commands"] = [  # type: ignore[assignment]
            {"argv": [f"fake-{index}"], "cwd": str(REPO_ROOT), "returncode": 0}
            for index in range(18)
        ]
        with self.assertRaisesRegex(validator.ValidationError, "Git preflight"):
            validator._verify_commands(manifest)

    def test_rejects_deleted_cleared_environment_record(self) -> None:
        temporary, _, manifest = _write_bundle()
        self.addCleanup(temporary.cleanup)
        del manifest["environment"]["threading"][  # type: ignore[index]
            "cleared_inherited_variables"
        ]
        with self.assertRaisesRegex(validator.ValidationError, "environment contract"):
            validator._verify_environment(manifest)

    def test_rejects_seal_enabled_cache_and_executable_hash_tamper(self) -> None:
        temporary, _, manifest = _write_bundle()
        self.addCleanup(temporary.cleanup)
        with (
            mock.patch.object(validator.m5_validator, "_verify_build_configuration"),
            mock.patch.object(
                validator.m5_validator,
                "_parse_cmake_cache",
                return_value={"MOAI_ENABLE_SEAL_REFERENCE": ("BOOL", "ON")},
            ),
            self.assertRaisesRegex(validator.ValidationError, "SEAL reference OFF"),
        ):
            validator._verify_build_configuration(manifest)
        executable = REPO_ROOT / "build-openfhe/openfhe_encoder_12_layer_smoke"
        if not executable.is_file():
            self.skipTest("benchmark executable is not built")
        manifest["workload"]["executable_sha256"] = "0" * 64  # type: ignore[index]
        with self.assertRaisesRegex(validator.ValidationError, "executable hash"):
            validator._verify_workload(manifest)

    def test_rejects_statistics_tamper(self) -> None:
        temporary, _, manifest = _write_bundle()
        self.addCleanup(temporary.cleanup)
        manifest["metrics"]["timing_ms"]["server_online"]["batch"][  # type: ignore[index]
            "median"
        ] += 1.0
        measured = manifest["samples"]["measured"]  # type: ignore[index]
        with self.assertRaisesRegex(validator.ValidationError, "statistics"):
            validator._verify_metrics(manifest, measured)

    def test_rejects_any_speedup_value_or_alias(self) -> None:
        with self.assertRaisesRegex(validator.ValidationError, "speedup"):
            validator._reject_speedup_claim(
                {"comparison": {"seal_reference": {"speedup": 1.1}}}
            )
        with self.assertRaisesRegex(validator.ValidationError, "speedup"):
            validator._reject_speedup_claim({"speedup_claim": False})
        validator._reject_speedup_claim(
            {"comparison": {"seal_reference": {"speedup": None}}}
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
