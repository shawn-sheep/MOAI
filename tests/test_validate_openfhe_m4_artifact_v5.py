#!/usr/bin/env python3
"""Positive and tamper-negative tests for the M4 schema-v5 validator."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_openfhe_m4_artifact_v5.py"
SCHEMA_PATH = REPO_ROOT / "docs" / "openfhe-m4-artifact-schema-v5.json"
spec = importlib.util.spec_from_file_location(
    "validate_openfhe_m4_artifact_v5", VALIDATOR_PATH
)
if spec is None or spec.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot import {VALIDATOR_PATH}")
validator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = validator
spec.loader.exec_module(validator)


def sentinel_ranges() -> dict[str, dict[str, float]]:
    return {
        "softmax_denominator": {"minimum": 0.01, "maximum": 80.0},
        "attention_layernorm_normalized_variance": {
            "minimum": 0.5,
            "maximum": 1536.0,
        },
        "output_layernorm_normalized_variance": {
            "minimum": 32.0,
            "maximum": 128.0,
        },
    }


def build_configuration() -> dict[str, object]:
    package_root = validator.OPENFHE_PREFIX / "lib" / "OpenFHE"
    return {
        "generator": "Unix Makefiles",
        "source_root": str(validator.REPO_ROOT),
        "build_root": str(validator.BUILD_ROOT),
        "openfhe_dir": str(package_root),
        "build_type": "Release",
        "build_testing": True,
        "cxx_compiler": validator.SYSTEM_CXX,
        "make_program": validator.SYSTEM_MAKE,
        "cxx_flags": "",
        "cxx_flags_release": "-O3 -DNDEBUG",
        "exe_linker_flags": "",
        "exe_linker_flags_release": "",
        "shared_linker_flags": "",
        "module_linker_flags": "",
        "static_linker_flags": "",
        "environment_inheritance": "ambient_minus_cleared_variables",
        "cleared_environment_variables": list(validator.CONFIGURE_ENV_UNSET),
        "forced_environment_variables": dict(validator.FORCED_SUBPROCESS_ENVIRONMENT),
        "cmake_cache": {
            "path": "build-openfhe/CMakeCache.txt",
            "bytes": 1,
            "sha256": "1" * 64,
        },
        "openfhe_cmake_package_files": [
            {
                "path": str(package_root / name),
                "bytes": 1,
                "sha256": f"{index + 2:064x}",
            }
            for index, name in enumerate(validator.OPENFHE_CMAKE_PACKAGE_FILES)
        ],
        "openfhe_include_tree": {
            "root": str(validator.OPENFHE_PREFIX / "include" / "openfhe"),
            "file_count": 1,
            "bytes": 1,
            "algorithm": "SHA-256",
            "canonicalization": validator.OPENFHE_INCLUDE_TREE_CANONICALIZATION,
            "sha256": "2" * 64,
        },
        "openfhe_linked_libraries": [
            {
                "soname": soname,
                "path": str(
                    validator.OPENFHE_PREFIX
                    / "lib"
                    / validator.OPENFHE_LINKED_LIBRARY_BASENAMES[soname]
                ),
                "bytes": 1,
                "sha256": f"{index + 20:064x}",
            }
            for index, soname in enumerate(validator.OPENFHE_LINKED_LIBRARY_SONAMES)
        ],
    }


def runtime_record() -> dict[str, object]:
    return {
        "test": "openfhe_encoder_layer",
        "profile": "paper_compat",
        "security_claim": "none",
        "parameter_sha256": validator.ENCODER_FEATURE_PROFILE_SHA256,
        "execution_mode": "server-only",
        "encoder_layers": 1,
        "layer_id": 1,
        "input_level": 29,
        "output_level": 29,
        "remaining_levels": 17,
        "multiplicative_depth": 47,
        "max_observed_level": 45,
        "max_polynomial_depth": 10,
        "actual_trace_shape": [5, 768],
        "actual_trace_value_count": 3840,
        "feature_block_size": 1024,
        "checkpoint_decryption_owner": "client",
        "server_private_key_present": False,
        "server_decryptions": 0,
        "server_plaintext_activations": False,
        "relative_l2": 0.004,
        "cosine": 0.9999,
        "inactive_max_abs": 2e-7,
        "layernorm_inactive_guard_max_error": 0.25,
        "inactive_polynomial_sentinel_ranges": {
            "softmax_denominator": {"minimum": 0.75, "maximum": 1.25},
            "attention_layernorm_normalized_variance": {
                "minimum": 45.0,
                "maximum": 90.0,
            },
            "output_layernorm_normalized_variance": {
                "minimum": 32.0,
                "maximum": 128.0,
            },
        },
        "checkpoints": copy.deepcopy(list(validator.M4_EXPECTED_CHECKPOINTS)),
        "operation_counts": {
            "rotations": 6300,
            "ct_pt_multiplications": 51885,
            "ct_ct_multiplications": 95,
            "explicit_rescale_requests": 810,
            "chebyshev_evaluations": 55,
            "estimated_polynomial_multiplications": 1150,
            "bootstraps": 25,
            "bootstrap_iterations": 50,
        },
    }


def diagnostic_record(target: dict[str, object]) -> dict[str, object]:
    counts = target["operation_counts"]
    assert isinstance(counts, dict)
    return {
        "test": "openfhe_encoder_layer_smoke",
        "profile": target["profile"],
        "security_claim": target["security_claim"],
        "parameter_sha256": target["parameter_sha256"],
        "layer": target["layer_id"],
        "input_level": target["input_level"],
        "tokens": 5,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "feature_block": 1024,
        "fixture_load_ms": 10.0,
        "setup_keygen_ms": 100.0,
        "client_encrypt_ms": 20.0,
        "server_online_ms": 500.0,
        "client_decrypt_validate_ms": 30.0,
        "relative_l2": target["relative_l2"],
        "cosine": target["cosine"],
        "max_absolute": 0.01,
        "exact_trace_relative_l2": 0.006,
        "exact_trace_cosine": 0.9998,
        "exact_trace_max_absolute": 0.02,
        "inactive_max_absolute": target["inactive_max_abs"],
        "peak_rss_bytes": 4096,
        **counts,
        "final_level": target["output_level"],
        "final_remaining_levels": target["remaining_levels"],
        "multiplicative_depth": target["multiplicative_depth"],
        "max_observed_level": target["max_observed_level"],
        "max_polynomial_depth": target["max_polynomial_depth"],
    }


def raw_stdout_bytes() -> bytes:
    groups: list[str] = []
    for _ in validator.TIME_PHASES:
        target = runtime_record()
        groups.extend(
            (
                validator.PROFILE_WARNING,
                json.dumps(
                    diagnostic_record(target),
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                json.dumps(target, sort_keys=True, separators=(",", ":")),
            )
        )
    return ("\n".join(groups) + "\n").encode("utf-8")


def raw_time_bytes() -> bytes:
    command = " ".join(validator._m4_runtime_argv(validator.REPO_ROOT))
    return "".join(
        f"{validator.TIME_RECORD_PREFIX}\t{phase}\t{1 + index}.25\t"
        f"{2048 + index}\t0\t{command}\n"
        for index, phase in enumerate(validator.TIME_PHASES)
    ).encode("utf-8")


class M4ArtifactV5ValidatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = validator.validate_schema(validator.load_json(SCHEMA_PATH))

    def test_schema_identity_is_frozen_to_m4_v5(self) -> None:
        self.assertEqual(self.schema["$id"], validator.SCHEMA_ID)
        self.assertEqual(self.schema["title"], validator.SCHEMA_TITLE)
        self.assertEqual(self.schema["properties"]["schema_version"]["const"], 5)
        self.assertEqual(self.schema["properties"]["milestone"]["const"], "M4")
        self.assertIn("build_configuration", self.schema["required"])
        self.assertEqual(
            self.schema["properties"]["build_configuration"]["$ref"],
            "#/$defs/build_configuration",
        )
        configure_command = self.schema["properties"]["commands"]["prefixItems"][5][
            "allOf"
        ][1]["properties"]["command"]["const"]
        self.assertIn("/usr/bin/cmake --fresh", configure_command)
        self.assertIn("--unset=CXXFLAGS", configure_command)
        self.assertEqual(
            self.schema["properties"]["git"]["properties"]["remote_url"]["const"],
            validator.REMOTE_URL,
        )
        self.assertEqual(
            self.schema["properties"]["source_build_provenance"]["$ref"],
            "#/$defs/source_build_provenance",
        )
        provenance = self.schema["$defs"]["source_build_provenance"]["properties"]
        self.assertEqual(
            (
                provenance["source_files"]["minItems"],
                provenance["source_files"]["maxItems"],
                provenance["build_products"]["minItems"],
                provenance["build_products"]["maxItems"],
            ),
            (
                len(validator.M4_SOURCE_PATHS),
                len(validator.M4_SOURCE_PATHS),
                len(validator.M4_BUILD_PRODUCT_PATHS),
                len(validator.M4_BUILD_PRODUCT_PATHS),
            ),
        )
        artifact_contracts = self.schema["properties"]["artifacts"]["prefixItems"]
        self.assertEqual(
            [
                (
                    item["allOf"][1]["properties"]["path"]["const"],
                    item["allOf"][1]["properties"]["role"]["const"],
                    item["allOf"][1]["properties"]["media_type"]["const"],
                )
                for item in artifact_contracts
            ],
            [
                ("stdout.log", "stdout", "text/plain"),
                ("stderr.log", "stderr", "text/plain"),
                ("time.log", "resource_metrics", "text/plain"),
                ("metrics.csv", "metrics", "text/csv"),
            ],
        )
        tampered_artifacts = copy.deepcopy(self.schema)
        tampered_artifacts["properties"]["artifacts"]["prefixItems"][1]["allOf"][1][
            "properties"
        ]["role"]["const"] = "stdout"
        with self.assertRaisesRegex(
            validator.ValidationError,
            "artifact order/roles drifted",
        ):
            validator.validate_schema(tampered_artifacts)
        self.assertEqual(len(self.schema["oneOf"]), 1)

    def test_claim_boundary_is_exact_and_tamper_closed(self) -> None:
        expected = [
            "M4 covers only single-layer layer-1 correctness on the fixed five-token "
            "BERT-base encoder trace.",
            "The server-only claim is an API/target trust boundary: server code "
            "receives only ciphertexts, public model weights, and evaluation keys; "
            "it is not a process-isolation claim.",
            "paper_compat OpenFHE CKKS CPU parameters are research-reproduction "
            "parameters with security_claim=none.",
            "This artifact makes no 128-bit-security or production-security claim.",
            "Timing is diagnostic only; this artifact makes no benchmark or speedup "
            "claim.",
            "GPU, Discrete CKKS/FBT, QDQ, tokenizer, classifier, and task-level "
            "inference are excluded.",
        ]
        claim_schema = self.schema["properties"]["claim_boundary"]
        self.assertEqual(list(validator.CLAIM_BOUNDARY), expected)
        self.assertEqual(claim_schema, {"type": "array", "const": expected})
        validator.validate_instance(expected, claim_schema, self.schema)
        validator._verify_claim_boundary(expected)

        for index in range(len(expected)):
            with self.subTest(index=index):
                tampered = copy.deepcopy(expected)
                tampered[index] += " Expanded claim."
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(tampered, claim_schema, self.schema)
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    "claim_boundary differs",
                ):
                    validator._verify_claim_boundary(tampered)

        expanded = [*expected, "Any additional claim is forbidden."]
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(expanded, claim_schema, self.schema)
        with self.assertRaises(validator.ValidationError):
            validator._verify_claim_boundary(expanded)

        tampered_schema = copy.deepcopy(self.schema)
        tampered_schema["properties"]["claim_boundary"]["const"][0] += " Expanded."
        with self.assertRaisesRegex(
            validator.ValidationError,
            "claim boundary contract drifted",
        ):
            validator.validate_schema(tampered_schema)

    def test_checksum_rehashes_complete_bundle_and_detects_manifest_tamper(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-checksum-") as directory:
            root = Path(directory) / "bundle"
            root.mkdir()

            def write_clean_bundle() -> dict[str, dict[str, str]]:
                contents = {
                    "stdout.log": "stdout\n",
                    "stderr.log": "",
                    "time.log": "time\n",
                    "metrics.csv": "metric,value\n",
                    "manifest.json": '{"schema_version": 5, "verdict": "GO"}\n',
                }
                for relative_path, content in contents.items():
                    (root / relative_path).write_text(content, encoding="utf-8")
                digests = {
                    path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                    for path in validator.M4_CHECKSUM_EVIDENCE_PATHS
                }
                (root / "SHA256SUMS").write_text(
                    "".join(
                        f"{digests[path]}  {path}\n"
                        for path in validator.M4_CHECKSUM_EVIDENCE_PATHS
                    ),
                    encoding="utf-8",
                )
                return {
                    path: {"sha256": digests[path]}
                    for path in validator.M4_MANIFEST_ARTIFACTS
                }

            artifact_records = write_clean_bundle()
            validator._verify_bundle_inventory(root)
            validator._verify_checksum_evidence(root, artifact_records)

            (root / "manifest.json").write_text(
                '{"schema_version": 5, "verdict": "NO-GO"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                validator.ValidationError,
                "digest mismatch for manifest.json",
            ):
                validator._verify_checksum_evidence(root, artifact_records)

            for name, transform in (
                ("missing", lambda lines: lines[:-1]),
                ("extra", lambda lines: [*lines, f"{'0' * 64}  extra.raw"]),
                ("duplicate", lambda lines: [*lines, lines[0]]),
            ):
                with self.subTest(name=name):
                    artifact_records = write_clean_bundle()
                    checksum_path = root / "SHA256SUMS"
                    lines = checksum_path.read_text(encoding="utf-8").splitlines()
                    checksum_path.write_text(
                        "\n".join(transform(lines)) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(validator.ValidationError):
                        validator._verify_checksum_evidence(root, artifact_records)

            artifact_records = write_clean_bundle()
            (root / "extra.raw").write_text("extra\n", encoding="utf-8")
            with self.assertRaisesRegex(validator.ValidationError, "extra.raw"):
                validator._verify_bundle_inventory(root)
            (root / "extra.raw").unlink()
            (root / "metrics.csv").unlink()
            with self.assertRaisesRegex(validator.ValidationError, "metrics.csv"):
                validator._verify_bundle_inventory(root)

            artifact_records = write_clean_bundle()
            checksum_path = root / "SHA256SUMS"
            checksum_path.unlink()
            outside_checksum = Path(directory) / "outside-SHA256SUMS"
            outside_checksum.write_text(
                "external checksum evidence\n",
                encoding="utf-8",
            )
            checksum_path.symlink_to(outside_checksum)
            validator._verify_bundle_inventory(root)
            with self.assertRaisesRegex(
                validator.ValidationError,
                "path contains a symlink",
            ):
                validator._verify_checksum_evidence(root, artifact_records)

    def test_raw_stdout_time_and_csv_reject_ambiguity(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-raw-") as directory:
            root = Path(directory)
            stdout_path = root / "stdout.log"
            time_path = root / "time.log"
            metrics_path = root / "metrics.csv"
            stdout_path.write_bytes(raw_stdout_bytes())
            time_path.write_bytes(raw_time_bytes())

            groups = validator._parse_m4_stdout_runs(stdout_path)
            timing = validator._parse_m4_time_log(
                time_path,
                validator.REPO_ROOT,
            )
            records = [target for target, _ in groups][1:]
            diagnostics = [diagnostic for _, diagnostic in groups][1:]
            measured_timing = timing[1:]
            metrics = validator._m4_metrics_bytes(
                records,
                diagnostics,
                measured_timing,
            )
            metrics_path.write_bytes(metrics)
            validator._read_encoder_metrics_csv(
                metrics_path,
                5,
                records,
                diagnostics,
            )

            lines = raw_stdout_bytes().decode("utf-8").splitlines()
            for injected in (
                '{"unknown":1,"unknown":2}',
                '{"unknown":NaN}',
                '{"unknown":',
                json.dumps(runtime_record(), separators=(",", ":")),
            ):
                with self.subTest(stdout=injected):
                    stdout_path.write_text(
                        "\n".join([lines[0], injected, *lines[1:]]) + "\n",
                        encoding="utf-8",
                    )
                    with self.assertRaises(validator.ValidationError):
                        validator._parse_m4_stdout_runs(stdout_path)
            stdout_path.write_bytes(raw_stdout_bytes())

            raw_time = raw_time_bytes().decode("utf-8")
            time_path.write_text(
                raw_time.replace("\t0\t", "\t1\t", 1),
                encoding="utf-8",
            )
            with self.assertRaises(validator.ValidationError):
                validator._parse_m4_time_log(time_path, validator.REPO_ROOT)
            time_path.write_bytes(raw_time_bytes())
            time_path.write_text(
                raw_time.replace("\t1.25\t", "\t01.25\t", 1),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                validator.ValidationError,
                "canonical C-locale",
            ):
                validator._parse_m4_time_log(time_path, validator.REPO_ROOT)
            time_path.write_bytes(raw_time_bytes())

            text = metrics.decode("utf-8")
            header, *rows = text.splitlines()
            metrics_path.write_text(
                f"run,{header}\n" + "\n".join(f"0,{row}" for row in rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                validator.ValidationError,
                "header/order",
            ):
                validator._read_encoder_metrics_csv(
                    metrics_path,
                    5,
                    records,
                    diagnostics,
                )

            metrics_path.write_bytes(metrics)
            csv_lines = metrics_path.read_text(encoding="utf-8").splitlines()
            csv_lines[1] += ",extra"
            metrics_path.write_text("\n".join(csv_lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(
                validator.ValidationError,
                "row width",
            ):
                validator._read_encoder_metrics_csv(
                    metrics_path,
                    5,
                    records,
                    diagnostics,
                )

    def test_json_files_reject_duplicate_nan_and_malformed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-json-") as directory:
            path = Path(directory) / "input.json"
            for content in (
                '{"key":1,"key":2}\n',
                '{"key":NaN}\n',
                '{"key":\n',
            ):
                with self.subTest(content=content):
                    path.write_text(content, encoding="utf-8")
                    with self.assertRaises(validator.ValidationError):
                        validator.load_json(path)

    def test_success_stderr_is_bounded_raw_evidence_not_required_empty(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-stderr-") as directory:
            path = Path(directory) / "stderr.log"
            expected = b"OpenFHE warning\n\x00binary detail\n"
            path.write_bytes(expected)
            self.assertEqual(validator._verify_m4_stderr_log(path), expected)

            path.write_bytes(b"x" * (validator.RAW_STDERR_MAX_BYTES + 1))
            with self.assertRaisesRegex(
                validator.ValidationError,
                "exceeds the frozen M4 bound",
            ):
                validator._verify_m4_stderr_log(path)

    def test_git_validation_freezes_binary_environment_and_origin(self) -> None:
        poisoned = {
            "PATH": "/tmp/fake-bin:/usr/bin",
            "GIT_DIR": "/tmp/fake.git",
            "GIT_CONFIG_COUNT": "1",
        }
        with mock.patch.dict(os.environ, poisoned, clear=False):
            environment = validator._git_environment()
        self.assertEqual(environment["PATH"], "/usr/bin:/bin")
        self.assertFalse(any(name.startswith("GIT_") for name in environment))

        completed = subprocess.CompletedProcess(
            [validator.SYSTEM_GIT, "rev-parse", "HEAD"],
            0,
            stdout="a" * 40 + "\n",
            stderr="",
        )
        with mock.patch.object(
            validator.subprocess,
            "run",
            return_value=completed,
        ) as run:
            output = validator._run_git(validator.REPO_ROOT, ["rev-parse", "HEAD"])
        self.assertEqual(output, "a" * 40)
        self.assertEqual(run.call_args.args[0][0], validator.SYSTEM_GIT)
        self.assertFalse(
            any(
                name.startswith("GIT_")
                for name in run.call_args.kwargs["env"]
            )
        )

        head = "a" * 40
        manifest = {
            "git": {
                "local_commit": head,
                "branch": "refactor/openfhe-cpu",
                "clean": True,
                "remote_name": validator.REMOTE_NAME,
                "remote_url": validator.REMOTE_URL,
                "remote_ref": validator.REMOTE_REF,
                "remote_commit": head,
            },
            "verdict": "GO",
            "gate": {"passed": True, "checks": {"all": "PASS"}},
            "commands": [],
        }
        live = [
            head,
            "refactor/openfhe-cpu",
            "",
            validator.REMOTE_URL,
            f"{head}\t{validator.REMOTE_REF}",
        ]
        with mock.patch.object(
            validator,
            "_run_git",
            side_effect=live,
        ) as run_git:
            validator._verify_git(manifest, validator.REPO_ROOT, True)
        self.assertEqual(
            run_git.call_args_list[-1].args[1],
            [
                "ls-remote",
                "--exit-code",
                validator.REMOTE_URL,
                validator.REMOTE_REF,
            ],
        )

        poisoned_live = [*live]
        poisoned_live[3] = "file:///tmp/fake.git"
        with (
            mock.patch.object(
                validator,
                "_run_git",
                side_effect=poisoned_live,
            ),
            self.assertRaisesRegex(validator.ValidationError, "URL differs"),
        ):
            validator._verify_git(manifest, validator.REPO_ROOT, True)

    def test_source_build_provenance_binds_head_source_and_live_products(self) -> None:
        with tempfile.TemporaryDirectory(prefix="m4-v5-provenance-") as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "build").mkdir()
            source = root / "src" / "source.cpp"
            product = root / "build" / "product.o"
            executable = root / "build" / "encoder"
            source.write_text("int source = 1;\n", encoding="utf-8")
            product.write_bytes(b"object")
            executable.write_bytes(b"executable")
            executable.chmod(0o755)
            subprocess.run([validator.SYSTEM_GIT, "init", "-q"], cwd=root, check=True)
            subprocess.run(
                [validator.SYSTEM_GIT, "config", "user.email", "tests@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [validator.SYSTEM_GIT, "config", "user.name", "MOAI tests"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [validator.SYSTEM_GIT, "add", "src/source.cpp"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                [validator.SYSTEM_GIT, "commit", "-qm", "fixture"],
                cwd=root,
                check=True,
            )
            head = subprocess.run(
                [validator.SYSTEM_GIT, "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            with (
                mock.patch.object(
                    validator,
                    "M4_SOURCE_PATHS",
                    ("src/source.cpp",),
                ),
                mock.patch.object(
                    validator,
                    "M4_BUILD_PRODUCT_PATHS",
                    ("build/product.o",),
                ),
                mock.patch.object(
                    validator,
                    "M4_WORKLOAD_EXECUTABLE_PATH",
                    "build/encoder",
                ),
            ):
                provenance = validator._live_source_build_provenance(root, head)
                self.assertEqual(
                    provenance["source_files"][0]["sha256"],
                    provenance["source_files"][0]["head_blob_sha256"],
                )
                source.write_text("int source = 2;\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    validator.ValidationError,
                    "differs from HEAD blob",
                ):
                    validator._live_source_build_provenance(root, head)

    def test_artifact_environment_and_schema_are_fail_closed(self) -> None:
        poisoned = {name: f"poison-{name}" for name in validator.CONFIGURE_ENV_UNSET}
        poisoned["MOAI_SAFE_SENTINEL"] = "preserved"
        with mock.patch.dict(os.environ, poisoned, clear=False):
            environment = validator._artifact_environment()
        for name in validator.CONFIGURE_ENV_UNSET:
            with self.subTest(name=name):
                self.assertNotIn(name, environment)
        self.assertEqual(
            {
                name: environment[name]
                for name in validator.FORCED_SUBPROCESS_ENVIRONMENT
            },
            validator.FORCED_SUBPROCESS_ENVIRONMENT,
        )
        self.assertEqual(environment["MOAI_SAFE_SENTINEL"], "preserved")

        schema = self.schema["$defs"]["build_configuration"]
        record = build_configuration()
        validator.validate_instance(record, schema, self.schema)
        for name, replacement in (
            ("environment_inheritance", "ambient"),
            ("cleared_environment_variables", ["CXXFLAGS"]),
            ("forced_environment_variables", {"LANG": "C"}),
        ):
            with self.subTest(field=name):
                tampered = copy.deepcopy(record)
                tampered[name] = replacement
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(tampered, schema, self.schema)

    def test_narrow_ctest_contract_is_frozen_and_exact(self) -> None:
        expected = (
            "openfhe_m4_v5_artifact_schema_contract",
            "openfhe_m4_v5_artifact_validator_contract",
            "openfhe_m4_v5_encoder_artifact_runner_contract",
            "openfhe_feature_layernorm_smoke",
        )
        self.assertEqual(validator.M4_V5_REQUIRED_CTEST_CONTRACTS, expected)
        for name in expected:
            with self.subTest(name=name):
                self.assertIsNotNone(re.fullmatch(validator.M4_CTEST_PATTERN, name))
                self.assertIsNone(
                    re.fullmatch(validator.M4_CTEST_PATTERN, f"{name}_not_frozen")
                )
        execution_schema = self.schema["$defs"]["m4_execution"]
        execution = {
            "mode": "server-only",
            "quality_reference": (
                "config/moai_encoder_trace.json single-layer frozen polynomial oracle"
            ),
            "encoder_layers": 1,
            "layer_id": 1,
            "trace_shape": [5, 768],
            "feature_block_size": 1024,
            "checkpoint_decryption_owner": "client",
            "server_private_key_present": False,
            "server_decryptions": 0,
            "server_plaintext_activations": False,
            "multiplicative_depth": 47,
            "max_observed_level": 45,
            "max_polynomial_depth": 10,
            "post_bootstrap_mask": "single_normal_scale_mask",
            "internal_checkpoint_metadata": copy.deepcopy(
                validator.M4_INTERNAL_CHECKPOINT_METADATA
            ),
            "required_checkpoints": list(validator.M4_CHECKPOINT_NAMES),
            "required_ctest_contracts": list(expected),
        }
        validator.validate_instance(execution, execution_schema, self.schema)
        execution["required_ctest_contracts"] = [
            "openfhe_artifact_schema_contract",
            "openfhe_artifact_validator_contract",
            "openfhe_encoder_artifact_runner_contract",
        ]
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(execution, execution_schema, self.schema)

    def test_schema_rejects_historical_only_ctest_binding(self) -> None:
        schema = copy.deepcopy(self.schema)
        schema["$defs"]["m4_execution"]["properties"]["required_ctest_contracts"][
            "const"
        ] = [
            "openfhe_artifact_schema_contract",
            "openfhe_artifact_validator_contract",
            "openfhe_encoder_artifact_runner_contract",
        ]
        with self.assertRaisesRegex(
            validator.ValidationError, "required CTest contracts drifted"
        ):
            validator.validate_schema(schema)

    def test_schema_rejects_old_mask_and_internal_checkpoint_schedule(self) -> None:
        mask_schema = copy.deepcopy(self.schema)
        mask_schema["$defs"]["m4_execution"]["properties"]["post_bootstrap_mask"][
            "const"
        ] = "double_mask"
        with self.assertRaisesRegex(
            validator.ValidationError, "post-bootstrap mask contract drifted"
        ):
            validator.validate_schema(mask_schema)

        metadata_schema = copy.deepcopy(self.schema)
        metadata_schema["$defs"]["m4_execution"]["properties"][
            "internal_checkpoint_metadata"
        ]["const"]["layernorm_normalized_variance"] = {
            "level": 20,
            "noise_scale_degree": 2,
            "remaining_levels": 26,
        }
        with self.assertRaisesRegex(
            validator.ValidationError, "internal checkpoint metadata drifted"
        ):
            validator.validate_schema(metadata_schema)

    def test_schema_identity_version_and_milestone_tamper_are_rejected(self) -> None:
        cases = (
            (("$id",), "https://local.moai/wrong.json"),
            (("title",), "wrong title"),
            (("properties", "schema_version", "const"), 4),
            (("properties", "milestone", "const"), "M3"),
        )
        for path, replacement in cases:
            with self.subTest(path=path):
                schema = copy.deepcopy(self.schema)
                current = schema
                for key in path[:-1]:
                    current = current[key]
                current[path[-1]] = replacement
                with self.assertRaises(validator.ValidationError):
                    validator.validate_schema(schema)

    def test_checkpoint_digest_and_live_schedule_are_frozen(self) -> None:
        checkpoints = copy.deepcopy(list(validator.M4_EXPECTED_CHECKPOINTS))
        self.assertEqual(checkpoints[-1]["level"], 29)
        self.assertEqual(checkpoints[-1]["remaining_levels"], 17)
        self.assertEqual(
            validator.checkpoint_metadata_sha256(checkpoints),
            "c4c1c85e52154215784b9fa93a584d824dde94a644e5af997882486aac01a6db",
        )
        for index, checkpoint in enumerate(checkpoints):
            validator._verify_checkpoint(
                checkpoint,
                validator.M4_EXPECTED_CHECKPOINTS[index],
                f"checkpoint[{index}]",
            )
        checkpoints[-1]["level"] = 30
        checkpoints[-1]["remaining_levels"] = 16
        with self.assertRaisesRegex(validator.ValidationError, "level must be 29"):
            validator._verify_checkpoint(
                checkpoints[-1], validator.M4_EXPECTED_CHECKPOINTS[-1], "output"
            )

    def test_command_transcript_rejects_configure_environment_tamper(self) -> None:
        timestamp = "2026-07-31T10:00:00+09:00"

        def command(tokens: list[str], *, timed: bool) -> dict[str, object]:
            record: dict[str, object] = {
                "command": shlex.join(tokens),
                "cwd": str(validator.REPO_ROOT),
                "exit_code": 0,
                "phase": "artifact_generation",
            }
            if timed:
                record["started_at"] = timestamp
                record["finished_at"] = timestamp
            return record

        executable = validator.REPO_ROOT / validator.M4_WORKLOAD_EXECUTABLE_PATH
        commands = [
            command(
                [
                    validator.SYSTEM_GIT,
                    "status",
                    "--porcelain=v1",
                    "--untracked-files=normal",
                ],
                timed=False,
            ),
            command(
                [validator.SYSTEM_GIT, "branch", "--show-current"],
                timed=False,
            ),
            command(
                [validator.SYSTEM_GIT, "rev-parse", "HEAD"],
                timed=False,
            ),
            command(
                [
                    validator.SYSTEM_GIT,
                    "remote",
                    "get-url",
                    "--all",
                    validator.REMOTE_NAME,
                ],
                timed=False,
            ),
            command(
                [
                    validator.SYSTEM_GIT,
                    "ls-remote",
                    "--exit-code",
                    validator.REMOTE_URL,
                    validator.REMOTE_REF,
                ],
                timed=False,
            ),
            command(validator._expected_configure_command(), timed=True),
            command(
                [
                    validator.SYSTEM_CMAKE,
                    "--build",
                    str(validator.BUILD_ROOT),
                    "--clean-first",
                    "-j",
                    "4",
                ],
                timed=True,
            ),
            command([validator.SYSTEM_LDD, str(executable)], timed=True),
            command(
                [
                    validator.SYSTEM_CTEST,
                    "--test-dir",
                    str(validator.BUILD_ROOT),
                    "--output-on-failure",
                    "--no-tests=error",
                    "-R",
                    validator.M4_CTEST_PATTERN,
                ],
                timed=True,
            ),
        ]
        manifest_path = Path("/tmp/m4-v5-manifest.json")
        time_path = manifest_path.resolve().parent / "time.log"
        for phase in validator.TIME_PHASES:
            commands.append(
                command(
                    [
                        "/usr/bin/time",
                        "--append",
                        f"--format={validator._m4_time_format(phase)}",
                        f"--output={time_path}",
                        *validator._m4_runtime_argv(validator.REPO_ROOT),
                    ],
                    timed=True,
                )
            )
        commands.append(
            command(
                [
                    sys.executable,
                    str(validator.REPO_ROOT / validator.VALIDATOR_RELATIVE_PATH),
                    "--schema",
                    str(validator.REPO_ROOT / validator.SCHEMA_RELATIVE_PATH),
                    "--manifest",
                    str(manifest_path),
                    "--verify-git",
                ],
                timed=False,
            )
        )
        manifest = {"milestone": "M4", "verdict": "GO", "commands": commands}
        validator._verify_m4_go_command_transcript(manifest, manifest_path)

        tampered = copy.deepcopy(manifest)
        tokens = shlex.split(tampered["commands"][5]["command"])
        tokens.remove("--unset=CPATH")
        tampered["commands"][5]["command"] = shlex.join(tokens)
        with self.assertRaisesRegex(validator.ValidationError, "commands\\[5\\]"):
            validator._verify_m4_go_command_transcript(tampered, manifest_path)

    def test_semantic_build_provenance_rejects_each_live_drift(self) -> None:
        record = build_configuration()
        manifest = {"build_configuration": record}
        package_records = record["openfhe_cmake_package_files"]
        include_record = record["openfhe_include_tree"]
        linked_records = record["openfhe_linked_libraries"]

        def patches(
            *,
            packages: object = package_records,
            include: object = include_record,
            linked: object = linked_records,
        ) -> tuple[mock._patch, ...]:
            return (
                mock.patch.object(validator, "_verify_file_record"),
                mock.patch.object(validator, "_verify_cmake_cache"),
                mock.patch.object(
                    validator,
                    "_absolute_regular_file_record",
                    side_effect=copy.deepcopy(packages),
                ),
                mock.patch.object(
                    validator,
                    "_openfhe_include_tree_record",
                    return_value=copy.deepcopy(include),
                ),
                mock.patch.object(
                    validator,
                    "_live_openfhe_linked_libraries",
                    return_value=copy.deepcopy(linked),
                ),
            )

        baseline = patches()
        with baseline[0], baseline[1], baseline[2], baseline[3], baseline[4]:
            validator._verify_build_configuration(manifest)

        drifted_packages = copy.deepcopy(package_records)
        drifted_packages[0]["sha256"] = "0" * 64
        package_patches = patches(packages=drifted_packages)
        with (
            package_patches[0],
            package_patches[1],
            package_patches[2],
            package_patches[3],
            package_patches[4],
            self.assertRaisesRegex(
                validator.ValidationError,
                "CMake package file provenance drifted",
            ),
        ):
            validator._verify_build_configuration(manifest)

        drifted_include = copy.deepcopy(include_record)
        drifted_include["sha256"] = "0" * 64
        include_patches = patches(include=drifted_include)
        with (
            include_patches[0],
            include_patches[1],
            include_patches[2],
            include_patches[3],
            include_patches[4],
            self.assertRaisesRegex(
                validator.ValidationError,
                "include tree provenance drifted",
            ),
        ):
            validator._verify_build_configuration(manifest)

        drifted_linked = copy.deepcopy(linked_records)
        drifted_linked[0]["sha256"] = "0" * 64
        linked_patches = patches(linked=drifted_linked)
        with (
            linked_patches[0],
            linked_patches[1],
            linked_patches[2],
            linked_patches[3],
            linked_patches[4],
            self.assertRaisesRegex(
                validator.ValidationError,
                "linked-library provenance drifted",
            ),
        ):
            validator._verify_build_configuration(manifest)

        cache_patches = patches()
        with (
            mock.patch.object(
                validator,
                "_verify_file_record",
                side_effect=validator.ValidationError(
                    "build_configuration.cmake_cache: SHA-256 mismatch"
                ),
            ),
            cache_patches[1],
            cache_patches[2],
            cache_patches[3],
            cache_patches[4],
            self.assertRaisesRegex(
                validator.ValidationError,
                "cmake_cache.*SHA-256 mismatch",
            ),
        ):
            validator._verify_build_configuration(manifest)

    def test_operation_counts_reject_every_drift(self) -> None:
        expected = {
            "rotations": 6300,
            "ct_pt_multiplications": 51885,
            "ct_ct_multiplications": 95,
            "explicit_rescale_requests": 810,
            "chebyshev_evaluations": 55,
            "estimated_polynomial_multiplications": 1150,
            "bootstraps": 25,
            "bootstrap_iterations": 50,
        }
        self.assertEqual(
            validator._verify_encoder_operation_counts(
                expected, "counts", 6300, 25, 50
            ),
            expected,
        )
        for key in expected:
            with self.subTest(key=key):
                tampered = copy.deepcopy(expected)
                tampered[key] += 1
                with self.assertRaisesRegex(validator.ValidationError, key):
                    validator._verify_encoder_operation_counts(
                        tampered, "counts", 6300, 25, 50
                    )

    def test_layernorm_registered_ranges_accept_exact_edges(self) -> None:
        self.assertEqual(
            validator._verify_inactive_sentinel_ranges(sentinel_ranges(), "sentinels"),
            sentinel_ranges(),
        )
        validator.validate_instance(
            sentinel_ranges(),
            self.schema["$defs"]["m4_inactive_polynomial_sentinel_ranges"],
            self.schema,
        )

    def test_layernorm_registered_interval_cannot_be_relaxed(self) -> None:
        cases = (
            ("attention_layernorm_normalized_variance", "minimum", 0.499999),
            ("attention_layernorm_normalized_variance", "maximum", 1536.0001),
            ("output_layernorm_normalized_variance", "minimum", float("nan")),
            ("output_layernorm_normalized_variance", "maximum", float("inf")),
        )
        for site, field, replacement in cases:
            with self.subTest(site=site, field=field):
                ranges = sentinel_ranges()
                ranges[site][field] = replacement
                with self.assertRaises(validator.ValidationError):
                    validator._verify_inactive_sentinel_ranges(ranges, "sentinels")
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(
                        ranges,
                        self.schema["$defs"]["m4_inactive_polynomial_sentinel_ranges"],
                        self.schema,
                    )
        ranges = sentinel_ranges()
        ranges["attention_layernorm_normalized_variance"] = {
            "minimum": 64.0,
            "maximum": 63.0,
        }
        with self.assertRaisesRegex(validator.ValidationError, "minimum exceeds"):
            validator._verify_inactive_sentinel_ranges(ranges, "sentinels")

    def test_softmax_keeps_reciprocal_interval(self) -> None:
        ranges = sentinel_ranges()
        ranges["softmax_denominator"] = {"minimum": 0.5, "maximum": 2.0}
        validator._verify_inactive_sentinel_ranges(ranges, "sentinels")
        ranges["softmax_denominator"]["minimum"] = 0.009999
        with self.assertRaises(validator.ValidationError):
            validator._verify_inactive_sentinel_ranges(ranges, "sentinels")

    def test_trace_scale_provenance_schema_and_source_hashes_fail_closed(
        self,
    ) -> None:
        provenance = {
            "source_path": validator.TRACE_SCALE_SOURCE_PATH,
            "json_locator": validator.TRACE_SCALE_JSON_LOCATOR,
            "contract_id": validator.TRACE_SCALE_CONTRACT_ID,
            "contract_sha256": validator.TRACE_SCALE_CONTRACT_SHA256,
            "values_sha256": validator.TRACE_SCALE_VALUES_SHA256,
            "raw_variance_sha256": validator.TRACE_SCALE_RAW_VARIANCE_SHA256,
        }
        provenance_schema = self.schema["$defs"]["m4_trace_scale_contract_provenance"]
        validator.validate_instance(provenance, provenance_schema, self.schema)
        self.assertEqual(
            validator._trace_scale_contract_provenance(REPO_ROOT),
            provenance,
        )

        approximation = json.loads(
            (REPO_ROOT / validator.TRACE_SCALE_SOURCE_PATH).read_text(encoding="utf-8")
        )
        contract = approximation["operators"]["layernorm"][
            "feature_packed_trace_scale_contract"
        ]
        self.assertEqual(
            validator._binary64_tensor_sha256(
                contract["values"],
                validator.TRACE_SCALE_SHAPE,
                "values",
            ),
            validator.TRACE_SCALE_VALUES_SHA256,
        )
        profile = json.loads(
            (REPO_ROOT / validator.ENCODER_FEATURE_PROFILE_PATH).read_text(
                encoding="utf-8"
            )
        )
        for field in (
            "contract_sha256",
            "values_sha256",
            "raw_variance_sha256",
        ):
            with self.subTest(field=field):
                tampered_provenance = copy.deepcopy(provenance)
                tampered_provenance[field] = "0" * 64
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(
                        tampered_provenance,
                        provenance_schema,
                        self.schema,
                    )

                tampered_approximation = copy.deepcopy(approximation)
                tampered_profile = copy.deepcopy(profile)
                tampered_approximation["operators"]["layernorm"][
                    "feature_packed_trace_scale_contract"
                ][field] = "0" * 64
                tampered_profile["feature_packed_layernorm_override"][
                    "trace_scale_contract"
                ][field] = "0" * 64
                with mock.patch.object(
                    validator,
                    "load_json",
                    side_effect=[tampered_approximation, tampered_profile],
                ):
                    with self.assertRaisesRegex(
                        validator.ValidationError,
                        "trace-scale",
                    ):
                        validator._trace_scale_contract_provenance(REPO_ROOT)

        invalid_length = copy.deepcopy(provenance)
        invalid_length["contract_sha256"] = validator.TRACE_SCALE_CONTRACT_SHA256[:-1]
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(
                invalid_length,
                provenance_schema,
                self.schema,
            )

    def test_unsealed_profile_schedule_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            validator.ValidationError,
            "metadata schedule is not sealed",
        ):
            validator._require_m4_schedule_sealed(REPO_ROOT)

    def test_schema_binding_shape_and_hash_tamper(self) -> None:
        binding = {
            "schema": {
                "id": validator.SCHEMA_ID,
                "title": validator.SCHEMA_TITLE,
                "version": 5,
                "path": validator.SCHEMA_RELATIVE_PATH,
                "sha256": "a" * 64,
            },
            "validator": {
                "path": validator.VALIDATOR_RELATIVE_PATH,
                "sha256": "b" * 64,
            },
            "runner": {
                "path": validator.RUNNER_RELATIVE_PATH,
                "sha256": "c" * 64,
            },
        }
        validator.validate_instance(
            binding, self.schema["$defs"]["schema_binding"], self.schema
        )
        for path, value in (
            (("schema", "id"), "https://local.moai/wrong.json"),
            (("schema", "version"), 4),
            (("schema", "sha256"), "0" * 63),
            (("validator", "path"), "scripts/validate_openfhe_artifact.py"),
            (("runner", "sha256"), "not-a-hash"),
        ):
            with self.subTest(path=path):
                tampered = copy.deepcopy(binding)
                tampered[path[0]][path[1]] = value
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(
                        tampered,
                        self.schema["$defs"]["schema_binding"],
                        self.schema,
                    )

    def test_head_blob_and_manifest_binding_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(
            prefix="m4-v5-validator-binding-"
        ) as directory:
            root = Path(directory)
            files = {
                validator.SCHEMA_RELATIVE_PATH: json.dumps(
                    {
                        "$id": validator.SCHEMA_ID,
                        "title": validator.SCHEMA_TITLE,
                        "properties": {
                            "schema_version": {"const": validator.SCHEMA_VERSION}
                        },
                    }
                )
                + "\n",
                validator.VALIDATOR_RELATIVE_PATH: "# validator\n",
                validator.RUNNER_RELATIVE_PATH: "# runner\n",
            }
            for relative, content in files.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "tests@example.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "MOAI tests"], cwd=root, check=True
            )
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            head = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            binding = validator._schema_binding_for_repository(root, head)
            for role, relative in (
                ("schema", validator.SCHEMA_RELATIVE_PATH),
                ("validator", validator.VALIDATOR_RELATIVE_PATH),
                ("runner", validator.RUNNER_RELATIVE_PATH),
            ):
                self.assertEqual(binding[role]["path"], relative)
                self.assertEqual(
                    binding[role]["sha256"],
                    hashlib.sha256((root / relative).read_bytes()).hexdigest(),
                )
            manifest = {
                "git": {"local_commit": head},
                "schema_binding": copy.deepcopy(binding),
            }
            validator._verify_schema_binding(manifest, root)
            manifest["schema_binding"]["runner"]["sha256"] = "0" * 64
            with self.assertRaisesRegex(
                validator.ValidationError, "manifest schema_binding differs"
            ):
                validator._verify_schema_binding(manifest, root)
            (root / validator.VALIDATOR_RELATIVE_PATH).write_text(
                "# tampered validator\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(
                validator.ValidationError, "differ from HEAD blob"
            ):
                validator._schema_binding_for_repository(root, head)

    def test_threshold_contract_is_frozen(self) -> None:
        self.assertEqual(
            validator.M4_EXPECTED_THRESHOLDS,
            {
                "relative_l2_max": 1e-2,
                "cosine_min": 0.999,
                "inactive_max_abs": 1e-6,
            },
        )
        threshold_schema = self.schema["$defs"]["m4_thresholds"]
        validator.validate_instance(
            copy.deepcopy(validator.M4_EXPECTED_THRESHOLDS),
            threshold_schema,
            self.schema,
        )
        relaxed = copy.deepcopy(validator.M4_EXPECTED_THRESHOLDS)
        relaxed["inactive_max_abs"] = 1e-5
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(relaxed, threshold_schema, self.schema)

        quality = {
            "relative_l2_max": 0.001,
            "cosine_min": 0.9999,
            "inactive_max_abs_max": 1e-6,
            "layernorm_inactive_guard_max_error": 100.0,
        }
        quality_schema = self.schema["$defs"]["m4_quality"]
        validator.validate_instance(quality, quality_schema, self.schema)
        quality["inactive_max_abs_max"] = 1.000001e-6
        with self.assertRaises(validator.ValidationError):
            validator.validate_instance(quality, quality_schema, self.schema)
        for invalid_guard in (-1e-12, float("nan"), float("inf")):
            with self.subTest(invalid_guard=invalid_guard):
                invalid_quality = {
                    **quality,
                    "inactive_max_abs_max": 1e-6,
                    "layernorm_inactive_guard_max_error": invalid_guard,
                }
                with self.assertRaises(validator.ValidationError):
                    validator.validate_instance(
                        invalid_quality,
                        quality_schema,
                        self.schema,
                    )


if __name__ == "__main__":
    unittest.main()
