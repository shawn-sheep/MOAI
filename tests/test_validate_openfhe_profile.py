#!/usr/bin/env python3
"""Fail-closed tests for the frozen OpenFHE profile validator."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_openfhe_profile.py"
BASE_CONFIG = REPO_ROOT / "config" / "paper_compat.json"
FEATURE_CONFIG = REPO_ROOT / "config" / "paper_compat_feature_packed.json"


class OpenFHEProfileValidatorTests(unittest.TestCase):
    def run_validator(self, feature_config: dict[str, object]) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary_directory:
            feature_path = Path(temporary_directory) / "paper_compat_feature_packed.json"
            feature_path.write_text(
                json.dumps(feature_config, ensure_ascii=False),
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
                ],
                check=False,
                capture_output=True,
                text=True,
            )

    def load_feature_config(self) -> dict[str, object]:
        return json.loads(FEATURE_CONFIG.read_text(encoding="utf-8"))

    def test_frozen_feature_profile_passes(self) -> None:
        result = self.run_validator(self.load_feature_config())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('"passed":true', result.stdout)

    def test_validation_status_drift_fails_closed(self) -> None:
        feature_config = self.load_feature_config()
        feature_config["validation_status"] = "five-token encoder gate required"
        result = self.run_validator(feature_config)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(
            "feature-packed paper_compat identity/security contract drifted",
            result.stderr,
        )


if __name__ == "__main__":
    unittest.main()
