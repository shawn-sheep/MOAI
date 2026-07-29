#!/usr/bin/env python3
"""Validate the frozen M3 effective OpenFHE CryptoProfile payload and hash."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "paper_compat.json"
EXPECTED_HASH = "965ded1a76b064c738f55b87b151df60c605d62eaeaad55f6cc753e149f8af59"
EXPECTED_WARNING = (
    "Research reproduction parameters only. Do not claim 128-bit security."
)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    with arguments.config.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if (
        config.get("profile_id") != "paper_compat"
        or config.get("security_claim") != "none"
        or config.get("warning") != EXPECTED_WARNING
    ):
        raise RuntimeError("paper_compat identity/security contract drifted")
    runtime = config["openfhe_translation"]["m3_smoke_effective_runtime"]
    payload = canonical_json(runtime["parameters"])
    calculated = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if (
        runtime["effective_profile_sha256"] != EXPECTED_HASH
        or calculated != EXPECTED_HASH
    ):
        raise RuntimeError(
            "M3 effective CryptoProfile hash drifted: "
            "frozen="
            f"{runtime['effective_profile_sha256']} calculated={calculated}"
        )
    print(
        json.dumps(
            {
                "test": "validate_openfhe_profile",
                "profile": "paper_compat",
                "security_claim": "none",
                "parameter_sha256": calculated,
                "passed": True,
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"validate_openfhe_profile failed: {error}", file=sys.stderr)
        sys.exit(1)
