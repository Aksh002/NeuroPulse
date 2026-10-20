"""Stage-1 configuration diagnostic; does not train or download data."""

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from yaml import YAMLError

from neuropulse.config import load_data_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=version("neuropulse"))
    parser.add_argument("--config", type=Path, required=True, help="FD001 YAML configuration")
    args = parser.parse_args()
    try:
        config = load_data_config(args.config)
    except (OSError, ValueError, YAMLError) as exc:
        parser.exit(2, f"Configuration error: {exc}\n")
    print(
        json.dumps(
            {
                "stage": "phase_1_foundation",
                "configuration_valid": True,
                "models_available": False,
                "config": config.model_dump(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
