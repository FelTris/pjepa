"""Run portable LEMON self-supervised training."""

import argparse
import json
from pjepa.training.lemon import train
from pjepa.utils.runtime import load_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pretrain P-JEPA on sharded LEMON PL-Stitch features."
    )
    parser.add_argument("--config", required=True, help="Training JSON configuration.")
    args = parser.parse_args()
    config = load_json(args.config)
    if str(config.get("action", "train")).lower() != "train":
        raise ValueError("training.lemon_ssl only supports action='train'.")
    result = train(config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
