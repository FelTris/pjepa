"""Encode a tensor feature sequence using a registered or original checkpoint."""

import argparse
from pathlib import Path
import torch
from pjepa.checkpoints import load_model
from pjepa.inference.encoder import encode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--input",
        required=True,
        help="Torch dict with features [B,T,D], optional valid_mask and segment_lengths.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    model, meta = load_model(args.checkpoint, device=args.device)
    payload = torch.load(args.input, map_location="cpu", weights_only=True)
    features = encode(
        model,
        payload["features"],
        payload.get("valid_mask"),
        segment_lengths=payload.get("segment_lengths"),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"features": features.cpu(), "checkpoint": meta, "valid_mask": payload.get("valid_mask")},
        output,
    )


if __name__ == "__main__":
    main()
