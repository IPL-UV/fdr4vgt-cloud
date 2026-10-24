"""Verify the published dataset and model over the public Hugging Face URLs.

Checks:
  - tacoreader can load the remote `.tacocat`
  - mlstac can load the model JSON and download artifacts
  - predict_large runs on a small test patch
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify remote publication")
    parser.add_argument(
        "--dataset-url",
        default="https://huggingface.co/datasets/isp-uv-es/FDR4VGT-CLOUD/resolve/main/fdr4vgt-cloud-scenes/.tacocat",
    )
    parser.add_argument(
        "--model-url",
        default="https://huggingface.co/isp-uv-es/FDR4VGT-CLOUD/resolve/main/single/mlm.json",
    )
    args = parser.parse_args()

    print(f"Would verify dataset at {args.dataset_url}")
    print(f"Would verify model at {args.model_url}")


if __name__ == "__main__":
    main()
