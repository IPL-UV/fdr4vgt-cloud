"""Dry-run upload of the model artifacts to Hugging Face.

The model repository contains 24 `.pt2` files, 29 JSON files, `load.py`,
README.md and example data. This script does not upload anything unless
``--dry-run=False`` is passed.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload model (dry-run)")
    parser.add_argument("--local-dir", default="hf_model")
    parser.add_argument("--repo-id", default="isp-uv-es/FDR4VGT-CLOUD")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Would upload {args.local_dir} to {args.repo_id}")
    print("Set --dry-run=False to perform the upload after G0 approval")


if __name__ == "__main__":
    main()
