"""Dry-run upload of the TACO dataset to Hugging Face.

Requires ``HF_TOKEN`` to be set in the environment. No credentials are
stored in this file.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload dataset (dry-run)")
    parser.add_argument("--local-dir", default="/data/databases/Julio/FDR4VGT_RELEASE/taco")
    parser.add_argument("--repo-id", default="isp-uv-es/FDR4VGT-CLOUD")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Would upload {args.local_dir} to {args.repo_id}")
    print("Set --dry-run=False to perform the upload after G0 approval")


if __name__ == "__main__":
    main()
