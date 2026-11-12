"""Upload the model artifacts to Hugging Face.

The upload takes the `.pt2` files and the metadata from a local staging
directory and the shared `load.py` from `notebooks/fdr4vgt_model/`, so the
published helper is always the version tracked in this repository.

Usage:

    python pipeline/15_upload_model.py --staging hf_model --dry-run
    python pipeline/15_upload_model.py --staging hf_model --yes
"""

from __future__ import annotations

import argparse
import pathlib

from huggingface_hub import HfApi

REPO = "isp-uv-es/FDR4VGT-CLOUD"
LOAD = pathlib.Path("notebooks/fdr4vgt_model/load.py")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload artifacts to Hugging Face")
    parser.add_argument("--staging", default="hf_model")
    parser.add_argument("--repo-id", default=REPO)
    parser.add_argument("--yes", action="store_true", help="Perform the upload")
    args = parser.parse_args()

    staging = pathlib.Path(args.staging)
    files = sorted(p for p in staging.rglob("*") if p.suffix in {".pt2", ".json", ".safetensor"})
    if not LOAD.exists():
        raise FileNotFoundError(LOAD)

    for path in files:
        print(f"{path.relative_to(staging)}")
    print("load.py -> single/load.py and ensemble/load.py")

    if not args.yes:
        print(f"{len(files) + 2} files ready. Run again with --yes to upload.")
        return

    api = HfApi()
    for path in files:
        api.upload_file(
            path_or_fileobj=str(path),
            path_in_repo=str(path.relative_to(staging)),
            repo_id=args.repo_id,
            commit_message=f"update {path.name}",
        )
    for prefix in ("single", "ensemble"):
        api.upload_file(
            path_or_fileobj=str(LOAD),
            path_in_repo=f"{prefix}/load.py",
            repo_id=args.repo_id,
            commit_message="update load helper",
        )
    print("upload done")


if __name__ == "__main__":
    main()
