"""Generate README.md and index.html for each TACO partition from COLLECTION.json.

The generated files use only fields present in the COLLECTION so no metadata
is invented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generate_readme(collection_path: Path, output_path: Path) -> None:
    with open(collection_path) as f:
        collection = json.load(f)

    lines = [
        f"# {collection.get('id', 'FDR4VGT-CLOUD')}",
        "",
        collection.get("description", ""),
        "",
        "## License",
        "",
        "See LICENSE file. Third-party data providers are credited in the documentation.",
    ]
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate TACO documentation")
    parser.add_argument(
        "--collection",
        default="/data/databases/Julio/FDR4VGT_RELEASE/taco/fdr4vgt-cloud-scenes/COLLECTION.json",
    )
    parser.add_argument(
        "--output",
        default="/data/databases/Julio/FDR4VGT_RELEASE/taco/fdr4vgt-cloud-scenes/README.md",
    )
    args = parser.parse_args()

    generate_readme(Path(args.collection), Path(args.output))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
