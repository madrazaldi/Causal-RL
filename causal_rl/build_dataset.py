from __future__ import annotations

import argparse

from .config import DECISION_LOG_PATH, METADATA_PATH, RAW_DATA_PATH
from .dataset import build_and_save_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the offline decision log dataset.")
    parser.add_argument("--raw-path", default=str(RAW_DATA_PATH))
    args = parser.parse_args()

    bundle = build_and_save_dataset(args.raw_path)
    print(f"Built decision log with {len(bundle.df):,} rows")
    print(f"Saved decision log to {DECISION_LOG_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
