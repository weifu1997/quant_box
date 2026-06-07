from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adj_factor_metadata import build_adj_factor_metadata, write_adj_factor_metadata
from src.config_loader import load_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description="Build version metadata for raw adj_factor columns.")
    parser.add_argument("--output", help="Override data_governance.adj_factor_meta_file.")
    args = parser.parse_args()

    metadata = build_adj_factor_metadata(config)
    path = write_adj_factor_metadata(metadata, config, path=args.output)
    logger.info("Saved adj-factor metadata to %s", path)
    logger.info(
        "Coverage: %s/%s files, %s symbols, %s to %s, digest=%s",
        metadata.files_with_adj_factor,
        metadata.raw_file_count,
        metadata.symbol_count,
        metadata.start_date,
        metadata.end_date,
        metadata.digest,
    )
    if metadata.issues:
        logger.warning("Issues: %s", ", ".join(metadata.issues[:10]))


if __name__ == "__main__":
    main()
