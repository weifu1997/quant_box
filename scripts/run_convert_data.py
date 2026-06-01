from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_converter import convert_to_qlib_format

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    result = convert_to_qlib_format()
    logger.info("Conversion finished.")
    for key, value in result.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
