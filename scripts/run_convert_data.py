"""模块说明：提供 run_convert_data 命令行入口。"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_converter import convert_to_qlib_format

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """函数说明：解析命令行参数并执行主流程。"""
    parser = argparse.ArgumentParser(description="Convert raw stock CSV files to Qlib data and local price panels.")
    parser.parse_args()

    result = convert_to_qlib_format()
    logger.info("Conversion finished.")
    for key, value in result.items():
        logger.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
