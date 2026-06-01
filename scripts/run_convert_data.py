from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_converter import convert_to_qlib_format


def main() -> None:
    result = convert_to_qlib_format()
    print("Conversion finished.")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
