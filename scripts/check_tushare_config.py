from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_fetcher import DAILY_FIELDS, TushareHttpClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    client = TushareHttpClient.from_config()
    preview = client.redacted_request_preview(
        api_name="daily",
        params={"ts_code": "000001.SZ", "start_date": "20240102", "end_date": "20240105"},
        fields=DAILY_FIELDS,
    )
    logger.info("Request preview:\n%s", json.dumps(preview, ensure_ascii=False, indent=2))
    logger.info("This command does not send a network request and does not print your token.")


if __name__ == "__main__":
    main()
