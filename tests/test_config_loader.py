from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from src.config_loader import _expand_env_values


class ConfigLoaderTests(unittest.TestCase):
    def test_expand_env_values_supports_whole_value_and_embedded_values(self) -> None:
        value = {
            "token": "${TUSHARE_TOKEN}",
            "url": "http://${TUSHARE_HOST}:${TUSHARE_PORT}/api",
            "items": ["${TUSHARE_HOST}", "plain"],
        }

        with patch.dict(
            os.environ,
            {"TUSHARE_TOKEN": "secret", "TUSHARE_HOST": "127.0.0.1", "TUSHARE_PORT": "8020"},
            clear=False,
        ):
            expanded = _expand_env_values(value)

        self.assertEqual(expanded["token"], "secret")
        self.assertEqual(expanded["url"], "http://127.0.0.1:8020/api")
        self.assertEqual(expanded["items"], ["127.0.0.1", "plain"])

    def test_expand_env_values_missing_variable_becomes_empty_string(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_expand_env_values("http://${MISSING_HOST}:8020/"), "http://:8020/")


if __name__ == "__main__":
    unittest.main()
