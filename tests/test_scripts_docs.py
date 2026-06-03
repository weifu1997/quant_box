from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ScriptsDocsTests(unittest.TestCase):
    def test_quick_signal_bat_and_legacy_wrapper_are_documented(self) -> None:
        quick = (ROOT / "02_快速更新并生成信号.bat").read_text(encoding="utf-8")
        legacy = (ROOT / "02_自动调参并生成信号.bat").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("--skip-optimize --skip-backtest", quick)
        self.assertIn("CHUNK_SIZE=15", quick)
        self.assertIn("SLEEP_SECONDS=10", quick)
        self.assertIn("02_快速更新并生成信号.bat", legacy)
        self.assertIn("02_快速更新并生成信号.bat", readme)
        self.assertIn("data/raw/failed_fetches.csv", readme)


if __name__ == "__main__":
    unittest.main()
