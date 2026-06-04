from __future__ import annotations

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
BAT_FILES = sorted(ROOT.glob("*.bat"))
MOJIBAKE_MARKERS = ("鍙傛暟", "蹇", "鑷", "璋", "鐢", "淇")


class ScriptsDocsTests(unittest.TestCase):
    def test_quick_signal_bat_and_legacy_wrapper_are_documented(self) -> None:
        quick = (ROOT / "02_快速更新并生成信号.bat").read_text(encoding="utf-8")
        legacy = (ROOT / "02_自动调参并生成信号.bat").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("--skip-optimize --skip-backtest", quick)
        self.assertIn("CHUNK_SIZE=300", quick)
        self.assertIn("SLEEP_SECONDS=0", quick)
        self.assertIn("chcp 65001 >nul", quick)
        self.assertIn("chcp 65001 >nul", legacy)
        self.assertIn("02_快速更新并生成信号.bat", legacy)
        self.assertIn("--skip-optimize --skip-backtest", legacy)
        self.assertIn("02_快速更新并生成信号.bat", readme)
        self.assertIn("直接运行同一套快速流程", readme)
        self.assertIn("data/raw/failed_fetches.csv", readme)

    def test_all_bat_files_are_documented(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for path in BAT_FILES:
            self.assertIn(path.name, readme)

        self.assertIn("不用把所有 `.bat` 顺序执行", readme)
        self.assertIn("04 -> 06 -> 07 -> 08 -> 09 -> 10", readme)
        self.assertIn("不含 walk-forward 参数优化", readme)

    def test_legacy_run_all_bat_is_explicit_about_optimization(self) -> None:
        run_all = (ROOT / "run_all.bat").read_text(encoding="utf-8")
        legacy = (ROOT / "11_旧版全流程_补数据到信号.bat").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("scripts\\run_auto_signal.py", run_all)
        self.assertIn("--skip-optimize", run_all)
        self.assertNotIn("--skip-backtest", run_all)
        self.assertIn("without walk-forward optimization", run_all)
        self.assertIn("refresh missing and stale raw data", run_all)
        self.assertIn("check data health", run_all)
        self.assertIn("latest candidate signal", run_all)
        self.assertIn("without walk-forward optimization", legacy)
        self.assertIn("刷新缺失和过期股票", readme)
        self.assertIn("data health", readme)

    def test_bat_files_are_utf8_and_use_crlf_line_endings(self) -> None:
        for path in BAT_FILES:
            data = path.read_bytes()
            data.decode("utf-8")
            self.assertGreater(data.count(b"\r\n"), 0)
            self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))

    def test_chinese_bat_output_switches_to_utf8_before_echo(self) -> None:
        for path in BAT_FILES:
            text = path.read_text(encoding="utf-8")
            if not any(ord(char) > 127 for char in text):
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual("@echo off", lines[0])
            self.assertIn("chcp 65001 >nul", lines[:3])

    def test_scripts_and_docs_do_not_contain_mojibake_markers(self) -> None:
        files = [*BAT_FILES, ROOT / "README.md"]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

        for marker in MOJIBAKE_MARKERS:
            self.assertNotIn(marker, combined)


if __name__ == "__main__":
    unittest.main()
