"""模块说明：覆盖 test_scripts_docs 相关行为的测试用例。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
BAT_FILES = sorted(ROOT.glob("*.bat"))
MOJIBAKE_MARKERS = ("鍙傛暟", "蹇", "鑷", "璋", "鐢", "淇")


class ScriptsDocsTests(unittest.TestCase):
    """类说明：组织 ScriptsDocsTests 测试用例。"""
    def test_quick_signal_bat_is_documented(self) -> None:
        """函数说明：验证 test_quick_signal_bat_is_documented 覆盖的行为场景。"""
        quick = (ROOT / "02_快速更新并生成信号.bat").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("--skip-optimize --skip-backtest", quick)
        self.assertIn("CHUNK_SIZE=300", quick)
        self.assertIn("SLEEP_SECONDS=0", quick)
        self.assertIn("chcp 65001 >nul", quick)
        self.assertIn("02_快速更新并生成信号.bat", readme)
        self.assertNotIn("02_自动调参并生成信号.bat", readme)
        self.assertIn("data/raw/failed_fetches.csv", readme)

    def test_all_bat_files_are_documented(self) -> None:
        """函数说明：验证 test_all_bat_files_are_documented 覆盖的行为场景。"""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        for path in BAT_FILES:
            self.assertIn(path.name, readme)

        self.assertIn("不用把所有 `.bat` 顺序执行", readme)
        self.assertIn("04 -> 06 -> 07 -> 08 -> 09 -> 10", readme)
        self.assertIn("不含 walk-forward 参数优化", readme)

    def test_run_all_bat_is_explicit_about_optimization(self) -> None:
        """函数说明：验证 test_run_all_bat_is_explicit_about_optimization 覆盖的行为场景。"""
        run_all = (ROOT / "run_all.bat").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("scripts\\run_auto_signal.py", run_all)
        self.assertIn("--skip-optimize", run_all)
        self.assertNotIn("--skip-backtest", run_all)
        self.assertIn("without walk-forward optimization", run_all)
        self.assertIn("refresh missing and stale raw data", run_all)
        self.assertIn("check data health", run_all)
        self.assertIn("latest candidate signal", run_all)
        self.assertNotIn("11_旧版全流程_补数据到信号.bat", readme)
        self.assertIn("刷新缺失和过期股票", readme)
        self.assertIn("data health", readme)

    def test_supervised_auto_signal_entrypoint_is_documented(self) -> None:
        """函数说明：验证 test_supervised_auto_signal_entrypoint_is_documented 覆盖的行为场景。"""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("scripts\\run_auto_signal_supervised.py start", readme)
        self.assertIn("scripts\\run_auto_signal_supervised.py status", readme)
        self.assertIn("outputs/logs/auto_signal_*.log", readme)

        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_auto_signal_supervised.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Start and inspect long-running auto signal jobs", proc.stdout)

    def test_dashboard_bat_starts_backend_and_frontend(self) -> None:
        """函数说明：验证 Web 仪表盘一键启动脚本覆盖前后端入口。"""
        dashboard = (ROOT / "15_启动Web仪表盘.bat").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")

        self.assertIn("scripts\\run_dashboard.py", dashboard)
        self.assertIn("npm run dev", dashboard)
        self.assertIn("http://127.0.0.1:8000/api/health", dashboard)
        self.assertIn("http://127.0.0.1:5173", dashboard)
        self.assertIn("15_启动Web仪表盘.bat", readme)

    def test_convert_data_help_does_not_start_conversion(self) -> None:
        """函数说明：验证 test_convert_data_help_does_not_start_conversion 覆盖的行为场景。"""
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "run_convert_data.py"), "--help"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
        )

        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("Convert raw stock CSV files", proc.stdout)

    def test_bat_files_are_utf8_and_use_crlf_line_endings(self) -> None:
        """函数说明：验证 test_bat_files_are_utf8_and_use_crlf_line_endings 覆盖的行为场景。"""
        for path in BAT_FILES:
            data = path.read_bytes()
            data.decode("utf-8")
            self.assertGreater(data.count(b"\r\n"), 0)
            self.assertEqual(data.count(b"\n"), data.count(b"\r\n"))

    def test_chinese_bat_output_switches_to_utf8_before_echo(self) -> None:
        """函数说明：验证 test_chinese_bat_output_switches_to_utf8_before_echo 覆盖的行为场景。"""
        for path in BAT_FILES:
            text = path.read_text(encoding="utf-8")
            if not any(ord(char) > 127 for char in text):
                continue

            lines = [line.strip() for line in text.splitlines() if line.strip()]
            self.assertGreaterEqual(len(lines), 2)
            self.assertEqual("@echo off", lines[0])
            self.assertIn("chcp 65001 >nul", lines[:3])

    def test_scripts_and_docs_do_not_contain_mojibake_markers(self) -> None:
        """函数说明：验证 test_scripts_and_docs_do_not_contain_mojibake_markers 覆盖的行为场景。"""
        files = [*BAT_FILES, ROOT / "README.md"]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in files)

        for marker in MOJIBAKE_MARKERS:
            self.assertNotIn(marker, combined)


if __name__ == "__main__":
    unittest.main()
