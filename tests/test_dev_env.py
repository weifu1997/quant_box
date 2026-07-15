"""Tests for the cross-platform development environment contract."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.dev_env import (
    BUILD_STAMP,
    _build_checks,
    compare_locked_packages,
    frontend_build_fingerprint,
    node_version_supported,
    normalize_package_name,
    parse_locked_requirements,
    parse_version,
    read_stamp,
    sha256_paths,
    stamp_status,
    venv_python,
    write_stamp,
)


class DevEnvironmentTests(unittest.TestCase):
    def test_parse_locked_requirements_normalizes_direct_dependencies(self) -> None:
        locked = parse_locked_requirements("FastAPI==0.136.3\na_trade_calendar==2028.4.11.1\nnumpy>=2\n")

        self.assertEqual(locked, {"fastapi": "0.136.3", "a-trade-calendar": "2028.4.11.1"})
        self.assertEqual(normalize_package_name("a_trade.calendar"), "a-trade-calendar")

    def test_compare_locked_packages_reports_missing_and_mismatch(self) -> None:
        missing, mismatched = compare_locked_packages(
            {"fastapi": "1.0", "uvicorn": "2.0", "pyyaml": "3.0"},
            {"FastAPI": "1.0", "uvicorn": "1.9"},
        )

        self.assertEqual(missing, ["pyyaml==3.0"])
        self.assertEqual(mismatched, ["uvicorn: installed=1.9, required=2.0"])

    def test_parse_and_validate_supported_node_versions(self) -> None:
        self.assertEqual(parse_version("v22.12.1"), (22, 12, 1))
        self.assertTrue(node_version_supported((20, 19, 0)))
        self.assertTrue(node_version_supported((22, 12, 0)))
        self.assertTrue(node_version_supported((24, 0, 0)))
        self.assertFalse(node_version_supported((20, 18, 0)))
        self.assertFalse(node_version_supported((21, 7, 0)))
        self.assertFalse(node_version_supported((22, 11, 0)))

    def test_venv_python_uses_platform_specific_layout(self) -> None:
        root = Path("project")

        self.assertEqual(venv_python(root, "nt"), root / ".venv" / "Scripts" / "python.exe")
        self.assertEqual(venv_python(root, "posix"), root / ".venv" / "bin" / "python")

    def test_path_hash_changes_with_content_and_filename(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "first.txt"
            first.write_text("one", encoding="utf-8")
            initial = sha256_paths(root, [first])
            first.write_text("two", encoding="utf-8")
            changed = sha256_paths(root, [first])
            renamed = root / "renamed.txt"
            first.rename(renamed)

            self.assertNotEqual(initial, changed)
            self.assertNotEqual(changed, sha256_paths(root, [renamed]))

    def test_frontend_fingerprint_tracks_source_but_not_e2e(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            web = root / "web"
            source = web / "src"
            e2e = web / "e2e"
            source.mkdir(parents=True)
            e2e.mkdir()
            for name in ("package.json", "package-lock.json", "index.html", "tsconfig.json", "vite.config.ts"):
                (web / name).write_text(name, encoding="utf-8")
            app = source / "App.tsx"
            app.write_text("first", encoding="utf-8")
            initial = frontend_build_fingerprint(root)
            (e2e / "workspace.spec.ts").write_text("changed test", encoding="utf-8")
            self.assertEqual(initial, frontend_build_fingerprint(root))
            app.write_text("second", encoding="utf-8")
            self.assertNotEqual(initial, frontend_build_fingerprint(root))

    def test_stamp_round_trip_and_status(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "dist" / BUILD_STAMP

            self.assertIsNone(read_stamp(path))
            self.assertEqual(stamp_status(path, "expected"), "missing")
            write_stamp(path, "old")
            self.assertEqual(stamp_status(path, "expected"), "stale")
            write_stamp(path, "expected")
            self.assertEqual(read_stamp(path), "expected")
            self.assertEqual(stamp_status(path, "expected"), "current")

    def test_build_check_rejects_missing_and_stale_artifacts(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            web = root / "web"
            source = web / "src"
            source.mkdir(parents=True)
            for name in ("package.json", "package-lock.json", "index.html", "tsconfig.json", "vite.config.ts"):
                (web / name).write_text(name, encoding="utf-8")
            app = source / "App.tsx"
            app.write_text("first", encoding="utf-8")

            self.assertEqual(_build_checks(root)[0].status, "fail")
            (web / "dist").mkdir()
            (web / "dist" / "index.html").write_text("built", encoding="utf-8")
            write_stamp(web / "dist" / BUILD_STAMP, frontend_build_fingerprint(root))
            self.assertEqual(_build_checks(root)[0].status, "pass")
            app.write_text("changed", encoding="utf-8")
            self.assertEqual(_build_checks(root)[0].status, "fail")


if __name__ == "__main__":
    unittest.main()
