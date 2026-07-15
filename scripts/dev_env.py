"""Synchronize and diagnose the cross-platform quant_box development environment."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PYTHON_VERSION = (3, 11)
PYTHON_STAMP = ".quant_box_requirements.sha256"
NODE_STAMP = ".quant_box_package_lock.sha256"
BUILD_STAMP = ".quant_box_build.sha256"
LOCK_PATTERN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s;]+)$")
VERSION_PATTERN = re.compile(r"v?(\d+)\.(\d+)(?:\.(\d+))?")


@dataclass(frozen=True)
class CheckResult:
    """One actionable environment diagnostic."""

    key: str
    status: str
    message: str


def normalize_package_name(value: str) -> str:
    """Normalize a Python distribution name using PEP 503 rules."""
    return re.sub(r"[-_.]+", "-", value).lower()


def parse_locked_requirements(text: str) -> dict[str, str]:
    """Return exact direct dependencies from the project lock file."""
    locked: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = LOCK_PATTERN.fullmatch(line)
        if match:
            locked[normalize_package_name(match.group(1))] = match.group(2)
    return locked


def compare_locked_packages(
    locked: Mapping[str, str], installed: Mapping[str, str]
) -> tuple[list[str], list[str]]:
    """Return missing and version-mismatched direct dependencies."""
    normalized = {normalize_package_name(name): version for name, version in installed.items()}
    missing: list[str] = []
    mismatched: list[str] = []
    for name, required in sorted(locked.items()):
        observed = normalized.get(normalize_package_name(name))
        if observed is None:
            missing.append(f"{name}=={required}")
        elif observed != required:
            mismatched.append(f"{name}: installed={observed}, required={required}")
    return missing, mismatched


def parse_version(value: str) -> tuple[int, int, int] | None:
    """Parse a Python/Node-style version string."""
    match = VERSION_PATTERN.search(value.strip())
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def node_version_supported(version: tuple[int, int, int]) -> bool:
    """Match Vite 8's supported Node version ranges."""
    major, minor, _patch = version
    if major == 20:
        return minor >= 19
    if major == 22:
        return minor >= 12
    return major > 22


def venv_python(root: Path, platform_name: str | None = None) -> Path:
    """Resolve the virtual-environment Python executable for an OS family."""
    platform_name = platform_name or os.name
    if platform_name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def sha256_file(path: Path) -> str:
    """Hash one file without loading it wholly into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(root: Path, paths: Iterable[Path]) -> str:
    """Hash relative paths and contents so renames also invalidate a stamp."""
    digest = hashlib.sha256()
    unique_paths = sorted({path.resolve() for path in paths}, key=lambda path: str(path).lower())
    for path in unique_paths:
        try:
            relative = path.relative_to(root.resolve()).as_posix()
        except ValueError:
            relative = str(path)
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        if not path.is_file():
            digest.update(b"<missing>")
        else:
            digest.update(sha256_file(path).encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def frontend_build_fingerprint(root: Path) -> str:
    """Hash every source/config input used by the production frontend build."""
    web = root / "web"
    paths = [
        web / "package.json",
        web / "package-lock.json",
        web / "index.html",
        web / "tsconfig.json",
        web / "vite.config.ts",
    ]
    source = web / "src"
    if source.exists():
        paths.extend(path for path in source.rglob("*") if path.is_file())
    return sha256_paths(root, paths)


def read_stamp(path: Path) -> str | None:
    """Read a local environment stamp, returning None for missing/empty files."""
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def write_stamp(path: Path, value: str) -> None:
    """Write an ignored environment stamp after a successful operation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value + "\n", encoding="utf-8")


def stamp_status(path: Path, expected: str) -> str:
    """Classify an ignored stamp as missing, current, or stale."""
    observed = read_stamp(path)
    if observed is None:
        return "missing"
    return "current" if observed == expected else "stale"


def _run(
    command: Sequence[str | Path],
    *,
    cwd: Path,
    capture: bool = False,
    check: bool = False,
    timeout_seconds: int | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [str(item) for item in command]
    if not capture:
        print(f"RUN {' '.join(args)}", flush=True)
    return subprocess.run(
        args,
        cwd=cwd,
        text=True,
        capture_output=capture,
        check=check,
        timeout=timeout_seconds,
    )


def _command(name: str) -> str | None:
    candidates = [name]
    if os.name == "nt" and not name.lower().endswith(".cmd"):
        candidates.insert(0, f"{name}.cmd")
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return None


def _installed_packages(python: Path, root: Path) -> dict[str, str] | None:
    result = _run([python, "-m", "pip", "list", "--format=json"], cwd=root, capture=True)
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
        return {normalize_package_name(row["name"]): str(row["version"]) for row in payload}
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _python_version(python: Path, root: Path) -> tuple[int, int, int] | None:
    result = _run(
        [python, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        cwd=root,
        capture=True,
    )
    if result.returncode != 0:
        return None
    return parse_version(result.stdout)


def _python_checks(root: Path, python: Path | None = None) -> list[CheckResult]:
    results: list[CheckResult] = []
    uses_project_venv = python is None
    python = python or venv_python(root)
    if not python.is_file():
        return [
            CheckResult(
                "python.venv",
                "fail",
                f"Virtual environment is missing: {python}. Run: python scripts/dev_env.py sync",
            )
        ]

    version = _python_version(python, root)
    if version is None:
        results.append(CheckResult("python.version", "fail", f"Cannot run {python}. Recreate the environment with sync."))
    elif version[:2] != PYTHON_VERSION:
        results.append(
            CheckResult(
                "python.version",
                "fail",
                f"Python {version[0]}.{version[1]} is unsupported; expected {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}.",
            )
        )
    else:
        results.append(CheckResult("python.version", "pass", f"Python {version[0]}.{version[1]}.{version[2]}"))

    lock_path = root / "requirements-lock.txt"
    if not lock_path.is_file():
        results.append(CheckResult("python.lock", "fail", f"Missing lock file: {lock_path}"))
        return results
    locked = parse_locked_requirements(lock_path.read_text(encoding="utf-8"))
    installed = _installed_packages(python, root)
    if installed is None:
        results.append(CheckResult("python.packages", "fail", "Unable to inspect installed Python packages."))
    else:
        missing, mismatched = compare_locked_packages(locked, installed)
        if missing or mismatched:
            details = "; ".join([*(f"missing {item}" for item in missing), *mismatched])
            results.append(
                CheckResult(
                    "python.packages",
                    "fail",
                    f"Locked direct dependencies are not synchronized: {details}. Run sync.",
                )
            )
        else:
            results.append(CheckResult("python.packages", "pass", f"{len(locked)} locked direct dependencies match."))

    import_result = _run([python, "-c", "import fastapi, uvicorn"], cwd=root, capture=True)
    if import_result.returncode != 0:
        results.append(CheckResult("python.runtime", "fail", "FastAPI/Uvicorn imports failed. Run sync."))
    else:
        results.append(CheckResult("python.runtime", "pass", "FastAPI and Uvicorn imports succeed."))

    if uses_project_venv:
        expected = sha256_file(lock_path)
        stamp = stamp_status(root / ".venv" / PYTHON_STAMP, expected)
        if stamp == "stale":
            results.append(CheckResult("python.stamp", "fail", "requirements-lock.txt changed after the last sync."))
        elif stamp == "missing":
            results.append(CheckResult("python.stamp", "warn", "Python sync stamp is missing; run sync to establish it."))
        else:
            results.append(CheckResult("python.stamp", "pass", "Python dependency stamp is current."))
    return results


def _frontend_checks(root: Path) -> list[CheckResult]:
    results: list[CheckResult] = []
    web = root / "web"
    npm = _command("npm")
    if npm is None:
        return [CheckResult("node.npm", "fail", "npm is unavailable. Install Node 22 LTS and run sync.")]
    node = _command("node")
    if node is None:
        return [CheckResult("node.version", "fail", "Node.js is unavailable. Install Node 22 LTS and run sync.")]

    node_result = _run([node, "--version"], cwd=root, capture=True)
    version = parse_version(node_result.stdout) if node_result.returncode == 0 else None
    if version is None or not node_version_supported(version):
        shown = node_result.stdout.strip() or "unknown"
        results.append(CheckResult("node.version", "fail", f"Unsupported Node version {shown}; use Node 22 LTS."))
    else:
        results.append(CheckResult("node.version", "pass", f"Node {version[0]}.{version[1]}.{version[2]}"))

    lock_path = web / "package-lock.json"
    if not lock_path.is_file():
        results.append(CheckResult("node.lock", "fail", f"Missing frontend lock file: {lock_path}"))
        return results

    node_modules = web / "node_modules"
    if not node_modules.is_dir():
        results.append(CheckResult("node.packages", "fail", "web/node_modules is missing. Run sync."))
    else:
        npm_list = _run([npm, "ls", "--depth=0", "--json"], cwd=web, capture=True)
        if npm_list.returncode != 0:
            results.append(CheckResult("node.packages", "fail", "Frontend dependencies do not satisfy package-lock.json. Run sync."))
        else:
            results.append(CheckResult("node.packages", "pass", "Frontend dependencies are installed."))

    expected_lock = sha256_file(lock_path)
    node_stamp = stamp_status(node_modules / NODE_STAMP, expected_lock)
    if node_stamp == "stale":
        results.append(CheckResult("node.stamp", "fail", "web/package-lock.json changed after the last sync."))
    elif node_stamp == "missing":
        results.append(CheckResult("node.stamp", "warn", "Frontend sync stamp is missing; run sync to establish it."))
    else:
        results.append(CheckResult("node.stamp", "pass", "Frontend dependency stamp is current."))

    return results


def _build_checks(root: Path) -> list[CheckResult]:
    """Validate production frontend artifacts without requiring Node at runtime."""
    web = root / "web"
    index = web / "dist" / "index.html"
    if not index.is_file():
        return [CheckResult("web.build", "fail", "web/dist is missing. Run: python scripts/dev_env.py sync --build-web")]
    expected_build = frontend_build_fingerprint(root)
    build_stamp = stamp_status(web / "dist" / BUILD_STAMP, expected_build)
    if build_stamp != "current":
        return [
            CheckResult(
                "web.build",
                "fail",
                "Production frontend is stale or untracked. Run: python scripts/dev_env.py sync --build-web",
            )
        ]
    return [CheckResult("web.build", "pass", "Production frontend build is current.")]


def doctor_environment(
    root: Path,
    *,
    backend: bool = True,
    frontend: bool = True,
    require_build: bool = False,
    python: Path | None = None,
) -> list[CheckResult]:
    """Run read-only environment diagnostics."""
    results: list[CheckResult] = []
    if backend:
        results.extend(_python_checks(root, python=python))
    if frontend:
        results.extend(_frontend_checks(root))
    if require_build:
        results.extend(_build_checks(root))
    return results


def _print_checks(results: Sequence[CheckResult]) -> None:
    for result in results:
        print(f"{result.status.upper():4} [{result.key}] {result.message}")
    counts = {status: sum(result.status == status for result in results) for status in ("pass", "warn", "fail")}
    print(f"SUMMARY pass={counts['pass']} warn={counts['warn']} fail={counts['fail']}")


def _sync_python(root: Path, force: bool) -> None:
    if sys.version_info[:2] != PYTHON_VERSION:
        raise RuntimeError(
            f"Bootstrap Python must be {PYTHON_VERSION[0]}.{PYTHON_VERSION[1]}; "
            f"got {sys.version_info[0]}.{sys.version_info[1]}."
        )
    python = venv_python(root)
    if not python.is_file():
        print(f"Creating virtual environment: {root / '.venv'}")
        _run([sys.executable, "-m", "venv", root / ".venv"], cwd=root, check=True)

    lock_path = root / "requirements-lock.txt"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Python lock file not found: {lock_path}")
    expected = sha256_file(lock_path)
    stamp_path = root / ".venv" / PYTHON_STAMP
    packages = _installed_packages(python, root)
    locked = parse_locked_requirements(lock_path.read_text(encoding="utf-8"))
    missing, mismatched = compare_locked_packages(locked, packages or {})
    needs_install = force or stamp_status(stamp_path, expected) != "current" or bool(missing or mismatched)
    if needs_install:
        _run([python, "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "60", "--retries", "10", "pip<24.1"], cwd=root, check=True)
        _run(
            [python, "-m", "pip", "install", "--disable-pip-version-check", "--timeout", "60", "--retries", "10", "-r", lock_path],
            cwd=root,
            check=True,
        )
        write_stamp(stamp_path, expected)
    else:
        print("Python dependencies are already synchronized.")


def _sync_frontend(root: Path, force: bool, build_web: bool, with_playwright: bool) -> None:
    web = root / "web"
    npm = _command("npm")
    node = _command("node")
    if npm is None or node is None:
        raise RuntimeError("Node/npm is unavailable. Install Node 22 LTS and retry.")
    node_result = _run([node, "--version"], cwd=root, capture=True)
    version = parse_version(node_result.stdout)
    if version is None or not node_version_supported(version):
        raise RuntimeError(f"Unsupported Node version {node_result.stdout.strip() or 'unknown'}; use Node 22 LTS.")

    lock_path = web / "package-lock.json"
    if not lock_path.is_file():
        raise FileNotFoundError(f"Frontend lock file not found: {lock_path}")
    expected = sha256_file(lock_path)
    stamp_path = web / "node_modules" / NODE_STAMP
    npm_list = _run([npm, "ls", "--depth=0", "--json"], cwd=web, capture=True)
    dependencies_ready = npm_list.returncode == 0 and (web / "node_modules").is_dir()
    if force or stamp_status(stamp_path, expected) != "current" or not dependencies_ready:
        _run([npm, "ci"], cwd=web, check=True)
        write_stamp(stamp_path, expected)
    else:
        print("Frontend dependencies are already synchronized.")

    if build_web:
        _run([npm, "run", "build"], cwd=web, check=True)
        write_stamp(web / "dist" / BUILD_STAMP, frontend_build_fingerprint(root))
    if with_playwright:
        npx = _command("npx")
        if npx is None:
            raise RuntimeError("npx is unavailable. Reinstall Node 22 LTS and retry.")
        _run([npx, "playwright", "install", "chromium"], cwd=web, check=True, timeout_seconds=300)


def sync_environment(root: Path, *, force: bool, build_web: bool, with_playwright: bool) -> None:
    """Synchronize Python and frontend dependencies, then verify the result."""
    _sync_python(root, force=force)
    _sync_frontend(root, force=force, build_web=build_web, with_playwright=with_playwright)
    results = doctor_environment(root, require_build=build_web)
    _print_checks(results)
    failures = [result for result in results if result.status == "fail"]
    if failures:
        raise RuntimeError(f"Environment synchronization finished with {len(failures)} failed checks.")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synchronize and diagnose the quant_box development environment.")
    parser.add_argument("--root", type=Path, default=ROOT, help="Project root; defaults to the repository containing this script.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sync = subparsers.add_parser("sync", help="Idempotently synchronize Python and frontend dependencies.")
    sync.add_argument("--force", action="store_true", help="Reinstall dependencies even when lock stamps are current.")
    sync.add_argument("--build-web", action="store_true", help="Build web/dist and record its source fingerprint.")
    sync.add_argument("--with-playwright", action="store_true", help="Install the Playwright Chromium browser.")

    subparsers.add_parser("stamp-web-build", help="Record the current frontend source fingerprint after a successful build.")

    doctor = subparsers.add_parser("doctor", help="Run read-only environment diagnostics.")
    doctor.add_argument("--strict", action="store_true", help="Exit non-zero when any check fails.")
    doctor.add_argument(
        "--current-python",
        action="store_true",
        help="Validate the current interpreter instead of the project .venv (intended for CI).",
    )
    scope = doctor.add_mutually_exclusive_group()
    scope.add_argument("--backend-only", action="store_true", help="Check only the Python environment.")
    scope.add_argument("--frontend-only", action="store_true", help="Check only Node and frontend dependencies.")
    scope.add_argument(
        "--runtime-only",
        action="store_true",
        help="Check the Python runtime and production frontend build without requiring Node/npm.",
    )
    doctor.add_argument("--require-web-dist", action="store_true", help="Require a current production frontend build.")
    return parser


def main() -> None:
    args = _parser().parse_args()
    root = args.root.resolve()
    try:
        if args.command == "sync":
            sync_environment(root, force=args.force, build_web=args.build_web, with_playwright=args.with_playwright)
            return
        if args.command == "stamp-web-build":
            index = root / "web" / "dist" / "index.html"
            if not index.is_file():
                raise FileNotFoundError(f"Production frontend entry is missing: {index}")
            write_stamp(root / "web" / "dist" / BUILD_STAMP, frontend_build_fingerprint(root))
            print("Recorded production frontend source fingerprint.")
            return
        backend = not args.frontend_only
        frontend = not args.backend_only and not args.runtime_only
        require_build = args.require_web_dist or args.runtime_only
        python = Path(sys.executable) if args.current_python else None
        results = doctor_environment(root, backend=backend, frontend=frontend, require_build=require_build, python=python)
        _print_checks(results)
        if args.strict and any(result.status == "fail" for result in results):
            raise SystemExit(1)
    except (FileNotFoundError, RuntimeError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
