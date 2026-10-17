"""Discover Aurora's tests, with optional component selection."""

import argparse
from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_ROOT = PROJECT_ROOT / "tests"


def main() -> int:
    groups = sorted(
        path.name
        for path in TEST_ROOT.iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suites", nargs="*", help="Component folders; default: all")
    parser.add_argument("--list", action="store_true", help="List available suites")
    parser.add_argument("-p", "--pattern", default="test_*.py", help="Test filename pattern")
    parser.add_argument("--failfast", action="store_true", help="Stop after the first failure")
    args = parser.parse_args()
    unknown = sorted(set(args.suites) - set(groups))
    if unknown:
        parser.error(f"Unknown suites: {', '.join(unknown)}. Available: {', '.join(groups)}")
    if args.list:
        print("\n".join(groups))
        return 0

    sys.path.insert(0, str(SOURCE_ROOT))
    loader = unittest.TestLoader()
    if args.suites:
        selected = list(dict.fromkeys(args.suites))
        suite = unittest.TestSuite()
        for group in selected:
            suite.addTests(loader.discover(
                start_dir=str(TEST_ROOT / group),
                pattern=args.pattern,
                top_level_dir=str(PROJECT_ROOT),
            ))
    else:
        selected = groups
        suite = loader.discover(
            start_dir=str(TEST_ROOT),
            pattern=args.pattern,
            top_level_dir=str(PROJECT_ROOT),
        )
    if suite.countTestCases() == 0:
        parser.error(f"No tests matched pattern {args.pattern!r}")
    print(f"Suites: {', '.join(selected)} ({suite.countTestCases()} tests)", flush=True)
    result = unittest.TextTestRunner(verbosity=2, failfast=args.failfast).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
