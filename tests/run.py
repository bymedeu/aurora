"""Run every Aurora unittest module from the repository checkout."""

from pathlib import Path
import sys
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
TEST_ROOT = PROJECT_ROOT / "tests"


def main() -> int:
    sys.path.insert(0, str(SOURCE_ROOT))
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(TEST_ROOT),
        pattern="test_*.py",
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
