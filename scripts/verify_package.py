"""Health checks for the automl-pipeline-optimizer package."""

from __future__ import annotations

import sys


CHECKS: list[tuple[str, callable]] = []


def check(name: str, fn: callable) -> None:
    CHECKS.append((name, fn))


def run_all() -> int:
    passed = 0
    failed = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"  [OK] {name}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {name}: {exc}")
            failed += 1
    print(f"\n{passed}/{passed + failed} checks passed")
    return 0 if failed == 0 else 1


check("version", lambda: __import__("automl_pipeline").__version__)
check("data imports", lambda: __import__("automl_pipeline.data"))
check("config imports", lambda: __import__("automl_pipeline.config"))
check("preprocessing imports", lambda: __import__("automl_pipeline.preprocessing"))
check("models imports", lambda: __import__("automl_pipeline.models"))
check("optimization imports", lambda: __import__("automl_pipeline.optimization"))
check("pipeline imports", lambda: __import__("automl_pipeline.pipeline"))
check("cli imports", lambda: __import__("automl_pipeline.cli"))
check("report imports", lambda: __import__("automl_pipeline.report"))

if __name__ == "__main__":
    sys.exit(run_all())
