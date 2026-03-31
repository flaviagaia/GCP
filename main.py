from __future__ import annotations

import json
from pathlib import Path

from src.projects import bundle_reports


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    report = bundle_reports()
    output_dir = base_dir / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "classic_ml_portfolio_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
