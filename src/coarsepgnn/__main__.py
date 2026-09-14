"""输出 CoarsePGNN 包和 schema 摘要的轻量入口。"""

from __future__ import annotations

import json

from .schema import package_schema


def main() -> int:
    """打印冻结 schema，便于部署后做只读契约检查。"""

    print(json.dumps(package_schema(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
