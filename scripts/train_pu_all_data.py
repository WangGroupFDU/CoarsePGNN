#!/usr/bin/env python3
"""一键训练入口；科学协议与预算控制均由独立 workflow 模块执行。"""
from pathlib import Path
import sys

# 支持直接从仓库执行，不要求事先安装研究包。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coarsepgnn_retrain.workflow import main


if __name__ == "__main__":
    main()
