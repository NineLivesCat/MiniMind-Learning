"""PyInstaller 入口：打包后双击/命令行运行这个 exe。"""
import sys

from mymm.desktop import main

if __name__ == "__main__":
    sys.exit(main())
