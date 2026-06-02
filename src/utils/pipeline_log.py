"""Pipeline 运行日志：将终端输出同步写入按次生成的 .log 文件。"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class _TeeStdout:
    """同时写入控制台与文件（不改动 print 调用处）。"""

    def __init__(self, console, file_obj):
        self._console = console
        self._file = file_obj

    def write(self, data):
        if isinstance(data, bytes):
            enc = getattr(self._console, "encoding", None) or "utf-8"
            data = data.decode(enc, errors="replace")
        self._console.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self) -> None:
        self._console.flush()
        self._file.flush()

    def __getattr__(self, name):
        return getattr(self._console, name)


def default_log_path(log_dir: Path, video_stem: str) -> Path:
    """每次运行一个文件：时间戳 + 视频名（截断）。"""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in video_stem)[:80]
    return log_dir / f"run_{ts}_{safe or 'video'}.log"


@contextmanager
def tee_stdout(log_path: Path):
    """
    将 sys.stdout 重定向到「控制台 + log_path」。
    进入时写入一行开始时间；退出时写入结束时间并关闭文件。
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    old = sys.stdout
    f = open(log_path, "w", encoding="utf-8")
    try:
        start = datetime.now().isoformat(timespec="seconds")
        f.write(f"# pipeline log started {start}\n")
        f.write(f"# log file: {log_path.resolve()}\n\n")
        f.flush()
        sys.stdout = _TeeStdout(old, f)
        yield log_path
    finally:
        sys.stdout = old
        try:
            end = datetime.now().isoformat(timespec="seconds")
            f.write(f"\n# pipeline log ended {end}\n")
        except OSError:
            pass
        f.close()
