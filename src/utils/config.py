import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Pipeline 运行日志目录；可用 .env 覆盖为绝对路径
PIPELINE_LOG_DIR = Path(
    os.getenv("PIPELINE_LOG_DIR", str(_PROJECT_ROOT / "logs"))
).expanduser()

SERPER_KEY = os.getenv("SERPER_KEY", "").strip()

MAX_COMPRESSED_VIDEO_BYTES = int(os.getenv("MAX_COMPRESSED_VIDEO_BYTES", str(18 * 1024 * 1024)))

# 检索由模型决定何时停止；此处仅为防止死循环的「安全上限」（步数）
SEARCH_SAFETY_MAX_STEPS = int(os.getenv("SEARCH_SAFETY_MAX_STEPS", "30"))
# 「handle + 关键词叠加」每轮最多几批新关键词（每批 2-3 词由模型给出）
SEARCH_KEYWORD_BATCH_MAX_ROUNDS = int(os.getenv("SEARCH_KEYWORD_BATCH_MAX_ROUNDS", "5"))

# YouTube 下载：解决 "Sign in to confirm you're not a bot"
# 方式 A（推荐）：用浏览器扩展导出 Netscape 格式 cookies.txt，填绝对路径
YT_DLP_COOKIES_FILE = os.getenv("YT_DLP_COOKIES_FILE", "").strip()
# 方式 A2：多个 cookies 文件，逗号分隔；每次下载轮询使用，减轻单账号限流
YT_DLP_COOKIES_FILES = os.getenv("YT_DLP_COOKIES_FILES", "").strip()
# 方式 A3：目录内所有 cookies*.txt（按文件名排序），与 YT_DLP_COOKIES_FILES 二选一
YT_DLP_COOKIES_DIR = os.getenv("YT_DLP_COOKIES_DIR", "").strip()
# 方式 B：从本机浏览器读已登录 YouTube 的 Cookie，格式 browser 或 browser:profile，如 chrome、safari、firefox
YT_DLP_COOKIES_FROM_BROWSER = os.getenv("YT_DLP_COOKIES_FROM_BROWSER", "").strip()

# YouTube 解签 / n challenge 需要 JS 运行时；默认可用 node（需本机已安装 node 且在 PATH）
# 多个用逗号：node,deno,bun。也可 node:/opt/homebrew/bin/node
YT_DLP_JS_RUNTIMES = os.getenv("YT_DLP_JS_RUNTIMES", "node,deno").strip()


def load_api_config() -> dict[str, str]:
    """Load provider API keys from environment (multi-vendor routing)."""
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "GEMINI_API_KEY": os.getenv("GEMINI_API_KEY", "").strip(),
        "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", "").strip(),
        "DASHSCOPE_API_KEY": os.getenv("DASHSCOPE_API_KEY", "").strip(),
        "DASHSCOPE_BASE_URL": os.getenv("DASHSCOPE_BASE_URL", "").strip(),
    }


def resolve_yt_dlp_cookies_files() -> list[str]:
    """Resolve cookie file paths for yt-dlp rotation.

    Priority: YT_DLP_COOKIES_FILES > YT_DLP_COOKIES_DIR > YT_DLP_COOKIES_FILE
    """
    def _existing(paths: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for raw in paths:
            path = str(Path(raw).expanduser())
            if path in seen:
                continue
            seen.add(path)
            if Path(path).is_file():
                out.append(path)
        return out

    files_env = os.getenv("YT_DLP_COOKIES_FILES", "").strip()
    if files_env:
        return _existing([part.strip() for part in files_env.split(",") if part.strip()])

    dir_env = os.getenv("YT_DLP_COOKIES_DIR", "").strip()
    if dir_env:
        cookie_dir = Path(dir_env).expanduser()
        if cookie_dir.is_dir():
            return _existing([str(path) for path in sorted(cookie_dir.glob("cookies*.txt"))])

    single = os.getenv("YT_DLP_COOKIES_FILE", "").strip()
    if single:
        return _existing([single])
    return []
