from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import suppress
from pathlib import Path

import streamlit as st

from eduecon_radar.config import Settings
from eduecon_radar.db import Database
from eduecon_radar.product import PAGE_TITLE
from eduecon_radar.services.product_access import local_request_allowed
from eduecon_radar.ui.theme import apply_app_theme
from eduecon_radar.ui.views import personal_mode

ROOT = Path(__file__).resolve().parents[3]
LOCAL_PAGES = ["文献工作台"]
SETTINGS: Settings
DB: Database

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📗",
    layout="wide",
)
apply_app_theme()


def _request_host() -> str | None:
    try:
        return st.context.headers.get("host")
    except Exception:
        return None


def _render_local_routes() -> None:
    personal_mode.render(DB, SETTINGS)


def main() -> None:
    global DB, SETTINGS
    mode = os.environ.get("LITERATURE_RADAR_MODE", "hosted")
    if mode == "hosted":
        from eduecon_radar.ui.collection import HostedSettings, WorkspaceLimit
        from eduecon_radar.ui.hosted_workspace import session_workspace

        try:
            _, _, DB = session_workspace()
        except WorkspaceLimit:
            st.info("当前阅读会话已满。稍后有空位时即可进入。")
            st.button("重试")
            return
        except (OSError, sqlite3.Error):
            logging.getLogger(__name__).exception("Could not initialize hosted workspace")
            st.error("暂时无法打开阅读工作区，请稍后重试。")
            st.button("重试")
            return
        SETTINGS = HostedSettings()
    elif mode == "local":
        if not local_request_allowed(_request_host()):
            st.error("本地个人应用仅允许通过 localhost 或 127.0.0.1 访问。")
            return
        try:
            SETTINGS = Settings.load(ROOT)
            if not SETTINGS.database_path.is_file():
                path = SETTINGS.database_path
                st.header("打开个人论文库")
                st.caption(f"本机保存位置 · `{path}`")
                if any(Path(str(path) + suffix).exists() for suffix in (
                    ".reading.sqlite", ".jobs.sqlite", "-wal", "-shm", "-journal",
                )):
                    st.warning("发现已有历史文件。请先恢复原数据库，或在 .env 中确认库的位置。")
                    return
                st.write("首次使用可以创建空库。已有论文库请在 .env 中确认保存位置。")
                if st.button("创建个人论文库", type="primary"):
                    from eduecon_radar.services.migration_integrity import create_personal_database

                    with suppress(FileExistsError):  # Another tab may finish setup first.
                        create_personal_database(path)
                    st.rerun()
                return
            DB = Database(SETTINGS.database_path)
        except Exception as exc:
            st.error(f"应用初始化失败：{str(exc)[:1000]}")
            return
    else:
        st.error("LITERATURE_RADAR_MODE 必须为 local 或 hosted。")
        return
    try:
        _render_local_routes()
    except Exception as exc:
        if mode == "hosted":
            logging.getLogger(__name__).exception("Could not render hosted workspace")
            st.error("暂时无法读取工作区。请重试，或重新打开开始新的阅读会话。")
            st.button("重试")
        else:
            st.error(f"页面加载失败：{type(exc).__name__}: {str(exc)[:800]}")


if __name__ == "__main__":
    main()
