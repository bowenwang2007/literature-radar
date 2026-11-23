from __future__ import annotations

import os
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
            st.info("当前阅读会话已满，请稍后重新打开。")
            return
        SETTINGS = HostedSettings()
    elif mode == "local":
        if not local_request_allowed(_request_host()):
            st.error("本地个人应用仅允许通过 localhost 或 127.0.0.1 访问。")
            return
        try:
            SETTINGS = Settings.load(ROOT)
            if not SETTINGS.database_path.is_file():
                st.error(f"数据库不存在：{SETTINGS.database_path}。请先使用 CLI 显式初始化数据库。")
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
        st.error(f"页面加载失败：{type(exc).__name__}: {str(exc)[:800]}")


if __name__ == "__main__":
    main()
