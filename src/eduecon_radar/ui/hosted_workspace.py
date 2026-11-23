"""Allocate one bounded, disposable SQLite workspace per Streamlit session."""
from pathlib import Path

import streamlit as st

from eduecon_radar.resources import resource
from eduecon_radar.ui.collection import WorkspacePool


@st.cache_resource(show_spinner=False)
def _pool():
    return WorkspacePool(Path(str(resource("config/public_papers.json"))))


def _release(workspace):
    pool, directory, _ = workspace
    pool.close(directory)


@st.cache_resource(scope="session", on_release=_release, show_spinner=False)
def session_workspace():
    pool = _pool()
    directory, database = pool.open()
    return pool, directory, database
