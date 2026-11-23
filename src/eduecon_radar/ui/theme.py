# ruff: noqa: E501
"""Sea-fog visual system for the Literature Radar research workbench.

The design is deliberately not an admin dashboard: a thin masthead, one open
gradient workbench surface where retrieval is the center of the page, quiet
counts instead of BI cards, and paper-sheet surfaces for reading.  Anything a
researcher does not act on today is folded away rather than styled to compete.
"""

from __future__ import annotations

from html import escape
from typing import Literal

import streamlit as st

StatusTone = Literal["pending", "missing", "retry", "review", "complete", "failed", "info"]


MISTY_BLUE_CSS = """
<style id="eduecon-misty-blue-theme">
:root {
  --sea-mist-50: #F6F9FB;
  --sea-mist-100: #EAF1F6;
  --sea-mist-200: #DCE6EE;
  --sea-mist-300: #C3D2DE;
  --sea-mist-400: #A2B7C9;
  --sea-mist-500: #7E97AC;
  --sea-mist-600: #5F7B93;
  --sea-mist-700: #48647D;
  --sea-mist-800: #334D63;
  --sea-mist-900: #24394C;
  --text-primary: #24394C;
  --text-secondary: #51697D;
  --text-muted: #74899A;
  --surface: rgba(251, 253, 254, 0.84);
  --surface-solid: #FBFDFE;
  --surface-soft: rgba(241, 246, 250, 0.72);
  --surface-muted: #EAF1F6;
  --border: rgba(126, 150, 171, 0.20);
  --border-strong: rgba(104, 132, 153, 0.34);
  --focus-ring: rgba(79, 110, 136, 0.20);
  --accent: #4F6E88;
  --accent-hover: #3B5871;
  --complete: #52708B;
  --complete-bg: #E8EFF5;
  --success: #4E6C80;
  --success-bg: #E9F0F5;
  --warning: #8A7049;
  --warning-bg: #F6EFE3;
  --error: #955D68;
  --error-bg: #F8ECEE;
  --retry: #7C5E55;
  --retry-bg: #F5ECE8;
  --info: #4F6E88;
  --info-bg: #E9F0F5;
  --shadow-soft: 0 24px 64px rgba(53, 78, 98, 0.055);
  --shadow-raised: 0 30px 72px rgba(53, 78, 98, 0.085);
  --radius-paper: 26px;
  --radius-control: 14px;
  --font-body: 16px;
  --font-caption: 14px;
  --font-meta: 15px;
  --font-label: 16px;
  --font-abstract: 19px;
  --workbench-max: 1280px;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
  color: var(--text-primary);
  font-family: "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "Segoe UI", sans-serif;
  font-size: var(--font-body);
}

[data-testid="stAppViewContainer"], .stApp {
  background-color: #D4E0E9;
  background-attachment: fixed;
  background-image:
    radial-gradient(ellipse 68% 46% at 88% 3%, rgba(255, 255, 255, 0.72) 0%, rgba(243, 248, 252, 0.42) 24%, rgba(221, 232, 241, 0.15) 50%, transparent 74%),
    radial-gradient(ellipse 30% 68% at 84% 64%, rgba(235, 243, 249, 0.22) 0%, rgba(218, 231, 240, 0.09) 48%, transparent 78%),
    radial-gradient(ellipse 94% 32% at 46% 35%, rgba(251, 253, 254, 0.40) 0%, rgba(235, 242, 247, 0.18) 50%, transparent 80%),
    radial-gradient(ellipse 86% 28% at 18% 61%, rgba(235, 242, 247, 0.18) 0%, transparent 76%),
    radial-gradient(ellipse 96% 42% at 58% 92%, rgba(83, 111, 134, 0.15) 0%, rgba(110, 138, 159, 0.07) 46%, transparent 78%),
    linear-gradient(180deg, #E9EFF4 0%, #EDF2F6 18%, #E1E9F0 39%, #CBD9E4 61%, #E0E9EF 80%, #D7E3EC 100%);
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

[data-testid="stAppViewContainer"]::before {
  background:
    radial-gradient(ellipse 78% 28% at 82% 19%, rgba(255, 255, 255, 0.26) 0%, rgba(241, 247, 251, 0.11) 48%, transparent 76%),
    radial-gradient(ellipse 98% 32% at 36% 50%, rgba(247, 250, 252, 0.20) 0%, transparent 80%),
    linear-gradient(104deg, rgba(250, 252, 254, 0.18) 0%, transparent 38%, rgba(229, 238, 245, 0.08) 68%, rgba(248, 251, 253, 0.17) 100%);
  content: "";
  inset: 0;
  opacity: 0.74;
  pointer-events: none;
  position: fixed;
  z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
  background:
    radial-gradient(ellipse 70% 42% at 62% 12%, rgba(255, 255, 255, 0.48) 0%, rgba(242, 247, 251, 0.22) 28%, rgba(220, 232, 241, 0.08) 54%, transparent 78%),
    linear-gradient(180deg, rgba(246, 250, 252, 0.18) 0%, rgba(232, 240, 246, 0.08) 52%, transparent 88%);
  content: "";
  filter: blur(34px);
  inset: -8% -5% -10% 40%;
  opacity: 0.78;
  pointer-events: none;
  position: fixed;
  z-index: 0;
}
[data-testid="stAppViewContainer"] > section { position: relative; z-index: 1; }

[data-testid="stHeader"] { background: transparent; height: 1.5rem; }
[data-testid="stToolbar"], .stDeployButton { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

[data-testid="stMainBlockContainer"] {
  box-sizing: border-box;
  margin-inline: auto;
  max-width: var(--workbench-max);
  padding: 1.4rem 2.25rem 4rem;
  width: 100%;
}

h1, h2, h3, h4, h5, h6,
[data-testid="stHeadingWithActionElements"] {
  color: var(--sea-mist-900);
  letter-spacing: -0.022em;
}
h1 {
  font-size: clamp(3rem, 3.8vw, 4.1rem);
  font-weight: 650;
  line-height: 1.08;
}
h2 {
  font-size: clamp(2rem, 2.5vw, 2.5rem) !important;
  font-weight: 660 !important;
  line-height: 1.22 !important;
}
h3 {
  font-size: clamp(1.55rem, 2vw, 1.82rem);
  font-weight: 650;
  line-height: 1.38;
}
[data-testid="stHeadingWithActionElements"] {
  margin-block: 0;
  margin-top: .85rem;
  padding-block: 0;
}
p, li, [data-testid="stMarkdownContainer"] { font-size: 1.04rem; line-height: 1.9; }
[data-testid="stCaptionContainer"] {
  color: var(--text-secondary);
  font-size: var(--font-caption);
  line-height: 1.72;
}
hr { border-color: var(--border); opacity: .72; }

/* ---------- Masthead: thin, open, no card ---------- */
.st-key-edr_page_header {
  margin: 0;
  padding: 0;
  position: relative;
}
.st-key-edr_page_header::after {
  background: linear-gradient(90deg, rgba(104, 132, 153, 0.30), rgba(104, 132, 153, 0.08) 34%, transparent 72%);
  bottom: 0;
  content: "";
  height: 1px;
  left: 0;
  position: absolute;
  width: 100%;
}
.st-key-edr_page_header > div { gap: 0 !important; }
.st-key-edr_page_header [data-testid="stHorizontalBlock"] { align-items: flex-end; }
.edr-masthead-eyebrow {
  color: var(--sea-mist-600);
  font-size: 0.82rem;
  font-weight: 680;
  letter-spacing: 0.26em;
  line-height: 1.5;
  margin: 0 0 0.85rem;
  text-transform: uppercase;
}
.st-key-edr_page_header [data-testid="stHeadingWithActionElements"] {
  margin-top: 0;
}
.st-key-edr_page_header [data-testid="stHeadingWithActionElements"] h1 {
  font-family: Georgia, "Times New Roman", "Songti SC", serif;
  font-size: clamp(1.6rem, 2.1vw, 2.1rem);
  font-weight: 540;
  letter-spacing: 0.014em;
  line-height: 1.05;
  margin: 0;
  padding-left: 3.6rem;
  position: relative;
}
.st-key-edr_page_header h1::before {
  background: radial-gradient(circle at 50% 50%, rgba(95, 123, 147, 0.92) 0 13%, transparent 14%);
  border: 1.5px solid rgba(94, 126, 151, 0.48);
  border-radius: 50%;
  box-shadow:
    inset 0 0 0 4px rgba(234, 241, 246, 0.95),
    inset 0 0 0 5px rgba(126, 150, 171, 0.28);
  content: "";
  height: 2.3rem;
  left: 0;
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  width: 2.3rem;
}
.st-key-edr_page_header [data-testid="stCaptionContainer"] {
  color: var(--text-secondary);
  font-size: 1.12rem;
  font-weight: 470;
  letter-spacing: 0.1em;
  margin: 0.25rem 0 0;
}
/* ---------- Open section headings ---------- */
.edr-section-description {
  color: var(--text-secondary);
  font-size: 1.02rem;
  line-height: 1.78;
  margin: 0.35rem 0 1.55rem;
  max-width: 74ch;
}
.edr-section-meta {
  color: var(--text-muted);
  font-size: 0.98rem;
  font-variant-numeric: tabular-nums;
  font-weight: 500;
  line-height: 1.7;
  text-align: right;
}

/* ---------- Hero: the retrieval workbench ---------- */
.st-key-edr_retrieval_panel {
  background:
    radial-gradient(780px 360px at 10% 0%, rgba(255, 255, 255, 0.78), transparent 72%),
    radial-gradient(940px 440px at 94% 10%, rgba(196, 213, 226, 0.36), transparent 74%),
    linear-gradient(116deg, rgba(224, 235, 243, 0.55), rgba(246, 249, 251, 0.30) 48%, rgba(212, 228, 237, 0.48));
  border: 0;
  border-radius: 34px;
  box-shadow:
    inset 0 1px 0 rgba(255, 255, 255, 0.82),
    0 34px 84px rgba(53, 78, 98, 0.065);
  margin: 0.6rem 0 0;
  padding: 2.7rem 3rem 2.6rem;
}
.edr-hero-eyebrow {
  color: var(--sea-mist-600);
  font-size: 0.82rem;
  font-weight: 680;
  letter-spacing: 0.24em;
  line-height: 1.5;
  margin: 0 0 0.9rem;
  text-transform: uppercase;
}
.st-key-edr_retrieval_panel [data-testid="stHeadingWithActionElements"] {
  margin-top: 0;
}
.st-key-edr_retrieval_panel [data-testid="stHeadingWithActionElements"] h2 {
  font-size: clamp(2.3rem, 3vw, 3.1rem) !important;
  font-weight: 620 !important;
  letter-spacing: -0.024em !important;
  line-height: 1.12 !important;
  margin: 0;
}
.edr-hero-intro {
  color: var(--text-secondary);
  font-size: 1.08rem;
  line-height: 1.82;
  margin: 0.5rem 0 1.4rem;
  max-width: 60ch;
}
.edr-hero-note {
  color: var(--text-muted);
  font-size: 0.94rem;
  line-height: 1.7;
  margin: 1.55rem 0.1rem 0;
}
.st-key-edr_retrieval_panel [data-testid="stHorizontalBlock"] {
  align-items: flex-end;
  gap: 1.5rem !important;
  margin-block: 0;
}
.st-key-edr_retrieval_panel[data-testid="stVerticalBlock"],
.st-key-edr_retrieval_panel [data-testid="stVerticalBlock"] {
  gap: 0.8rem !important;
}
.st-key-edr_page_header[data-testid="stVerticalBlock"] {
  gap: 0.25rem !important;
}
.st-key-edr_retrieval_panel [data-testid="stWidgetLabel"] p {
  margin-bottom: 0.5rem !important;
}
.st-key-edr_retrieval_panel [data-testid="stExpander"] summary {
  min-height: 2.9rem;
}

/* ---------- Workflow line: quiet, not a dashboard ---------- */
.edr-flow {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin-top: 1.9rem;
  position: relative;
}
.edr-flow::before {
  background: linear-gradient(90deg, transparent, rgba(126, 150, 171, 0.34) 12%, rgba(126, 150, 171, 0.34) 88%, transparent);
  content: "";
  height: 1px;
  left: 7%;
  position: absolute;
  right: 7%;
  top: 5px;
}
.edr-flow-note {
  color: var(--text-muted);
  font-size: 0.95rem;
  line-height: 1.65;
  margin: 0.85rem 0 0;
}
.edr-flow-node {
  align-items: center;
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 0 1.1rem;
  position: relative;
  text-align: center;
}
.edr-flow-node::before {
  background: var(--sea-mist-200);
  border: 2px solid var(--sea-mist-400);
  border-radius: 50%;
  box-shadow: 0 0 0 4px rgba(234, 241, 246, 0.92);
  content: "";
  height: 11px;
  width: 11px;
  z-index: 1;
}
.edr-flow-node--info::before { border-color: var(--sea-mist-500); }
.edr-flow-node--missing::before { border-color: var(--warning); }
.edr-flow-node--review::before { border-color: #7A8CA6; }
.edr-flow-node--complete::before { background: var(--sea-mist-500); border-color: var(--sea-mist-600); }
.edr-flow-node b {
  color: var(--sea-mist-800);
  font-size: 1.4rem;
  font-variant-numeric: tabular-nums;
  font-weight: 660;
  line-height: 1.12;
}
.edr-flow-node span {
  color: var(--text-secondary);
  font-size: 0.95rem;
  font-weight: 560;
  line-height: 1.5;
}

/* ---------- Today's glance: which queue, how much, what next ---------- */
.edr-status-bar {
  align-items: stretch;
  border-bottom: 1px solid rgba(126, 150, 171, 0.18);
  border-top: 1px solid rgba(126, 150, 171, 0.18);
  display: flex;
  flex-wrap: wrap;
  gap: 1.6rem;
  justify-content: space-between;
  margin: .25rem 0 0;
  padding: .6rem .2rem;
}
.edr-status-profile {
  display: flex;
  flex-direction: column;
  gap: 0.25rem;
  justify-content: center;
  min-width: 12rem;
}
.edr-status-label {
  color: var(--text-muted);
  font-size: 0.9rem;
  font-weight: 560;
  letter-spacing: 0.06em;
}
.edr-status-name {
  color: var(--sea-mist-900);
  font-size: 1.5rem;
  font-weight: 640;
  letter-spacing: -0.01em;
  line-height: 1.25;
}
.edr-status-period {
  color: var(--text-muted);
  font-size: 0.92rem;
  font-variant-numeric: tabular-nums;
}
.edr-status-metrics {
  display: grid;
  flex: 1 1 26rem;
  gap: 1.2rem;
  grid-template-columns: repeat(4, minmax(0, 1fr));
}
.edr-status-metric {
  border-left: 1px solid rgba(126, 150, 171, 0.2);
  display: flex;
  flex-direction: column;
  gap: 0.2rem;
  padding-left: 1.1rem;
}
.edr-status-metric b {
  color: var(--sea-mist-800);
  font-size: 1.85rem;
  font-variant-numeric: tabular-nums;
  font-weight: 660;
  line-height: 1.1;
}
.edr-status-metric span {
  color: var(--text-secondary);
  font-size: 0.95rem;
  font-weight: 540;
  line-height: 1.45;
}
.edr-next-action {
  align-items: baseline;
  border-left: 3px solid var(--sea-mist-500);
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem 1rem;
  margin: .35rem 0 .15rem;
  padding: 0.2rem 0 0.2rem 1.05rem;
}
.edr-next-action b {
  color: var(--sea-mist-900);
  font-size: 1.22rem;
  font-weight: 640;
  line-height: 1.5;
}
.edr-next-action span {
  color: var(--text-secondary);
  font-size: 1rem;
  line-height: 1.62;
}
.edr-next-action--review { border-left-color: #7A8CA6; }
.edr-next-action--missing { border-left-color: var(--warning); }
.edr-next-action--pending { border-left-color: var(--sea-mist-500); }
.edr-next-action--complete { border-left-color: var(--complete); }

/* ---------- Audit line: one sentence, panel folded away ---------- */
.edr-audit-line {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem 0.85rem;
  margin: 1.5rem 0 0;
}
.edr-audit-label {
  color: var(--text-muted);
  font-size: 0.95rem;
  font-weight: 560;
  letter-spacing: 0.04em;
}
.edr-audit-line .edr-badges { margin-top: 0; }
.edr-audit-note {
  color: var(--text-muted);
  font-size: 0.9rem;
  line-height: 1.6;
}
.edr-action-hint {
  color: var(--text-secondary);
  font-size: 1.02rem;
  line-height: 1.75;
  margin: 0.2rem 0 0.9rem;
  max-width: 76ch;
}

/* ---------- Working sections: open, box-free ---------- */
.st-key-edr_processing_panel,
.st-key-edr_export_panel,
.st-key-edr_review_workspace,
.st-key-edr_library_filters {
  background: transparent;
  border: 0;
  box-shadow: none;
}
.edr-run-label {
  color: var(--sea-mist-700);
  font-size: 1.06rem;
  font-weight: 640;
  letter-spacing: 0.02em;
  margin: 0.4rem 0 0.5rem;
}
.edr-process-summary {
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  margin: 0.2rem 0 1.15rem;
}

/* ---------- Retrieval verification: conclusion first ---------- */
.st-key-edr_verification_result {
  background: linear-gradient(145deg, rgba(250, 253, 255, 0.82), rgba(235, 243, 248, 0.66));
  border: 1px solid rgba(126, 150, 171, 0.18);
  border-radius: 22px;
  box-shadow: 0 18px 42px rgba(53, 78, 98, 0.045);
  margin: 0.7rem 0 1.25rem;
  padding: 1.5rem 1.7rem 1.35rem;
}
.st-key-edr_verification_result[data-testid="stVerticalBlock"],
.st-key-edr_verification_result [data-testid="stVerticalBlock"] {
  gap: 0.55rem !important;
}
.edr-verification-title {
  color: var(--sea-mist-900);
  font-size: 1.28rem;
  font-weight: 680;
  letter-spacing: -0.01em;
  line-height: 1.4;
}
.edr-verification-summary,
.edr-verification-empty {
  color: var(--text-secondary);
  font-size: 1rem;
  line-height: 1.72;
  max-width: 72ch;
}
.edr-verification-stats {
  border-bottom: 1px solid rgba(126, 150, 171, 0.15);
  border-top: 1px solid rgba(126, 150, 171, 0.15);
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  margin: 0.75rem 0 0.55rem;
  padding: 0.9rem 0;
}
.edr-verification-stats > div {
  border-right: 1px solid rgba(126, 150, 171, 0.13);
  display: flex;
  flex-direction: column;
  gap: 0.22rem;
  padding: 0 1.1rem;
}
.edr-verification-stats > div:first-child { padding-left: 0; }
.edr-verification-stats > div:last-child { border-right: 0; }
.edr-verification-stats b {
  color: var(--sea-mist-800);
  font-size: 1.45rem;
  font-variant-numeric: tabular-nums;
  font-weight: 680;
  line-height: 1.2;
}
.edr-verification-stats span {
  color: var(--text-muted);
  font-size: 0.88rem;
  font-weight: 540;
  line-height: 1.45;
}
.edr-verification-list-title {
  color: var(--sea-mist-800);
  font-size: 1.04rem;
  font-weight: 660;
  margin: 0.75rem 0 0.25rem;
}
.edr-verification-missing {
  background: rgba(250, 252, 254, 0.72);
  border: 1px solid rgba(138, 112, 73, 0.16);
  border-radius: 15px;
  margin-top: 0.65rem;
  padding: 1rem 1.15rem;
}
.edr-verification-missing-title {
  color: var(--sea-mist-900);
  font-size: 1.03rem;
  font-weight: 650;
  line-height: 1.55;
}
.edr-verification-missing-meta,
.edr-verification-reason {
  color: var(--text-muted);
  font-size: 0.88rem;
  line-height: 1.62;
  margin-top: 0.3rem;
}
.edr-verification-missing-meta a {
  color: var(--sea-mist-700);
  font-weight: 620;
  text-decoration: none;
}
.edr-verification-missing-meta a:hover { text-decoration: underline; }
.edr-verification-reason { color: var(--text-secondary); }
.st-key-edr_verification_result [data-testid="stButton"] { margin-top: 0.35rem; }
.st-key-edr_verification_result [data-testid="stButton"] button {
  min-height: 2.8rem;
  padding-inline: 1rem;
}
.st-key-edr_verification_result [data-testid="stExpander"] { margin-top: 0.2rem; }

/* ---------- Badges ---------- */
.edr-badges { display: flex; flex-wrap: wrap; gap: 0.55rem; margin-top: 0.9rem; }
.edr-badge {
  align-items: center;
  border: 1px solid transparent;
  border-radius: 999px;
  display: inline-flex;
  font-size: 0.96rem;
  font-weight: 620;
  line-height: 1.35;
  padding: 0.38rem 0.9rem;
  white-space: nowrap;
}
.edr-badge--pending, .edr-badge--info { background: var(--info-bg); border-color: rgba(79, 110, 136, 0.16); color: var(--info); }
.edr-badge--missing { background: var(--warning-bg); border-color: rgba(138, 112, 73, 0.18); color: var(--warning); }
.edr-badge--retry { background: var(--retry-bg); border-color: rgba(124, 94, 85, 0.18); color: var(--retry); }
.edr-badge--review { background: #EFF2F6; border-color: rgba(90, 108, 130, 0.16); color: #56647A; }
.edr-badge--complete { background: var(--complete-bg); border-color: rgba(82, 112, 139, 0.18); color: var(--complete); }
.edr-badge--failed { background: var(--error-bg); border-color: rgba(149, 93, 104, 0.18); color: var(--error); }

/* ---------- Reading surfaces: paper sheets ---------- */
.edr-paper-card,
.edr-library-card {
  background:
    radial-gradient(720px 240px at 6% 0%, rgba(226, 236, 244, 0.55), transparent 72%),
    linear-gradient(180deg, rgba(253, 254, 255, 0.92), rgba(243, 247, 250, 0.82));
  border: 1px solid rgba(126, 150, 171, 0.15);
  border-radius: var(--radius-paper);
  box-shadow: var(--shadow-soft);
  margin: 1.2rem 0 1.5rem;
  padding: 2.5rem 3rem 2.7rem;
}
.edr-paper-title {
  color: var(--sea-mist-900);
  font-size: 1.85rem;
  font-weight: 680;
  line-height: 1.42;
  margin: 0 0 0.5rem;
}
.edr-library-title {
  color: var(--sea-mist-900);
  font-size: 1.55rem;
  font-weight: 670;
  line-height: 1.46;
  margin: 0 0 0.35rem;
}
.edr-paper-title-en {
  color: var(--text-secondary);
  font-size: 1.16rem;
  font-weight: 450;
  line-height: 1.68;
  margin: 0 0 0.9rem;
  max-width: 88ch;
}
.edr-library-title-en {
  color: var(--text-secondary);
  font-size: 1.08rem;
  font-weight: 450;
  line-height: 1.66;
  margin: 0 0 0.85rem;
  max-width: 92ch;
}
.edr-paper-meta, .edr-library-meta {
  color: var(--text-muted);
  font-size: var(--font-meta);
  font-weight: 470;
  line-height: 1.68;
  margin: 0 0 1rem;
}
.edr-abstract, .edr-reading-text {
  color: var(--text-primary);
  font-size: var(--font-abstract);
  font-weight: 440;
  line-height: 1.96;
  max-width: 74ch;
}
.edr-quiet {
  color: var(--text-secondary);
  font-size: var(--font-caption);
  line-height: 1.74;
  margin: 0.45rem 0 0.55rem;
}
.st-key-edr_review_workspace { margin-inline: auto; max-width: 1180px; }
.st-key-edr_review_workspace [data-testid="stProgress"] { margin: 0.35rem 0 0.5rem; }
.st-key-edr_review_workspace [data-testid="stProgress"] p {
  color: var(--text-secondary);
  font-size: 1rem;
}
.st-key-edr_review_workspace [data-testid="stRadio"] { margin-top: .35rem; }
/* The AI's reason and the human decision are one panel: the decision belongs
   directly under the reason it is accepting or overruling. */
.edr-reason {
  border-top: 1px solid rgba(126, 150, 171, .2);
  margin-top: .35rem;
  padding-top: .85rem;
}
.edr-reason-title {
  color: var(--sea-mist-900);
  font-size: 1.06rem;
  font-weight: 640;
  letter-spacing: .01em;
  margin-bottom: .3rem;
}
.edr-reason-text {
  color: var(--text-primary);
  font-size: 1.02rem;
  line-height: 1.78;
  margin: 0 0 .7rem;
}
.edr-reason-text--quiet { color: var(--text-muted); }
.edr-reason-grid {
  display: grid;
  gap: .3rem .9rem;
  grid-template-columns: minmax(0, 1fr);
  margin-bottom: .55rem;
}
.edr-reason-row {
  display: flex;
  font-size: .96rem;
  gap: .55rem;
  line-height: 1.62;
}
.edr-reason-row b {
  color: var(--text-muted);
  flex: 0 0 6.4rem;
  font-weight: 560;
}
.edr-reason-row span { color: var(--text-primary); }
.edr-reason-quotes {
  color: var(--text-secondary);
  font-size: .96rem;
  line-height: 1.72;
}
.edr-reason-quotes b {
  color: var(--text-muted);
  display: block;
  font-weight: 560;
  margin-bottom: .2rem;
}
.edr-reason-quotes ul { margin: .1rem 0 0; padding-left: 1.15rem; }
.edr-reason-quotes li { margin-bottom: .25rem; }
.edr-reason-quotes--quiet { color: var(--text-muted); }
.edr-review-position {
  color: var(--text-muted);
  font-size: .94rem;
  line-height: 3.3rem;
  text-align: center;
}
.edr-library-scope {
  color: var(--text-muted);
  font-size: .95rem;
  line-height: 1.7;
  margin: -0.6rem 0 1.3rem;
}
.edr-export-intro {
  color: var(--text-secondary);
  font-size: 1rem;
  line-height: 1.72;
  margin: .15rem 0 1.1rem;
}
.edr-export-choice {
  display: flex;
  flex-direction: column;
  gap: .18rem;
  margin: 1.05rem 0 .45rem;
}
.edr-export-choice b {
  color: var(--sea-mist-900);
  font-size: 1.1rem;
  font-weight: 640;
}
.edr-export-choice span {
  color: var(--text-secondary);
  font-size: .96rem;
  line-height: 1.68;
  max-width: 76ch;
}
.st-key-edr_review_reading_panel .edr-paper-card {
  margin: .35rem 0 .75rem;
  /* The reading pane is sized against the viewport so that, once the review
     section is in view, the paper and the decision panel share the screen; the
     card scrolls internally rather than pushing the decision off it. */
  max-height: max(18rem, calc(100vh - 26rem));
  min-height: 0;
  overflow-y: auto;
  padding: 1.7rem 1.9rem 1.8rem;
  scrollbar-color: rgba(93, 119, 142, .34) transparent;
  scrollbar-gutter: stable;
}
.st-key-edr_review_reading_panel .edr-paper-title { font-size: 1.6rem; }
.st-key-edr_review_reading_panel .edr-paper-title-en {
  font-size: 1.03rem;
  line-height: 1.58;
}
.st-key-edr_review_reading_panel .edr-abstract { line-height: 1.88; }
.st-key-edr_review_decision_panel {
  background: rgba(249, 252, 254, .76);
  border: 1px solid rgba(126, 150, 171, .18);
  border-radius: 20px;
  box-shadow: 0 14px 34px rgba(45, 68, 88, .08);
  padding: 1.05rem 1.15rem .9rem;
  position: sticky;
  top: .75rem;
}
.st-key-edr_review_decision_panel > div > [data-testid="stVerticalBlock"] {
  gap: .45rem;
}
/* The head is one compact row: the number the researcher acts on, plus how much
   evidence stood behind it.  Everything the model also produced is folded away. */
.edr-decision-head {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: .2rem .7rem;
  justify-content: space-between;
}
.edr-decision-head .edr-badges { margin-top: 0; }
.edr-decision-badges {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: .45rem;
}
.edr-decision-basis {
  color: var(--text-muted);
  font-size: .92rem;
  line-height: 1.5;
}
.edr-decision-hint {
  color: var(--text-secondary);
  font-size: .98rem;
  line-height: 1.6;
  margin: .1rem 0 .2rem;
}
.edr-decision-hint--quiet { color: var(--text-muted); }
.edr-review-panel-title {
  color: var(--sea-mist-900);
  font-size: 1.06rem;
  font-weight: 660;
}
.st-key-edr_review_decision_panel [data-testid="stRadio"] { margin-top: 0; }
.st-key-edr_review_decision_panel [data-testid="stRadio"] [role="radiogroup"] {
  background: rgba(236, 242, 247, .62);
  border-radius: 14px;
  display: grid;
  gap: .2rem .5rem;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  padding: .4rem .6rem;
}
.st-key-edr_review_decision_panel [data-testid="stRadio"] [role="radiogroup"] label {
  min-height: 2rem;
}
.st-key-edr_review_decision_panel [data-testid="stExpander"] { margin-top: 0; }
.st-key-edr_review_decision_panel [data-testid="stExpander"] summary {
  font-size: .94rem;
  min-height: 2.4rem;
}
.st-key-edr_review_decision_panel [data-testid="stExpander"] details {
  border-top: 0;
}
/* The reasoning block is audit material now, so it keeps its readable layout
   without borrowing space from the decision. */
.st-key-edr_review_decision_panel .edr-reason { border-top: 0; margin-top: 0; padding-top: 0; }
.st-key-edr_review_decision_panel [data-testid="stButton"] button {
  min-height: 3rem;
}
/* Navigation is a cursor, not a call to action: it should cost almost no height
   and leave the space to the paper and the decision. */
.st-key-edr_review_nav [data-testid="stButton"] button {
  min-height: 2.5rem;
  padding-inline: .8rem;
}
.st-key-edr_review_nav [data-testid="stProgress"] { margin: 0; }
.st-key-edr_review_nav [data-testid="stProgress"] p {
  font-size: .95rem;
  line-height: 1.5;
}
.edr-access-inline,
.edr-access-link {
  color: var(--sea-mist-700);
  font-weight: 620;
  text-decoration: none;
  text-underline-offset: .18em;
}
.edr-access-inline:hover,
.edr-access-link:hover { text-decoration: underline; }
.edr-access-link {
  display: inline-flex;
  margin: .45rem 0 .75rem;
}
.edr-reading-text--source { color: var(--text-secondary); }
.edr-verification-date-compare {
  color: var(--text-muted);
  font-size: .84rem;
  line-height: 1.55;
  margin-top: .25rem;
}
.edr-library-count {
  color: var(--text-secondary);
  font-size: var(--font-caption);
  margin: 0.95rem 0 0.6rem;
}

/* ---------- Controls ---------- */
[data-testid="stButton"] button,
[data-testid="stFormSubmitButton"] button,
[data-testid="stDownloadButton"] button,
[data-testid="stLinkButton"] a {
  border: 1px solid var(--border-strong);
  border-radius: var(--radius-control);
  font-size: 1.02rem;
  font-weight: 610;
  min-height: 3.3rem;
  padding-inline: 1.25rem;
  transition: background-color 140ms ease, border-color 140ms ease, box-shadow 140ms ease, transform 80ms ease;
}
[data-testid="stButton"] button:not([kind="primary"]),
[data-testid="stFormSubmitButton"] button:not([kind="primary"]),
[data-testid="stDownloadButton"] button,
[data-testid="stLinkButton"] a {
  background: rgba(251, 253, 254, 0.86);
  color: var(--sea-mist-700);
}
[data-testid="stButton"] button:not([kind="primary"]):hover,
[data-testid="stFormSubmitButton"] button:not([kind="primary"]):hover,
[data-testid="stDownloadButton"] button:hover,
[data-testid="stLinkButton"] a:hover {
  background: var(--sea-mist-100);
  border-color: var(--sea-mist-500);
  box-shadow: 0 4px 14px rgba(35, 54, 75, 0.08);
}
button[kind="primary"] {
  background: linear-gradient(115deg, #58748D, #3F5D76) !important;
  border-color: #46647D !important;
  color: #FFFFFF !important;
  box-shadow: 0 10px 28px rgba(59, 88, 113, 0.20);
}
button[kind="primary"]:hover {
  background: linear-gradient(115deg, #4D6B84, #344F68) !important;
  border-color: var(--accent-hover) !important;
  box-shadow: 0 12px 32px rgba(59, 88, 113, 0.25);
}
button[kind="primary"]:active { transform: translateY(1px); }
.st-key-edr_retrieval_panel button[kind="primary"] {
  border-radius: 999px;
  font-size: 1.18rem;
  font-weight: 650;
  letter-spacing: 0.035em;
  min-height: 4.3rem;
  padding-inline: 2rem;
}
button:focus-visible, a:focus-visible { outline: 3px solid var(--focus-ring); outline-offset: 2px; }
button:disabled {
  background: #EEF2F6 !important;
  border-color: #D5DEE8 !important;
  color: #66788A !important;
  opacity: 1 !important;
}

div[data-baseweb="input"], div[data-baseweb="select"], div[data-baseweb="textarea"],
[data-testid="stDateInput"] > div > div {
  background: rgba(252, 254, 255, 0.92);
  border-color: var(--border-strong);
  border-radius: var(--radius-control);
  min-height: 3.3rem;
}
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea,
div[data-baseweb="select"] span {
  color: var(--text-primary);
  font-size: 1.02rem;
}
div[data-baseweb="input"] input, div[data-baseweb="textarea"] textarea { caret-color: var(--accent); }
div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"]:focus-within,
div[data-baseweb="textarea"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--focus-ring);
}
[data-baseweb="tag"] {
  background: var(--info-bg) !important;
  border-radius: 10px !important;
  min-height: 2.45rem;
}
[data-baseweb="tag"] span { color: var(--sea-mist-800) !important; font-size: 0.98rem !important; }
[data-baseweb="tag"] button { min-height: 2rem !important; min-width: 2rem; }
[data-testid="stWidgetLabel"] p,
[data-testid="stRadio"] label,
[data-testid="stCheckbox"] label {
  color: var(--text-primary);
  font-size: var(--font-label);
  font-weight: 530;
}
[data-testid="stRadio"] [role="radiogroup"] { gap: 0.95rem; }

[data-testid="stExpander"] details {
  background: transparent;
  border: 0;
  border-radius: 0;
  box-shadow: none;
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  border-bottom: 1px solid rgba(126, 150, 171, 0.16);
  color: var(--sea-mist-700);
  font-size: 1.04rem;
  font-weight: 600;
  min-height: 3.5rem;
  padding-inline: 0.1rem;
  transition: color 120ms ease;
}
[data-testid="stExpander"] summary:hover { color: var(--accent); }
[data-testid="stExpander"] [data-testid="stVerticalBlock"] {
  padding-top: 0.85rem;
}
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: rgba(251, 253, 254, 0.78);
  border: 1px solid var(--border);
  border-radius: 16px;
  font-size: 0.98rem;
  overflow: hidden;
}

.st-key-edr_primary_navigation {
  border-bottom: 1px solid rgba(126, 150, 171, 0.16);
  margin: 0 0 .4rem;
  padding: 0 0.2rem 0.85rem;
}
[class*="st-key-radar_card_"] [data-testid="stButton"] button:not([kind="primary"]) {
  background: transparent;
  border: 0;
  box-shadow: none;
  min-height: 2.1rem;
  justify-content: flex-start;
  text-align: left;
  padding: 0;
  color: var(--text);
}
[class*="st-key-radar_card_"] [data-testid="stButton"] button p {
  font-size: 1.13rem;
  line-height: 1.45;
  font-weight: 620;
  text-align: left;
}
[class*="st-key-radar_card_"] [data-testid="stButton"] button:hover p {
  text-decoration: underline;
  text-underline-offset: 0.2em;
}
[class*="st-key-radar_card_"] [data-testid="stVerticalBlock"] {gap: .25rem;}
.st-key-edr_primary_navigation > div { gap: 0 !important; }
.st-key-edr_primary_view [data-testid="stButtonGroup"] {
  width: fit-content;
}
.st-key-edr_primary_view button {
  font-size: 1.06rem;
  font-weight: 590;
  letter-spacing: 0.02em;
  min-height: 2.9rem;
}

[data-testid="stAlert"], div[data-baseweb="notification"] {
  border: 1px solid var(--border);
  border-left-width: 3px;
  border-radius: 16px;
  box-shadow: none;
  font-size: 1rem;
}
div[data-baseweb="notification"][kind="info"] { background: var(--info-bg); color: var(--info); }
div[data-baseweb="notification"][kind="success"] { background: var(--success-bg); color: var(--success); }
div[data-baseweb="notification"][kind="warning"] { background: var(--warning-bg); color: var(--warning); }
div[data-baseweb="notification"][kind="error"] { background: var(--error-bg); color: var(--error); }
[data-testid="stStatusWidget"] {
  background: rgba(248, 251, 253, 0.88);
  border: 1px solid var(--border);
  border-radius: 18px;
}
[data-testid="stProgress"] [role="progressbar"] > div > div {
  background: var(--sea-mist-500);
}
[data-testid="stProgress"] [role="progressbar"] > div {
  background: rgba(126, 150, 171, 0.16);
}

.edr-empty-state {
  background: rgba(248, 251, 253, 0.66);
  border: 1px dashed var(--border-strong);
  border-radius: 18px;
  color: var(--text-secondary);
  font-size: var(--font-label);
  padding: 1.5rem 1.7rem;
}
.edr-doi {
  color: var(--sea-mist-700);
  font-family: Consolas, "SFMono-Regular", monospace;
}
a.edr-doi { text-decoration: none; text-underline-offset: 0.18em; }
a.edr-doi:hover { text-decoration: underline; }

@media (min-width: 1041px) {
  .edr-status-profile { flex-direction: row; align-items: baseline; gap: .6rem; }
  .edr-status-label, .edr-status-period { display: none; }
  .edr-status-name { font-size: 1.1rem; }
  .edr-status-metric { flex-direction: row; align-items: baseline; gap: .45rem; }
  .edr-status-metric b { font-size: 1.3rem; }
}
@media (max-width: 1180px) {
  [data-testid="stMainBlockContainer"] {
    max-width: 100%;
    padding: 1.25rem 1.8rem 4rem;
  }
  .st-key-edr_retrieval_panel { padding: 2.3rem 2rem 2rem; }
  [data-testid="stHorizontalBlock"] { flex-wrap: wrap; gap: 1.1rem; }
  [data-testid="column"] { flex: 1 1 17rem !important; min-width: min(100%, 17rem) !important; }
  .st-key-edr_page_header [data-testid="stHorizontalBlock"] { gap: 1.2rem; }
}

@media (max-width: 1040px) {
  .edr-status-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .edr-status-metric:nth-child(odd) { border-left: 0; padding-left: 0; }
  .st-key-edr_review_workspace [data-testid="stHorizontalBlock"]:has(.st-key-edr_review_reading_panel) {
    display: block;
  }
  .st-key-edr_review_workspace [data-testid="stHorizontalBlock"]:has(.st-key-edr_review_reading_panel) > [data-testid="stColumn"] {
    min-width: 100% !important;
    width: 100% !important;
  }
  .st-key-edr_review_reading_panel .edr-paper-card {
    max-height: none;
    min-height: 0;
    overflow: visible;
  }
  .st-key-edr_review_decision_panel {
    margin-top: 1rem;
    position: static;
  }
}

@media (max-width: 860px) {
  :root { --font-body: 20px; }
  [data-testid="stAppViewContainer"]::after {
    height: 9rem;
    opacity: 0.42;
    right: 1.2rem;
    top: 0.8rem;
    width: 9rem;
  }
  [data-testid="stMainBlockContainer"] { padding: 1.2rem 1rem 3.5rem; }
  .st-key-edr_page_header { margin-bottom: 1.1rem; padding: 0.3rem 0 1.1rem; }
  .st-key-edr_page_header [data-testid="stHeadingWithActionElements"] h1 {
    font-size: 2.1rem;
    padding-left: 3.1rem;
  }
  .st-key-edr_page_header h1::before { height: 2rem; width: 2rem; }
  .st-key-edr_retrieval_panel {
    border-radius: 24px;
    padding: 2rem 1.4rem 1.8rem;
  }
  .st-key-edr_retrieval_panel [data-testid="stHeadingWithActionElements"] h2 {
    font-size: 2.15rem !important;
  }
  .edr-flow {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    row-gap: 1.5rem;
  }
  .edr-flow::before { display: none; }
  .st-key-edr_verification_result { border-radius: 18px; padding: 1.25rem 1.1rem; }
  .edr-verification-stats { grid-template-columns: repeat(2, minmax(0, 1fr)); row-gap: 0.9rem; }
  .edr-verification-stats > div { padding: 0 0.7rem; }
  .edr-verification-stats > div:nth-child(2) { border-right: 0; }
  .edr-verification-stats > div:nth-child(3) { padding-left: 0; }
  .edr-section-meta { text-align: left; }
  [data-testid="column"] { flex-basis: 100% !important; min-width: 100% !important; }
  .edr-paper-card, .edr-library-card { padding: 1.6rem 1.3rem 1.7rem; }
  .edr-paper-title { font-size: 1.55rem; }
  .edr-library-title { font-size: 1.35rem; }
  [data-testid="stButton"] button,
  [data-testid="stFormSubmitButton"] button,
  [data-testid="stDownloadButton"] button,
  [data-testid="stLinkButton"] a { width: 100%; }
}
@media (max-width: 640px) {
  :root { --font-body: 16px; }
  [data-testid="stMainBlockContainer"] { padding: .6rem .9rem 2rem; }
  .st-key-edr_primary_navigation { padding: 0 0 .4rem; margin-bottom: 0; }
  .st-key-edr_primary_navigation [data-testid="stHorizontalBlock"] { gap: .3rem; }
  .st-key-edr_page_header { margin: 0; padding: 0; border: 0; }
  .st-key-edr_page_header [data-testid="stCaptionContainer"] { display: none; }
  .st-key-edr_page_header [data-testid="stHeadingWithActionElements"] h1 {
    font-size: 1.5rem !important; padding: .15rem 0 .15rem 1.9rem; white-space: nowrap;
  }
  .st-key-edr_page_header h1::before { height: 1.3rem; width: 1.3rem; }
  .st-key-edr_primary_view button, [class*="st-key-radar_section_"] button {
    font-size: .85rem; min-height: 2rem; padding: .2rem .45rem;
  }
  .st-key-edr_primary_view button p, [class*="st-key-radar_section_"] button p {
    font-size: .85rem;
  }
  .st-key-edr_primary_navigation [data-testid="stWidgetLabel"] { display: none; }
}
</style>
"""


def apply_app_theme() -> None:
    """Inject the side-effect-free visual theme for the current render."""
    st.markdown(MISTY_BLUE_CSS, unsafe_allow_html=True)


def render_page_header(
    *,
    eyebrow: str,
    title: str,
    subtitle: str | None,
    heading: Literal["title", "header"] = "header",
) -> None:
    """Render a thin open masthead: wordmark and one subtitle only."""
    with st.container(key="edr_page_header"):
        if eyebrow:
            st.markdown(
                f'<div class="edr-masthead-eyebrow">{escape(eyebrow)}</div>',
                unsafe_allow_html=True,
            )
        (st.title if heading == "title" else st.header)(title)
        if subtitle:
            st.caption(subtitle)


def render_section_header(
    title: str,
    description: str | None = None,
    meta: str | None = None,
) -> None:
    """Open section heading with optional right-aligned workflow meta."""
    if meta:
        title_col, meta_col = st.columns([0.72, 0.28], vertical_alignment="bottom")
        with title_col:
            st.header(title)
        with meta_col:
            st.markdown(
                f'<div class="edr-section-meta">{escape(meta)}</div>',
                unsafe_allow_html=True,
            )
    else:
        st.header(title)
    if description:
        st.markdown(
            f'<div class="edr-section-description">{escape(description)}</div>',
            unsafe_allow_html=True,
        )


def workflow_progress_html(
    stages: tuple[tuple[str, object, StatusTone], ...],
) -> str:
    """A quiet horizontal workflow line with counts, not dashboard cards."""
    nodes = "".join(
        '<div class="edr-flow-node edr-flow-node--'
        f'{tone}"><b>{value}</b><span>{escape(label)}</span></div>'
        for label, value, tone in stages
    )
    return (
        '<div class="edr-flow" aria-label="本次工作流状态">'
        f"{nodes}</div>"
    )


def render_status_badge(label: str, tone: StatusTone = "info") -> None:
    st.markdown(
        f'<div class="edr-badges">{status_badge_html(label, tone)}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(message: str) -> None:
    st.markdown(
        f'<div class="edr-empty-state">{escape(message)}</div>',
        unsafe_allow_html=True,
    )


def status_badge_html(label: str, tone: StatusTone = "info") -> str:
    return f'<span class="edr-badge edr-badge--{tone}">{escape(label)}</span>'
