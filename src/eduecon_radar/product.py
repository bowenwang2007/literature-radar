"""Product identity: one definition of what this tool is called.

The application is no longer a single education-economics filter.  It keeps
several research profiles over one shared paper store, proposes a judgement
with AI, records the human decision, and keeps every step auditable.  The name
therefore has to say "literature workbench for a research direction", not the
name of one field.

Nothing here is persisted: these strings are display-only, so renaming the
product can never invalidate stored data or a stored judgement.
"""

from __future__ import annotations

#: Latin wordmark shown in the masthead and as the browser page title.
PRODUCT_NAME = "Literature Radar"

#: The browser tab reads product + scope, so a bookmarked tab is self-describing.
PAGE_TITLE = f"{PRODUCT_NAME} · 研究方向文献工作台"

#: Chinese scope line under the wordmark.  Says the three things the user must
#: be able to trust at a glance: several profiles, AI proposes, a person decides.
PRODUCT_TAGLINE = "多研究方向文献发现与人工审核工作台"

#: One-line explanation used where the tool has to introduce itself (README,
#: exported files).  Deliberately plain: no marketing vocabulary.
PRODUCT_SUMMARY = (
    "本地、单用户的研究文献工作台：按研究方向检索期刊论文，AI 提出初筛建议，"
    "人工审核决定保留与分类，全部依据与决定可复核。"
)

__all__ = ["PAGE_TITLE", "PRODUCT_NAME", "PRODUCT_SUMMARY", "PRODUCT_TAGLINE"]
