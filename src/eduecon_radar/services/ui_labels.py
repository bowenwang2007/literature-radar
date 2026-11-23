"""Centralized user-visible Chinese labels.

The database and services keep stable English enums; only the UI layer maps
them to Chinese here. Never put these labels into persisted columns.
"""

from __future__ import annotations

AUTO_DECISION_ZH = {
    "relevant": "相关",
    "irrelevant": "无关",
    "uncertain": "不确定",
    "needs_abstract": "缺摘要",
    "insufficient_evidence": "证据不足",
    "needs_retry": "处理失败",
    "processing": "处理中",
}

HUMAN_DECISION_ZH = {
    "undecided": "未决定",
    "related": "相关",
    "irrelevant": "无关",
    "uncertain": "不确定",
}
HUMAN_DECISION_OPTIONS = tuple(HUMAN_DECISION_ZH.values())
HUMAN_DECISION_FROM_ZH = {value: key for key, value in HUMAN_DECISION_ZH.items()}

STATUS_ZH = {
    "pending": "待处理",
    "processing": "处理中",
    "abstract_ready": "摘要已获取",
    "needs_retry": "需要重试",
    "needs_manual_abstract": "需要人工补充摘要",
    "classification_ready": "待筛选",
    "classified": "已完成",
    "failed": "处理失败",
    "translated": "已翻译",
    "manual_title_preserved": "人工保留",
    "title_translation_pending": "待翻译",
    "title_translation_retry": "待重试",
    "abstract_translation_pending": "待翻译",
    "unavailable": "不可用",
    "ai_generated": "AI生成",
    "human_edited": "人工编辑",
}

STOP_REASON_ZH = {
    "requested_limit_reached": "已达到每刊检索上限",
    "provider_exhausted": "数据源已返回全部结果",
    "rate_limited": "数据源请求受限，稍后可重试",
    "timeout": "请求超时",
    "provider_error": "数据源错误",
    "provider_not_configured": "未配置该数据源",
    "unauthorized": "未授权",
    "forbidden": "无权限",
    "not_found": "未找到记录",
    "no_match": "无匹配记录",
    "matched_without_abstract": "匹配成功但无摘要",
    "low_confidence_match": "低置信度匹配被拒绝",
    "parse_error": "响应解析失败",
    "temporary_error": "暂时性错误",
    "permanent_error": "永久性错误",
    "success": "成功",
    "interrupted": "已中断",
    "provider_fused_skip": "数据源已熔断，本轮跳过",
    "identity_mismatch": "身份匹配失败",
    "invalid_content": "内容无效",
    "invalid_request": "请求无效",
    "retryable_provider_errors": "数据源暂时不可用，可稍后重试",
    "matched_records_without_abstract": "数据源匹配成功但未提供摘要",
    "provider_access_denied": "数据源无权限",
    "strategy_exhausted_no_match": "数据源未找到记录",
    "needs_manual_abstract": "需要人工补充摘要",
    "provider_budget_exhausted_pending": "本轮调用预算已用完",
    "interrupted_recoverable": "处理被中断，可恢复",
    "provider_response_parse_error": "数据源响应解析失败",
    "title_candidates_require_manual_review": "标题候选需人工复核",
    "non_research_front_matter_no_abstract_expected": "期刊前置内容通常不提供摘要",
}

ABSTRACT_STATUS_ZH = {
    "abstract_ready": "摘要已获取",
    "needs_retry": "需要重试",
    "needs_manual_abstract": "需要人工补充摘要",
    "failed": "处理失败",
}

PROVIDER_STATUS_ZH = {
    "success": "成功",
    "provider_not_configured": "未配置",
    "unauthorized": "未授权",
    "forbidden": "无权限",
    "rate_limited": "请求受限",
    "timeout": "超时",
    "not_found": "未找到",
    "no_match": "无匹配",
    "matched_without_abstract": "匹配但无摘要",
    "low_confidence_match": "低置信度拒绝",
    "parse_error": "解析失败",
    "temporary_error": "暂时性错误",
    "permanent_error": "永久性错误",
}

#: What the AI actually read before deciding.  This is the difference between
#: "the model judged the abstract" and "the model only saw a title", which a
#: reviewer has to know before accepting or overriding a suggestion.
SCREENING_BASIS_ZH = {
    "title_and_abstract": "依据标题与英文摘要",
    "title_only": "仅依据标题（无摘要）",
    "deterministic_metadata_rule": "依据期刊元数据规则",
    "historical_unknown": "历史判断（依据未记录）",
}

#: How sure the screening call said it was.  Shown next to the suggestion so a
#: low-confidence "相关" is reviewed differently from a high-confidence one.
CONFIDENCE_ZH = {
    "high": "高",
    "medium": "中",
    "low": "低",
}

#: Error classes recorded by the evidence validator, in plain language.
VALIDATION_STATUS_ZH = {
    "valid": "校验通过",
    "invalid": "校验未通过",
    "needs_recheck": "已复核",
    "recheck_failed": "复核未通过",
    "parse_failed": "响应解析失败",
}

#: Stored judgement dimensions.  The column names are stable, but their
#: *meaning* is profile content, so the label a reader sees is the profile's own
#: (see ``ProductProfile.dimensions.centrality_label``).
CENTRALITY_VALUE_ZH = {
    "central": "核心议题",
    "secondary": "次要涉及",
    "none": "未涉及",
}

RELEVANCE_VALUE_ZH = {
    "substantive": "有实质经济学内容",
    "incidental": "仅顺带提及",
    "none": "无明显经济学内容",
}

EVIDENCE_SUFFICIENCY_ZH = {
    "sufficient": "证据充分",
    "partial": "证据不完整",
    "insufficient": "证据不足",
}



def zh_auto_decision(decision: object) -> str:
    return AUTO_DECISION_ZH.get(str(decision or ""), str(decision or "未判断"))


def zh_human_decision(decision: object) -> str:
    return HUMAN_DECISION_ZH.get(str(decision or "undecided"), str(decision or ""))


def zh_status(status: object) -> str:
    return STATUS_ZH.get(str(status or ""), str(status or ""))


def zh_stop_reason(reason: object) -> str:
    value = str(reason or "")
    return STOP_REASON_ZH.get(value, "原因待核实" if value else "")


def zh_provider_status(status: object) -> str:
    return PROVIDER_STATUS_ZH.get(str(status or ""), str(status or ""))


def zh_screening_basis(basis: object) -> str:
    """What the screening call read.  Unknown basis is stated, never guessed."""
    value = str(basis or "").strip()
    if not value:
        return "历史判断（当时未记录依据）"
    return SCREENING_BASIS_ZH.get(value, "判断依据待核实")


def zh_confidence(confidence: object) -> str:
    value = str(confidence or "").strip()
    return CONFIDENCE_ZH.get(value, "")


def zh_validation_status(status: object) -> str:
    value = str(status or "").strip()
    return VALIDATION_STATUS_ZH.get(value, "")


ABSTRACT_SOURCE_ZH = {
    "crossref": "自动获取",
    "openalex": "自动获取",
    "semantic_scholar": "自动获取",
    "doaj": "自动获取",
    "openaire": "自动获取",
    "publisher": "自动获取",
    "doi": "自动获取",
    "title_fallback": "标题匹配获取",
    "manual": "人工粘贴",
    "pasted": "人工粘贴",
}


def zh_abstract_source(source: object) -> str:
    value = str(source or "").strip().lower()
    if value == "publisher_excerpt":
        return "期刊摘要节选"
    if not value:
        return "未记录"
    return ABSTRACT_SOURCE_ZH.get(value, "自动获取")


def human_decision_from_zh(label: str) -> str:
    return HUMAN_DECISION_FROM_ZH.get(str(label).strip(), "undecided")
